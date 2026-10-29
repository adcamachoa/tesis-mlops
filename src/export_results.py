"""Export: arma los datos de la página con las tablas metricas_modelos, predicciones, panel_semanal y la venta diaria.

Se corre en Databricks como spark_python_task, al final de cada corrida semanal:
    python src/export_results.py --schema tesis_dev

Arma un results.json por cada origen de predicciones, con la última corrida de ese origen (si predict corrió varias
veces con un origen, solo la última, sin mezclar), y manifest.json con la lista, en el formato 1 de tesis-mlops-web. En
cada corrida vuelve a generar todos los orígenes desde las tablas.

El pronóstico de cada origen se arma solo con lo que se sabía hasta su datos_hasta: la venta real de las 52 semanas
hasta el origen, las métricas de la corrida de train que registró la versión que pronosticó (nunca las de un train
posterior) y, en esas 52 semanas, el pronóstico que estaba vigente: el de la última corrida del origen anterior más
reciente que la cubría. La versión se liga con su train por su run en Unity Catalog, que es el run_id_modelo de su
modelo en metricas_modelos; por eso no se borran versiones del modelo. Después del origen solo entra la evaluación: la
venta real de las semanas y de las ventanas pronosticadas que ya cerraron hasta real_hasta, el último domingo con datos.
En las semanas sin reentrenamiento, el pronóstico publicado no cambia: solo se agregan esas semanas reales.

Publica el árbol total → submarca → departamento → categoría → serie (common.arbol), podado en el índice mínimo, con
nodos `otras` para lo chico. Cada nodo trae la venta real de las 52 semanas, el pronóstico de las 13 semanas y de las 3
ventanas, su venta real, el pronóstico vigente en las semanas pasadas y el formato estándar de cada modelo en su
backtest, sumado al nodo: sale de las predicciones del backtest que train.py guarda en el run de cada modelo, así que
tampoco se borran esos runs. La página es pública, así que todo va como índice: 100 es la venta semanal promedio del
total en las 52 semanas hasta el origen, y el mismo factor se aplica a todos los nodos, a todos los campos y al MAE y el
RMSE. La tabla predicciones sigue en piezas.

Todo se valida con common.results_schema antes de escribir en /Volumes/<catalogo>/<schema>/raw/output/:
datos/<origen>/results.json (compacto), datos/manifest.json y results.json, que es el mismo archivo del origen más
reciente. Si algo no cumple el esquema, no se escribe nada y lo anterior queda intacto.
"""
import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession

from common import arbol, metrics, results_schema
from common.backtest import CRITERIO
from common.features import HORIZONTE, SERIE
from common.metrics import METRICAS, VENTANAS_DIAS
from train import ARTEFACTO_BACKTEST, venta_diaria_serie

SEMANAS_REALES = 52
DECIMALES_INDICE = 1  # más fino, una pieza se vería como un escalón fijo y se podría reconstruir la escala
DECIMALES_METRICA = 4
EN_PIEZAS = ("mae", "rmse")  # se escalan con el mismo factor que la venta
DATOS = "datos"  # la carpeta que publica pages.yml
RESULTS = "results.json"


def ultimas_corridas(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Las filas de la última corrida de cada origen."""
    ultima = predicciones.groupby("origen")["fecha_corrida"].transform("max")
    return predicciones[predicciones["fecha_corrida"] == ultima]


def entrenamiento_de(metricas: pd.DataFrame, version: int, run_version: str) -> pd.DataFrame:
    """Las filas de metricas_modelos de la corrida de train que registró `version`, ordenadas por el criterio del
    champion: la que tiene como run de uno de sus modelos a `run_version`, el run de la versión en Unity Catalog."""
    train = metricas.loc[metricas["run_id_modelo"] == run_version, "run_id"].unique()
    if len(train) != 1:
        raise ValueError(
            f"metricas_modelos no tiene la corrida de train que registró la versión {version} (run {run_version}): "
            f"sin sus métricas no se exporta nada"
        )
    return metricas[metricas["run_id"] == train[0]].sort_values(CRITERIO)


def _fecha(valor) -> str:
    return pd.Timestamp(valor).date().isoformat()


def _fecha_hora(valor) -> str:
    return pd.Timestamp(valor).isoformat(timespec="seconds")


def _indice(valores: pd.Series, factor: float) -> list[float]:
    return [round(float(v) * factor, DECIMALES_INDICE) for v in valores]


def _metrica(valor, factor: float = 1.0) -> float | None:
    return None if pd.isna(valor) else round(float(valor) * factor, DECIMALES_METRICA)


def pronostico_vigente(anteriores: pd.DataFrame) -> pd.DataFrame:
    """Las filas semanales del pronóstico que estaba vigente en cada semana: de las últimas corridas de `anteriores`,
    las del origen más reciente cuyo horizonte cubre esa semana."""
    semanal = anteriores[anteriores["granularidad"] == "semana"]
    return semanal[semanal["origen"] == semanal.groupby("fecha_inicio")["origen"].transform("max")]


def metricas_por_nodo(ventanas_backtest: dict[str, pd.DataFrame], nodos: list[arbol.Nodo]) -> dict[str, list[dict]]:
    """Por nodo, el formato estándar de cada modelo en su backtest, con la venta y el pronóstico de cada origen y
    ventana sumados al nodo. `ventanas_backtest` trae, por modelo y en el orden de `metricas`, el resultado de
    metrics.a_ventanas: todas las series, con las que no se pronosticaban en pronóstico 0."""
    por_nodo = {nodo.id: [] for nodo in nodos}
    for modelo, ventanas in ventanas_backtest.items():
        for nodo in nodos:
            del_nodo = ventanas if nodo.nivel == "total" else ventanas[ventanas["serie"].isin(nodo.series)]
            sumadas = del_nodo.groupby(["origen", "ventana"], as_index=False)[["real", "pronostico"]].sum()
            estandar = metrics.formato_estandar(sumadas.assign(serie=nodo.id))
            por_nodo[nodo.id].append(
                {"modelo": modelo, **{m: _metrica(estandar[m]) for m in results_schema.METRICAS_NODO}}
            )
    return por_nodo


def construir_resultados(
    entrenamiento: pd.DataFrame,
    corrida: pd.DataFrame,
    anteriores: pd.DataFrame,
    panel: pd.DataFrame,
    venta_diaria: pd.DataFrame,
    ventanas_backtest: dict[str, pd.DataFrame],
    generado: pd.Timestamp,
    minimo: float = arbol.INDICE_MINIMO,
) -> dict:
    """El results.json de una corrida de predict (ver common.results_schema).

    `corrida` son las filas de predicciones de un solo origen y una sola fecha_corrida, `anteriores` las de las últimas
    corridas de los orígenes anteriores, y `entrenamiento` las de metricas_modelos de la corrida de train que registró
    la versión que pronosticó (ver `entrenamiento_de`). `panel` es la venta semanal de cada serie de panel_semanal, con
    su jerarquía, y `venta_diaria` la de cada serie por día (serie, fecha y unidades). `ventanas_backtest` trae, por
    modelo, las ventanas de su backtest en ese train (ver `metricas_por_nodo`).

    El árbol (common.arbol) publica los nodos cuyo índice promedio semanal de venta real en las 52 semanas llega a
    `minimo`. Cada nodo suma la venta real de todas sus series, también las que no se pronostican, como en la métrica
    del backtest. La venta real llega hasta la última semana del panel, `real_hasta`:
    después del origen solo entran las semanas y las ventanas que ya cerraron.
    """
    origen, champion = corrida["origen"].iloc[0], corrida["modelo"].iloc[0]
    real_hasta = panel["semana"].max() + pd.Timedelta(days=6)
    semanas = pd.date_range(end=origen, periods=SEMANAS_REALES, freq="7D")
    real = panel[panel["semana"].isin(semanas)]
    factor = results_schema.BASE_INDICE * SEMANAS_REALES / real["venta"].sum()
    semanal = corrida[corrida["granularidad"] == "semana"]
    ventanas = corrida[corrida["granularidad"] == "ventana"]
    semanas_pronostico = pd.DatetimeIndex(semanal.sort_values("periodo")["fecha_inicio"].unique())
    vigente = pronostico_vigente(anteriores)
    real_ventanas = metrics.real_por_ventana(venta_diaria, origen).reset_index()
    # qué valores ya se conocen: las semanas y ventanas cerradas hasta real_hasta, y las semanas que un pronóstico
    # anterior cubría
    cerradas = semanas_pronostico + pd.Timedelta(days=6) <= real_hasta
    ventanas_cerradas = [hasta <= real_hasta for _, hasta in metrics.fechas_ventanas(origen)]
    cubiertas = semanas.isin(vigente["fecha_inicio"])

    # el peso de cada serie para publicar su nodo: el índice promedio semanal de su venta real
    pesos = real.groupby("serie")["venta"].sum() * factor / SEMANAS_REALES
    nodos = arbol.construir(panel[["serie", *SERIE]], pesos, minimo)
    metricas_nodo = metricas_por_nodo(ventanas_backtest, nodos)

    def serie(nodo: arbol.Nodo) -> dict:
        """El nodo con sus valores como índice."""

        def suma(df: pd.DataFrame, valor: str, por: str, claves, conocidas=None) -> list[float | None]:
            """Con `conocidas`, nulo donde el valor todavía no se conoce."""
            filas = df if nodo.nivel == "total" else df[df["serie"].isin(nodo.series)]
            valores = _indice(filas.groupby(por)[valor].sum().reindex(claves, fill_value=0), factor)
            if conocidas is None:
                return valores
            return [v if conocida else None for v, conocida in zip(valores, conocidas, strict=True)]

        return {
            "id": nodo.id,
            "padre": nodo.padre,
            "nivel": nodo.nivel,
            "nombre": nodo.nombre,
            "etiqueta": nodo.etiqueta,
            "real": suma(real, "venta", "semana", semanas),
            "pronostico": suma(semanal, "pronostico", "periodo", range(1, HORIZONTE + 1)),
            "ventanas": suma(ventanas, "pronostico", "periodo", range(1, len(VENTANAS_DIAS) + 1)),
            "real_horizonte": suma(panel, "venta", "semana", semanas_pronostico, cerradas),
            "pronostico_anterior": suma(vigente, "pronostico", "fecha_inicio", semanas, cubiertas),
            "ventanas_real": suma(real_ventanas, "real", "ventana", range(1, len(VENTANAS_DIAS) + 1),
                                  ventanas_cerradas),
            "metricas": metricas_nodo[nodo.id],
        }

    primera = entrenamiento.iloc[0]
    return {
        "version_esquema": results_schema.VERSION,
        "generado": _fecha_hora(generado),
        "escala": results_schema.ESCALA,
        "indice_minimo": minimo,
        "real_hasta": _fecha(real_hasta),
        "corrida": {
            "fecha_prediccion": _fecha_hora(corrida["fecha_corrida"].iloc[0]),
            "fecha_entrenamiento": _fecha_hora(primera["fecha_corrida"]),
            "origen": _fecha(origen),
            "datos_hasta": _fecha(origen + pd.Timedelta(days=6)),
        },
        "champion": {"modelo": champion, "version": int(corrida["modelo_version"].iloc[0])},
        "backtest": {
            "origenes": int(primera["origenes"]),
            "primer_origen": _fecha(primera["primer_origen"]),
            "ultimo_origen": _fecha(primera["ultimo_origen"]),
        },
        "metricas": [
            {
                "modelo": fila["modelo"],
                **{m: _metrica(fila[m], factor if m in EN_PIEZAS else 1.0) for m in METRICAS},
                "es_mejor": bool(fila["es_mejor"]),
                "es_champion": fila["modelo"] == champion,
            }
            for _, fila in entrenamiento.iterrows()
        ],
        "semanas": {"real": [_fecha(s) for s in semanas], "pronostico": [_fecha(s) for s in semanas_pronostico]},
        "ventanas": [
            {"numero": int(v["periodo"]), "desde": _fecha(v["fecha_inicio"]), "hasta": _fecha(v["fecha_fin"]),
             "dias": (v["fecha_fin"] - v["fecha_inicio"]).days + 1}
            for _, v in ventanas.drop_duplicates("periodo").sort_values("periodo").iterrows()
        ],
        "series": [serie(nodo) for nodo in nodos],
    }


def construir_publicacion(
    metricas: pd.DataFrame,
    predicciones: pd.DataFrame,
    panel: pd.DataFrame,
    venta_diaria: pd.DataFrame,
    runs: dict[int, str],
    backtests: dict[str, pd.DataFrame],
    generado: pd.Timestamp,
    minimo: float = arbol.INDICE_MINIMO,
) -> tuple[dict, dict[str, dict]]:
    """manifest.json y los results.json por su ruta relativa al manifest: uno por origen de predicciones, con su última
    corrida, las métricas del train de su versión y el pronóstico vigente de los orígenes anteriores.

    `metricas` y `predicciones` son las tablas metricas_modelos y predicciones con todas sus corridas, `panel` y
    `venta_diaria` la venta semanal y diaria por serie, `runs` el run en Unity Catalog de cada versión que pronosticó y
    `backtests` las predicciones del backtest de cada modelo por su run_id_modelo (el artefacto que deja train.py).
    """
    ultimas = ultimas_corridas(predicciones)
    archivos, pronosticos, ventanas_de = {}, [], {}
    for origen, corrida in ultimas.groupby("origen"):
        version = int(corrida["modelo_version"].iloc[0])
        entrenamiento = entrenamiento_de(metricas, version, runs[version])
        for run in entrenamiento["run_id_modelo"]:  # un mismo train puede servir a varios orígenes
            if run not in ventanas_de:
                if run not in backtests:
                    raise ValueError(f"falta el backtest del run {run} (artefacto {ARTEFACTO_BACKTEST}): sin él no "
                                     f"hay métricas por nodo")
                ventanas_de[run] = metrics.a_ventanas(backtests[run], venta_diaria)
        ventanas_backtest = {f.modelo: ventanas_de[f.run_id_modelo] for f in entrenamiento.itertuples()}
        anteriores = ultimas[ultimas["origen"] < origen]
        resultados = construir_resultados(entrenamiento, corrida, anteriores, panel, venta_diaria, ventanas_backtest,
                                          generado, minimo)
        ruta = f"{_fecha(origen)}/{RESULTS}"
        archivos[ruta] = resultados
        pronosticos.append(
            {"id": _fecha(origen), "ruta": ruta, "version_esquema": resultados["version_esquema"],
             "origen": resultados["corrida"]["origen"], "fecha_prediccion": resultados["corrida"]["fecha_prediccion"]}
        )
    manifest = {
        "version_manifest": results_schema.VERSION_MANIFEST,
        "generado": _fecha_hora(generado),
        "pronosticos": pronosticos[::-1],  # del origen más reciente al más antiguo, como los ordena la página
    }
    return manifest, archivos


def _texto(contenido: dict, compacto: bool = False) -> str:
    """El JSON, compacto en los results.json: con sangría pesaría el doble (cada número en su línea)."""
    formato = {"separators": (",", ":")} if compacto else {"indent": 2}
    return json.dumps(contenido, ensure_ascii=False, allow_nan=False, **formato)


def escribir(manifest: dict, archivos: dict[str, dict], salida: Path) -> None:
    """Valida el manifest y cada results.json, y solo entonces escribe en `salida`: datos/<ruta> de cada uno, después
    datos/manifest.json y al final results.json, el del origen más reciente. Si algo no cumple el esquema, no escribe
    nada y lo anterior queda intacto."""
    results_schema.validar_manifest(manifest, archivos)
    reciente = max(manifest["pronosticos"], key=lambda entrada: entrada["origen"])["ruta"]
    textos = {f"{DATOS}/{ruta}": _texto(resultados, compacto=True) for ruta, resultados in archivos.items()}
    textos[f"{DATOS}/{results_schema.MANIFEST}"] = _texto(manifest)
    textos[RESULTS] = textos[f"{DATOS}/{reciente}"]
    for ruta, texto in textos.items():
        destino = salida / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")


def run_de_version(cliente: MlflowClient, modelo_uc: str, version: int) -> str:
    """El run de MLflow donde train.py registró `version`: el de su modelo en la corrida de train."""
    try:
        return cliente.get_model_version(modelo_uc, str(version)).run_id
    except MlflowException as error:
        raise ValueError(
            f"{modelo_uc} no tiene la versión {version}, que pronosticó un origen de predicciones: sin ella no se "
            f"encuentran sus métricas. Las versiones del modelo no se borran."
        ) from error


def backtest_de_run(run_id: str) -> pd.DataFrame:
    """Las predicciones del backtest que train.py guardó en el run de un modelo (serie, origen, h y pronostico)."""
    with tempfile.TemporaryDirectory() as carpeta:
        try:
            ruta = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=ARTEFACTO_BACKTEST, dst_path=carpeta
            )
        except MlflowException as error:
            raise ValueError(
                f"el run {run_id} no tiene {ARTEFACTO_BACKTEST}: sin él no hay métricas por nodo"
            ) from error
        backtest = pd.read_parquet(ruta)
    return backtest.assign(origen=pd.to_datetime(backtest["origen"]), serie=backtest["serie"].astype(str))


def main():
    parser = argparse.ArgumentParser(description="Arma los datos de la página con las tablas de las corridas.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> DataFrame:
        return spark.table(f"{args.catalogo}.{args.schema}.{nombre}")

    columnas = [
        "fecha_corrida", "run_id", "run_id_modelo", "modelo", *METRICAS, "es_mejor", "origenes", "primer_origen",
        "ultimo_origen",
    ]
    metricas = tabla("metricas_modelos").select(*columnas).toPandas()
    predicciones = tabla("predicciones").toPandas()
    panel = tabla("panel_semanal").select("serie", "semana", *SERIE, "venta").toPandas()
    venta_diaria = venta_diaria_serie(tabla("ventas_bronze"), tabla("productos_bronze")).toPandas()
    for df, fechas in (
        (predicciones, ("origen", "fecha_inicio", "fecha_fin")), (panel, ("semana",)), (venta_diaria, ("fecha",))
    ):
        for columna in fechas:
            df[columna] = pd.to_datetime(df[columna])

    mlflow.set_registry_uri("databricks-uc")
    cliente, modelo_uc = MlflowClient(), f"{args.catalogo}.{args.schema}.demanda_forecast"
    versiones = ultimas_corridas(predicciones)["modelo_version"].unique()
    runs = {int(v): run_de_version(cliente, modelo_uc, int(v)) for v in versiones}
    # los backtests de cada modelo en los trains que registraron esas versiones, para las métricas por nodo
    modelos_de = {run for version, run_version in runs.items()
                  for run in entrenamiento_de(metricas, version, run_version)["run_id_modelo"]}
    backtests = {run: backtest_de_run(run) for run in sorted(modelos_de)}
    manifest, archivos = construir_publicacion(
        metricas, predicciones, panel, venta_diaria, runs, backtests, pd.Timestamp.now().floor("s")
    )
    salida = Path(f"/Volumes/{args.catalogo}/{args.schema}/raw/output")
    escribir(manifest, archivos, salida)
    real_hasta = next(iter(archivos.values()))["real_hasta"]
    print(f"{salida / DATOS}: {results_schema.MANIFEST} con {len(archivos)} pronóstico(s), venta real hasta el "
          f"{real_hasta}; {salida / RESULTS} es el del origen más reciente")
    for entrada in manifest["pronosticos"]:
        resultados = archivos[entrada["ruta"]]
        champion, corrida = resultados["champion"], resultados["corrida"]
        semanas = sum(v is not None for v in resultados["series"][0]["real_horizonte"])
        por_nivel = pd.Series([s["nivel"] for s in resultados["series"]]).value_counts()
        print(f"  origen {corrida['origen']}: {champion['modelo']} versión {champion['version']}, predicción del "
              f"{corrida['fecha_prediccion']}, entrenamiento del {corrida['fecha_entrenamiento']}, "
              f"{len(resultados['series'])} nodos ({', '.join(f'{n} {k}' for k, n in por_nivel.items())}), "
              f"{semanas} de {HORIZONTE} semanas con venta real")


if __name__ == "__main__":
    main()
