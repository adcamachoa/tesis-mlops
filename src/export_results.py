"""Export: arma los datos de la página con las tablas metricas_modelos, predicciones y panel_semanal.

Se corre en Databricks como spark_python_task:
    python src/export_results.py --schema tesis_dev

Arma un results.json por cada origen de predicciones, con la última corrida de ese origen (si predict corrió varias
veces con un origen, solo la última, sin mezclar), y manifest.json con la lista, en el formato 1 de tesis-mlops-web. En
cada corrida vuelve a generar todos los orígenes desde las tablas.

Cada origen se arma solo con lo que se sabía hasta su datos_hasta: la venta real de las 52 semanas hasta el origen y las
métricas de la corrida de train que registró la versión que pronosticó, nunca las de un train posterior. La versión se
liga con su train por su run en Unity Catalog, que es el run_id_modelo de su modelo en metricas_modelos; por eso no se
borran versiones del modelo. Para el total, cada submarca y las 20 series de más venta deja la venta real de las 52
semanas y el pronóstico de las 13 semanas y de las 3 ventanas.

La página es pública, así que la venta y el pronóstico van como índice: 100 es la venta semanal promedio del total en
esas 52 semanas, y el mismo factor se aplica a todas las series y al MAE y el RMSE. La tabla predicciones sigue en
piezas.

Todo se valida con common.results_schema antes de escribir en /Volumes/<catalogo>/<schema>/raw/output/:
datos/<origen>/results.json, datos/manifest.json y results.json, que es el mismo archivo del origen más reciente. Si
algo no cumple el esquema, no se escribe nada y lo anterior queda intacto.
"""
import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession

from common import results_schema
from common.backtest import CRITERIO
from common.features import HORIZONTE
from common.metrics import METRICAS, VENTANAS_DIAS

SEMANAS_REALES = 52
SERIES_TOP = 20
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


def construir_resultados(
    entrenamiento: pd.DataFrame,
    corrida: pd.DataFrame,
    panel: pd.DataFrame,
    generado: pd.Timestamp,
    top: int = SERIES_TOP,
) -> dict:
    """El results.json de una corrida de predict (ver common.results_schema).

    `corrida` son las filas de predicciones de un solo origen y una sola fecha_corrida, y `entrenamiento` las de
    metricas_modelos de la corrida de train que registró la versión que pronosticó (ver `entrenamiento_de`). `panel` es
    la venta semanal de cada serie (serie, semana, submarca y venta) de panel_semanal: solo cuentan las 52 semanas hasta
    el origen. El total y las submarcas suman la venta real de todas las series, también las que no se pronostican, como
    en la métrica del backtest.
    """
    origen, champion = corrida["origen"].iloc[0], corrida["modelo"].iloc[0]
    semanas = pd.date_range(end=origen, periods=SEMANAS_REALES, freq="7D")
    real = panel[panel["semana"].isin(semanas)]
    factor = results_schema.BASE_INDICE * SEMANAS_REALES / real["venta"].sum()
    semanal = corrida[corrida["granularidad"] == "semana"]
    ventanas = corrida[corrida["granularidad"] == "ventana"]

    def serie(nivel: str, nombre: str, columna: str | None = None) -> dict:
        """El total (sin `columna`) o las filas donde `columna` vale `nombre`, como índice."""

        def suma(df: pd.DataFrame, valor: str, por: str, claves) -> list[float]:
            filas = df if columna is None else df[df[columna] == nombre]
            return _indice(filas.groupby(por)[valor].sum().reindex(claves, fill_value=0), factor)

        return {
            "nivel": nivel,
            "nombre": nombre,
            "real": suma(real, "venta", "semana", semanas),
            "pronostico": suma(semanal, "pronostico", "periodo", range(1, HORIZONTE + 1)),
            "ventanas": suma(ventanas, "pronostico", "periodo", range(1, len(VENTANAS_DIAS) + 1)),
        }

    # las de más venta real en las 52 semanas, entre las que tienen pronóstico
    venta_serie = real.groupby("serie")["venta"].sum()
    principales = venta_serie[venta_serie.index.isin(corrida["serie"])].sort_values(ascending=False, kind="stable")
    primera = entrenamiento.iloc[0]
    return {
        "version_esquema": results_schema.VERSION,
        "generado": _fecha_hora(generado),
        "escala": results_schema.ESCALA,
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
        "semanas": {
            "real": [_fecha(s) for s in semanas],
            "pronostico": [_fecha(s) for s in semanal.sort_values("periodo")["fecha_inicio"].unique()],
        },
        "ventanas": [
            {"numero": int(v["periodo"]), "desde": _fecha(v["fecha_inicio"]), "hasta": _fecha(v["fecha_fin"]),
             "dias": (v["fecha_fin"] - v["fecha_inicio"]).days + 1}
            for _, v in ventanas.drop_duplicates("periodo").sort_values("periodo").iterrows()
        ],
        "series": [
            serie("total", "Total"),
            *(serie("submarca", s, "submarca") for s in sorted(corrida["submarca"].unique())),
            *(serie("serie", s, "serie") for s in principales.index[:top]),
        ],
    }


def construir_publicacion(
    metricas: pd.DataFrame,
    predicciones: pd.DataFrame,
    panel: pd.DataFrame,
    runs: dict[int, str],
    generado: pd.Timestamp,
    top: int = SERIES_TOP,
) -> tuple[dict, dict[str, dict]]:
    """manifest.json y los results.json por su ruta relativa al manifest: uno por origen de predicciones, con su última
    corrida y las métricas del train de su versión.

    `metricas` y `predicciones` son las tablas metricas_modelos y predicciones con todas sus corridas, y `runs` el run
    en Unity Catalog de cada versión que pronosticó.
    """
    archivos, pronosticos = {}, []
    for origen, corrida in ultimas_corridas(predicciones).groupby("origen"):
        version = int(corrida["modelo_version"].iloc[0])
        entrenamiento = entrenamiento_de(metricas, version, runs[version])
        resultados = construir_resultados(entrenamiento, corrida, panel, generado, top)
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


def _texto(contenido: dict) -> str:
    return json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False)


def escribir(manifest: dict, archivos: dict[str, dict], salida: Path) -> None:
    """Valida el manifest y cada results.json, y solo entonces escribe en `salida`: datos/<ruta> de cada uno, después
    datos/manifest.json y al final results.json, el del origen más reciente. Si algo no cumple el esquema, no escribe
    nada y lo anterior queda intacto."""
    results_schema.validar_manifest(manifest, archivos)
    reciente = max(manifest["pronosticos"], key=lambda entrada: entrada["origen"])["ruta"]
    textos = {f"{DATOS}/{ruta}": _texto(resultados) for ruta, resultados in archivos.items()}
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
    panel = tabla("panel_semanal").select("serie", "semana", "submarca", "venta").toPandas()
    for df, fechas in ((predicciones, ("origen", "fecha_inicio", "fecha_fin")), (panel, ("semana",))):
        for columna in fechas:
            df[columna] = pd.to_datetime(df[columna])

    mlflow.set_registry_uri("databricks-uc")
    cliente, modelo_uc = MlflowClient(), f"{args.catalogo}.{args.schema}.demanda_forecast"
    versiones = ultimas_corridas(predicciones)["modelo_version"].unique()
    runs = {int(v): run_de_version(cliente, modelo_uc, int(v)) for v in versiones}
    manifest, archivos = construir_publicacion(metricas, predicciones, panel, runs, pd.Timestamp.now().floor("s"))
    salida = Path(f"/Volumes/{args.catalogo}/{args.schema}/raw/output")
    escribir(manifest, archivos, salida)
    print(f"{salida / DATOS}: {results_schema.MANIFEST} con {len(archivos)} pronóstico(s); {salida / RESULTS} es el "
          f"del origen más reciente")
    for entrada in manifest["pronosticos"]:
        resultados = archivos[entrada["ruta"]]
        champion, corrida = resultados["champion"], resultados["corrida"]
        print(f"  origen {corrida['origen']}: {champion['modelo']} versión {champion['version']}, predicción del "
              f"{corrida['fecha_prediccion']}, entrenamiento del {corrida['fecha_entrenamiento']}, "
              f"{len(resultados['series'])} series")


if __name__ == "__main__":
    main()
