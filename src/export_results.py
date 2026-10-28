"""Export: arma results.json para la página con las tablas metricas_modelos, predicciones y panel_semanal.

Se corre en Databricks como spark_python_task:
    python src/export_results.py --schema tesis_dev

Usa la última corrida de predicciones del origen más reciente (si predict corrió varias veces con ese origen, solo la
última, sin mezclar) y la última corrida de metricas_modelos. Para el total, cada submarca y las 20 series de más venta
deja la venta real de las 52 semanas hasta el origen y el pronóstico de las 13 semanas y de las 3 ventanas.

La página es pública, así que la venta y el pronóstico van como índice: 100 es la venta semanal promedio del total en
esas 52 semanas, y el mismo factor se aplica a todas las series y al MAE y el RMSE. La tabla predicciones sigue en
piezas. El JSON se valida con common.results_schema antes de escribirlo en
/Volumes/<catalogo>/<schema>/raw/output/results.json: si no cumple el esquema, el archivo anterior queda intacto.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
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


def ultima_corrida(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Las filas de la última corrida del origen más reciente."""
    del_origen = predicciones[predicciones["origen"] == predicciones["origen"].max()]
    return del_origen[del_origen["fecha_corrida"] == del_origen["fecha_corrida"].max()]


def _fecha(valor) -> str:
    return pd.Timestamp(valor).date().isoformat()


def _fecha_hora(valor) -> str:
    return pd.Timestamp(valor).isoformat(timespec="seconds")


def _indice(valores: pd.Series, factor: float) -> list[float]:
    return [round(float(v) * factor, DECIMALES_INDICE) for v in valores]


def _metrica(valor, factor: float = 1.0) -> float | None:
    return None if pd.isna(valor) else round(float(valor) * factor, DECIMALES_METRICA)


def construir_resultados(
    metricas: pd.DataFrame,
    predicciones: pd.DataFrame,
    panel: pd.DataFrame,
    generado: pd.Timestamp,
    top: int = SERIES_TOP,
) -> dict:
    """El contenido de results.json (ver common.results_schema).

    `metricas` y `predicciones` son las tablas metricas_modelos y predicciones con todas sus corridas, y `panel` la
    venta semanal de cada serie (serie, semana, submarca y venta) de panel_semanal. El total y las submarcas suman la
    venta real de todas las series, también las que no se pronostican, como en la métrica del backtest.
    """
    corrida = ultima_corrida(predicciones)
    entrenamiento = metricas[metricas["fecha_corrida"] == metricas["fecha_corrida"].max()].sort_values(CRITERIO)
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


def escribir(resultados: dict, ruta: Path) -> None:
    """Valida `resultados` y solo entonces escribe el archivo: si no cumple el esquema, el anterior queda intacto."""
    results_schema.validar(resultados)
    texto = json.dumps(resultados, ensure_ascii=False, indent=2, allow_nan=False)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Arma results.json para la página con las tablas de la corrida.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> DataFrame:
        return spark.table(f"{args.catalogo}.{args.schema}.{nombre}")

    columnas = ["fecha_corrida", "modelo", *METRICAS, "es_mejor", "origenes", "primer_origen", "ultimo_origen"]
    metricas = tabla("metricas_modelos").select(*columnas).toPandas()
    predicciones = tabla("predicciones").toPandas()
    panel = tabla("panel_semanal").select("serie", "semana", "submarca", "venta").toPandas()
    for df, fechas in ((predicciones, ("origen", "fecha_inicio", "fecha_fin")), (panel, ("semana",))):
        for columna in fechas:
            df[columna] = pd.to_datetime(df[columna])

    resultados = construir_resultados(metricas, predicciones, panel, pd.Timestamp.now().floor("s"))
    ruta = Path(f"/Volumes/{args.catalogo}/{args.schema}/raw/output/results.json")
    escribir(resultados, ruta)
    champion, corrida = resultados["champion"], resultados["corrida"]
    print(
        f"{ruta}: {champion['modelo']} versión {champion['version']}, origen {corrida['origen']}, predicción del "
        f"{corrida['fecha_prediccion']}, {len(resultados['series'])} series"
    )


if __name__ == "__main__":
    main()
