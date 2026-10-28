"""Predicción: pronóstico del champion para las 13 semanas siguientes a la última semana completa.

Se corre en Databricks como spark_python_task:
    python src/predict.py --schema tesis_dev

Busca qué versión de <catalogo>.<schema>.demanda_forecast tiene el alias champion y carga esa versión exacta, para que
la versión anotada sea la que pronosticó aunque el alias cambie a mitad de la corrida. Pronostica cada serie de la tabla
features en el último origen (el lunes de la última semana completa): las series activas, igual que en el backtest. El
pronóstico semanal se reparte en las ventanas de 31, 30 y 30 días con common.metrics, el mismo reparto con que se mide
el error. Agrega a <catalogo>.<schema>.predicciones una fila por serie y semana y otra por serie y ventana, en piezas;
las corridas anteriores se conservan.
"""
import argparse

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common import metrics, modelos
from common.features import JERARQUIA

COLUMNAS = [
    "fecha_corrida", "origen", "modelo", "modelo_version", "serie", *JERARQUIA,
    "granularidad", "periodo", "fecha_inicio", "fecha_fin", "pronostico",
]


def revisar_entrenamiento(entrenado_hasta: pd.Timestamp, origen: pd.Timestamp) -> None:
    """Se detiene si el modelo se entrenó con semanas posteriores al origen, porque ya vio lo que pronostica. Si se
    entrenó con menos semanas solo avisa: el modelo está atrasado, pero el pronóstico no usa información futura."""
    if entrenado_hasta > origen:
        raise ValueError(
            f"El champion se entrenó hasta la semana del {entrenado_hasta.date()}, posterior al origen "
            f"{origen.date()}: conoce semanas que tendría que pronosticar"
        )
    if entrenado_hasta < origen:
        print(f"aviso: el champion se entrenó hasta la semana del {entrenado_hasta.date()} y el origen es el "
              f"{origen.date()}; el pronóstico no aprovecha las semanas intermedias")


def tabla_predicciones(
    filas: pd.DataFrame,
    pronostico: np.ndarray,
    venta_total: pd.Series,
    fecha_corrida: pd.Timestamp,
    modelo: str,
    version: int,
) -> pd.DataFrame:
    """La tabla predicciones de una corrida: 13 filas semanales y 3 de ventana por serie.

    `filas` son las de features de un solo origen, y `pronostico` la venta semanal de cada una. `venta_total` es la
    venta diaria del total, indexada por fecha, para el perfil con que se reparten las semanas 5 y 9 en las ventanas.
    """
    origen = filas["origen"].iloc[0]
    texto = filas[["serie", *JERARQUIA]].astype(str)
    semanal = texto.assign(h=filas["h"].to_numpy(int), pronostico=np.asarray(pronostico, dtype=float))
    semanas = semanal.assign(
        granularidad="semana",
        periodo=semanal["h"],
        fecha_inicio=filas["semana_objetivo"].to_numpy(),
        fecha_fin=(filas["semana_objetivo"] + pd.Timedelta(days=6)).to_numpy(),
    )
    fechas = pd.DataFrame(metrics.fechas_ventanas(origen), columns=["fecha_inicio", "fecha_fin"])
    fechas["periodo"] = np.arange(1, len(fechas) + 1)
    ventanas = (
        metrics.pronostico_por_ventana(semanal, venta_total, origen)
        .rename_axis(["serie", "periodo"])
        .reset_index()
        .merge(texto.drop_duplicates("serie"), on="serie")
        .merge(fechas, on="periodo")
        .assign(granularidad="ventana")
    )
    tabla = pd.concat([semanas, ventanas], ignore_index=True).assign(
        fecha_corrida=fecha_corrida, origen=origen, modelo=modelo, modelo_version=version
    )
    return tabla.sort_values(["serie", "granularidad", "periodo"], ignore_index=True)[COLUMNAS]


def main():
    parser = argparse.ArgumentParser(description="Pronóstico del champion para las 13 semanas siguientes.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> str:
        return f"{args.catalogo}.{args.schema}.{nombre}"

    mlflow.set_registry_uri("databricks-uc")
    modelo_uc = tabla("demanda_forecast")
    champion = MlflowClient().get_model_version_by_alias(modelo_uc, "champion")
    version, modelo = int(champion.version), champion.tags["modelo"]

    features = spark.table(tabla("features"))
    ultimo = features.agg(F.max("origen")).first()[0]
    filas = features.where(F.col("origen") == ultimo).toPandas()
    for columna in ("origen", "semana_objetivo"):
        filas[columna] = pd.to_datetime(filas[columna])
    origen = pd.Timestamp(ultimo)
    revisar_entrenamiento(pd.Timestamp(champion.tags["entrenado_hasta"]), origen)
    print(f"{modelo_uc} versión {version} ({modelo}); origen {origen.date()}: {filas['serie'].nunique()} series")

    cargado = mlflow.pyfunc.load_model(f"models:/{modelo_uc}/{version}")
    pronostico = np.asarray(cargado.predict(modelos.a_entrada(filas)), dtype=float)

    venta = spark.table(tabla("ventas_bronze")).groupBy("fecha").agg(F.sum("unidades").alias("unidades")).toPandas()
    venta_total = pd.Series(venta["unidades"].to_numpy(), index=pd.to_datetime(venta["fecha"])).sort_index()

    fecha_corrida = pd.Timestamp.now().floor("s")
    predicciones = tabla_predicciones(filas, pronostico, venta_total, fecha_corrida, modelo, version)
    a_spark = predicciones.assign(**{c: predicciones[c].dt.date for c in ("origen", "fecha_inicio", "fecha_fin")})
    spark.createDataFrame(a_spark).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
        tabla("predicciones")
    )
    ventanas = predicciones[predicciones["granularidad"] == "ventana"].groupby("periodo")["pronostico"].sum()
    print(f"{tabla('predicciones')}: {len(predicciones):,} filas de la corrida {fecha_corrida}")
    print("pronóstico del total por ventana: " + ", ".join(f"{v:,.0f}" for v in ventanas))


if __name__ == "__main__":
    main()
