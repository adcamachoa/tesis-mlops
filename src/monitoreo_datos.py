"""Monitoreo de datos: drift del último origen contra la ventana de entrenamiento, antes de entrenar.

Se corre en Databricks como spark_python_task, entre features y train:
    python src/monitoreo_datos.py --schema tesis_dev
    python src/monitoreo_datos.py --schema tesis_dev --aceptar-drift 2027-01-11   # sigue con un cambio ya revisado

Revisa el último origen de features con common.monitoreo:
- el PSI y los nulos de las variables principales, contra la ventana de entrenamiento del champion;
- la cobertura de la semana del origen: SKU con stock (de panel_semanal) y filas de venta (de ventas_bronze);
- que el stock llegue a la última semana completa con venta.
Agrega las revisiones a <catalogo>.<schema>.monitoreo_datos, que es de solo agregar. Solo después, si alguna es extrema
y el cambio no se aceptó para este origen, se detiene, y el job no entrena ni pronostica. Los umbrales están en
CLAUDE.md. Imprime PSI, proporciones, fechas y nombres de variables, nunca piezas.
"""
import argparse

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common import monitoreo
from predict import proteger_solo_agregar

COLUMNAS = ["fecha_corrida", *monitoreo.COLUMNAS_DATOS, "aceptado"]


def leer_aceptado(texto: str) -> pd.Timestamp | None:
    """La fecha del parámetro aceptar_drift, o None si viene vacío, como en cada corrida normal."""
    return pd.Timestamp(texto) if texto.strip() else None


def main():
    parser = argparse.ArgumentParser(description="Drift de los datos del último origen; se detiene si es extremo.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    parser.add_argument("--aceptar-drift", default="", help="fecha del origen cuyo cambio extremo ya se revisó")
    args = parser.parse_args()
    aceptado = leer_aceptado(args.aceptar_drift)

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> str:
        return f"{args.catalogo}.{args.schema}.{nombre}"

    columnas = ["serie", "origen", "semana_objetivo", "y", *monitoreo.VARIABLES]
    base = spark.table(tabla("features")).select(*columnas).toPandas()
    panel = spark.table(tabla("panel_semanal")).select("semana", "skus_con_stock").toPandas()
    for df, fechas in ((base, ("origen", "semana_objetivo")), (panel, ("semana",))):
        for columna in fechas:
            df[columna] = pd.to_datetime(df[columna])
    origen = base["origen"].max()

    ventas, stock = spark.table(tabla("ventas_bronze")), spark.table(tabla("stock_bronze"))
    desde = (origen - pd.Timedelta(weeks=monitoreo.SEMANAS_COBERTURA)).date()
    fechas_venta = ventas.where(F.col("fecha") >= F.lit(desde)).select("fecha").toPandas()["fecha"]
    corte_venta, corte_stock = (pd.Timestamp(df.agg(F.max("fecha")).first()[0]) for df in (ventas, stock))
    conteos = {
        "skus_con_stock": monitoreo.skus_con_stock_por_semana(panel),
        "filas_venta": monitoreo.filas_por_semana(fechas_venta),
    }
    revisiones = pd.concat(
        [
            monitoreo.drift_datos(base, origen),
            monitoreo.revisar_cobertura(origen, conteos),
            monitoreo.revisar_corte(origen, corte_venta, corte_stock),
        ],
        ignore_index=True,
    )

    destino = tabla("monitoreo_datos")
    filas = revisiones.assign(fecha_corrida=pd.Timestamp.now().floor("s"), origen=origen.date(),
                              aceptado=aceptado == origen)
    spark.createDataFrame(filas[COLUMNAS]).write.format("delta").mode("append").saveAsTable(destino)
    proteger_solo_agregar(spark, destino)
    print(f"{destino}: {len(filas)} revisiones del origen {origen.date()}")
    for revision in revisiones.itertuples():
        print(f"  {monitoreo.describir(revision)}")
    if aceptado is not None and aceptado != origen:
        print(f"aceptar_drift={aceptado.date()} no es el origen de esta corrida ({origen.date()}): no se aplica")
    print(monitoreo.revisar_detencion(revisiones, origen, aceptado))


if __name__ == "__main__":
    main()
