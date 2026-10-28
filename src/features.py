"""Features: lee las tablas bronze, arma el panel serie × semana y la base del pronóstico directo.

Se corre en Databricks como spark_python_task:
    python src/features.py --schema tesis_dev

La agregación de diario a semana × SKU se hace en Spark, porque el stock tiene 12,6 M filas; el resto, con las
funciones puras de common.features en pandas. Escribe <catalogo>.<schema>.panel_semanal y <catalogo>.<schema>.features.
"""
import argparse

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.features import construir_base, construir_panel


def _semana():
    """Lunes de la semana de `fecha`."""
    return F.expr("date_sub(to_date(fecha), weekday(fecha))").alias("semana")


def agregar_semana_sku(ventas: DataFrame, stock: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Ventas y stock diarios a semana × SKU. Un día cuenta como disponible si el stock al cierre es mayor que 0 o si
    hubo venta, porque el stock al cierre deja en 0 el día en que un SKU vendió su última unidad."""
    ventas_sku = ventas.groupBy(_semana(), "sku").agg(F.sum("unidades").alias("unidades"))
    diario = stock.join(ventas.select("fecha", "sku", "unidades"), ["fecha", "sku"], "left")
    disponible = (F.col("stock") > 0) | (F.coalesce("unidades", F.lit(0)) > 0)
    stock_sku = diario.groupBy(_semana(), "sku").agg(
        F.max_by("stock", "fecha").alias("stock_cierre"),
        F.sum(disponible.cast("int")).alias("dias_disponibles"),
    )
    return ventas_sku, stock_sku


def _a_pandas(df: DataFrame) -> pd.DataFrame:
    pdf = df.toPandas()
    pdf["semana"] = pd.to_datetime(pdf["semana"])
    return pdf


def _a_spark(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
    """Las semanas se guardan como date."""
    pdf = pdf.copy()
    for columna in pdf.select_dtypes("datetime").columns:
        pdf[columna] = pdf[columna].dt.date
    return spark.createDataFrame(pdf)


def main():
    parser = argparse.ArgumentParser(description="Arma el panel semanal y la base de features desde las tablas bronze.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> DataFrame:
        return spark.table(f"{args.catalogo}.{args.schema}.{nombre}")

    ventas, stock = tabla("ventas_bronze"), tabla("stock_bronze")
    # el corte es el último día con datos en las dos tablas; solo entran semanas completas
    corte = pd.Timestamp(min(df.agg(F.max("fecha")).first()[0] for df in (ventas, stock)))
    ventas_sku, stock_sku = (_a_pandas(df) for df in agregar_semana_sku(ventas, stock))
    print(f"corte {corte.date()}: {len(ventas_sku):,} filas de venta y {len(stock_sku):,} de stock por semana × SKU")

    panel = construir_panel(ventas_sku, stock_sku, tabla("productos_bronze").toPandas(), corte)
    base = construir_base(panel, tabla("calendario_bronze").toPandas(), corte)
    print(f"panel: {panel['serie'].nunique()} series × {panel['semana'].nunique()} semanas")
    print(f"features: {base['origen'].nunique()} orígenes, {base['y'].notna().sum():,} filas con y")

    for nombre, pdf in {"panel_semanal": panel, "features": base}.items():
        destino = f"{args.catalogo}.{args.schema}.{nombre}"
        _a_spark(spark, pdf).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            destino
        )
        print(f"{destino}: {spark.table(destino).count():,} filas")


if __name__ == "__main__":
    main()
