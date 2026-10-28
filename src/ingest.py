"""Ingesta: valida los parquet anonimizados del Volume y escribe las tablas bronze en Delta.

Se corre en Databricks como spark_python_task:
    python src/ingest.py --schema tesis_dev

Lee /Volumes/<catalogo>/<schema>/raw/<archivo>.parquet y valida los datos en pandas con common.validation. Solo si todo
pasa escribe <catalogo>.<schema>.<tabla>; si algo falla, se detiene sin escribir ninguna tabla. El stock (12,6 M filas)
se valida con todas las filas de una muestra de SKU, y sus SKU distintos se cruzan completos con productos; las demás
tablas se validan completas. Al final publica la versión Delta de ventas_bronze como task value `bronze_version`, para
que el entrenamiento la registre en MLflow.
"""
import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common import validation

# archivo del Volume -> tabla bronze
TABLAS = {
    "ventas_diarias": "ventas_bronze",
    "stock_diario": "stock_bronze",
    "quiebres_diarios": "quiebres_bronze",
    "productos": "productos_bronze",
    "calendario_eventos": "calendario_bronze",
}


def muestra_por_sku(df: DataFrame, skus_muestra: int, total_skus: int) -> DataFrame:
    """Todas las filas de unos `skus_muestra` SKU elegidos por hash: la muestra es siempre la misma y, como trae cada
    SKU completo, sus duplicados por fecha + sku se pueden detectar."""
    cada = max(1, total_skus // skus_muestra)
    return df.where(F.pmod(F.xxhash64("sku"), F.lit(cada)) == 0)


def main():
    parser = argparse.ArgumentParser(description="Valida los parquet del Volume y los carga como tablas bronze.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    parser.add_argument("--skus-muestra", type=int, default=2000, help="SKU del stock que se validan en pandas")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    dfs = {a: spark.read.parquet(f"/Volumes/{args.catalogo}/{args.schema}/raw/{a}.parquet") for a in TABLAS}

    # Se valida todo antes de escribir, para que una falla deje intactas las tablas bronze anteriores
    skus_stock = dfs["stock_diario"].select("sku").distinct().toPandas()["sku"]
    muestras = {}
    for archivo, df in dfs.items():
        if archivo == "stock_diario":
            df = muestra_por_sku(df, args.skus_muestra, len(skus_stock))
        muestras[archivo] = df.toPandas()
        print(f"{archivo}: {len(muestras[archivo]):,} filas a validar")
    validation.validar(muestras, skus={"stock_diario": skus_stock})
    print("validación sin problemas")

    for archivo, tabla in TABLAS.items():
        destino = f"{args.catalogo}.{args.schema}.{tabla}"
        dfs[archivo].write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(destino)
        print(f"{destino}: {spark.table(destino).count():,} filas")

    version = spark.sql(f"DESCRIBE HISTORY {args.catalogo}.{args.schema}.ventas_bronze LIMIT 1").first()["version"]
    try:
        from databricks.sdk.runtime import dbutils
        dbutils.jobs.taskValues.set(key="bronze_version", value=int(version))
    except Exception:  # fuera de un job (por ejemplo, en local) no hay task values
        pass
    print(f"bronze_version = {version}")


if __name__ == "__main__":
    main()
