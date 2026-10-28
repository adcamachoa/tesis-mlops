"""Ingesta: lee los parquet anonimizados del Volume y escribe las tablas bronze en Delta.

Se corre en Databricks como spark_python_task:
    python src/ingest.py --schema tesis_dev

Lee /Volumes/<catalogo>/<schema>/raw/<archivo>.parquet y escribe <catalogo>.<schema>.<tabla>. Al final publica la
versión Delta de ventas_bronze como task value `bronze_version`, para que el entrenamiento la registre en MLflow.
La validación de los datos se agrega en la fase 2 (src/common/validation.py).
"""
import argparse

from pyspark.sql import SparkSession

# archivo del Volume -> tabla bronze
TABLAS = {
    "ventas_diarias": "ventas_bronze",
    "stock_diario": "stock_bronze",
    "quiebres_diarios": "quiebres_bronze",
    "productos": "productos_bronze",
    "calendario_eventos": "calendario_bronze",
}


def main():
    parser = argparse.ArgumentParser(description="Carga los parquet del Volume como tablas bronze.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    for archivo, tabla in TABLAS.items():
        df = spark.read.parquet(f"/Volumes/{args.catalogo}/{args.schema}/raw/{archivo}.parquet")
        destino = f"{args.catalogo}.{args.schema}.{tabla}"
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(destino)
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
