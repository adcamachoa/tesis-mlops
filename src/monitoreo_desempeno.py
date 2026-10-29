"""Monitoreo de desempeño: error en vivo de cada pronóstico cuyas ventanas ya cerraron, contra el rango de su backtest.

Se corre en Databricks como spark_python_task, al final del job:
    python src/monitoreo_desempeno.py --schema tesis_dev

Busca en <catalogo>.<schema>.predicciones las corridas cuya ventana 3 ya terminó (origen + 13 semanas a lo más el último
origen de features) y que todavía no están en <catalogo>.<schema>.monitoreo_desempeno. Mide cada una desde sus filas
semanales con el mismo código del backtest (common.metrics) y la compara con el mínimo y el máximo por origen del
backtest donde se registró la versión que pronosticó: el artefacto predicciones_backtest.parquet de ese run de MLflow,
que se guardó al entrenar. Agrega una fila por corrida, con su estado (common.monitoreo.estados), a una tabla de solo
agregar. Para las corridas que todavía no cierran, imprime el rango con que se medirán.

No detiene el job y no imprime piezas: la tabla y el log solo llevan WAPE, MAPE y sesgo.
"""
import argparse
import tempfile

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common import metrics, monitoreo
from predict import proteger_solo_agregar
from train import venta_diaria_serie

ARTEFACTO = "predicciones_backtest.parquet"  # lo deja train.py en el run de cada modelo
CORRIDA = ["fecha_corrida", "origen", "modelo", "modelo_version"]
RANGO = [f"{m}_{f}" for m in monitoreo.METRICAS_ALERTA for f in ("min", "max")]
COLUMNAS = [
    "fecha_evaluacion", *CORRIDA, "run_id_backtest", *monitoreo.METRICAS_VIVO, *RANGO, "fuera_de_rango", "alerta",
    "estado",
]


def rango_de_version(cliente: MlflowClient, modelo_uc: str, version: int, venta_diaria: pd.DataFrame):
    """El rango del backtest donde se registró `version`, y el run de ese backtest."""
    run_id = cliente.get_model_version(modelo_uc, str(version)).run_id
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTEFACTO, dst_path=carpeta)
        del_backtest = pd.read_parquet(ruta)
    del_backtest["origen"] = pd.to_datetime(del_backtest["origen"])
    ventanas = metrics.a_ventanas(del_backtest, venta_diaria)
    return monitoreo.rango_backtest(metrics.por_origen(ventanas)), run_id


def main():
    parser = argparse.ArgumentParser(description="Error en vivo de los pronósticos cuyas ventanas ya cerraron.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> str:
        return f"{args.catalogo}.{args.schema}.{nombre}"

    ultimo_origen = pd.Timestamp(spark.table(tabla("features")).agg(F.max("origen")).first()[0])
    semanales = (
        spark.table(tabla("predicciones"))
        .where(F.col("granularidad") == "semana")
        .select(*CORRIDA, "serie", F.col("periodo").alias("h"), "pronostico")
        .toPandas()
    )
    destino = tabla("monitoreo_desempeno")
    if spark.catalog.tableExists(destino):
        previas = spark.table(destino).select("fecha_corrida", "origen", "alerta").toPandas()
    else:
        previas = pd.DataFrame({"fecha_corrida": pd.Series(dtype="datetime64[ns]"),
                                "origen": pd.Series(dtype="datetime64[ns]"), "alerta": pd.Series(dtype=bool)})
    for df in (semanales, previas):
        df["origen"] = pd.to_datetime(df["origen"])
    corridas = semanales[CORRIDA].drop_duplicates().sort_values(["origen", "fecha_corrida"], ignore_index=True)
    cerradas, abiertas = monitoreo.por_medir(corridas, previas, ultimo_origen)
    print(f"último origen {ultimo_origen.date()}: {len(cerradas)} corridas por medir y {len(abiertas)} con ventanas "
          f"abiertas")
    if cerradas.empty and abiertas.empty:
        return

    venta_diaria = venta_diaria_serie(spark.table(tabla("ventas_bronze")), spark.table(tabla("productos_bronze")))
    venta_diaria = venta_diaria.toPandas()
    venta_diaria["fecha"] = pd.to_datetime(venta_diaria["fecha"])
    mlflow.set_registry_uri("databricks-uc")
    cliente, modelo_uc = MlflowClient(), tabla("demanda_forecast")
    rangos = {
        int(v): rango_de_version(cliente, modelo_uc, int(v), venta_diaria)
        for v in pd.concat([cerradas, abiertas])["modelo_version"].unique()
    }

    for (origen, version), _ in abiertas.groupby(["origen", "modelo_version"]):
        cierre = metrics.fechas_ventanas(origen)[-1][1]
        print(f"origen {origen.date()} (versión {version}): cierra el {cierre.date()}; rango de su backtest: "
              f"{monitoreo.describir_rango(rangos[int(version)][0])}")
    if cerradas.empty:
        return

    medidas = []
    for corrida in cerradas.itertuples(index=False):
        filas = semanales[(semanales["fecha_corrida"] == corrida.fecha_corrida)
                          & (semanales["origen"] == corrida.origen)]
        rango, run_id = rangos[int(corrida.modelo_version)]
        medida = monitoreo.evaluar(metrics.a_ventanas(filas, venta_diaria), rango)
        medidas.append(corrida._asdict() | medida | {"run_id_backtest": run_id})
    medidas = pd.DataFrame(medidas)
    todas = pd.concat([previas, medidas[["fecha_corrida", "origen", "alerta"]]], ignore_index=True)
    medidas["estado"] = monitoreo.estados(todas).iloc[len(previas):].to_numpy()

    nuevas = medidas.assign(fecha_evaluacion=pd.Timestamp.now().floor("s"), origen=medidas["origen"].dt.date,
                            modelo_version=medidas["modelo_version"].astype(int))
    spark.createDataFrame(nuevas[COLUMNAS]).write.format("delta").mode("append").saveAsTable(destino)
    proteger_solo_agregar(spark, destino)
    print(f"{destino}: {len(nuevas)} corridas medidas")
    for _, medida in medidas.iterrows():
        print(f"  corrida {medida['fecha_corrida']}, {monitoreo.describir_medida(medida)}")


if __name__ == "__main__":
    main()
