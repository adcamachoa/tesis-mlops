"""Monitoreo de desempeño: mide cada ventana de los pronósticos apenas cierra, y publica si la corrida reentrena.

Se corre en Databricks como spark_python_task, después de monitoreo_datos y antes de la condición toca_reentrenar:
    python src/monitoreo_desempeno.py --schema tesis_dev
    python src/monitoreo_desempeno.py --schema tesis_dev --reentrenar si   # entrena y pronostica aunque no toque

Busca en <catalogo>.<schema>.predicciones las mediciones de cada corrida (mes 1, mes 2 y total) cuya ventana ya cerró
hasta el último domingo de features y que todavía no están en <catalogo>.<schema>.monitoreo_desempeno. Mide cada una
desde las filas semanales, con el mismo código del backtest (common.metrics), y la compara con el mínimo y el máximo
por origen del backtest donde se registró la versión que pronosticó: el artefacto predicciones_backtest.parquet de ese
run de MLflow, que se guardó al entrenar. Agrega una fila por corrida × medición, con su estado
(common.monitoreo.estados), a una tabla de solo agregar. Para las mediciones que todavía no cierran, imprime cuándo
cierra la siguiente y el rango con que se medirá.

Al final decide si toca reentrenar (common.monitoreo.decidir_reentrenamiento: cada 4 semanas desde el último origen
pronosticado, o con reentrenar=si) y lo publica como task value `reentrenar`, que lee la condición toca_reentrenar. Una
alerta no detiene el job ni adelanta el reentrenamiento: queda en la tabla y en el log como señal para revisar el
modelo. El log no lleva piezas: solo WAPE, MAPE, sesgo y fechas.
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
    "fecha_evaluacion", *CORRIDA, "medicion", "cierre", "run_id_backtest", *monitoreo.METRICAS_VIVO, *RANGO,
    "fuera_de_rango", "alerta", "estado",
]
PREVIAS = ["fecha_corrida", "origen", "medicion", "cierre", "alerta"]


def rango_de_version(cliente: MlflowClient, modelo_uc: str, version: int, venta_diaria: pd.DataFrame):
    """El rango del backtest donde se registró `version`, y el run de ese backtest."""
    run_id = cliente.get_model_version(modelo_uc, str(version)).run_id
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTEFACTO, dst_path=carpeta)
        del_backtest = pd.read_parquet(ruta)
    del_backtest["origen"] = pd.to_datetime(del_backtest["origen"])
    ventanas = metrics.a_ventanas(del_backtest, venta_diaria)
    return monitoreo.rango_backtest(metrics.por_origen(ventanas)), run_id


def publicar_decision(reentrenar: bool) -> None:
    """La decisión como task value booleano para la condición toca_reentrenar. Si no se puede publicar, la tarea falla:
    sin el valor, la condición no sabría qué rama seguir."""
    from databricks.sdk.runtime import dbutils

    dbutils.jobs.taskValues.set(key="reentrenar", value=reentrenar)


def main():
    parser = argparse.ArgumentParser(description="Mide las ventanas cerradas y decide si la corrida reentrena.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    parser.add_argument("--reentrenar", default="", help="'si' para entrenar y pronosticar aunque no toque")
    args = parser.parse_args()
    forzado = monitoreo.leer_reentrenar(args.reentrenar)

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> str:
        return f"{args.catalogo}.{args.schema}.{nombre}"

    origen = pd.Timestamp(spark.table(tabla("features")).agg(F.max("origen")).first()[0])
    real_hasta = origen + pd.Timedelta(days=6)
    semanales = (
        spark.table(tabla("predicciones"))
        .where(F.col("granularidad") == "semana")
        .select(*CORRIDA, "serie", F.col("periodo").alias("h"), "pronostico")
        .toPandas()
        if spark.catalog.tableExists(tabla("predicciones")) else pd.DataFrame(columns=[*CORRIDA, "serie", "h"])
    )
    destino = tabla("monitoreo_desempeno")
    if spark.catalog.tableExists(destino):
        previas = spark.table(destino).select(*PREVIAS).toPandas()
    else:
        previas = pd.DataFrame({"fecha_corrida": pd.Series(dtype="datetime64[ns]"),
                                "origen": pd.Series(dtype="datetime64[ns]"), "medicion": pd.Series(dtype=object),
                                "cierre": pd.Series(dtype="datetime64[ns]"), "alerta": pd.Series(dtype=bool)})
    for df, fechas in ((semanales, ("origen",)), (previas, ("origen", "cierre"))):
        for columna in fechas:
            df[columna] = pd.to_datetime(df[columna])

    corridas = semanales[CORRIDA].drop_duplicates().sort_values(["origen", "fecha_corrida"], ignore_index=True)
    cerradas, abiertas = monitoreo.por_medir(corridas, previas, real_hasta)
    print(f"origen {origen.date()}, venta hasta el {real_hasta.date()}: {len(cerradas)} mediciones por hacer y "
          f"{len(abiertas)} con su ventana abierta")
    if len(cerradas) or len(abiertas):
        medir(spark, tabla, semanales, cerradas, abiertas, previas)

    ultimo_pronostico = corridas["origen"].max() if len(corridas) else None
    reentrenar, motivo = monitoreo.decidir_reentrenamiento(origen, ultimo_pronostico, forzado)
    publicar_decision(reentrenar)
    print(f"{'reentrena' if reentrenar else 'no reentrena'}: {motivo}")


def medir(spark, tabla, semanales, cerradas, abiertas, previas) -> None:
    """Mide las mediciones `cerradas`, las agrega a monitoreo_desempeno e imprime cuándo cierra la siguiente de cada
    corrida abierta."""
    venta_diaria = venta_diaria_serie(spark.table(tabla("ventas_bronze")), spark.table(tabla("productos_bronze")))
    venta_diaria = venta_diaria.toPandas()
    venta_diaria["fecha"] = pd.to_datetime(venta_diaria["fecha"])
    mlflow.set_registry_uri("databricks-uc")
    cliente, modelo_uc = MlflowClient(), tabla("demanda_forecast")
    rangos = {
        int(v): rango_de_version(cliente, modelo_uc, int(v), venta_diaria)
        for v in pd.concat([cerradas, abiertas])["modelo_version"].unique()
    }

    siguientes = abiertas.sort_values("cierre").drop_duplicates(["fecha_corrida", "origen"])
    for siguiente in siguientes.itertuples(index=False):
        corrida = monitoreo.corrida_de_medicion(siguiente.origen, siguiente.medicion)
        print(f"origen {siguiente.origen.date()} (versión {siguiente.modelo_version}): {siguiente.medicion} cierra el "
              f"{siguiente.cierre.date()} y se mide en la corrida del {corrida.date()}; rango de su backtest: "
              f"{monitoreo.describir_rango(rangos[int(siguiente.modelo_version)][0])}")
    if cerradas.empty:
        return

    medidas = []
    for medicion in cerradas.itertuples(index=False):
        filas = semanales[(semanales["fecha_corrida"] == medicion.fecha_corrida)
                          & (semanales["origen"] == medicion.origen)]
        rango, run_id = rangos[int(medicion.modelo_version)]
        medida = monitoreo.evaluar(metrics.a_ventanas(filas, venta_diaria), rango, medicion.medicion)
        medidas.append(medicion._asdict() | medida | {"run_id_backtest": run_id})
    medidas = pd.DataFrame(medidas)
    todas = pd.concat([previas, medidas[PREVIAS]], ignore_index=True)
    medidas["estado"] = monitoreo.estados(todas).iloc[len(previas):].to_numpy()

    destino = tabla("monitoreo_desempeno")
    nuevas = medidas.assign(fecha_evaluacion=pd.Timestamp.now().floor("s"), origen=medidas["origen"].dt.date,
                            cierre=medidas["cierre"].dt.date, modelo_version=medidas["modelo_version"].astype(int))
    spark.createDataFrame(nuevas[COLUMNAS]).write.format("delta").mode("append").saveAsTable(destino)
    proteger_solo_agregar(spark, destino)
    print(f"{destino}: {len(nuevas)} mediciones")
    for _, medida in medidas.iterrows():
        print(f"  corrida {medida['fecha_corrida']}, {monitoreo.describir_medida(medida)}")
    if medidas["alerta"].any():
        print(f"{int(medidas['alerta'].sum())} mediciones con alerta: revisar el modelo. La alerta no adelanta el "
              f"reentrenamiento, que va cada {monitoreo.SEMANAS_REENTRENAMIENTO} semanas")


if __name__ == "__main__":
    main()
