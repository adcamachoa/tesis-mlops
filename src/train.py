"""Entrenamiento: backtest de los modelos, runs de MLflow, metricas_modelos y registro del champion en Unity Catalog.

Se corre en Databricks como spark_python_task:
    python src/train.py --schema tesis_dev
    python src/train.py --schema tesis_dev --ultimo-origen 2026-02-23 --solo-evaluar   # réplica de la referencia

Lee la tabla features, con el texto como categóricas para que quepa en memoria, y la venta diaria por serie, agregada
en Spark desde ventas_bronze y productos_bronze. Corre el backtest de common.backtest y mide cada modelo en el formato
estándar de common.metrics. En el experimento /Shared/demanda_<schema> deja un run padre por corrida y un run hijo por
modelo. Después agrega el resumen a <catalogo>.<schema>.metricas_modelos, reentrena con todo lo conocido el algoritmo
que queda de champion y lo registra como <catalogo>.<schema>.demanda_forecast con el alias champion. El mejor modelo le
quita el alias al algoritmo del champion actual solo si lo mejora en este mismo backtest y le gana en 6 de cada 8
orígenes (common.backtest.supera_al_champion); si no, se reentrena el algoritmo del champion.
"""
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common import backtest, metrics, modelos
from common.features import HORIZONTE, SERIE

# paleta de referencia de las gráficas: serie 1, serie 2, texto secundario, ejes, cuadrícula y fondo
COLORES = {"real": "#2a78d6", "pronostico": "#eb6834", "texto": "#52514e", "eje": "#c3c2b7", "rejilla": "#e1e0d9"}
FONDO = "#fcfcfb"
SEMANAS_PREVIAS = 12  # semanas de venta real antes del origen en cada panel de la gráfica


def venta_diaria_serie(ventas: DataFrame, productos: DataFrame) -> DataFrame:
    """Venta diaria por serie, con la misma clave de serie que common.features.clave_serie."""
    serie = F.concat_ws(" | ", *SERIE).alias("serie")
    con_serie = ventas.join(productos.select("sku", *SERIE), "sku")
    return con_serie.groupBy(serie, "fecha").agg(F.sum("unidades").alias("unidades"))


def leer_base(spark: SparkSession, tabla: str) -> pd.DataFrame:
    """La tabla features en pandas, con el texto como categóricas y las semanas como datetime."""
    pdf = spark.table(tabla).toPandas()
    for columna in ("origen", "semana_objetivo"):
        pdf[columna] = pd.to_datetime(pdf[columna])
    return modelos.a_categoricas(pdf)


def _version_delta(spark: SparkSession, tabla: str) -> int:
    return int(spark.sql(f"DESCRIBE HISTORY {tabla} LIMIT 1").first()["version"])


def _bronze_version(spark: SparkSession, tabla: str) -> int:
    """La que publicó la tarea de ingesta como task value; fuera de un job, la versión actual de ventas_bronze."""
    try:
        from databricks.sdk.runtime import dbutils
        version = int(dbutils.jobs.taskValues.get(taskKey="ingest", key="bronze_version", default=-1, debugValue=-1))
    except Exception:  # fuera de un job, o en un job sin la tarea de ingesta
        version = -1
    return version if version >= 0 else _version_delta(spark, tabla)


def _champion_actual(cliente: MlflowClient, nombre: str) -> tuple[str | None, float | None]:
    """Algoritmo del champion y su WAPE guardado, o (None, None) si todavía no hay champion."""
    try:
        version = cliente.get_model_version_by_alias(nombre, "champion")
    except MlflowException:
        return None, None
    guardado = version.tags.get(backtest.CRITERIO)
    return version.tags.get("modelo"), float(guardado) if guardado else None


def tabla_comparativa(resumen: pd.DataFrame) -> str:
    """El resumen en el formato estándar, en porcentaje salvo MAE y RMSE, como tabla markdown."""
    filas = ["| modelo | " + " | ".join(resumen.columns) + " |", "|---" * (len(resumen.columns) + 1) + "|"]
    for modelo, valores in resumen.iterrows():
        celdas = [f"{v:,.1f}" if m in ("mae", "rmse") else f"{v:+.1%}" if m == "sesgo_total" else f"{v:.1%}"
                  for m, v in valores.items()]
        filas.append(f"| {modelo} | " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def grafica_total(predicciones: pd.DataFrame, venta_diaria: pd.DataFrame, titulo: str) -> Figure:
    """Un panel por origen: la venta semanal real del total de las series y el pronóstico de sus 13 semanas."""
    fechas = venta_diaria["fecha"]
    real = venta_diaria.groupby(fechas - pd.to_timedelta(fechas.dt.dayofweek, unit="D"))["unidades"].sum()
    pronostico = predicciones.groupby(["origen", "semana_objetivo"])["pronostico"].sum()
    origenes = sorted(predicciones["origen"].unique())
    columnas = min(4, len(origenes))
    filas = -(-len(origenes) // columnas)
    figura = Figure(figsize=(4 * columnas, 3 * filas + 0.6), facecolor=FONDO, layout="constrained")
    ejes = figura.subplots(filas, columnas, sharey=True, squeeze=False).ravel()
    for eje, origen in zip(ejes, origenes, strict=False):
        semanas = pd.date_range(origen - pd.Timedelta(weeks=SEMANAS_PREVIAS), origen + pd.Timedelta(weeks=HORIZONTE),
                                freq="7D")
        eje.plot(semanas, real.reindex(semanas, fill_value=0), color=COLORES["real"], linewidth=1.5, label="Real")
        eje.plot(pronostico[origen].index, pronostico[origen].to_numpy(), color=COLORES["pronostico"], linewidth=1.5,
                 label="Pronóstico")
        eje.axvline(origen, color=COLORES["eje"], linewidth=0.8)
        eje.set_title(f"Origen {origen:%d/%m/%Y}", fontsize=9, color=COLORES["texto"], loc="left")
        eje.set_facecolor(FONDO)
        eje.grid(axis="y", color=COLORES["rejilla"], linewidth=0.6)
        eje.set_axisbelow(True)
        eje.xaxis.set_major_formatter(DateFormatter("%d/%m"))
        eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        eje.tick_params(colors=COLORES["texto"], labelsize=8, length=0)
        for lado, borde in eje.spines.items():
            borde.set_visible(lado == "bottom")
            borde.set_color(COLORES["eje"])
    for eje in ejes[len(origenes):]:
        eje.set_visible(False)
    figura.suptitle(f"{titulo}: venta semanal del total, real y pronóstico", fontsize=11, color=COLORES["texto"],
                    x=0.01, ha="left")
    figura.legend(*ejes[0].get_legend_handles_labels(), loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    return figura


def main():
    parser = argparse.ArgumentParser(description="Backtest de los modelos, MLflow y registro del champion.")
    parser.add_argument("--catalogo", default="workspace")
    parser.add_argument("--schema", default="tesis_dev")
    parser.add_argument("--origenes", type=int, default=backtest.ORIGENES)
    parser.add_argument("--semillas", type=int, default=len(modelos.SEMILLAS), help="cuántas de las semillas 7, 8 y 9")
    parser.add_argument("--ultimo-origen", type=pd.Timestamp, help="lunes; por defecto, el más reciente posible")
    parser.add_argument("--modelos", help="solo estos, separados por coma; el algoritmo del champion entra siempre")
    parser.add_argument(
        "--solo-evaluar", action="store_true", help="solo los runs de MLflow: no escribe metricas_modelos ni registra"
    )
    args = parser.parse_args()
    inicio = time.perf_counter()

    spark = SparkSession.builder.getOrCreate()

    def tabla(nombre: str) -> str:
        return f"{args.catalogo}.{args.schema}.{nombre}"

    base = leer_base(spark, tabla("features"))
    ultima_semana = base.loc[base["y"].notna(), "semana_objetivo"].max()
    origenes = backtest.origenes(ultima_semana, args.origenes, ultimo=args.ultimo_origen)
    venta_diaria = venta_diaria_serie(spark.table(tabla("ventas_bronze")), spark.table(tabla("productos_bronze")))
    venta_diaria = venta_diaria.toPandas()
    venta_diaria["fecha"] = pd.to_datetime(venta_diaria["fecha"])
    print(f"features: {len(base):,} filas, {base.memory_usage(deep=True).sum() / 2**20:,.0f} MB en memoria")
    print(f"orígenes: {', '.join(f'{o.date()}' for o in origenes)}")

    mlflow.set_registry_uri("databricks-uc")
    modelo_uc = tabla("demanda_forecast")
    semillas = modelos.SEMILLAS[: args.semillas]
    disponibles = modelos.modelos(semillas)
    champion, criterio_guardado = (None, None) if args.solo_evaluar else _champion_actual(MlflowClient(), modelo_uc)
    elegidos = args.modelos.split(",") if args.modelos else list(disponibles)
    if champion in disponibles and champion not in elegidos:
        elegidos.append(champion)
    if desconocidos := set(elegidos) - set(disponibles):
        raise ValueError(f"Modelos desconocidos: {', '.join(sorted(desconocidos))}. Hay: {', '.join(disponibles)}")

    predicciones = backtest.correr({n: disponibles[n] for n in elegidos}, base, origenes)
    ventanas = {n: metrics.a_ventanas(p, venta_diaria) for n, p in predicciones.groupby("modelo")}
    resumen = pd.DataFrame({n: metrics.formato_estandar(ventanas[n]) for n in elegidos}).T
    por_origen = {n: metrics.por_origen(ventanas[n]) for n in elegidos}
    mejor = backtest.elegir_mejor(resumen)
    comparativo = tabla_comparativa(resumen)
    print(comparativo)

    comunes = {
        "origenes": len(origenes),
        "primer_origen": f"{origenes[0].date()}",
        "ultimo_origen": f"{origenes[-1].date()}",
        "cada_semanas": backtest.CADA,
        "horizonte": HORIZONTE,
        "semillas": ",".join(map(str, semillas)),
        "bronze_version": _bronze_version(spark, tabla("ventas_bronze")),
        "features_version": _version_delta(spark, tabla("features")),
    }
    fecha_corrida = pd.Timestamp.now().floor("s")
    mlflow.set_experiment(f"/Shared/demanda_{args.schema}")
    with mlflow.start_run(run_name=f"backtest {fecha_corrida:%Y-%m-%d %H:%M}") as padre:
        mlflow.log_params(comunes | {"mejor": mejor, "solo_evaluar": args.solo_evaluar})
        mlflow.log_text(comparativo, "comparativo.md")
        runs = {}
        for nombre in elegidos:
            with mlflow.start_run(run_name=nombre, nested=True) as hijo:
                mlflow.log_params(comunes | {"modelo": nombre} | disponibles[nombre].parametros())
                mlflow.log_metrics(resumen.loc[nombre].to_dict())
                for paso, (_, del_origen) in enumerate(por_origen[nombre].iterrows(), start=1):
                    mlflow.log_metrics({f"{m}_por_origen": del_origen[m] for m in ("wape_total", backtest.CRITERIO)},
                                       step=paso)
                del_modelo = predicciones[predicciones["modelo"] == nombre]
                mlflow.log_figure(grafica_total(del_modelo, venta_diaria, nombre), "real_vs_pronostico_total.png")
                with tempfile.TemporaryDirectory() as carpeta:
                    ruta = Path(carpeta) / "predicciones_backtest.parquet"
                    del_modelo.to_parquet(ruta)
                    mlflow.log_artifact(str(ruta))
                runs[nombre] = hijo.info.run_id

        if args.solo_evaluar:
            print(f"listo en {time.perf_counter() - inicio:,.0f} s, sin escribir metricas_modelos ni registrar")
            return

        metricas = resumen.reset_index(names="modelo").assign(
            fecha_corrida=fecha_corrida,
            run_id=padre.info.run_id,
            run_id_modelo=lambda d: d["modelo"].map(runs),
            origenes=len(origenes),
            primer_origen=origenes[0].date(),
            ultimo_origen=origenes[-1].date(),
            semillas=comunes["semillas"],
            bronze_version=comunes["bronze_version"],
            es_mejor=lambda d: d["modelo"] == mejor,
        )
        spark.createDataFrame(metricas).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
            tabla("metricas_modelos")
        )

        criterio_por_origen = pd.DataFrame({n: por_origen[n][backtest.CRITERIO] for n in elegidos})
        sube, motivo = backtest.supera_al_champion(resumen, criterio_por_origen, mejor, champion, criterio_guardado)
        if sube or champion not in resumen.index:
            registrado = mejor
        else:  # el mejor no le quita el alias: el algoritmo del champion se reentrena con las semanas nuevas
            registrado, motivo = champion, f"sigue {champion}: {motivo}"
        con_alias = sube or registrado == champion

        # el backtest eligió el algoritmo; el que se registra se reentrena con todas las semanas conocidas
        final = modelos.modelos(semillas)[registrado]
        final.ajustar(backtest.entrenamiento(base, ultima_semana), ultima_semana)
        envoltura = modelos.ModeloDemanda(registrado, final, modelos.categorias(base))
        ejemplo = modelos.a_entrada(backtest.prueba(base, ultima_semana).head(5))
        with mlflow.start_run(run_id=runs[registrado], nested=True):
            info = mlflow.pyfunc.log_model(
                name="modelo",
                python_model=envoltura,
                code_paths=[str(Path(modelos.__file__).parent)],
                signature=mlflow.models.infer_signature(ejemplo, envoltura.predict(None, ejemplo)),
                input_example=ejemplo,
                pip_requirements=modelos.requisitos(),
                registered_model_name=modelo_uc,
            )
        version = info.registered_model_version
        cliente = MlflowClient()
        etiquetas = {
            "modelo": registrado,
            backtest.CRITERIO: resumen.loc[registrado, backtest.CRITERIO],
            "wape_total": resumen.loc[registrado, "wape_total"],
            "primer_origen": comunes["primer_origen"],
            "ultimo_origen": comunes["ultimo_origen"],
            "entrenado_hasta": f"{ultima_semana.date()}",
            "bronze_version": comunes["bronze_version"],
        }
        for clave, valor in etiquetas.items():
            cliente.set_model_version_tag(modelo_uc, version, clave, str(valor))
        if con_alias:
            cliente.set_registered_model_alias(modelo_uc, "champion", version)
        mlflow.set_tags({"version_registrada": version, "modelo_registrado": registrado, "champion": con_alias})
        mlflow.set_tag("motivo_champion", motivo)
        print(f"{modelo_uc} versión {version} ({registrado}): {'champion' if con_alias else 'sin alias'}; {motivo}")
    print(f"listo en {time.perf_counter() - inicio:,.0f} s")


if __name__ == "__main__":
    main()
