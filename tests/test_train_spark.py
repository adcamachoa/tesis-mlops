"""Pruebas de src/train.py: la venta diaria por serie con Spark local, la tabla comparativa y la gráfica."""
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from common import backtest, metrics
from common.features import clave_serie
from common.modelos import MediaMovil
from train import grafica_total, tabla_comparativa, venta_diaria_serie


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def test_venta_diaria_por_serie_con_la_clave_de_features(spark):
    productos = pd.DataFrame(
        {
            "sku": ["SKU-1", "SKU-2", "SKU-3"],
            "submarca": ["Submarca A", "Submarca A", "Submarca B"],
            "departamento": ["Depto 1", "Depto 1", "Depto 2"],
            "categoria": ["Cat 1", "Cat 1", "Cat 2"],
            "tipo_producto": ["Tipo 1", "Tipo 1", "Sin dato"],
        }
    )
    ventas = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2025-03-03", "2025-03-03", "2025-03-03", "2025-03-04"]),
            "sku": ["SKU-1", "SKU-2", "SKU-3", "SKU-1"],
            "unidades": [1, 2, 5, 4],
        }
    )
    obtenida = (
        venta_diaria_serie(spark.createDataFrame(ventas), spark.createDataFrame(productos))
        .toPandas()
        .sort_values(["serie", "fecha"])
        .reset_index(drop=True)
    )
    claves = clave_serie(productos)
    assert obtenida["serie"].tolist() == [claves[0], claves[0], claves[2]]
    fechas = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-03"])
    assert pd.to_datetime(obtenida["fecha"]).tolist() == list(fechas)
    assert obtenida["unidades"].tolist() == [3, 4, 5]


def test_tabla_y_grafica(sinteticos):
    origenes = backtest.origenes(sinteticos["ultima_semana"], n=3)
    predicciones = backtest.correr({"media_movil_8": MediaMovil(8)}, sinteticos["base"], origenes)
    ventanas = metrics.a_ventanas(predicciones, sinteticos["venta_diaria"])
    resumen = pd.DataFrame({"media_movil_8": metrics.formato_estandar(ventanas)}).T
    tabla = tabla_comparativa(resumen).splitlines()
    # el formato obligatorio va primero: WAPE y MAPE del mes 1, el mes 2 y el total, y el sesgo total
    formato = "| modelo | wape_m1 | wape_m2 | wape_total | mape_m1 | mape_m2 | mape_total | sesgo_total |"
    assert tabla[0].startswith(formato) and tabla[2].startswith("| media_movil_8 | ")
    figura = grafica_total(predicciones, sinteticos["venta_diaria"], "media_movil_8")
    assert sum(eje.get_visible() for eje in figura.axes) == len(origenes)
