"""Prueba de la agregación a semana × SKU de src/features.py con Spark local."""
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from features import agregar_semana_sku


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


def test_agrega_a_semana_sku_con_dias_disponibles(spark):
    # semana del 3 al 9 de marzo y el lunes siguiente; el jueves vende sus últimas 4 unidades y cierra en 0
    stock = pd.DataFrame(
        {"fecha": pd.date_range("2025-03-03", "2025-03-10"), "sku": "SKU-1", "stock": [5, 4, 4, 0, 0, 0, 2, 1]}
    )
    ventas = pd.DataFrame(
        {"fecha": pd.to_datetime(["2025-03-04", "2025-03-06", "2025-03-10"]), "sku": "SKU-1", "unidades": [1, 4, 1]}
    )
    ventas_sku, stock_sku = (
        df.toPandas().sort_values("semana").reset_index(drop=True)
        for df in agregar_semana_sku(spark.createDataFrame(ventas), spark.createDataFrame(stock))
    )
    semanas = [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-10")]
    assert pd.to_datetime(ventas_sku["semana"]).tolist() == semanas
    assert ventas_sku["unidades"].tolist() == [5, 1]
    assert pd.to_datetime(stock_sku["semana"]).tolist() == semanas
    assert stock_sku["stock_cierre"].tolist() == [2, 1]  # el del domingo
    assert stock_sku["dias_disponibles"].tolist() == [5, 1]  # lunes a miércoles, jueves por la venta y domingo
