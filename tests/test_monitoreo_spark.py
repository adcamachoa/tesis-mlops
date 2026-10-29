"""Prueba de las salidas de stock de src/monitoreo_datos.py con Spark local: dan lo mismo que
common.monitoreo.salidas_por_semana."""
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from common import monitoreo
from monitoreo_datos import salidas_por_semana


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


def test_salidas_de_stock_en_spark_y_en_pandas(spark):
    stock = pd.DataFrame({
        # SKU-1: baja 2, se queda, sube (llega mercancía), baja 7, se queda, baja 1 y, ya el lunes siguiente, baja 1
        "fecha": [*pd.date_range("2026-10-05", "2026-10-12"), *pd.to_datetime(["2026-10-05", "2026-10-06",
                                                                              "2026-10-09"])],
        "sku": ["SKU-1"] * 8 + ["SKU-2"] * 3,
        # SKU-2: baja 2 y después le falta el día anterior, así que no cuenta como salida
        "stock": [10, 8, 8, 12, 5, 5, 4, 3, 5, 3, 1],
    })
    esperadas = pd.Series([12.0, 1.0], index=pd.to_datetime(["2026-10-05", "2026-10-12"]))
    pd.testing.assert_series_equal(monitoreo.salidas_por_semana(stock), esperadas, check_names=False,
                                   check_index_type=False, check_freq=False)
    obtenidas = salidas_por_semana(spark.createDataFrame(stock)).toPandas()
    obtenidas = pd.Series(obtenidas["salidas"].to_numpy(dtype=float), index=pd.to_datetime(obtenidas["semana"]))
    pd.testing.assert_series_equal(obtenidas.sort_index(), esperadas, check_names=False, check_index_type=False,
                                   check_freq=False)
