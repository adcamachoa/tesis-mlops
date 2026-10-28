"""Pruebas de common.metrics: fórmulas, reparto de las semanas en ventanas de 31-30-30 días y formato estándar."""
import numpy as np
import pandas as pd
import pytest

from common.metrics import (
    METRICAS,
    a_ventanas,
    formato_estandar,
    mae,
    mape,
    perfil_semanal,
    pesos_ventanas,
    por_origen,
    rmse,
    sesgo,
    wape,
)

LUNES = pd.Timestamp("2025-03-03")


def test_formulas():
    real, pronostico = [10, 0, 5], [8, 2, 5]
    assert wape(real, pronostico) == pytest.approx(4 / 15)
    assert mape(real, pronostico) == pytest.approx((2 / 10 + 0 / 5) / 2)  # la celda sin venta no cuenta
    assert sesgo(real, pronostico) == pytest.approx(0)
    assert sesgo([10], [12]) == pytest.approx(0.2)  # positivo = pronóstico de más
    assert mae(real, pronostico) == pytest.approx(4 / 3)
    assert rmse(real, pronostico) == pytest.approx(np.sqrt(8 / 3))


def test_pesos_con_perfil_uniforme():
    pesos = pesos_ventanas(np.full(7, 1 / 7))
    assert pesos.shape == (13, 3)
    np.testing.assert_allclose(pesos.sum(axis=1), 1)
    np.testing.assert_allclose(pesos.sum(axis=0) * 7, [31, 30, 30])
    np.testing.assert_allclose(pesos[4], [3 / 7, 4 / 7, 0])  # semana 5: lunes a miércoles en el mes 1
    np.testing.assert_allclose(pesos[8], [0, 5 / 7, 2 / 7])  # semana 9: sábado y domingo en el mes 3
    enteras = np.delete(pesos, [4, 8], axis=0)
    np.testing.assert_allclose(enteras, np.round(enteras))  # las demás semanas caen enteras en una ventana


def test_pesos_reparten_con_el_perfil():
    perfil = np.arange(1, 8) / 28
    pesos = pesos_ventanas(perfil)
    np.testing.assert_allclose(pesos[4], [6 / 28, 22 / 28, 0])
    np.testing.assert_allclose(pesos[8], [0, 15 / 28, 13 / 28])


def test_perfil_usa_solo_las_52_semanas_anteriores():
    fechas = pd.date_range(LUNES - pd.Timedelta(weeks=60), LUNES + pd.Timedelta(weeks=2))
    venta = pd.Series(0.0, index=fechas)
    dentro = (fechas >= LUNES - pd.Timedelta(weeks=52)) & (fechas < LUNES)
    venta[dentro & (fechas.dayofweek == 0)] = 1
    venta[dentro & (fechas.dayofweek == 5)] = 3
    venta[~dentro] = 100  # antes de las 52 semanas y desde el lunes: no cuentan
    np.testing.assert_allclose(perfil_semanal(venta, LUNES), [1 / 4, 0, 0, 0, 0, 3 / 4, 0])
    np.testing.assert_allclose(perfil_semanal(venta[~dentro & (fechas >= LUNES)], LUNES), np.full(7, 1 / 7))


def test_a_ventanas_cuenta_las_series_fuera_de_la_base():
    origen = LUNES - pd.Timedelta(weeks=1)
    dias = pd.date_range(LUNES, periods=91)
    venta_diaria = pd.concat(
        [
            pd.DataFrame({"serie": "A", "fecha": dias, "unidades": 1}),
            pd.DataFrame({"serie": "B", "fecha": dias[31:61], "unidades": 2}),  # no está en la base
        ]
    )
    predicciones = pd.DataFrame(
        {"serie": np.repeat(["A", "C"], 13), "origen": origen, "h": np.tile(np.arange(1, 14), 2), "pronostico": 7.0}
    )
    ventanas = a_ventanas(predicciones, venta_diaria).set_index(["serie", "ventana"])
    assert ventanas["origen"].eq(origen).all()
    assert ventanas.loc["A", "real"].tolist() == [31, 30, 30]
    assert ventanas.loc["A", "pronostico"].tolist() == pytest.approx([31, 30, 30])  # sin venta previa, 1/7 por día
    assert ventanas.loc["B", "real"].tolist() == [0, 60, 0]
    assert ventanas.loc["B", "pronostico"].tolist() == [0, 0, 0]
    assert ventanas.loc["C", "real"].tolist() == [0, 0, 0]


def test_formato_estandar_no_cancela_errores_entre_meses():
    ventanas = pd.DataFrame(
        {"origen": LUNES, "serie": "A", "ventana": [1, 2, 3], "real": 100.0, "pronostico": [200.0, 0.0, 100.0]}
    )
    metricas = formato_estandar(ventanas)
    assert tuple(metricas) == METRICAS
    assert metricas["wape_m1"] == metricas["wape_m2"] == pytest.approx(1)
    assert metricas["wape_total"] == pytest.approx(0)  # el mes 1 y el mes 2 se cancelan en el trimestre
    assert metricas["wape_tipo_mes"] == pytest.approx(2 / 3)  # en el criterio del champion no
    assert metricas["mape_m1"] == pytest.approx(1)
    assert metricas["mape_total"] == pytest.approx(0)
    assert metricas["sesgo_total"] == pytest.approx(0)
    assert metricas["mae"] == pytest.approx(200 / 3)
    assert metricas["rmse"] == pytest.approx(np.sqrt(20000 / 3))


def test_por_origen_mide_cada_origen_por_separado():
    otro = LUNES + pd.Timedelta(weeks=4)
    ventanas = pd.DataFrame(
        {
            "origen": [LUNES] * 3 + [otro] * 3,
            "serie": "A",
            "ventana": [1, 2, 3] * 2,
            "real": 100.0,
            "pronostico": [100.0, 100.0, 100.0, 150.0, 150.0, 150.0],
        }
    )
    tabla = por_origen(ventanas)
    assert tabla.index.tolist() == [LUNES, otro] and tuple(tabla.columns) == METRICAS
    assert tabla["wape_tipo_mes"].tolist() == pytest.approx([0, 0.5])
    assert tabla.loc[otro].to_dict() == pytest.approx(formato_estandar(ventanas[ventanas["origen"] == otro]))


def test_con_pronostico_perfecto_solo_falla_la_serie_nueva(sinteticos):
    base, venta_diaria = sinteticos["base"], sinteticos["venta_diaria"]
    origen = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=100)  # la serie Tipo 3 empieza en la semana 105
    filas = base[base["origen"] == origen]
    assert not filas["tipo_producto"].eq("Tipo 3").any()
    ventanas = a_ventanas(filas.assign(pronostico=filas["y"]), venta_diaria)
    total = ventanas.groupby("serie")[["real", "pronostico"]].sum()
    nueva = total.index.str.endswith("Tipo 3")
    assert total.loc[nueva, "real"].item() > 0
    np.testing.assert_allclose(total.loc[~nueva, "pronostico"], total.loc[~nueva, "real"])  # 13 semanas = 91 días
    metricas = formato_estandar(ventanas)
    assert metricas["wape_total"] == pytest.approx(total.loc[nueva, "real"].item() / total["real"].sum())
