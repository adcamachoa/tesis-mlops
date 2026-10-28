"""Pruebas de src/export_results.py y common.results_schema: results.json cumple el esquema, usa solo la última corrida,
lleva la venta como índice y no se escribe si algo falla."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from common import backtest, metrics
from common.modelos import MediaMovil, NaiveEstacional
from common.results_schema import ErrorEsquema, validar
from export_results import DECIMALES_INDICE, SEMANAS_REALES, construir_resultados, escribir
from predict import tabla_predicciones

ENTRENAMIENTO = pd.Timestamp("2026-05-20 08:00:00")
PREDICCION = pd.Timestamp("2026-05-20 08:30:00")
GENERADO = pd.Timestamp("2026-05-20 09:00:00")
REDONDEO = 0.5 * 10**-DECIMALES_INDICE + 1e-9


def _prediccion(sinteticos, origen: pd.Timestamp, fecha: pd.Timestamp, escala: float = 1.0) -> pd.DataFrame:
    """Una corrida de predict.py en `origen`, con la media móvil de 8 semanas como champion."""
    base = sinteticos["base"]
    filas = base[base["origen"] == origen]
    venta_total = sinteticos["venta_diaria"].groupby("fecha")["unidades"].sum()
    return tabla_predicciones(filas, MediaMovil(8).predecir(filas) * escala, venta_total, fecha, "media_movil_8", 4)


@pytest.fixture(scope="module")
def tablas(sinteticos) -> dict[str, pd.DataFrame]:
    """metricas_modelos con dos corridas de train, predicciones con una y el panel, como los lee export_results.py."""
    origenes = backtest.origenes(sinteticos["ultima_semana"], n=2)
    candidatos = {"naive_estacional": NaiveEstacional(), "media_movil_8": MediaMovil(8)}
    del_backtest = backtest.correr(candidatos, sinteticos["base"], origenes)
    resumen = pd.DataFrame(
        {n: metrics.formato_estandar(metrics.a_ventanas(p, sinteticos["venta_diaria"]))
         for n, p in del_backtest.groupby("modelo")}
    ).T

    def corrida(fecha: pd.Timestamp, resumen: pd.DataFrame) -> pd.DataFrame:
        return resumen.reset_index(names="modelo").assign(
            fecha_corrida=fecha,
            origenes=len(origenes),
            primer_origen=origenes[0].date(),
            ultimo_origen=origenes[-1].date(),
            es_mejor=lambda d: d["modelo"] == backtest.elegir_mejor(resumen),
        )

    anterior = corrida(ENTRENAMIENTO - pd.Timedelta(weeks=4), resumen.loc[["naive_estacional"]] * 2)
    return {
        "metricas": pd.concat([anterior, corrida(ENTRENAMIENTO, resumen)], ignore_index=True),
        "resumen": resumen,
        "predicciones": _prediccion(sinteticos, sinteticos["ultima_semana"], PREDICCION),
        "panel": sinteticos["panel"],
    }


@pytest.fixture(scope="module")
def resultados(tablas) -> dict:
    return construir_resultados(tablas["metricas"], tablas["predicciones"], tablas["panel"], GENERADO)


def _venta_real(tablas, resultados) -> pd.DataFrame:
    panel = tablas["panel"]
    return panel[panel["semana"].isin(pd.to_datetime(resultados["semanas"]["real"]))]


def test_results_json_cumple_el_esquema(resultados, tmp_path):
    ruta = tmp_path / "output" / "results.json"
    escribir(resultados, ruta)
    leido = json.loads(ruta.read_text(encoding="utf-8"))
    validar(leido)
    assert leido == resultados
    assert leido["champion"] == {"modelo": "media_movil_8", "version": 4}
    assert leido["corrida"]["fecha_prediccion"] == "2026-05-20T08:30:00"
    assert leido["corrida"]["fecha_entrenamiento"] == "2026-05-20T08:00:00"
    assert len(leido["semanas"]["real"]) == SEMANAS_REALES and len(leido["semanas"]["pronostico"]) == 13
    assert leido["semanas"]["real"][-1] == leido["corrida"]["origen"]
    assert [v["dias"] for v in leido["ventanas"]] == [31, 30, 30]
    # solo la última corrida de train, con el champion marcado aparte del mejor
    assert {m["modelo"]: m["es_champion"] for m in leido["metricas"]} == {
        "naive_estacional": False,
        "media_movil_8": True,
    }
    assert sum(m["es_mejor"] for m in leido["metricas"]) == 1


def test_venta_y_pronostico_como_indice(tablas, resultados):
    real, predicciones = _venta_real(tablas, resultados), tablas["predicciones"]
    factor = 100 / (real["venta"].sum() / SEMANAS_REALES)
    total = resultados["series"][0]
    assert np.mean(total["real"]) == pytest.approx(100, abs=REDONDEO)
    # el mismo factor para cada serie, en la venta real, el pronóstico y las ventanas
    semanas = pd.to_datetime(resultados["semanas"]["real"])
    for serie in (s for s in resultados["series"] if s["nivel"] == "serie"):
        venta = real[real["serie"] == serie["nombre"]].set_index("semana")["venta"].reindex(semanas, fill_value=0)
        np.testing.assert_allclose(serie["real"], venta * factor, atol=REDONDEO)
        de_la_serie = predicciones[predicciones["serie"] == serie["nombre"]]
        for granularidad, campo in (("semana", "pronostico"), ("ventana", "ventanas")):
            filas = de_la_serie[de_la_serie["granularidad"] == granularidad].sort_values("periodo")
            np.testing.assert_allclose(serie[campo], filas["pronostico"] * factor, atol=REDONDEO)
    # el MAE y el RMSE van en piezas y se escalan; los porcentajes no cambian
    for fila in resultados["metricas"]:
        esperado = tablas["resumen"].loc[fila["modelo"]]
        for metrica in ("mae", "rmse"):
            assert fila[metrica] == pytest.approx(esperado[metrica] * factor, abs=1e-4)
        for metrica in ("wape_m1", "wape_tipo_mes", "mape_total", "sesgo_total"):
            assert fila[metrica] == pytest.approx(esperado[metrica], abs=1e-4)


def test_el_total_es_la_suma_de_las_submarcas(resultados):
    total, *resto = resultados["series"]
    submarcas = [s for s in resto if s["nivel"] == "submarca"]
    assert [s["nombre"] for s in submarcas] == ["Submarca A", "Submarca B"]
    for campo in ("real", "pronostico", "ventanas"):
        suma = np.sum([s[campo] for s in submarcas], axis=0)
        np.testing.assert_allclose(total[campo], suma, atol=REDONDEO * (len(submarcas) + 1))


def test_las_series_de_mas_venta(tablas, resultados):
    venta = _venta_real(tablas, resultados).groupby("serie")["venta"].sum().sort_values(ascending=False)
    pocas = construir_resultados(tablas["metricas"], tablas["predicciones"], tablas["panel"], GENERADO, top=2)
    assert [s["nombre"] for s in pocas["series"] if s["nivel"] == "serie"] == venta.index[:2].tolist()
    # con el tope de 20 entran todas las que tienen pronóstico, y no la que dejó de vender
    todas = [s["nombre"] for s in resultados["series"] if s["nivel"] == "serie"]
    assert sorted(todas) == sorted(tablas["predicciones"]["serie"].unique()) and len(todas) == 3


def test_solo_la_ultima_corrida_del_origen_mas_reciente(sinteticos, tablas):
    ultima = sinteticos["ultima_semana"]
    repetida = _prediccion(sinteticos, ultima, PREDICCION + pd.Timedelta(hours=1), escala=2.0)
    # una corrida posterior pero de un origen anterior tampoco se mezcla
    vieja = _prediccion(sinteticos, ultima - pd.Timedelta(weeks=4), PREDICCION + pd.Timedelta(hours=2))
    todas = pd.concat([tablas["predicciones"], repetida, vieja], ignore_index=True)
    obtenido = construir_resultados(tablas["metricas"], todas, tablas["panel"], GENERADO)
    assert obtenido == construir_resultados(tablas["metricas"], repetida, tablas["panel"], GENERADO)
    assert obtenido["corrida"]["fecha_prediccion"] == "2026-05-20T09:30:00"


CAMBIOS = {
    "falta un campo": (lambda r: r.pop("champion"), "champion"),
    "tipo equivocado": (lambda r: r["champion"].update(version="4"), "champion/version"),
    "serie con otro largo": (lambda r: r["series"][1]["real"].pop(), "semanas reales"),
    "número no finito": (lambda r: r["series"][0]["pronostico"].__setitem__(0, float("nan")), "no es un número válido"),
    "dos totales": (lambda r: r["series"].append(copy.deepcopy(r["series"][0])), "una sola serie total"),
    "venta en piezas": (lambda r: r["series"][0].update(real=[v * 50 for v in r["series"][0]["real"]]), "como índice"),
}


@pytest.mark.parametrize(("cambio", "mensaje"), CAMBIOS.values(), ids=list(CAMBIOS))
def test_si_no_cumple_el_esquema_no_se_escribe(resultados, tmp_path, cambio, mensaje):
    ruta = tmp_path / "results.json"
    escribir(resultados, ruta)
    anterior = ruta.read_text(encoding="utf-8")
    malo = copy.deepcopy(resultados)
    cambio(malo)
    with pytest.raises(ErrorEsquema, match=mensaje):
        escribir(malo, ruta)
    assert ruta.read_text(encoding="utf-8") == anterior  # el archivo anterior queda intacto
