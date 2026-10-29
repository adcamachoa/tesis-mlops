"""Pruebas de common.monitoreo: PSI, nulos, cobertura, corte del stock, detención y desempeño contra el rango."""
import numpy as np
import pandas as pd
import pytest

from common import monitoreo
from common.monitoreo import DriftExtremo

ORIGENES = pd.date_range("2025-06-02", periods=30, freq="7D")
ULTIMO = ORIGENES[-1]


@pytest.fixture
def base() -> pd.DataFrame:
    """300 series con las variables principales estables en 30 orígenes, una fila por serie × origen (h = 1)."""
    rng = np.random.default_rng(0)
    tamano = rng.lognormal(2, 1, 300)  # cada serie tiene su escala, como en los datos reales
    filas = []
    for origen in ORIGENES:
        valores = {v: tamano * rng.lognormal(0, 0.3, len(tamano)) for v in monitoreo.VARIABLES}
        filas.append(pd.DataFrame({"serie": [f"S{i}" for i in range(len(tamano))], "origen": origen,
                                   "semana_objetivo": origen + pd.Timedelta(weeks=1), "y": 1.0, **valores}))
    return pd.concat(filas, ignore_index=True)


def _con_origen(base: pd.DataFrame, **cambios) -> pd.DataFrame:
    """`base` con las variables del último origen reemplazadas por funciones de su valor."""
    base = base.copy()
    ultimo = base["origen"] == ULTIMO
    for variable, cambio in cambios.items():
        base.loc[ultimo, variable] = cambio(base.loc[ultimo, variable])
    return base


def _nivel(revisiones: pd.DataFrame, revision: str, variable: str) -> str:
    return revisiones.set_index(["revision", "variable"]).loc[(revision, variable), "nivel"]


def test_psi_es_cero_con_la_misma_distribucion_y_alto_con_un_desplazamiento():
    rng = np.random.default_rng(1)
    referencia = rng.lognormal(0, 1, 20_000)
    assert monitoreo.psi(referencia, rng.lognormal(0, 1, 5_000)) < 0.02
    assert monitoreo.psi(referencia, rng.lognormal(1.5, 1, 5_000)) > 1
    # un valor fuera del rango de la referencia cae en el tramo de la punta, no se pierde
    assert monitoreo.psi(referencia, np.full(100, 1e9)) > 5


def test_psi_con_empates_y_nulos():
    referencia = np.r_[np.zeros(600), np.arange(1, 401)]  # 60 % en 0: los deciles repetidos quedan en un solo tramo
    assert monitoreo.psi(referencia, referencia) == pytest.approx(0)
    con_nulos = np.r_[referencia, np.full(500, np.nan)]
    assert monitoreo.psi(con_nulos, referencia) == pytest.approx(0)  # los nulos no cuentan
    assert monitoreo.psi(referencia, np.zeros(100)) > 0.25
    assert np.isnan(monitoreo.psi(referencia, np.full(10, np.nan)))


def test_un_origen_normal_queda_estable_y_tendencia_solo_informa(base):
    revisiones = monitoreo.drift_datos(base, ULTIMO)
    assert len(revisiones) == 2 * len(monitoreo.VARIABLES)
    controladas = revisiones[~revisiones["variable"].isin(monitoreo.INFORMATIVAS)]
    assert set(controladas["nivel"]) == {"estable"}
    # tendencia queda con su PSI, pero sin aviso ni detención aunque cambie mucho
    revisiones = monitoreo.drift_datos(_con_origen(base, tendencia=lambda v: v * 50), ULTIMO)
    tendencia = revisiones[revisiones["variable"] == "tendencia"]
    assert set(tendencia["nivel"]) == {"informativa"}
    assert tendencia.loc[tendencia["revision"] == "psi", "valor"].iloc[0] > monitoreo.PSI_EXTREMO


def test_niveles_del_psi(base):
    revisiones = monitoreo.drift_datos(_con_origen(base, media_4=lambda v: v * 2.2, stock_cierre=lambda v: v * 0),
                                       ULTIMO)
    assert _nivel(revisiones, "psi", "media_4") == "aviso"
    assert _nivel(revisiones, "psi", "stock_cierre") == "extremo"


def test_nulos_contra_los_ultimos_13_origenes(base):
    # la ventana completa tiene nulos por diseño en disponibilidad, pero los 13 orígenes recientes no
    base.loc[base["origen"] < ULTIMO - pd.Timedelta(weeks=13), "disponibilidad"] = np.nan
    assert _nivel(monitoreo.drift_datos(base, ULTIMO), "nulos", "disponibilidad") == "estable"

    def nulos(parte):
        return lambda v: v.where(np.arange(len(v)) >= parte * len(v))

    revisiones = monitoreo.drift_datos(_con_origen(base, disponibilidad=nulos(0.2), stock_cierre=nulos(1)), ULTIMO)
    assert _nivel(revisiones, "nulos", "disponibilidad") == "aviso"
    assert _nivel(revisiones, "nulos", "stock_cierre") == "extremo"
    # sin datos en el origen el PSI no se puede calcular: lo marca la revisión de nulos
    assert _nivel(revisiones, "psi", "stock_cierre") == "sin datos"


def test_toda_nula_no_detiene_si_tampoco_habia_datos_recientes(base):
    base["lanzamientos_8"] = np.nan  # como en el primer año, cuando no se distingue un lanzamiento
    assert _nivel(monitoreo.drift_datos(base, ULTIMO), "nulos", "lanzamientos_8") == "estable"


def test_cobertura_contra_la_semana_mas_baja_de_las_4_anteriores():
    semanas = pd.date_range("2026-04-06", periods=6, freq="7D")
    normal = pd.Series([100, 105, 98, 102, 110, 101], index=semanas)
    assert monitoreo.cobertura(normal, semanas[-1]) == pytest.approx(101 / 98)
    # después de un pico, contra el promedio bajaría a la mitad; contra la más baja sigue normal
    pico = pd.Series([100, 100, 300, 300, 100, 95], index=semanas)
    assert monitoreo.cobertura(pico, semanas[-1]) == pytest.approx(0.95)
    a_medias = normal.copy()
    a_medias.iloc[-1] = 50
    revision = monitoreo.revisar_cobertura(semanas[-1], {"filas_venta": a_medias}).iloc[0]
    assert revision["valor"] == pytest.approx(50 / 98) and revision["nivel"] == "extremo"
    assert monitoreo.cobertura(normal.iloc[:-1], semanas[-1]) == 0  # una semana sin filas cuenta 0
    assert monitoreo.cobertura(normal.astype(float).where(normal.index < semanas[-1]), semanas[-1]) == 0  # ni dato
    sin_stock = normal.astype(float).where(normal.index >= semanas[3])  # sin dato en las semanas anteriores
    assert np.isnan(monitoreo.cobertura(sin_stock, semanas[-1]))
    assert monitoreo.nivel_cobertura(0.7) == "aviso" and monitoreo.nivel_cobertura(0.9) == "estable"


def test_conteos_semanales():
    fechas = pd.Series(pd.to_datetime(["2026-10-05", "2026-10-11", "2026-10-12", "2026-10-12"]))
    filas = monitoreo.filas_por_semana(fechas)
    assert filas.to_dict() == {pd.Timestamp("2026-10-05"): 2, pd.Timestamp("2026-10-12"): 2}
    panel = pd.DataFrame({"semana": pd.to_datetime(["2025-04-28", "2025-04-28", "2025-05-05", "2025-05-05"]),
                          "skus_con_stock": [np.nan, np.nan, 3, 4]})
    skus = monitoreo.skus_con_stock_por_semana(panel)
    assert np.isnan(skus.iloc[0]) and skus.iloc[1] == 7


def test_corte_del_stock():
    origen = pd.Timestamp("2026-10-12")
    venta = pd.Timestamp("2026-10-18")
    assert monitoreo.revisar_corte(origen, venta, pd.Timestamp("2026-10-21")).iloc[0]["nivel"] == "estable"
    assert monitoreo.revisar_corte(origen, venta, pd.Timestamp("2026-10-18")).iloc[0]["nivel"] == "estable"
    # el stock llega al domingo 11: su última semana completa es la anterior a la del origen
    atrasado = monitoreo.revisar_corte(origen, venta, pd.Timestamp("2026-10-11")).iloc[0]
    assert atrasado["valor"] == 1 and atrasado["nivel"] == "extremo"


def test_se_detiene_con_un_extremo_salvo_que_se_acepte_para_ese_origen(base):
    revisiones = monitoreo.drift_datos(_con_origen(base, stock_cierre=lambda v: v * 0), ULTIMO)
    with pytest.raises(DriftExtremo, match=f"aceptar_drift={ULTIMO.date()}"):
        monitoreo.revisar_detencion(revisiones, ULTIMO, None)
    with pytest.raises(DriftExtremo):  # aceptado para otro origen
        monitoreo.revisar_detencion(revisiones, ULTIMO, ULTIMO - pd.Timedelta(weeks=1))
    assert "aceptado" in monitoreo.revisar_detencion(revisiones, ULTIMO, ULTIMO)
    assert "sin cambios extremos" in monitoreo.revisar_detencion(monitoreo.drift_datos(base, ULTIMO), ULTIMO, None)


def test_aceptar_un_stock_atrasado_avisa_que_repite_el_origen():
    origen = pd.Timestamp("2026-10-12")
    corte = monitoreo.revisar_corte(origen, pd.Timestamp("2027-01-17"), pd.Timestamp("2026-10-18"))
    with pytest.raises(DriftExtremo, match="repite el pronóstico desde este mismo origen"):
        monitoreo.revisar_detencion(corte, origen, None)
    assert "volver a extraer el stock" in monitoreo.revisar_detencion(corte, origen, origen)


def _ventanas(real: list[float], pronostico: list[float], origen=ULTIMO) -> pd.DataFrame:
    """Dos series × 3 ventanas de un origen, en el formato de metrics.a_ventanas."""
    return pd.DataFrame({"origen": origen, "serie": np.repeat(["A", "B"], 3), "ventana": np.tile([1, 2, 3], 2),
                         "real": real, "pronostico": pronostico})


RANGO = {"wape_m1_min": 0.2, "wape_m1_max": 0.4, "wape_m2_min": 0.2, "wape_m2_max": 0.5, "wape_total_min": 0.2,
         "wape_total_max": 0.4, "sesgo_total_min": -0.05, "sesgo_total_max": 0.10}


def test_alerta_si_el_wape_pasa_el_maximo_o_el_sesgo_sale_del_rango():
    real = [100, 100, 100, 50, 50, 50]
    en_rango = monitoreo.evaluar(_ventanas(real, [130, 70, 100, 50, 65, 40]), RANGO)
    assert en_rango["fuera_de_rango"] == "" and not en_rango["alerta"]
    assert not {"mae", "rmse"} & set(en_rango)  # van en piezas
    # un WAPE mejor que el mejor origen del backtest no es alerta
    perfecto = monitoreo.evaluar(_ventanas(real, real), RANGO)
    assert perfecto["wape_total"] == 0 and not perfecto["alerta"]
    # el sesgo se sale por abajo: pronosticó 20 % de menos en todo
    corto = monitoreo.evaluar(_ventanas(real, [v * 0.8 for v in real]), RANGO)
    assert corto["fuera_de_rango"] == "sesgo_total" and corto["alerta"]
    # el mes 1 muy errado
    errado = monitoreo.evaluar(_ventanas(real, [200, 100, 100, 0, 50, 50]), RANGO)
    assert "wape_m1" in errado["fuera_de_rango"].split(",")


def test_rango_del_backtest():
    por_origen = pd.DataFrame({"wape_m1": [0.3, 0.2, 0.4], "wape_m2": [0.4, 0.5, 0.3], "wape_total": [0.3, 0.2, 0.25],
                               "sesgo_total": [0.02, -0.04, 0.08]})
    rango = monitoreo.rango_backtest(por_origen)
    assert rango["wape_m1_min"] == 0.2 and rango["wape_m1_max"] == 0.4
    assert rango["sesgo_total_min"] == -0.04 and rango["sesgo_total_max"] == 0.08


def test_revision_manual_con_dos_origenes_seguidos_en_alerta():
    medidas = pd.DataFrame({
        "origen": pd.date_range("2026-01-05", periods=5, freq="28D"),
        "fecha_corrida": pd.date_range("2026-01-12", periods=5, freq="28D"),
        "alerta": [False, True, True, False, True],
    })
    assert monitoreo.estados(medidas).tolist() == ["en rango", "alerta", "revisión manual", "en rango", "alerta"]


def test_de_cada_origen_cuenta_su_ultima_corrida():
    medidas = pd.DataFrame({
        "origen": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-02-02"]),
        "fecha_corrida": pd.to_datetime(["2026-01-06 10:00", "2026-01-06 12:00", "2026-02-03 10:00"]),
        "alerta": [True, False, True],
    })
    # la última corrida del 5-ene no tuvo alerta, así que la del 2-feb es solo alerta
    assert monitoreo.estados(medidas).tolist() == ["alerta", "en rango", "alerta"]


def test_corridas_por_medir():
    corridas = pd.DataFrame({
        "fecha_corrida": pd.to_datetime(["2026-07-14", "2026-07-15", "2026-10-13"]),
        "origen": pd.to_datetime(["2026-07-13", "2026-07-13", "2026-10-12"]),
    })
    medidas = corridas.iloc[:1]
    cerradas, abiertas = monitoreo.por_medir(corridas, medidas, pd.Timestamp("2026-10-12"))
    assert cerradas["fecha_corrida"].tolist() == [pd.Timestamp("2026-07-15")]  # la primera ya se midió
    assert abiertas["origen"].tolist() == [pd.Timestamp("2026-10-12")]
    assert monitoreo.ventanas_cerradas(pd.Timestamp("2026-07-13"), pd.Timestamp("2026-10-12"))
    assert not monitoreo.ventanas_cerradas(pd.Timestamp("2026-07-20"), pd.Timestamp("2026-10-12"))


def test_el_log_no_muestra_piezas(base):
    real = [12_345, 23_456, 34_567, 4_321, 5_432, 6_543]
    medida = monitoreo.evaluar(_ventanas(real, [v * 1.3 for v in real]), RANGO) | {"origen": ULTIMO, "estado": "alerta"}
    revisiones = monitoreo.drift_datos(_con_origen(base, stock_cierre=lambda v: v * 0), ULTIMO)
    texto = " ".join([
        monitoreo.describir_medida(medida),
        monitoreo.describir_rango(RANGO),
        *(monitoreo.describir(r) for r in revisiones.itertuples()),
    ])
    with pytest.raises(DriftExtremo) as error:
        monitoreo.revisar_detencion(revisiones, ULTIMO, None)
    texto += str(error.value)
    totales = [sum(real), sum(real) * 1.3, *real]
    assert not any(f"{v:,.0f}" in texto or f"{v:.0f}" in texto for v in totales)
    assert "%" in texto and "PSI" in texto
