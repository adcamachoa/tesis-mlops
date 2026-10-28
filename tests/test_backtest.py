"""Pruebas de common.backtest. La principal comprueba que ningún modelo entrena con una fecha posterior al origen: un
modelo espía anota lo que recibe, y alterar todo lo posterior al origen no cambia los pronósticos."""
import numpy as np
import pandas as pd
import pytest

from common import backtest
from common.modelos import LIGHTGBM_1, LightGBMTweedie, Mezcla, RandomForest

SEMANA = pd.Timedelta(weeks=1)


@pytest.fixture(scope="module")
def origenes(sinteticos):
    return backtest.origenes(sinteticos["ultima_semana"], n=3)


class Espia:
    """Anota qué filas recibe en cada origen y pronostica 1."""

    def __init__(self):
        self.vistas = {}

    def ajustar(self, filas, origen):
        self.vistas[origen] = {"ultima_objetivo": filas["semana_objetivo"].max(), "y_nula": filas["y"].isna().any()}

    def predecir(self, filas):
        self.vistas[filas["origen"].iloc[0]]["origenes_predichos"] = set(filas["origen"])
        return np.ones(len(filas))


def test_origenes_cada_4_semanas_con_el_horizonte_completo():
    ultima = pd.Timestamp("2026-10-12")
    assert backtest.origenes(ultima)[-1] == pd.Timestamp("2026-07-13")  # sus 13 semanas terminan el 18-oct
    assert backtest.origenes(ultima, n=3) == list(pd.to_datetime(["2026-05-18", "2026-06-15", "2026-07-13"]))
    referencia = backtest.origenes(ultima, ultimo=pd.Timestamp("2026-02-23"))
    assert referencia == list(pd.date_range("2025-08-11", "2026-02-23", freq="28D"))
    with pytest.raises(ValueError, match="no se conocen"):
        backtest.origenes(ultima, ultimo=pd.Timestamp("2026-07-20"))
    with pytest.raises(ValueError, match="lunes"):
        backtest.origenes(ultima, ultimo=pd.Timestamp("2026-02-24"))


def test_entrenamiento_solo_con_semanas_objetivo_hasta_el_origen(sinteticos, origenes):
    base = sinteticos["base"]
    for origen in origenes:
        filas = backtest.entrenamiento(base, origen)
        assert filas["semana_objetivo"].max() == origen
        assert filas["y"].notna().all()
        # hay filas de orígenes anteriores que quedan fuera porque su semana objetivo cae después del origen
        anteriores = base[(base["origen"] < origen) & (base["semana_objetivo"] > origen)]
        assert len(anteriores) > 0 and not anteriores.index.isin(filas.index).any()
        assert backtest.prueba(base, origen)["origen"].eq(origen).all()


def test_examen_interno(sinteticos, origenes):
    origen = origenes[-1]
    filas = backtest.entrenamiento(sinteticos["base"], origen)
    ajuste, examen = backtest.examen_interno(filas, origen)
    internos = pd.date_range(origen - 25 * SEMANA, origen - 13 * SEMANA, freq="7D")
    assert len(internos) == 13 and sorted(examen["origen"].unique()) == list(internos)
    assert examen["semana_objetivo"].max() == origen  # todo el horizonte del examen cae antes del origen
    assert ajuste["semana_objetivo"].max() == origen - 25 * SEMANA
    assert ajuste["semana_objetivo"].max() < examen["semana_objetivo"].min()


def test_ningun_modelo_ve_fechas_posteriores_al_origen(sinteticos, origenes):
    espia = Espia()
    predicciones = backtest.correr({"espia": espia}, sinteticos["base"], origenes)
    for origen in origenes:
        vistas = espia.vistas[origen]
        assert vistas["ultima_objetivo"] <= origen
        assert not vistas["y_nula"]
        assert vistas["origenes_predichos"] == {origen}
    assert set(predicciones["origen"]) == set(origenes)
    assert (predicciones["semana_objetivo"] > predicciones["origen"]).all()


def _alterar_despues_de(base: pd.DataFrame, origen: pd.Timestamp) -> pd.DataFrame:
    """Cambia la venta de las semanas posteriores al origen y todas las variables de los orígenes posteriores."""
    rng = np.random.default_rng(1)
    alterada = base.copy()
    futuro = alterada["semana_objetivo"] > origen
    alterada.loc[futuro, "y"] = rng.poisson(50, futuro.sum()).astype(float)
    posteriores = alterada["origen"] > origen
    for columna in alterada.select_dtypes("float32").columns:
        alterada.loc[posteriores, columna] = rng.random(posteriores.sum()).astype("float32") * 100
    return alterada


def test_alterar_el_futuro_no_cambia_el_pronostico(sinteticos, origenes):
    origen = origenes[0]
    base = sinteticos["base"]

    def pronosticos(datos):
        modelos = {
            # la parada temprana también tiene que quedar antes del origen
            "lightgbm": LightGBMTweedie(LIGHTGBM_1 | {"num_threads": 1, "deterministic": True}, semillas=(7, 8)),
            "random_forest": RandomForest(n_estimators=10, n_jobs=1),
        }
        return backtest.correr(modelos, datos, [origen])["pronostico"].to_numpy()

    np.testing.assert_array_equal(pronosticos(base), pronosticos(_alterar_despues_de(base, origen)))


class Contador:
    def __init__(self, valor):
        self.valor, self.entrenamientos = valor, 0

    def ajustar(self, filas, origen):
        self.entrenamientos += 1

    def predecir(self, filas):
        return np.full(len(filas), self.valor)


def test_la_mezcla_reutiliza_a_sus_componentes(sinteticos, origenes):
    a, b = Contador(2.0), Contador(4.0)
    modelos = {"a": a, "b": b, "mezcla": Mezcla({"a": Contador(0.0), "b": Contador(0.0)})}
    predicciones = backtest.correr(modelos, sinteticos["base"], origenes)
    assert a.entrenamientos == b.entrenamientos == len(origenes)  # una vez por origen, no dos
    assert predicciones.loc[predicciones["modelo"] == "mezcla", "pronostico"].eq(3.0).all()


def test_eleccion_del_champion():
    resumen = pd.DataFrame({"wape_tipo_mes": [0.40, 0.35, 0.70]}, index=["lightgbm", "produccion", "naive"])
    por_origen = pd.DataFrame({"lightgbm": [0.40] * 8, "produccion": [0.35] * 8, "naive": [0.70] * 8})
    mejor = backtest.elegir_mejor(resumen)
    assert mejor == "produccion"
    assert backtest.supera_al_champion(resumen, por_origen, mejor, None)[0]
    assert backtest.supera_al_champion(resumen, por_origen, mejor, "lightgbm")[0]
    assert backtest.supera_al_champion(resumen, por_origen, mejor, "produccion")[0]  # el mismo algoritmo, reentrenado
    assert not backtest.supera_al_champion(resumen, por_origen, "naive", "lightgbm")[0]
    # el algoritmo del champion ya no existe: se usa el WAPE guardado en su versión
    assert not backtest.supera_al_champion(resumen, por_origen, mejor, "retirado", criterio_guardado=0.30)[0]
    assert backtest.supera_al_champion(resumen, por_origen, mejor, "retirado", criterio_guardado=0.38)[0]
    assert not backtest.supera_al_champion(resumen, por_origen, mejor, "retirado")[0]


def test_un_algoritmo_distinto_tiene_que_ganar_en_6_de_8_origenes():
    resumen = pd.DataFrame({"wape_tipo_mes": [0.40, 0.39]}, index=["actual", "retador"])

    def por_origen(ganados, empatados=0):
        retador = [0.38] * ganados + [0.40] * empatados + [0.42] * (8 - ganados - empatados)
        return pd.DataFrame({"actual": [0.40] * 8, "retador": retador})

    sube, motivo = backtest.supera_al_champion(resumen, por_origen(5), "retador", "actual")
    assert not sube and "le gana en 5 de 8 orígenes y necesita 6" in motivo
    assert not backtest.supera_al_champion(resumen, por_origen(5, empatados=3), "retador", "actual")[0]  # empatar no
    assert backtest.supera_al_champion(resumen, por_origen(6), "retador", "actual")[0]
    # ganar en los 8 orígenes no basta si no mejora el WAPE de todo el backtest
    empate = pd.DataFrame({"wape_tipo_mes": [0.40, 0.40]}, index=["actual", "retador"])
    assert not backtest.supera_al_champion(empate, por_origen(8), "retador", "actual")[0]
    # el algoritmo del champion, reentrenado, se queda con el alias aunque haya empeorado
    peor = pd.DataFrame({"wape_tipo_mes": [0.45]}, index=["actual"])
    assert backtest.supera_al_champion(peor, por_origen(0), "actual", "actual")[0]
