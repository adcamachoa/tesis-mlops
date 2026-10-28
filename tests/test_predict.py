"""Pruebas de src/predict.py: la tabla predicciones, el reparto en ventanas, el control de fuga y el modelo guardado
con MLflow de punta a punta."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from common import backtest, metrics, modelos
from common.modelos import LIGHTGBM_1, LightGBMTweedie, ModeloDemanda, a_entrada, categorias
from predict import COLUMNAS, revisar_entrenamiento, tabla_predicciones

FECHA = pd.Timestamp("2026-05-20 08:30:00")


@pytest.fixture(scope="module")
def ultimo(sinteticos):
    """Las filas del último origen, las que pronostica predict.py, y la venta diaria del total."""
    base = sinteticos["base"]
    filas = base[base["origen"] == base["origen"].max()]
    return filas, sinteticos["venta_diaria"].groupby("fecha")["unidades"].sum()


def test_13_semanas_y_3_ventanas_por_serie(ultimo):
    filas, venta_total = ultimo
    origen = filas["origen"].iloc[0]
    assert filas["y"].isna().all()  # del último origen todavía no se conoce ninguna semana
    pronostico = np.arange(len(filas), dtype=float)
    tabla = tabla_predicciones(filas, pronostico, venta_total, FECHA, "media_movil_8", 3)

    assert tabla.columns.tolist() == COLUMNAS
    assert tabla[["fecha_corrida", "origen", "modelo", "modelo_version"]].drop_duplicates().values.tolist() == [
        [FECHA, origen, "media_movil_8", 3]
    ]
    conteo = tabla.groupby(["serie", "granularidad"]).size().unstack()
    assert len(conteo) == filas["serie"].nunique()
    assert conteo["semana"].eq(13).all() and conteo["ventana"].eq(3).all()

    semanas = tabla[tabla["granularidad"] == "semana"]
    assert semanas["fecha_inicio"].eq(origen + pd.to_timedelta(7 * semanas["periodo"], unit="D")).all()
    assert (semanas["fecha_fin"] - semanas["fecha_inicio"]).eq(pd.Timedelta(days=6)).all()
    esperado = pd.Series(pronostico, index=[filas["serie"].astype(str), filas["h"].astype(int)]).sort_index()
    np.testing.assert_array_equal(semanas.set_index(["serie", "periodo"])["pronostico"].sort_index(), esperado)

    ventanas = tabla[tabla["granularidad"] == "ventana"]
    fechas = ventanas[["periodo", "fecha_inicio", "fecha_fin"]].drop_duplicates().sort_values("periodo")
    assert list(fechas[["fecha_inicio", "fecha_fin"]].itertuples(index=False, name=None)) == metrics.fechas_ventanas(
        origen
    )
    # las ventanas solo reparten las semanas: por serie suman lo mismo
    sumas = tabla.groupby(["serie", "granularidad"])["pronostico"].sum().unstack()
    np.testing.assert_allclose(sumas["ventana"], sumas["semana"])


def test_las_semanas_partidas_se_reparten_con_el_perfil(ultimo):
    filas, venta_total = ultimo
    origen = filas["origen"].iloc[0]
    pronostico = np.where(filas["h"].isin([5, 9]), 7.0, 0.0)
    tabla = tabla_predicciones(filas, pronostico, venta_total, FECHA, "media_movil_8", 3)
    por_ventana = tabla[tabla["granularidad"] == "ventana"].groupby("periodo")["pronostico"].sum()
    perfil = metrics.perfil_semanal(venta_total, origen + pd.Timedelta(weeks=1))
    assert not np.allclose(perfil, 1 / 7)
    # la semana 5 tiene de lunes a miércoles en el mes 1; la 9, sábado y domingo en el mes 3
    esperado = 7 * np.array([perfil[:3].sum(), perfil[3:].sum() + perfil[:5].sum(), perfil[5:].sum()])
    np.testing.assert_allclose(por_ventana / filas["serie"].nunique(), esperado)


def test_el_modelo_guardado_pronostica_el_ultimo_origen(sinteticos, ultimo, tmp_path):
    """El camino de predict.py: las filas del último origen con a_entrada, al modelo cargado con MLflow."""
    filas, venta_total = ultimo
    base, origen = sinteticos["base"], filas["origen"].iloc[0]
    modelo = LightGBMTweedie(LIGHTGBM_1, rondas=20, semillas=(7,))
    modelo.ajustar(backtest.entrenamiento(base, origen), origen)
    entrada = a_entrada(filas)
    ruta = tmp_path / "modelo"
    mlflow.pyfunc.save_model(
        ruta,
        python_model=ModeloDemanda("lightgbm_tweedie", modelo, categorias(base)),
        code_paths=[str(Path(modelos.__file__).parent)],
        signature=mlflow.models.infer_signature(entrada, modelo.predecir(filas)),
        pip_requirements=modelos.requisitos(),
    )
    pronostico = np.asarray(mlflow.pyfunc.load_model(str(ruta)).predict(entrada), dtype=float)
    np.testing.assert_allclose(pronostico, modelo.predecir(filas), rtol=1e-6)
    tabla = tabla_predicciones(filas, pronostico, venta_total, FECHA, "lightgbm_tweedie", 1)
    assert np.isfinite(tabla["pronostico"]).all() and tabla["pronostico"].ge(0).all()


def test_se_detiene_si_el_modelo_conoce_semanas_posteriores_al_origen(capsys):
    origen = pd.Timestamp("2026-10-12")
    revisar_entrenamiento(origen, origen)
    assert capsys.readouterr().out == ""
    revisar_entrenamiento(origen - pd.Timedelta(weeks=1), origen)  # atrasado: solo avisa
    assert "aviso" in capsys.readouterr().out
    with pytest.raises(ValueError, match="posterior al origen"):
        revisar_entrenamiento(origen + pd.Timedelta(weeks=1), origen)
