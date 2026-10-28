"""Pruebas de common.modelos: cada modelo entrena y pronostica, la parada temprana de LightGBM, las categorías y el
modelo de MLflow que se registra."""
from pathlib import Path

import mlflow
import numpy as np
import pytest

from common import backtest, modelos
from common.features import JERARQUIA
from common.modelos import (
    LIGHTGBM_1,
    LIGHTGBM_2,
    LightGBMTweedie,
    MediaMovil,
    ModeloDemanda,
    NaiveEstacional,
    a_categoricas,
    a_entrada,
    categorias,
)


@pytest.fixture(scope="module")
def partes(sinteticos):
    """Filas de entrenamiento y a pronosticar del último origen posible."""
    base = sinteticos["base"]
    origen = backtest.origenes(sinteticos["ultima_semana"], n=1)[0]
    return backtest.entrenamiento(base, origen), backtest.prueba(base, origen), origen


def test_cada_modelo_pronostica_sin_negativos(partes):
    entrena, filas, origen = partes
    for nombre, modelo in modelos.modelos(semillas=(7,)).items():
        modelo.ajustar(entrena, origen)
        pronostico = modelo.predecir(filas)
        assert pronostico.shape == (len(filas),), nombre
        assert np.isfinite(pronostico).all() and (pronostico >= 0).all(), nombre


def test_naive_estacional_y_media_movil(partes):
    _, filas, _ = partes
    np.testing.assert_array_equal(NaiveEstacional().predecir(filas), filas["venta_anio_anterior"].fillna(0))
    np.testing.assert_array_equal(MediaMovil(8).predecir(filas), filas["media_8"].fillna(0))


def test_una_parada_temprana_y_todas_las_semillas_con_esas_rondas(partes):
    entrena, filas, origen = partes
    modelo = LightGBMTweedie(LIGHTGBM_1, semillas=(7, 8, 9))
    modelo.ajustar(entrena, origen)
    assert isinstance(modelo.rondas_elegidas, int) and modelo.rondas_elegidas >= 50
    assert [b.num_trees() for b in modelo.boosters] == [modelo.rondas_elegidas] * 3
    por_semilla = [b.predict(filas[modelo.variables]) for b in modelo.boosters]
    assert not np.allclose(por_semilla[0], por_semilla[1])  # el muestreo cambia con la semilla
    np.testing.assert_allclose(modelo.predecir(filas), np.mean(por_semilla, axis=0))


def test_rondas_fijas_sin_parada_temprana(partes, monkeypatch):
    entrena, _, origen = partes
    modelo = LightGBMTweedie(LIGHTGBM_2, rondas=449, semillas=(7, 8))
    monkeypatch.setattr(modelo, "_parada_temprana", lambda *_: pytest.fail("no debía hacer parada temprana"))
    modelo.ajustar(entrena, origen)
    assert modelo.rondas_elegidas == 449 and len(modelo.boosters) == 2


def test_las_categorias_del_entrenamiento_se_mantienen(sinteticos):
    base = sinteticos["base"]
    guardadas = categorias(base)
    filas = base.iloc[::-1].head(50)  # otro orden, y no aparecen todas las series
    entrada = a_entrada(filas)
    entrada.loc[entrada.index[0], "tipo_producto"] = "Tipo nuevo"
    convertida = a_categoricas(entrada, guardadas)
    assert convertida["tipo_producto"].cat.categories.tolist() == guardadas["tipo_producto"]
    assert convertida["tipo_producto"].isna().tolist() == [True] + [False] * 49
    np.testing.assert_array_equal(convertida["submarca"].cat.codes, filas["submarca"].cat.codes)
    assert (convertida.dtypes[entrada.select_dtypes("number").columns] == "float32").all()


def test_el_modelo_registrado_se_guarda_y_carga(partes, tmp_path):
    entrena, filas, origen = partes
    modelo = LightGBMTweedie(LIGHTGBM_1, rondas=20, semillas=(7,))
    modelo.ajustar(entrena, origen)
    entrada = a_entrada(filas)
    ruta = tmp_path / "modelo"
    mlflow.pyfunc.save_model(
        ruta,
        python_model=ModeloDemanda("lightgbm_tweedie", modelo, categorias(entrena)),
        code_paths=[str(Path(modelos.__file__).parent)],
        signature=mlflow.models.infer_signature(entrada, modelo.predecir(filas)),
        input_example=entrada.head(5),
        pip_requirements=modelos.requisitos(),
    )
    assert entrada.select_dtypes(exclude="float64").columns.tolist() == JERARQUIA  # texto, como en la tabla features
    cargado = mlflow.pyfunc.load_model(str(ruta))
    np.testing.assert_allclose(np.asarray(cargado.predict(entrada)), modelo.predecir(filas), rtol=1e-6)
