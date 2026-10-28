"""Modelos que compara el backtest y la envoltura de MLflow con la que se registra el elegido.

Todos tienen la misma forma: `ajustar(filas, origen)` entrena con las filas de entrenamiento de un origen y
`predecir(filas)` devuelve la venta semanal pronosticada de cada fila, sin negativos. Para comparar un modelo más, basta
con agregarlo a `modelos()`.
"""
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from common.backtest import examen_interno
from common.features import JERARQUIA

CATEGORICAS = ["serie", *JERARQUIA]  # la serie no es variable, pero como texto ocupa mucha memoria
NO_VARIABLES = ["serie", "origen", "semana_objetivo", "y"]
SEMILLAS = (7, 8, 9)

LIGHTGBM_1 = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEMILLAS[0],  # la de la parada temprana; al entrenar, cada semilla la reemplaza
}
PARADA = {"maximo": 3000, "paciencia": 100, "minimo": 50}  # parada temprana de la configuración 1
LIGHTGBM_2 = LIGHTGBM_1 | {
    "tweedie_variance_power": 1.7597162498650851,
    "learning_rate": 0.04091976178025285,
    "num_leaves": 211,
    "min_data_in_leaf": 41,
    "feature_fraction": 0.3775166630691353,
    "bagging_fraction": 0.6647265231110455,
    "lambda_l2": 0.8069730415760346,
    "cat_smooth": 57.21805018347323,
}
RONDAS_2 = 449
RANDOM_FOREST = {
    "n_estimators": 100,
    "min_samples_leaf": 50,
    "max_features": 0.5,
    "max_samples": 0.3,
    "n_jobs": -1,
    "random_state": SEMILLAS[0],
}


def a_categoricas(base: pd.DataFrame, categorias: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Texto a categóricas y variables numéricas a float32, para que la base quepa en memoria.

    Con `categorias`, cada columna toma esas categorías en ese orden, como al predecir con un modelo ya entrenado: un
    texto que no esté queda nulo. Devuelve una tabla nueva; `y` queda en float64.
    """
    columnas = {}
    for nombre, columna in base.items():
        if nombre in CATEGORICAS:
            columna = pd.Categorical(columna, categories=None if categorias is None else categorias[nombre])
        elif nombre != "y" and pd.api.types.is_numeric_dtype(columna) and not pd.api.types.is_bool_dtype(columna):
            columna = columna.astype("float32")
        columnas[nombre] = columna
    return pd.DataFrame(columnas, index=base.index)


def categorias(base: pd.DataFrame) -> dict[str, list[str]]:
    return {c: base[c].cat.categories.tolist() for c in CATEGORICAS if c in base}


def variables(filas: pd.DataFrame) -> list[str]:
    return [c for c in filas.columns if c not in NO_VARIABLES]


def a_entrada(filas: pd.DataFrame) -> pd.DataFrame:
    """Las variables como llegan de la tabla features (texto y float64): la entrada del modelo registrado."""
    entrada = filas[variables(filas)]
    return entrada.astype({c: str if c in CATEGORICAS else "float64" for c in entrada.columns})


def _sin_negativos(pronostico) -> np.ndarray:
    return np.clip(np.asarray(pronostico, dtype=float), 0, None)


class NaiveEstacional:
    """La venta de la misma semana del año anterior."""

    def parametros(self) -> dict:
        return {"columna": "venta_anio_anterior"}

    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None:
        pass

    def predecir(self, filas: pd.DataFrame) -> np.ndarray:
        return _sin_negativos(filas["venta_anio_anterior"].fillna(0))


class MediaMovil:
    """El promedio de las últimas `semanas` semanas hasta el origen, igual para los 13 horizontes."""

    def __init__(self, semanas: int = 8):
        self.semanas = semanas

    def parametros(self) -> dict:
        return {"semanas": self.semanas}

    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None:
        pass

    def predecir(self, filas: pd.DataFrame) -> np.ndarray:
        return _sin_negativos(filas[f"media_{self.semanas}"].fillna(0))


class RandomForest:
    """RandomForest de scikit-learn. No entiende categóricas: recibe su código, con los nulos como NaN."""

    def __init__(self, **params):
        self.params = RANDOM_FOREST | params

    def parametros(self) -> dict:
        return self.params

    def _matriz(self, filas: pd.DataFrame) -> pd.DataFrame:
        matriz = filas[self.variables]
        codigos = matriz.select_dtypes("category").apply(lambda c: c.cat.codes.where(c.cat.codes >= 0))
        return matriz.assign(**codigos.astype("float32"))

    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None:
        self.variables = variables(filas)
        self.modelo = RandomForestRegressor(**self.params).fit(self._matriz(filas), filas["y"])

    def predecir(self, filas: pd.DataFrame) -> np.ndarray:
        return _sin_negativos(self.modelo.predict(self._matriz(filas)))


class LightGBMTweedie:
    """LightGBM con objetivo Tweedie: un modelo por semilla, todos con las mismas rondas, y el promedio de sus
    predicciones.

    Con `rondas`, ese número fijo. Sin `rondas`, una sola parada temprana por origen, con la semilla de `params`:
    entrena con el ajuste y se detiene cuando el error del examen deja de bajar (ver `backtest.examen_interno`). Después
    cada semilla entrena esas rondas con todas las filas hasta el origen.
    """

    def __init__(self, params: dict, rondas: int | None = None, semillas=SEMILLAS, parada: dict = PARADA):
        self.params, self.rondas, self.semillas, self.parada = params, rondas, tuple(semillas), parada

    def parametros(self) -> dict:
        rondas = {"rondas": self.rondas} if self.rondas else {f"parada_{k}": v for k, v in self.parada.items()}
        return self.params | rondas | {"semillas": ",".join(map(str, self.semillas))}

    def _datos(self, filas: pd.DataFrame, referencia: lgb.Dataset | None = None) -> lgb.Dataset:
        return lgb.Dataset(filas[self.variables], filas["y"], reference=referencia)

    def _parada_temprana(self, filas: pd.DataFrame, origen: pd.Timestamp) -> int:
        ajuste, examen = examen_interno(filas, origen)
        datos = self._datos(ajuste)
        prueba = lgb.train(
            self.params | {"verbosity": -1},
            datos,
            num_boost_round=self.parada["maximo"],
            valid_sets=[self._datos(examen, referencia=datos)],
            callbacks=[lgb.early_stopping(self.parada["paciencia"], verbose=False)],
        )
        return max(prueba.best_iteration, self.parada["minimo"])

    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None:
        self.variables = variables(filas)
        self.rondas_elegidas = rondas = self.rondas or self._parada_temprana(filas, origen)
        self.boosters = [
            lgb.train(self.params | {"seed": s, "verbosity": -1}, self._datos(filas), num_boost_round=rondas)
            for s in self.semillas
        ]

    def predecir(self, filas: pd.DataFrame) -> np.ndarray:
        matriz = filas[self.variables]
        return _sin_negativos(np.mean([b.predict(matriz) for b in self.boosters], axis=0))


class Mezcla:
    """Promedio de las predicciones de sus componentes. En el backtest, cada componente se entrena una sola vez por
    origen aunque también se evalúe por separado (ver `backtest.correr`)."""

    def __init__(self, componentes: dict):
        self.componentes = componentes

    def parametros(self) -> dict:
        return {f"{n}.{k}": v for n, m in self.componentes.items() for k, v in m.parametros().items()}

    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None:
        for modelo in self.componentes.values():
            modelo.ajustar(filas, origen)

    def predecir(self, filas: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predecir(filas) for m in self.componentes.values()], axis=0)


def modelos(semillas=SEMILLAS) -> dict:
    """Los modelos del backtest, nuevos y sin entrenar."""

    def lightgbm_1():
        return LightGBMTweedie(LIGHTGBM_1, semillas=semillas)

    def lightgbm_2():
        return LightGBMTweedie(LIGHTGBM_2, rondas=RONDAS_2, semillas=semillas)

    return {
        "naive_estacional": NaiveEstacional(),
        "media_movil_8": MediaMovil(8),
        "random_forest": RandomForest(),
        "lightgbm_tweedie": lightgbm_1(),
        "lightgbm_tweedie_211": lightgbm_2(),
        "produccion": Mezcla({"lightgbm_tweedie": lightgbm_1(), "lightgbm_tweedie_211": lightgbm_2()}),
    }


class ModeloDemanda(mlflow.pyfunc.PythonModel):
    """Envoltura de MLflow para registrar cualquiera de los modelos ya entrenado.

    Recibe las variables de la tabla features (ver `a_entrada`), les aplica las categorías del entrenamiento para que
    cada texto tenga el mismo código, y devuelve el pronóstico semanal de cada fila.
    """

    def __init__(self, nombre: str, modelo, categorias: dict[str, list[str]]):
        self.nombre, self.modelo, self.categorias = nombre, modelo, categorias

    def predict(self, context, model_input, params=None) -> np.ndarray:
        return self.modelo.predecir(a_categoricas(model_input, self.categorias))


def requisitos() -> list[str]:
    """Paquetes del modelo registrado, en las versiones con que se entrenó."""
    paquetes = {"lightgbm": lgb, "scikit-learn": sklearn, "pandas": pd, "numpy": np}
    return [f"{nombre}=={modulo.__version__}" for nombre, modulo in paquetes.items()]
