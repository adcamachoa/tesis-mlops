"""Validación temporal tipo rolling origin del pronóstico directo, y elección del champion.

Funciones puras en pandas. En cada origen del backtest, cada modelo entrena solo con las filas cuya semana objetivo ya
terminó al cierre del origen, y pronostica las 13 semanas siguientes con las filas de ese origen. Los orígenes caen
cada 4 semanas y, por defecto, el último es el más reciente con sus 13 semanas de venta conocidas.
"""
import math
import time
from typing import Protocol

import numpy as np
import pandas as pd

from common.features import HORIZONTE

ORIGENES = 8
CADA = 4  # semanas entre orígenes
ORIGENES_EXAMEN = 13  # orígenes internos del examen de la parada temprana
CRITERIO = "wape_tipo_mes"
GANA_EN = 0.75  # parte de los orígenes que un algoritmo distinto le tiene que ganar al del champion: 6 de 8


class Modelo(Protocol):
    def ajustar(self, filas: pd.DataFrame, origen: pd.Timestamp) -> None: ...

    def predecir(self, filas: pd.DataFrame) -> np.ndarray: ...


def origenes(
    ultima_semana: pd.Timestamp, n: int = ORIGENES, cada: int = CADA, ultimo: pd.Timestamp | None = None
) -> list[pd.Timestamp]:
    """Los `n` orígenes del backtest, uno cada `cada` semanas, del más antiguo al más reciente.

    `ultima_semana` es el lunes de la última semana con venta conocida. El último origen es `ultimo` o, si no se indica,
    el más reciente cuyas 13 semanas ya se conocen.
    """
    limite = ultima_semana - pd.Timedelta(weeks=HORIZONTE)
    ultimo = limite if ultimo is None else pd.Timestamp(ultimo)
    if ultimo.weekday() != 0:
        raise ValueError(f"El último origen tiene que ser un lunes, y el {ultimo.date()} no lo es")
    if ultimo > limite:
        raise ValueError(
            f"Del origen {ultimo.date()} todavía no se conocen sus {HORIZONTE} semanas; el último posible es "
            f"{limite.date()}"
        )
    return [ultimo - pd.Timedelta(weeks=cada * i) for i in reversed(range(n))]


def entrenamiento(base: pd.DataFrame, origen: pd.Timestamp) -> pd.DataFrame:
    """Filas cuya semana objetivo ya terminó al cierre del origen. Filtrar por fecha de origen no basta: una fila de
    un origen anterior con h grande pronostica una semana que en el origen todavía no ocurría."""
    return base[(base["semana_objetivo"] <= origen) & base["y"].notna()]


def prueba(base: pd.DataFrame, origen: pd.Timestamp) -> pd.DataFrame:
    return base[base["origen"] == origen]


def examen_interno(filas: pd.DataFrame, origen: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajuste y examen de la parada temprana, dentro de las filas de entrenamiento de `origen`.

    El examen son las filas de los 13 orígenes internos de origen − 25 a origen − 13 semanas, cuyo horizonte completo
    cae antes del origen. El ajuste son las filas con semana objetivo hasta origen − 25 semanas, así que no conoce
    ninguna semana que el examen pronostica.
    """
    hasta = origen - pd.Timedelta(weeks=HORIZONTE)
    desde = hasta - pd.Timedelta(weeks=ORIGENES_EXAMEN - 1)
    return filas[filas["semana_objetivo"] <= desde], filas[filas["origen"].between(desde, hasta)]


def _pronosticar(
    nombre: str, modelo: Modelo, entrena: pd.DataFrame, filas: pd.DataFrame, origen: pd.Timestamp, hechos: dict
) -> np.ndarray:
    """Pronóstico de `modelo` en `origen`, guardado en `hechos` por nombre para no entrenar dos veces lo mismo."""
    if nombre not in hechos:
        componentes = getattr(modelo, "componentes", None)
        if componentes:
            hechos[nombre] = np.mean(
                [_pronosticar(n, m, entrena, filas, origen, hechos) for n, m in componentes.items()], axis=0
            )
        else:
            inicio = time.perf_counter()
            modelo.ajustar(entrena, origen)
            hechos[nombre] = modelo.predecir(filas)
            rondas = getattr(modelo, "rondas_elegidas", None)
            detalle = f", rondas {rondas}" if rondas else ""
            print(f"  {origen.date()} {nombre}: {time.perf_counter() - inicio:.0f} s{detalle}")
    return hechos[nombre]


def correr(modelos: dict[str, Modelo], base: pd.DataFrame, origenes: list[pd.Timestamp]) -> pd.DataFrame:
    """Pronóstico de cada modelo en cada origen: modelo, serie, origen, h, semana_objetivo, la venta real `y` y
    pronostico.

    Una mezcla (un modelo con `componentes`) promedia las predicciones de sus componentes. Cada componente se entrena
    una sola vez por origen, aunque también esté en `modelos` con el mismo nombre.
    """
    partes = []
    for origen in origenes:
        entrena, filas = entrenamiento(base, origen), prueba(base, origen)
        print(f"origen {origen.date()}: {len(entrena):,} filas de entrenamiento, {len(filas):,} a pronosticar")
        hechos = {}
        for nombre, modelo in modelos.items():
            pronostico = _pronosticar(nombre, modelo, entrena, filas, origen, hechos)
            partes.append(
                filas[["serie", "origen", "h", "semana_objetivo", "y"]].assign(modelo=nombre, pronostico=pronostico)
            )
    columnas = ["modelo", "serie", "origen", "h", "semana_objetivo", "y", "pronostico"]
    return pd.concat(partes, ignore_index=True)[columnas]


def elegir_mejor(resumen: pd.DataFrame) -> str:
    """El modelo con menor WAPE tipo de producto × mes. `resumen` va indexado por modelo."""
    return resumen[CRITERIO].idxmin()


def supera_al_champion(
    resumen: pd.DataFrame,
    por_origen: pd.DataFrame,
    candidato: str,
    champion: str | None,
    criterio_guardado: float | None = None,
) -> tuple[bool, str]:
    """Si `candidato` se queda con el alias champion, y por qué.

    `por_origen` trae el criterio de cada modelo (columnas) en cada origen (filas). Si `candidato` es el algoritmo del
    champion, la versión nueva es el champion reentrenado con datos más recientes y se queda con el alias. Un algoritmo
    distinto tiene que mejorar el criterio del champion medido en este mismo backtest, porque el WAPE guardado se midió
    en otros orígenes y un trimestre puede ser más difícil que otro. Además tiene que ganarle en al menos 6 de cada 8
    orígenes, para que el alias no cambie por ruido. Solo si el algoritmo del champion ya no está entre los modelos se
    usa el WAPE guardado en su versión.
    """
    if champion is None:
        return True, "no hay champion"
    if candidato == champion:
        return True, f"{candidato} es el algoritmo del champion, reentrenado"
    nuevo = resumen.loc[candidato, CRITERIO]
    if champion in resumen.index:
        actual = resumen.loc[champion, CRITERIO]
        gana = int((por_origen[candidato] < por_origen[champion]).sum())
        minimo = math.ceil(GANA_EN * len(por_origen))
        motivo = (
            f"{candidato} {nuevo:.2%} contra {champion} {actual:.2%} en el mismo backtest; le gana en {gana} de "
            f"{len(por_origen)} orígenes y necesita {minimo}"
        )
        return bool(nuevo < actual and gana >= minimo), motivo
    if criterio_guardado is None:
        return False, f"el champion {champion} ya no está entre los modelos y no tiene su WAPE guardado"
    return (
        bool(nuevo < criterio_guardado),
        f"{candidato} {nuevo:.2%} contra el WAPE guardado de {champion}, {criterio_guardado:.2%} (otro backtest)",
    )
