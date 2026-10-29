"""El árbol de la jerarquía que se publica: total → submarca → departamento → categoría → serie (tipo de producto).

Funciones puras en pandas. Un nodo se publica si se publicó su padre y su peso llega al mínimo: el índice promedio
semanal de su venta real en las 52 semanas hasta el origen. Con 1 decimal de índice, un nodo más chico quedaría casi
siempre en 0,0, y sus valores de 1 a 3 piezas dejarían ver el escalón de una pieza. Los hijos que no llegan van juntos
a un nodo `otras` de su padre, si su suma llega al mínimo; si no, quedan solo dentro del total del padre.

No se publica por el pronóstico (lo nuevo o de temporada sin historia): en el origen 12-oct-2026 agregaba 32 nodos, casi
todos justo debajo del mínimo, y subía de 0,45 % a 2,24 % los valores publicados de 1 a 3 piezas (fase 9c de PLAN.md).

Cada nodo trae su `id` (la ruta en minúsculas y guiones, estable entre orígenes), el `id` de su padre, su nivel, su
nombre completo (como el de las series: "Submarca A | Departamento | Categoría | Tipo"), su etiqueta y el conjunto de
series que suma.
"""
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from common.features import SERIE

NIVELES = ("total", "submarca", "departamento", "categoria", "serie")  # una por columna de features.SERIE
OTRAS = "otras"
INDICE_MINIMO = 0.5  # índice promedio semanal; ver la fase 9c en PLAN.md


@dataclass(frozen=True)
class Nodo:
    id: str
    padre: str | None
    nivel: str
    nombre: str
    etiqueta: str
    series: frozenset[str]


def _id(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sin_acentos.lower()).strip("-")


def construir(catalogo: pd.DataFrame, pesos: pd.Series, minimo: float = INDICE_MINIMO) -> list[Nodo]:
    """Los nodos que se publican, en profundidad: cada padre justo antes de sus hijos.

    `catalogo` trae una fila por serie con su clave (`serie`) y su ruta (las columnas de features.SERIE). `pesos` es el
    índice promedio semanal de la venta real de cada serie; el de un nodo es la suma de sus series, y una serie sin peso
    cuenta 0. Los hijos de cada nodo van del más pesado al más liviano, con `otras` al final.
    """
    catalogo = catalogo.drop_duplicates("serie").set_index("serie")
    pesos = pesos.reindex(catalogo.index, fill_value=0.0)

    def peso(series) -> float:
        return float(pesos.loc[sorted(series)].sum())

    total = Nodo("total", None, "total", "Total", "Total", frozenset(catalogo.index))
    hijos: dict[str, list[Nodo]] = {}
    padres = [total]
    for profundidad, nivel in enumerate(NIVELES[1:], start=1):
        siguientes = []
        for padre in padres:
            grupos = catalogo.loc[sorted(padre.series)].groupby(list(SERIE[:profundidad]), sort=True)
            publicados, resto = [], set()
            for ruta, del_grupo in grupos:
                ruta = ruta if isinstance(ruta, tuple) else (ruta,)
                series = frozenset(del_grupo.index)
                if peso(series) < minimo:
                    resto |= series
                    continue
                publicados.append(Nodo("/".join(map(_id, ruta)), padre.id, nivel, " | ".join(ruta), ruta[-1], series))
            if not publicados:  # todos los hijos son chicos: el padre no se desglosa
                continue
            publicados.sort(key=lambda n: -peso(n.series))
            siguientes += publicados
            if resto and peso(resto) >= minimo:
                identificador = OTRAS if padre.id == "total" else f"{padre.id}/{OTRAS}"
                publicados.append(Nodo(identificador, padre.id, OTRAS, f"Otras de {padre.nombre}", "Otras",
                                       frozenset(resto)))
            hijos[padre.id] = publicados
        padres = siguientes

    nodos, pila = [], [total]
    while pila:
        nodo = pila.pop()
        nodos.append(nodo)
        pila.extend(reversed(hijos.get(nodo.id, [])))
    ids = [n.id for n in nodos]
    if len(set(ids)) != len(ids):
        repetidos = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"dos nodos del árbol dan el mismo id: {', '.join(repetidos)}")
    return nodos
