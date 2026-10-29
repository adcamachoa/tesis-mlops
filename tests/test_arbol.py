"""Pruebas de common.arbol: qué nodos se publican con el índice mínimo, los `otras`, el orden, los id y los padres."""
import pandas as pd
import pytest

from common.arbol import construir
from common.features import SERIE

FILAS = [  # submarca, departamento, categoría, tipo e índice promedio semanal de su venta real
    ("Submarca A", "Bebé Niño", "Conjuntos", "Conjunto de 2", 30.0),
    ("Submarca A", "Bebé Niño", "Conjuntos", "Conjunto de 3", 10.0),
    ("Submarca A", "Bebé Niño", "Conjuntos", "Chico 1", 0.3),  # con Chico 2 suman 0,6: su "otras" se publica
    ("Submarca A", "Bebé Niño", "Conjuntos", "Chico 2", 0.3),
    ("Submarca A", "Bebé Niño", "Pijamas", "Mameluco", 5.0),
    ("Submarca A", "Bebé Niño", "Pijamas", "Diminuto", 0.1),  # solo no llega: queda dentro de Pijamas
    ("Submarca A", "Bebé Niña", "Vestidos", "Vestido", 0.4),  # el departamento no llega, y su resto tampoco
    ("Submarca B", "Niño", "Playeras", "Playera", 0.2),  # la submarca entera no llega
    ("Submarca B", "Niño", "Playeras", "Camisa", 0.2),
    ("Submarca C", "Niña", "Nueva", "Lanzamiento", 0.0),  # nueva, sin historia: no se publica
]


def _datos(filas=FILAS) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame(filas, columns=[*SERIE, "peso"])
    df["serie"] = df[SERIE].agg(" | ".join, axis=1)
    return df[["serie", *SERIE]], df.set_index("serie")["peso"]


def test_se_publican_los_nodos_que_llegan_al_minimo_con_cada_padre_antes_de_sus_hijos():
    nodos = construir(*_datos(), minimo=0.5)
    assert [(n.id, n.padre, n.nivel) for n in nodos] == [
        ("total", None, "total"),
        ("submarca-a", "total", "submarca"),
        ("submarca-a/bebe-nino", "submarca-a", "departamento"),
        ("submarca-a/bebe-nino/conjuntos", "submarca-a/bebe-nino", "categoria"),
        ("submarca-a/bebe-nino/conjuntos/conjunto-de-2", "submarca-a/bebe-nino/conjuntos", "serie"),
        ("submarca-a/bebe-nino/conjuntos/conjunto-de-3", "submarca-a/bebe-nino/conjuntos", "serie"),
        ("submarca-a/bebe-nino/conjuntos/otras", "submarca-a/bebe-nino/conjuntos", "otras"),
        ("submarca-a/bebe-nino/pijamas", "submarca-a/bebe-nino", "categoria"),
        ("submarca-a/bebe-nino/pijamas/mameluco", "submarca-a/bebe-nino/pijamas", "serie"),
    ]


def test_cada_nodo_suma_sus_series_con_su_nombre_completo_y_su_etiqueta():
    catalogo, pesos = _datos()
    nodos = {n.id: n for n in construir(catalogo, pesos, minimo=0.5)}
    assert nodos["total"].series == frozenset(catalogo["serie"])  # también las que no se publican
    conjuntos = nodos["submarca-a/bebe-nino/conjuntos"]
    assert (conjuntos.nombre, conjuntos.etiqueta) == ("Submarca A | Bebé Niño | Conjuntos", "Conjuntos")
    assert len(conjuntos.series) == 4
    otras = nodos["submarca-a/bebe-nino/conjuntos/otras"]
    assert (otras.nombre, otras.etiqueta) == ("Otras de Submarca A | Bebé Niño | Conjuntos", "Otras")
    assert otras.series == {f"Submarca A | Bebé Niño | Conjuntos | Chico {i}" for i in (1, 2)}
    # una serie se llama como su clave, igual que en la versión 1
    assert nodos["submarca-a/bebe-nino/pijamas/mameluco"].nombre == "Submarca A | Bebé Niño | Pijamas | Mameluco"


def test_lo_que_no_tiene_historia_no_se_publica_aunque_sea_nuevo():
    """Publicar también por el pronóstico subía la exposición de 0,45 % a 2,24 % en el 12-oct (fase 9c)."""
    ids = {n.id for n in construir(*_datos(), minimo=0.5)}
    assert not {i for i in ids if i.startswith(("submarca-b", "submarca-c"))}


def test_un_nodo_sin_hijos_que_lleguen_no_se_desglosa():
    catalogo, pesos = _datos()
    # la submarca A (46,1) y Bebé Niño (45,7) llegan; ninguna de sus categorías, así que Bebé Niño no se desglosa
    nodos = construir(catalogo, pesos, minimo=45.0)
    assert [n.id for n in nodos] == ["total", "submarca-a", "submarca-a/bebe-nino"]


def test_dos_nodos_con_el_mismo_id_detienen_el_export():
    filas = FILAS[:2] + [("Submarca A", "Bebé Niño", "Conjuntos!", "Otro", 5.0)]  # "Conjuntos!" da el mismo id
    with pytest.raises(ValueError, match="mismo id: submarca-a/bebe-nino/conjuntos"):
        construir(*_datos(filas), minimo=0.5)
