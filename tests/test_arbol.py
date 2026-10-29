"""Pruebas de common.arbol: qué nodos se publican con el índice mínimo, los `otras`, que los hijos siempre sumen su
padre, el orden, los id y los padres."""
import pandas as pd
import pytest

from common.arbol import construir
from common.features import SERIE

FILAS = [  # submarca, departamento, categoría, tipo e índice promedio semanal de su venta real
    ("Submarca A", "Bebé Niño", "Conjuntos", "Conjunto de 2", 30.0),
    ("Submarca A", "Bebé Niño", "Conjuntos", "Conjunto de 3", 10.0),
    ("Submarca A", "Bebé Niño", "Conjuntos", "Chico 1", 0.3),  # con Chico 2 suman 0,6: van solas a "otras"
    ("Submarca A", "Bebé Niño", "Conjuntos", "Chico 2", 0.3),
    ("Submarca A", "Bebé Niño", "Pijamas", "Mameluco", 5.0),  # su único hermano no llega: Pijamas no se desglosa
    ("Submarca A", "Bebé Niño", "Pijamas", "Diminuto", 0.1),
    ("Submarca A", "Bebé Niña", "Vestidos", "Vestido", 8.0),  # sin hermanos: se publica igual que su categoría
    ("Submarca B", "Niño", "Playeras", "Playera", 2.0),  # la submarca publicada más chica: se junta con la C
    ("Submarca C", "Niña", "Nueva", "Lanzamiento", 0.3),  # no llega
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
        ("submarca-a/bebe-nina", "submarca-a", "departamento"),
        ("submarca-a/bebe-nina/vestidos", "submarca-a/bebe-nina", "categoria"),
        ("submarca-a/bebe-nina/vestidos/vestido", "submarca-a/bebe-nina/vestidos", "serie"),
        ("otras", "total", "otras"),
    ]


def test_los_hijos_de_un_nodo_desglosado_tienen_todas_sus_series():
    """Si no, la diferencia entre el padre y sus hijos dejaría ver lo que no se publicó."""
    nodos = construir(*_datos(), minimo=0.5)
    por_id = {n.id: n for n in nodos}
    hijos = {}
    for nodo in nodos:
        if nodo.padre:
            hijos.setdefault(nodo.padre, []).append(nodo)
    assert set(hijos) == {"total", "submarca-a", "submarca-a/bebe-nino", "submarca-a/bebe-nino/conjuntos",
                          "submarca-a/bebe-nina", "submarca-a/bebe-nina/vestidos"}
    for padre, de_padre in hijos.items():
        assert frozenset().union(*(h.series for h in de_padre)) == por_id[padre].series, padre
        assert sum(len(h.series) for h in de_padre) == len(por_id[padre].series), padre  # sin repetir series


def test_un_resto_que_no_llega_se_junta_con_el_hijo_mas_chico():
    nodos = {n.id: n for n in construir(*_datos(), minimo=0.5)}
    # la submarca C (0,3) no llega: se junta con la B, la publicada más chica, y ninguna de las dos se publica sola
    assert nodos["otras"].series == {
        "Submarca B | Niño | Playeras | Playera", "Submarca C | Niña | Nueva | Lanzamiento"}
    assert (nodos["otras"].nombre, nodos["otras"].etiqueta) == ("Otras de Total", "Otras")
    assert not [i for i in nodos if i.startswith(("submarca-b", "submarca-c"))]
    # con un solo hijo publicado, juntarlo dejaría todo en "otras": Pijamas no se desglosa
    assert not [i for i in nodos if i.startswith("submarca-a/bebe-nino/pijamas/")]
    assert len(nodos["submarca-a/bebe-nino/pijamas"].series) == 2


def test_cada_nodo_suma_sus_series_con_su_nombre_completo_y_su_etiqueta():
    catalogo, pesos = _datos()
    nodos = {n.id: n for n in construir(catalogo, pesos, minimo=0.5)}
    assert nodos["total"].series == frozenset(catalogo["serie"])
    conjuntos = nodos["submarca-a/bebe-nino/conjuntos"]
    assert (conjuntos.nombre, conjuntos.etiqueta) == ("Submarca A | Bebé Niño | Conjuntos", "Conjuntos")
    otras = nodos["submarca-a/bebe-nino/conjuntos/otras"]
    assert (otras.nombre, otras.etiqueta) == ("Otras de Submarca A | Bebé Niño | Conjuntos", "Otras")
    assert otras.series == {f"Submarca A | Bebé Niño | Conjuntos | Chico {i}" for i in (1, 2)}  # llegan solas
    # una serie se llama como su clave, igual que en la versión 1
    assert nodos["submarca-a/bebe-nina/vestidos/vestido"].nombre == "Submarca A | Bebé Niña | Vestidos | Vestido"


def test_lo_que_no_tiene_historia_no_se_publica_aunque_sea_nuevo():
    """Publicar también por el pronóstico subía la exposición de 0,45 % a 2,24 % en el 12-oct (fase 9c)."""
    catalogo, pesos = _datos(FILAS + [("Submarca A", "Bebé Niña", "Vestidos", "Nuevo", 0.0)])
    nodos = {n.id: n for n in construir(catalogo, pesos, minimo=0.5)}
    assert "submarca-a/bebe-nina/vestidos/nuevo" not in nodos
    # sin venta igual se junta: el árbol de un origen no cambia si después empieza a vender
    assert "Submarca A | Bebé Niña | Vestidos | Nuevo" in nodos["submarca-a/bebe-nina/vestidos"].series
    assert not [i for i in nodos if i.startswith("submarca-a/bebe-nina/vestidos/")]


def test_dos_nodos_con_el_mismo_id_detienen_el_export():
    filas = FILAS[:2] + [("Submarca A", "Bebé Niño", "Conjuntos!", "Otro", 5.0)]  # "Conjuntos!" da el mismo id
    with pytest.raises(ValueError, match="mismo id: submarca-a/bebe-nino/conjuntos"):
        construir(*_datos(filas), minimo=0.5)
