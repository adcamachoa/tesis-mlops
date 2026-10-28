"""Pruebas de common.validation: la muestra real de ventas pasa y cada regla falla con un caso sintético."""
from pathlib import Path

import pandas as pd
import pytest

from common.validation import ErrorValidacion, validar

MUESTRA = Path(__file__).parent / "data" / "muestra.parquet"
HOY = pd.Timestamp("2026-10-28")


def _productos(skus: list[str]) -> pd.DataFrame:
    n = len(skus)
    return pd.DataFrame(
        {
            "sku": skus,
            "estilo": [f"EST-{i:05d}" for i in range(n)],
            "estilo_color": [f"EC-{i:06d}" for i in range(n)],
            "talla": "Sin dato",
            "orden_talla": 1.0,
            "grupo_talla": "Sin dato",
            "submarca": "Submarca A",
            "edad": "Sin dato",
            "departamento": "Sin dato",
            "categoria": "Sin dato",
            "tipo_producto": "Sin dato",
            "es_basico": False,
            "temporada_tipo": "Sin dato",
            "temporada_anio": 2026,
        }
    )


@pytest.fixture
def tablas() -> dict[str, pd.DataFrame]:
    """Las cinco tablas válidas: ventas es la muestra real y las demás se arman con sus SKU."""
    ventas = pd.read_parquet(MUESTRA)
    skus = sorted(ventas["sku"].unique())
    fechas = pd.to_datetime(["2026-03-02", "2026-03-03"])
    stock = pd.MultiIndex.from_product([fechas, skus], names=["fecha", "sku"]).to_frame(index=False).assign(stock=5)
    quiebres = pd.DataFrame({"fecha": fechas, "sku": skus[:2], "unidades": [1, 3]})
    calendario = pd.DataFrame(
        {
            "evento": ["Evento 1", "Evento 2"],
            "tipo": ["festividad", "comercial"],
            # el calendario trae eventos anteriores a 2024 y futuros
            "fecha_inicio": pd.to_datetime(["2023-01-06", "2027-05-24"]),
            "fecha_fin": pd.to_datetime(["2023-01-06", "2027-06-01"]),
        }
    )
    return {
        "ventas_diarias": ventas,
        "stock_diario": stock,
        "quiebres_diarios": quiebres,
        "productos": _productos(skus),
        "calendario_eventos": calendario,
    }


def _poner(tabla, columna, valor):
    """Cambia el valor de la primera fila."""

    def cambiar(t):
        t[tabla].loc[0, columna] = valor

    return cambiar


def _a_texto(tabla, columna):
    def cambiar(t):
        t[tabla][columna] = t[tabla][columna].astype(str)

    return cambiar


def _quitar_columna(tabla, columna):
    def cambiar(t):
        t[tabla] = t[tabla].drop(columns=columna)

    return cambiar


def _quitar_primera_fila(tabla):
    def cambiar(t):
        t[tabla] = t[tabla].iloc[1:]

    return cambiar


def _duplicar_primera_fila(tabla):
    def cambiar(t):
        t[tabla] = pd.concat([t[tabla], t[tabla].head(1)], ignore_index=True)

    return cambiar


CASOS = [
    pytest.param(
        _quitar_columna("ventas_diarias", "unidades"), "ventas_diarias: falta la columna unidades", id="falta-columna"
    ),
    pytest.param(
        _a_texto("ventas_diarias", "unidades"),
        "ventas_diarias: la columna unidades debería ser entero",
        id="tipo-entero",
    ),
    pytest.param(
        _a_texto("stock_diario", "fecha"), "stock_diario: la columna fecha debería ser fecha", id="tipo-fecha"
    ),
    pytest.param(_poner("ventas_diarias", "sku", None), "ventas_diarias: filas con nulos en sku: 1", id="nulo-sku"),
    pytest.param(
        _poner("calendario_eventos", "fecha_fin", pd.NaT),
        "calendario_eventos: filas con nulos en fecha_fin: 1",
        id="nulo-calendario",
    ),
    pytest.param(
        _poner("ventas_diarias", "fecha", pd.Timestamp("2026-01-01 03:00")),
        "ventas_diarias: filas con hora distinta de 00:00 en fecha: 1",
        id="fecha-con-hora",
    ),
    pytest.param(
        _poner("quiebres_diarios", "fecha", pd.Timestamp("2023-12-31")),
        "quiebres_diarios: filas con fecha anterior a 2024-01-01: 1",
        id="fecha-anterior",
    ),
    pytest.param(
        _poner("stock_diario", "fecha", HOY + pd.Timedelta(days=1)),
        "stock_diario: filas con fecha posterior a 2026-10-28: 1",
        id="fecha-futura",
    ),
    pytest.param(
        _poner("ventas_diarias", "unidades", -1),
        "ventas_diarias: filas con valor negativo en unidades: 1",
        id="unidades-negativas",
    ),
    pytest.param(
        _poner("stock_diario", "stock", -2), "stock_diario: filas con valor negativo en stock: 1", id="stock-negativo"
    ),
    pytest.param(
        _duplicar_primera_fila("ventas_diarias"),
        "ventas_diarias: filas repetidas por fecha + sku: 1",
        id="ventas-repetidas",
    ),
    pytest.param(
        _duplicar_primera_fila("stock_diario"), "stock_diario: filas repetidas por fecha + sku: 1", id="stock-repetido"
    ),
    pytest.param(
        _duplicar_primera_fila("quiebres_diarios"),
        "quiebres_diarios: filas repetidas por fecha + sku: 1",
        id="quiebres-repetidos",
    ),
    pytest.param(
        _duplicar_primera_fila("productos"), "productos: filas repetidas por sku: 1", id="productos-repetidos"
    ),
    pytest.param(
        _quitar_primera_fila("productos"),
        "ventas_diarias: SKU que no existen en productos: 1",
        id="ventas-sin-producto",
    ),
    pytest.param(
        _poner("stock_diario", "sku", "SKU-999999"),
        "stock_diario: SKU que no existen en productos: 1 (ej.: SKU-999999)",
        id="stock-sin-producto",
    ),
    pytest.param(
        _poner("quiebres_diarios", "sku", "SKU-999999"),
        "quiebres_diarios: SKU que no existen en productos: 1 (ej.: SKU-999999)",
        id="quiebres-sin-producto",
    ),
    pytest.param(
        _poner("calendario_eventos", "fecha_fin", pd.Timestamp("2023-01-05")),
        "calendario_eventos: eventos con fecha_inicio posterior a fecha_fin: 1",
        id="evento-invertido",
    ),
]


def test_muestra_y_tablas_validas_pasan(tablas):
    validar(tablas, hoy=HOY)


@pytest.mark.parametrize(("cambio", "esperado"), CASOS)
def test_cada_regla_falla_con_un_mensaje_claro(tablas, cambio, esperado):
    cambio(tablas)
    with pytest.raises(ErrorValidacion) as error:
        validar(tablas, hoy=HOY)
    assert esperado in str(error.value)


def test_junta_los_problemas_de_todas_las_tablas(tablas):
    _poner("ventas_diarias", "unidades", -1)(tablas)
    _duplicar_primera_fila("productos")(tablas)
    with pytest.raises(ErrorValidacion) as error:
        validar(tablas, hoy=HOY)
    assert len(error.value.problemas) == 2


def test_cruza_con_productos_todos_los_skus_de_una_muestra(tablas):
    # el stock llega como muestra: sus SKU distintos se pasan aparte para que el cruce sea completo
    skus = pd.concat([tablas["stock_diario"]["sku"], pd.Series(["SKU-999999"])])
    with pytest.raises(ErrorValidacion, match="stock_diario: SKU que no existen en productos: 1"):
        validar(tablas, skus={"stock_diario": skus}, hoy=HOY)


def test_acepta_los_tipos_que_entrega_spark(tablas):
    # toPandas convierte los enteros con nulos a float64 y los booleanos con nulos a object
    productos = tablas["productos"]
    productos["temporada_anio"] = productos["temporada_anio"].astype("float64")
    productos.loc[0, "temporada_anio"] = float("nan")
    productos["es_basico"] = productos["es_basico"].astype(object)
    productos.loc[0, "es_basico"] = None
    productos.loc[0, "talla"] = None
    validar(tablas, hoy=HOY)
