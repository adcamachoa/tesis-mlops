"""Validación de los datos crudos antes de escribir las tablas bronze.

Funciones puras sobre pandas. Cada regla devuelve la lista de problemas que encontró, en vez de cortar en el primero;
`validar` junta los de todas las tablas y lanza ErrorValidacion con un solo mensaje.
"""
from dataclasses import dataclass

import pandas as pd

FECHA_MINIMA = pd.Timestamp("2024-01-01")
EJEMPLOS = 3  # filas de ejemplo en cada problema


@dataclass(frozen=True)
class Esquema:
    columnas: dict[str, str]  # columna -> tipo: fecha, texto, entero, numero o booleano
    no_nulas: tuple[str, ...]
    unica: tuple[str, ...] = ()  # clave que no se puede repetir
    no_negativas: tuple[str, ...] = ()
    diaria: bool = False  # fecha × sku: fechas entre FECHA_MINIMA y hoy, y SKU que existen en productos


ESQUEMAS = {
    "ventas_diarias": Esquema(
        columnas={"fecha": "fecha", "sku": "texto", "unidades": "entero"},
        no_nulas=("fecha", "sku", "unidades"),
        unica=("fecha", "sku"),
        no_negativas=("unidades",),
        diaria=True,
    ),
    "stock_diario": Esquema(
        columnas={"fecha": "fecha", "sku": "texto", "stock": "entero"},
        no_nulas=("fecha", "sku", "stock"),
        unica=("fecha", "sku"),
        no_negativas=("stock",),
        diaria=True,
    ),
    "quiebres_diarios": Esquema(
        columnas={"fecha": "fecha", "sku": "texto", "unidades": "entero"},
        no_nulas=("fecha", "sku", "unidades"),
        unica=("fecha", "sku"),
        no_negativas=("unidades",),
        diaria=True,
    ),
    "productos": Esquema(
        columnas={
            "sku": "texto",
            "estilo": "texto",
            "estilo_color": "texto",
            "talla": "texto",
            "orden_talla": "numero",
            "grupo_talla": "texto",
            "submarca": "texto",
            "edad": "texto",
            "departamento": "texto",
            "categoria": "texto",
            "tipo_producto": "texto",
            "es_basico": "booleano",
            "temporada_tipo": "texto",
            "temporada_anio": "entero",
        },
        # códigos y jerarquía de navegación; el resto puede venir vacío
        no_nulas=("sku", "estilo", "estilo_color", "submarca", "edad", "departamento", "categoria", "tipo_producto"),
        unica=("sku",),
    ),
    "calendario_eventos": Esquema(
        columnas={"evento": "texto", "tipo": "texto", "fecha_inicio": "fecha", "fecha_fin": "fecha"},
        no_nulas=("evento", "tipo", "fecha_inicio", "fecha_fin"),
    ),
}


class ErrorValidacion(Exception):
    """Los datos no pasaron la validación; `problemas` trae cada uno como texto."""

    def __init__(self, problemas: list[str]):
        self.problemas = problemas
        detalle = "\n".join(f"- {p}" for p in problemas)
        super().__init__(f"La validación encontró {len(problemas)} problema(s):\n{detalle}")


def _ejemplos(filas: pd.DataFrame) -> str:
    registros = filas.head(EJEMPLOS).to_dict("records")
    return "; ".join(", ".join(f"{c}={v}" for c, v in r.items()) for r in registros)


def _es_tipo(serie: pd.Series, tipo: str) -> bool:
    if tipo == "fecha":
        return pd.api.types.is_datetime64_any_dtype(serie)
    if tipo == "entero":
        # al pasar de Spark a pandas, una columna entera con nulos llega como float64
        if pd.api.types.is_float_dtype(serie):
            return bool((serie.dropna() % 1 == 0).all())
        return pd.api.types.is_integer_dtype(serie)
    if tipo == "numero":
        return pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    # texto y booleano pueden llegar como object, con None en los nulos: se infiere de los valores
    esperado = {"texto": "string", "booleano": "boolean"}[tipo]
    return pd.api.types.infer_dtype(serie, skipna=True) in (esperado, "empty")


def revisar_columnas(df: pd.DataFrame, columnas: dict[str, str]) -> list[str]:
    problemas = []
    for columna, tipo in columnas.items():
        if columna not in df.columns:
            problemas.append(f"falta la columna {columna}")
        elif not _es_tipo(df[columna], tipo):
            problemas.append(f"la columna {columna} debería ser {tipo} y es {df[columna].dtype}")
    return problemas


def revisar_nulos(df: pd.DataFrame, columnas: tuple[str, ...]) -> list[str]:
    return [f"filas con nulos en {c}: {n:,}" for c in columnas if (n := int(df[c].isna().sum()))]


def revisar_fechas(
    df: pd.DataFrame, columna: str, desde: pd.Timestamp | None = None, hasta: pd.Timestamp | None = None
) -> list[str]:
    """Fechas sin hora y, si se indican, dentro de [desde, hasta]. Los nulos los revisa revisar_nulos."""
    fechas = df[columna]
    malas = {f"filas con hora distinta de 00:00 en {columna}": fechas.notna() & (fechas != fechas.dt.normalize())}
    if desde is not None:
        malas[f"filas con {columna} anterior a {desde.date()}"] = fechas < desde
    if hasta is not None:
        malas[f"filas con {columna} posterior a {hasta.date()}"] = fechas > hasta
    return [f"{regla}: {filas.sum():,} (ej.: {_ejemplos(df[filas])})" for regla, filas in malas.items() if filas.any()]


def revisar_no_negativos(df: pd.DataFrame, columna: str) -> list[str]:
    negativos = df[columna] < 0
    if not negativos.any():
        return []
    return [f"filas con valor negativo en {columna}: {negativos.sum():,} (ej.: {_ejemplos(df[negativos])})"]


def revisar_duplicados(df: pd.DataFrame, claves: tuple[str, ...]) -> list[str]:
    repetidas = df.duplicated(list(claves))
    if not repetidas.any():
        return []
    ejemplos = _ejemplos(df.loc[repetidas, list(claves)])
    return [f"filas repetidas por {' + '.join(claves)}: {repetidas.sum():,} (ej.: {ejemplos})"]


def revisar_skus_en_productos(skus: pd.Series, skus_productos: pd.Series) -> list[str]:
    faltantes = pd.Index(skus.dropna().unique()).difference(pd.Index(skus_productos))
    if faltantes.empty:
        return []
    ejemplos = ", ".join(map(str, faltantes[:EJEMPLOS]))
    return [f"SKU que no existen en productos: {len(faltantes):,} (ej.: {ejemplos})"]


def revisar_periodos(df: pd.DataFrame) -> list[str]:
    invertidos = df["fecha_inicio"] > df["fecha_fin"]
    if not invertidos.any():
        return []
    return [f"eventos con fecha_inicio posterior a fecha_fin: {invertidos.sum():,} (ej.: {_ejemplos(df[invertidos])})"]


def validar_tabla(nombre: str, df: pd.DataFrame, hoy: pd.Timestamp) -> list[str]:
    """Aplica las reglas de la tabla `nombre`, salvo el cruce con productos, que hace `validar`."""
    esquema = ESQUEMAS[nombre]
    problemas = revisar_columnas(df, esquema.columnas)
    if problemas:  # sin las columnas y tipos correctos, las demás reglas no se pueden revisar
        return problemas

    problemas += revisar_nulos(df, esquema.no_nulas)
    if esquema.unica:
        problemas += revisar_duplicados(df, esquema.unica)
    for columna in esquema.no_negativas:
        problemas += revisar_no_negativos(df, columna)
    for columna in (c for c, tipo in esquema.columnas.items() if tipo == "fecha"):
        if esquema.diaria:
            problemas += revisar_fechas(df, columna, FECHA_MINIMA, hoy)
        else:  # el calendario trae eventos anteriores a FECHA_MINIMA y futuros
            problemas += revisar_fechas(df, columna)
    if nombre == "calendario_eventos":
        problemas += revisar_periodos(df)
    return problemas


def validar(
    tablas: dict[str, pd.DataFrame], skus: dict[str, pd.Series] | None = None, hoy: pd.Timestamp | None = None
) -> None:
    """Valida las tablas y lanza ErrorValidacion con todos los problemas juntos.

    `tablas` va del nombre del archivo a su DataFrame, completo o una muestra. Para una tabla que llega como muestra,
    `skus` puede traer todos sus SKU distintos, para que el cruce con productos sea completo.
    """
    hoy = pd.Timestamp.today().normalize() if hoy is None else hoy
    problemas = []
    for nombre, df in tablas.items():
        problemas += [f"{nombre}: {p}" for p in validar_tabla(nombre, df, hoy)]

    productos = tablas.get("productos")
    if productos is not None and "sku" in productos.columns:
        for nombre, df in tablas.items():
            if ESQUEMAS[nombre].diaria and "sku" in df.columns:
                skus_tabla = (skus or {}).get(nombre, df["sku"])
                problemas += [f"{nombre}: {p}" for p in revisar_skus_en_productos(skus_tabla, productos["sku"])]

    if problemas:
        raise ErrorValidacion(problemas)
