"""Validación de los datos crudos antes de escribir las tablas bronze.

Funciones puras sobre pandas. Cada regla devuelve la lista de problemas que encontró, en vez de cortar en el primero;
`validar` junta los de todas las tablas y lanza ErrorValidacion con un solo mensaje.

Cada lunes llega una exportación nueva con corte al domingo. Antes de validarla, `revisar_semana_nueva` detiene la
corrida si los datos no traen un domingo nuevo, y `revisar_llave` si la llave de los productos parece regenerada.
"""
from dataclasses import dataclass

import pandas as pd

from common.features import JERARQUIA, ultima_semana_completa

FECHA_MINIMA = pd.Timestamp("2024-01-01")
EJEMPLOS = 3  # filas de ejemplo en cada problema
LLAVE = ("estilo", "estilo_color")
# el catálogo se edita y un SKU puede cambiar de estilo; más del 1 % a la vez es una llave regenerada
CAMBIO_LLAVE_MAXIMO = 0.01


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


class SinSemanaNueva(Exception):
    """Los datos del Volume no traen un domingo nuevo respecto de la última corrida publicada."""


def revisar_semana_nueva(corte_venta: pd.Timestamp, corte_stock: pd.Timestamp, real_hasta: pd.Timestamp | None) -> str:
    """El freno de la corrida semanal: el último domingo completo de la venta y del stock (el corte de features) tiene
    que ser posterior a `real_hasta`, hasta donde llegó la venta real del último export. Si no, lanza SinSemanaNueva,
    antes de gastar la corrida. Sin `real_hasta`, como en la primera corrida semanal, no hay freno. Devuelve qué pasó,
    para el log.

    Se compara con el último export y no con las tablas bronze: una corrida que falló después de la ingesta no lo
    actualizó, así que repetirla pasa el freno.
    """
    domingo = ultima_semana_completa(min(corte_venta, corte_stock)) + pd.Timedelta(days=6)
    if real_hasta is None:
        return f"datos hasta el domingo {domingo.date()}; sin una corrida publicada con qué comparar"
    if domingo > real_hasta:
        return f"semana nueva: datos hasta el domingo {domingo.date()}, y lo publicado llega al {real_hasta.date()}"
    raise SinSemanaNueva(
        f"Los datos del Volume no traen un domingo nuevo: la venta llega al {corte_venta.date()} y el stock al "
        f"{corte_stock.date()}, y la última corrida ya publicó hasta el {real_hasta.date()}. El job se detiene sin "
        f"gastar la corrida: sube los parquet con corte al domingo y córrelo de nuevo."
    )


def comparar_productos(anteriores: pd.DataFrame, nuevos: pd.DataFrame) -> dict[str, int]:
    """Los productos ya cargados (`anteriores`) contra los de la exportación nueva: cuántos SKU había, cuántos faltan,
    cuántos cambiaron de estilo o estilo_color, cuántos cambiaron de jerarquía y cuántos son nuevos."""
    columnas = ["sku", *LLAVE, *JERARQUIA]
    juntos = anteriores[columnas].merge(nuevos[columnas], on="sku", how="left", suffixes=("", "_nuevo"),
                                        indicator=True)
    presentes = juntos[juntos["_merge"] == "both"]

    def cambiaron(grupo) -> int:
        antes = presentes[list(grupo)].to_numpy()
        despues = presentes[[f"{c}_nuevo" for c in grupo]].to_numpy()
        return int((antes != despues).any(axis=1).sum())

    return {
        "existentes": len(anteriores),
        "faltantes": int((juntos["_merge"] == "left_only").sum()),
        "cambio_llave": cambiaron(LLAVE),
        "cambio_jerarquia": cambiaron(JERARQUIA),
        "nuevos": int((~nuevos["sku"].isin(anteriores["sku"])).sum()),
    }


def revisar_llave(conteo: dict[str, int]) -> list[str]:
    """Los problemas de la llave (`comparar_productos`): que falte algún SKU que ya estaba, o que más del 1 % de los
    existentes cambie de estilo o estilo_color, como pasaría con una llave regenerada. La exportación reutiliza la
    llave: los cambios sueltos son ediciones del catálogo y solo van al log."""
    problemas = []
    if conteo["faltantes"]:
        problemas.append(f"productos: faltan {conteo['faltantes']:,} SKU que ya estaban cargados")
    parte = conteo["cambio_llave"] / conteo["existentes"] if conteo["existentes"] else 0.0
    if parte > CAMBIO_LLAVE_MAXIMO:
        problemas.append(
            f"productos: {conteo['cambio_llave']:,} de {conteo['existentes']:,} SKU cambiaron de estilo o estilo_color "
            f"({parte:.1%}), más del {CAMBIO_LLAVE_MAXIMO:.0%}: la llave parece regenerada"
        )
    return problemas


def describir_productos(conteo: dict[str, int]) -> str:
    """Los cambios del catálogo en una línea, solo con conteos."""
    return (f"productos: {conteo['existentes']:,} SKU cargados, {conteo['nuevos']:,} nuevos, {conteo['faltantes']:,} "
            f"faltantes, {conteo['cambio_llave']:,} con otro estilo o estilo_color y {conteo['cambio_jerarquia']:,} "
            f"con otra jerarquía")


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
