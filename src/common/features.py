"""Variables del pronóstico directo: panel serie × semana y base serie × origen × horizonte.

Funciones puras en pandas. Una serie es submarca × departamento × categoría × tipo de producto, y cada semana va de
lunes a domingo y se identifica por su lunes. Las variables de un origen usan solo información hasta el domingo de esa
semana; las de calendario son de la semana objetivo, porque el calendario se conoce por adelantado.
"""
import re
import unicodedata
import warnings
from datetime import date

import numpy as np
import pandas as pd

INICIO = pd.Timestamp("2024-01-01")  # lunes de la primera semana con ventas
INICIO_STOCK = pd.Timestamp("2025-02-17")
FIN_STOCK_SUBESTIMADO = pd.Timestamp("2025-05-01")  # en feb–abr 2025 faltan los SKU que todavía no se movían
SERIE = ["submarca", "departamento", "categoria", "tipo_producto"]
JERARQUIA = ["submarca", "edad", "departamento", "categoria", "tipo_producto"]  # la edad la fija el departamento
HORIZONTE = 13
REZAGOS = 4  # rezago_1 es la venta de la semana del origen
VENTANAS = (4, 8, 13, 26, 52)
SEMANAS_SURTIDO = 8  # estilos-color con venta, lanzamientos y peso de la disponibilidad
SEMANAS_ACTIVA = 52  # una serie entra a la base si vendió en las 52 semanas hasta el origen
SEMANAS_SIN_LANZAMIENTOS = 52  # sin ventas antes de INICIO, en el primer año no se distingue un lanzamiento
TOPE_SEMANAS_EVENTO = 13
EVENTOS_PROXIMOS = {"semanas_a_hot_sale": "Hot Sale", "semanas_a_buen_fin": "El Buen Fin"}


class ErrorCalendario(ValueError):
    """Al calendario le faltan eventos que el horizonte ya alcanza."""


def ultima_semana_completa(corte: pd.Timestamp) -> pd.Timestamp:
    """Lunes de la última semana cuyo domingo cae en el corte o antes."""
    dia = corte.normalize() - pd.Timedelta(days=6)
    return dia - pd.Timedelta(days=dia.weekday())


def clave_serie(df: pd.DataFrame) -> pd.Series:
    return df[SERIE[0]].str.cat(df[SERIE[1:]], sep=" | ")


def _por_serie(valores: pd.Series, semanas: pd.DatetimeIndex, series: pd.Series) -> pd.DataFrame:
    """De valores indexados por (semana, serie) a una tabla semana × serie con todas las semanas y series."""
    return valores.unstack("serie").reindex(index=semanas, columns=series)


def _surtido(ventas: pd.DataFrame, semanas: pd.DatetimeIndex, series: pd.Series) -> dict[str, pd.DataFrame]:
    estilos = ventas[["semana", "serie", "estilo_color"]].drop_duplicates()
    con_venta = (
        pd.crosstab(estilos["semana"], [estilos["serie"], estilos["estilo_color"]])
        .reindex(semanas, fill_value=0)
        .rolling(SEMANAS_SURTIDO, min_periods=1)
        .max()
        .T.groupby(level="serie")
        .sum()
        .T.reindex(columns=series, fill_value=0)
    )
    primera = estilos.groupby(["serie", "estilo_color"])["semana"].min().reset_index()
    nuevos = _por_serie(primera.groupby(["semana", "serie"]).size(), semanas, series).fillna(0)
    lanzamientos = nuevos.rolling(SEMANAS_SURTIDO, min_periods=1).sum()
    lanzamientos.loc[lanzamientos.index < INICIO + pd.Timedelta(weeks=SEMANAS_SIN_LANZAMIENTOS)] = np.nan
    return {f"estilos_color_con_venta_{SEMANAS_SURTIDO}": con_venta, f"lanzamientos_{SEMANAS_SURTIDO}": lanzamientos}


def _oferta(
    stock_sku: pd.DataFrame, ventas: pd.DataFrame, catalogo: pd.DataFrame, semanas: pd.DatetimeIndex, series: pd.Series
) -> dict[str, pd.DataFrame]:
    stock = stock_sku[stock_sku["semana"].isin(semanas)].merge(catalogo[["sku", "estilo_color", "serie"]], on="sku")
    con_stock = stock[stock["stock_cierre"] > 0]
    claves = ["semana", "serie"]
    niveles = {
        "stock_cierre": stock.groupby(claves)["stock_cierre"].sum(),
        "skus_con_stock": con_stock.groupby(claves)["sku"].nunique(),
        "estilos_color_con_stock": con_stock.groupby(claves)["estilo_color"].nunique(),
    }
    salida = {}
    for nombre, valores in niveles.items():
        # sin registros en la semana cuenta como 0; antes de mayo de 2025 quedan nulos porque están subestimados
        ancho = _por_serie(valores, semanas, series).fillna(0)
        ancho.loc[ancho.index < FIN_STOCK_SUBESTIMADO] = np.nan
        salida[nombre] = ancho

    # disponibilidad de la serie: días disponibles de cada SKU, ponderados por su venta en las 8 semanas anteriores
    venta_sku = ventas.pivot_table(index="semana", columns="sku", values="unidades", aggfunc="sum")
    peso = venta_sku.reindex(semanas).fillna(0).rolling(SEMANAS_SURTIDO, min_periods=1).sum().shift(1)
    stock = stock.join(peso.unstack().rename("peso"), on=["sku", "semana"])
    stock["peso"] = stock["peso"].fillna(0)
    stock["ponderado"] = stock["peso"] * stock["dias_disponibles"] / 7
    sumas = stock.groupby(claves)[["ponderado", "peso"]].sum()
    salida["disponibilidad"] = _por_serie(sumas["ponderado"] / sumas["peso"].where(sumas["peso"] > 0), semanas, series)
    return salida


def construir_panel(
    ventas_sku: pd.DataFrame, stock_sku: pd.DataFrame, productos: pd.DataFrame, corte: pd.Timestamp
) -> pd.DataFrame:
    """Panel serie × semana desde INICIO hasta la última semana completa, con venta 0 en las semanas sin venta.

    `ventas_sku` (semana, sku, unidades) y `stock_sku` (semana, sku, stock_cierre, dias_disponibles) llegan agregados a
    semana × SKU. Además de la venta, cada semana trae el surtido y la oferta de la serie.
    """
    semanas = pd.date_range(INICIO, ultima_semana_completa(corte), freq="7D")
    catalogo = productos[["sku", "estilo_color", *JERARQUIA]].assign(serie=clave_serie)
    ventas = ventas_sku[ventas_sku["semana"].isin(semanas)].merge(catalogo, on="sku")
    series = ventas[["serie", *JERARQUIA]].drop_duplicates("serie").sort_values("serie")

    columnas = {
        # float siempre: si ninguna serie tiene semanas sin venta, el tipo no puede depender del corte
        "venta": _por_serie(ventas.groupby(["semana", "serie"])["unidades"].sum(), semanas, series["serie"])
        .fillna(0)
        .astype(float),
        **_surtido(ventas, semanas, series["serie"]),
        **_oferta(stock_sku, ventas, catalogo, semanas, series["serie"]),
    }
    panel = pd.DataFrame({nombre: ancho.unstack() for nombre, ancho in columnas.items()})
    panel.index.names = ["serie", "semana"]
    panel = series.merge(panel.reset_index(), on="serie")[["serie", "semana", *JERARQUIA, *columnas]]
    return panel.sort_values(["serie", "semana"]).reset_index(drop=True)


def _venta_en(venta: pd.Series, series: pd.Series, semanas: pd.Series) -> np.ndarray:
    return venta.reindex(pd.MultiIndex.from_arrays([series, semanas])).to_numpy()


def construir_base(panel: pd.DataFrame, calendario: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Base del pronóstico directo: una fila por serie activa × semana de origen × horizonte h = 1..13.

    Las variables del origen salen de las semanas hasta el origen. `venta_anio_anterior` es la venta de la semana
    objetivo menos 52, que siempre cae antes del origen, e `y` es la venta de la semana objetivo, nula si todavía no
    ocurrió. Una serie está activa si vendió en las 52 semanas hasta el origen: la venta de las que no lo están no queda
    en la base y hay que sumarla al medir el error del total.
    """
    venta = panel.pivot(index="semana", columns="serie", values="venta")
    variables = {f"rezago_{k}": venta.shift(k - 1) for k in range(1, REZAGOS + 1)}
    for ventana in VENTANAS:
        variables[f"media_{ventana}"] = venta.rolling(ventana).mean()
    variables["tendencia"] = variables["media_4"] / variables["media_13"].where(variables["media_13"] > 0)
    suma_13 = venta.rolling(13).sum()
    suma_13_anterior = suma_13.shift(52)
    variables["crecimiento_anual"] = suma_13 / suma_13_anterior.where(suma_13_anterior > 0)
    variables["activa"] = (venta > 0).astype(float).rolling(SEMANAS_ACTIVA, min_periods=1).max()

    origenes = pd.DataFrame({nombre: ancho.unstack() for nombre, ancho in variables.items()})
    activa = origenes.pop("activa") > 0
    origenes = origenes[activa]
    origenes.index.names = ["serie", "origen"]
    # jerarquía, surtido y oferta de la semana de origen
    del_panel = panel.drop(columns="venta").rename(columns={"semana": "origen"})
    origenes = del_panel.merge(origenes.reset_index(), on=["serie", "origen"])

    base = origenes.loc[origenes.index.repeat(HORIZONTE)].reset_index(drop=True)
    base.insert(2, "h", np.tile(np.arange(1, HORIZONTE + 1), len(origenes)))
    base.insert(3, "semana_objetivo", base["origen"] + pd.to_timedelta(7 * base["h"], unit="D"))
    venta_larga = venta.unstack()
    hace_un_anio = base["semana_objetivo"] - pd.Timedelta(weeks=52)
    base["venta_anio_anterior"] = _venta_en(venta_larga, base["serie"], hace_un_anio)
    base = base.merge(variables_calendario(calendario, base["semana_objetivo"]), on="semana_objetivo", how="left")
    base["y"] = _venta_en(venta_larga, base["serie"], base["semana_objetivo"])
    return base


def _slug(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def revisar_cobertura(calendario: pd.DataFrame, semanas: pd.DatetimeIndex) -> None:
    """Se detiene si alguna semana llega a la fecha más temprana en que ha empezado un evento que falta ese año."""
    ultimo_dia = semanas.max() + pd.Timedelta(days=6)
    faltantes = []
    for evento, filas in calendario.groupby("evento"):
        anios = set(filas["fecha_inicio"].dt.year)
        mes, dia = min((inicio.month, inicio.day) for inicio in filas["fecha_inicio"])
        for anio in range(min(anios), ultimo_dia.year + 1):
            limite = pd.Timestamp(year=anio, month=mes, day=dia)
            if anio not in anios and ultimo_dia >= limite:
                faltantes.append(f"{evento} {anio} (puede empezar desde el {limite.date()})")
    if faltantes:
        raise ErrorCalendario(
            "El horizonte llega a eventos que faltan en calendario_eventos; agrega sus fechas: " + ", ".join(faltantes)
        )


def _fechas_evento(calendario: pd.DataFrame, evento: str, hasta: int) -> pd.DataFrame:
    """Inicio y fin del evento cada año hasta `hasta`. Los años que faltan se estiman con la misma semana ISO y el mismo
    día de la semana del año anterior."""
    filas = calendario[calendario["evento"] == evento]
    if filas.empty:
        raise ErrorCalendario(f"El calendario no tiene {evento}")
    fechas = zip(filas["fecha_inicio"], filas["fecha_fin"], strict=True)
    por_anio = {inicio.year: (inicio, fin, False) for inicio, fin in fechas}
    for anio in range(min(por_anio), hasta + 1):
        if anio not in por_anio:
            inicio, fin, _ = por_anio[anio - 1]
            iso = inicio.isocalendar()
            estimado = pd.Timestamp(date.fromisocalendar(anio, min(iso.week, 52), iso.weekday))
            por_anio[anio] = (estimado, estimado + (fin - inicio), True)
    return pd.DataFrame(sorted(por_anio.values()), columns=["inicio", "fin", "estimada"])


def _semanas_al_proximo(calendario: pd.DataFrame, evento: str, semanas: pd.DatetimeIndex) -> np.ndarray:
    """Semanas hasta la semana en que empieza el próximo `evento` (0 si está en curso), con tope TOPE_SEMANAS_EVENTO.

    Si el próximo evento todavía no está en el calendario se usa una fecha estimada, nunca un vacío: en el entrenamiento
    la variable nunca viene vacía, y LightGBM trata como 0 ("el evento es esta semana") un vacío que no vio al entrenar.
    """
    fechas = _fechas_evento(calendario, evento, hasta=semanas.max().year + 1)
    valores, estimados = [], set()
    for semana in semanas:
        proximo = fechas[fechas["fin"] >= semana].iloc[0]
        lunes_evento = proximo["inicio"] - pd.Timedelta(days=proximo["inicio"].weekday())
        semanas_al_evento = max(0, (lunes_evento - semana).days // 7)
        if proximo["estimada"] and semanas_al_evento < TOPE_SEMANAS_EVENTO:
            estimados.add(proximo["inicio"].date())
        valores.append(min(semanas_al_evento, TOPE_SEMANAS_EVENTO))
    if estimados:
        warnings.warn(
            f"{evento} no está en el calendario: para las semanas al próximo se usa como fecha estimada "
            f"{', '.join(map(str, sorted(estimados)))} (misma semana ISO del año anterior)",
            stacklevel=2,
        )
    return np.array(valores, dtype=float)


def variables_calendario(calendario: pd.DataFrame, semanas) -> pd.DataFrame:
    """Variables de calendario de cada semana objetivo: días de cada evento y de quincena (el 15 y el último día del
    mes), semana ISO, mes y semanas al próximo Hot Sale y Buen Fin."""
    semanas = pd.DatetimeIndex(pd.unique(np.asarray(semanas))).sort_values()
    revisar_cobertura(calendario, semanas)
    dias = [semanas + pd.Timedelta(days=d) for d in range(7)]
    salida = pd.DataFrame({"semana_objetivo": semanas})
    for evento, filas in calendario.groupby("evento"):
        salida[f"dias_{_slug(evento)}"] = sum(
            ((dia >= inicio) & (dia <= fin)).astype(int)
            for inicio, fin in zip(filas["fecha_inicio"], filas["fecha_fin"], strict=True)
            for dia in dias
        )
    salida["dias_quincena"] = sum(((dia.day == 15) | dia.is_month_end).astype(int) for dia in dias)
    salida["semana_iso"] = semanas.isocalendar()["week"].to_numpy(dtype="int64")
    salida["mes"] = (semanas + pd.Timedelta(days=3)).month  # el mes del jueves, que tiene la mayoría de los días
    for columna, evento in EVENTOS_PROXIMOS.items():
        salida[columna] = _semanas_al_proximo(calendario, evento, semanas)
    return salida
