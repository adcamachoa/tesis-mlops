"""Monitoreo del pronóstico: drift de los datos del origen y desempeño en vivo contra el rango del backtest.

Funciones puras en pandas. Los umbrales están documentados en CLAUDE.md.

- Drift de datos: compara las variables principales de las series en el origen con la ventana de entrenamiento del
  champion, con el PSI de cada variable sin nulos y, aparte, la parte de nulos contra los últimos 13 orígenes de la
  ventana. Revisa también lo que el PSI no ve: la cobertura de la semana del origen (SKU con stock y filas de venta) y
  que el stock llegue a la última semana con venta. Un nivel extremo detiene el job antes de entrenar, salvo que una
  persona lo acepte para ese origen.
- Desempeño: mide cada pronóstico en vivo con el formato estándar cuando cierran sus tres ventanas, y compara el WAPE y
  el sesgo con el mínimo y el máximo por origen del backtest de la versión que pronosticó.

Nada de esto lleva piezas: el PSI, la parte de nulos, el WAPE, el MAPE y el sesgo son proporciones.
"""
import numpy as np
import pandas as pd

from common.backtest import entrenamiento, prueba
from common.features import HORIZONTE, ultima_semana_completa
from common.metrics import formato_estandar

# historia de la serie, surtido y oferta; sin calendario (13 semanas siempre tienen otro calendario que el historial
# completo) ni jerarquía (no cambia de un origen a otro)
VARIABLES = (
    "rezago_1", "media_4", "media_13", "media_52", "tendencia", "crecimiento_anual",
    "estilos_color_con_venta_8", "lanzamientos_8",
    "stock_cierre", "disponibilidad",
)
# solo como dato, sin aviso ni detención: es la media de 4 semanas entre la de 13, así que sube después de cada evento
# (PSI de hasta 1,73 en orígenes normales), y una rotura de la venta ya la marcan rezago_1 y las medias
INFORMATIVAS = ("tendencia",)
TRAMOS = 10  # cortes en los deciles de la referencia
PARTE_MINIMA = 1e-4  # parte con que cuenta un tramo vacío, para que el logaritmo no se vaya a infinito
PSI_AVISO = 0.25
PSI_EXTREMO = 1.0
SEMANAS_NULOS = 13  # los nulos del origen se comparan con los de los últimos 13 orígenes de la ventana
NULOS_AVISO = 0.10  # aumento de la parte de nulos, en puntos
SEMANAS_COBERTURA = 4  # la semana del origen se compara con la más baja de las 4 anteriores
COBERTURA_AVISO = 0.75
COBERTURA_EXTREMO = 0.6
SEMANAS_ATRASO = 1  # semanas que el stock puede atrasar el origen antes de detener el job
COLUMNAS_DATOS = ["origen", "revision", "variable", "valor", "referencia", "umbral_aviso", "umbral_extremo", "nivel"]

METRICAS_VIVO = (  # el formato estándar y el criterio del champion, sin MAE ni RMSE, que van en piezas
    "wape_m1", "wape_m2", "wape_total", "mape_m1", "mape_m2", "mape_total", "sesgo_total", "wape_tipo_mes",
)
METRICAS_ALERTA = ("wape_m1", "wape_m2", "wape_total", "sesgo_total")
SOLO_ARRIBA = ("wape_m1", "wape_m2", "wape_total")  # un WAPE más bajo que el del mejor origen es buena noticia


class DriftExtremo(RuntimeError):
    """El origen tiene un cambio extremo en sus datos y nadie lo aceptó."""


def psi(referencia, actual, tramos: int = TRAMOS) -> float:
    """Índice de estabilidad poblacional (PSI) de `actual` contra `referencia`, sin nulos.

    Los tramos se cortan en los deciles de la referencia; con empates quedan menos tramos. Los tramos de las puntas
    quedan abiertos, así que un valor fuera del rango de la referencia cae en el primero o el último. Es nulo si alguno
    de los dos no tiene datos.
    """
    referencia, actual = (np.asarray(x, dtype=float) for x in (referencia, actual))
    referencia, actual = referencia[~np.isnan(referencia)], actual[~np.isnan(actual)]
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan
    cortes = np.unique(np.quantile(referencia, np.arange(1, tramos) / tramos))

    def partes(valores: np.ndarray) -> np.ndarray:
        conteo = np.bincount(np.searchsorted(cortes, valores), minlength=len(cortes) + 1)
        return np.maximum(conteo / len(valores), PARTE_MINIMA)

    esperada, observada = partes(referencia), partes(actual)
    return float(np.sum((observada - esperada) * np.log(observada / esperada)))


def nivel_psi(valor: float) -> str:
    if np.isnan(valor):
        return "sin datos"
    return "extremo" if valor >= PSI_EXTREMO else "aviso" if valor >= PSI_AVISO else "estable"


def nivel_nulos(origen: float, recientes: float) -> str:
    """Extremo si la variable viene toda nula y en los últimos orígenes de la ventana tenía datos; aviso si la parte de
    nulos sube 10 puntos o más."""
    if np.isnan(recientes):
        return "sin datos"
    if origen == 1 and recientes < 1:
        return "extremo"
    return "aviso" if origen - recientes >= NULOS_AVISO else "estable"


def drift_datos(base: pd.DataFrame, origen: pd.Timestamp, variables=VARIABLES) -> pd.DataFrame:
    """Revisiones de los datos del origen: una fila por variable y revisión (`psi` o `nulos`), con su nivel.

    El origen son las series activas en `origen`, una fila por serie. La ventana son las filas con que se entrena el
    champion (`backtest.entrenamiento`), una por serie × origen. El PSI compara el origen con toda la ventana. La parte
    de nulos se compara solo con los últimos 13 orígenes, porque el historial tiene nulos por diseño: el stock antes de
    mayo de 2025 y las medias largas en el primer año. En `nulos`, `valor` es la parte de nulos del origen y
    `referencia` la de esos 13 orígenes. Las variables de INFORMATIVAS quedan con nivel `informativa`.
    """
    actual = prueba(base, origen).drop_duplicates("serie")
    ventana = entrenamiento(base, origen).drop_duplicates(["serie", "origen"])
    recientes = ventana[ventana["origen"] >= origen - pd.Timedelta(weeks=SEMANAS_NULOS)]
    filas = []
    for variable in variables:
        informativa = variable in INFORMATIVAS
        valor = psi(ventana[variable], actual[variable])
        nivel = "informativa" if informativa else nivel_psi(valor)
        filas.append((origen, "psi", variable, valor, np.nan, PSI_AVISO, PSI_EXTREMO, nivel))
        nulos = float(actual[variable].isna().mean())
        previos = float(recientes[variable].isna().mean()) if len(recientes) else np.nan
        nivel = "informativa" if informativa else nivel_nulos(nulos, previos)
        filas.append((origen, "nulos", variable, nulos, previos, NULOS_AVISO, 1.0, nivel))
    return pd.DataFrame(filas, columns=COLUMNAS_DATOS)


def filas_por_semana(fechas: pd.Series) -> pd.Series:
    """Cuántas filas hay en cada semana, indexadas por su lunes: con las fechas de ventas_diarias, los SKU × día con
    venta."""
    fechas = pd.to_datetime(fechas)
    return (fechas - pd.to_timedelta(fechas.dt.dayofweek, unit="D")).value_counts().sort_index()


def skus_con_stock_por_semana(panel: pd.DataFrame) -> pd.Series:
    """SKU con stock al cierre de cada semana, sumados en todas las series de panel_semanal; nulo antes de que haya
    stock completo."""
    return panel.groupby("semana")["skus_con_stock"].sum(min_count=1)


def cobertura(conteos: pd.Series, origen: pd.Timestamp) -> float:
    """El conteo de la semana del origen entre el más bajo de las 4 semanas anteriores.

    Contra el promedio, la semana después de un Hot Sale baja hasta 0,25 en filas de venta, porque el promedio incluye
    el pico; contra la más baja, ninguna semana normal baja de 0,70. La semana del origen sin filas o sin dato cuenta 0.
    Es nulo si alguna de las 4 anteriores no tiene dato o si la más baja es 0.
    """
    semanas = pd.date_range(end=origen, periods=SEMANAS_COBERTURA + 1, freq="7D")
    valores = conteos.reindex(semanas, fill_value=0)
    previas = valores.iloc[:-1]
    if previas.isna().any() or previas.min() <= 0:
        return np.nan
    return float(np.nan_to_num(valores.iloc[-1]) / previas.min())


def revisar_cobertura(origen: pd.Timestamp, conteos: dict[str, pd.Series]) -> pd.DataFrame:
    """Revisiones de cobertura que el PSI no ve, una por conteo semanal (`skus_con_stock` y `filas_venta`): `valor` es
    la cobertura del origen contra las 4 semanas anteriores. Solo se revisan las caídas."""
    filas = []
    for variable, serie in conteos.items():
        valor = cobertura(serie, origen)
        filas.append((origen, "cobertura", variable, valor, np.nan, COBERTURA_AVISO, COBERTURA_EXTREMO,
                      nivel_cobertura(valor)))
    return pd.DataFrame(filas, columns=COLUMNAS_DATOS)


def nivel_cobertura(valor: float) -> str:
    if np.isnan(valor):
        return "sin datos"
    return "extremo" if valor <= COBERTURA_EXTREMO else "aviso" if valor <= COBERTURA_AVISO else "estable"


def revisar_corte(origen: pd.Timestamp, corte_venta: pd.Timestamp, corte_stock: pd.Timestamp) -> pd.DataFrame:
    """Revisión de que el stock llegue a la última semana completa con venta; `valor` son las semanas de atraso.

    features corta en la última semana completa de las dos tablas. Si una actualización llega sin stock reciente, no
    quedan nulos: el origen se queda atrás y el job vuelve a pronosticar desde un origen viejo sin avisar.
    """
    atraso = max(0, (ultima_semana_completa(corte_venta) - ultima_semana_completa(corte_stock)).days // 7)
    nivel = "extremo" if atraso >= SEMANAS_ATRASO else "estable"
    return pd.DataFrame([(origen, "corte", "stock", float(atraso), np.nan, np.nan, SEMANAS_ATRASO, nivel)],
                        columns=COLUMNAS_DATOS)


def describir(revision) -> str:
    """Una revisión de datos en una línea, sin piezas."""
    if revision.revision == "psi":
        valor = "sin datos" if np.isnan(revision.valor) else f"{revision.valor:.2f}"
        return f"PSI de {revision.variable} {valor} ({revision.nivel})"
    if revision.revision == "nulos":
        return (f"nulos de {revision.variable} {revision.valor:.1%} contra {revision.referencia:.1%} en los "
                f"{SEMANAS_NULOS} orígenes anteriores ({revision.nivel})")
    if revision.revision == "cobertura":
        valor = "sin datos" if np.isnan(revision.valor) else f"{revision.valor:.2f}"
        return (f"cobertura de {revision.variable} {valor} contra la semana más baja de las {SEMANAS_COBERTURA} "
                f"anteriores ({revision.nivel})")
    return f"el stock atrasa el origen {revision.valor:.0f} semanas ({revision.nivel})"


def revisar_detencion(revisiones: pd.DataFrame, origen: pd.Timestamp, aceptado: pd.Timestamp | None) -> str:
    """Se detiene con DriftExtremo si alguna revisión es extrema, salvo que el cambio se haya aceptado para este mismo
    origen. Devuelve qué pasó, para el log.

    Aceptar un atraso del stock no adelanta el origen: el job vuelve a pronosticar desde el mismo origen, con los mismos
    datos. Para un pronóstico nuevo hay que volver a extraer el stock.
    """
    extremas = revisiones[revisiones["nivel"] == "extremo"]
    if extremas.empty:
        return f"origen {origen.date()}: sin cambios extremos"
    detalle = "; ".join(describir(r) for r in extremas.itertuples())
    atraso = (extremas["revision"] == "corte").any()
    nota = (" Con el stock atrasado, aceptar_drift no adelanta el origen: repite el pronóstico desde este mismo origen;"
            " para uno nuevo hay que volver a extraer el stock.") if atraso else ""
    if aceptado is not None and aceptado == origen:
        return f"origen {origen.date()}: cambio extremo aceptado con aceptar_drift: {detalle}.{nota}"
    raise DriftExtremo(
        f"Cambio extremo en los datos del origen {origen.date()}: {detalle}. El job se detiene antes de entrenar. Si "
        f"se revisó y es legítimo, córrelo con el parámetro aceptar_drift={origen.date()}.{nota}"
    )


def ventanas_cerradas(origen: pd.Timestamp, ultimo_origen: pd.Timestamp) -> bool:
    """Si ya se conoce la venta de las tres ventanas del pronóstico de `origen`: la ventana 3 termina el domingo de la
    semana origen + 13, y `ultimo_origen` es el lunes de la última semana completa."""
    return origen + pd.Timedelta(weeks=HORIZONTE) <= ultimo_origen


def por_medir(
    corridas: pd.DataFrame, medidas: pd.DataFrame, ultimo_origen: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Las corridas de predicciones (una fila por fecha_corrida y origen) que todavía no están en `medidas`, separadas
    en las que ya cerraron sus tres ventanas y las que no."""
    ya = set(zip(medidas["fecha_corrida"], medidas["origen"], strict=True))
    nuevas = corridas[[c not in ya for c in zip(corridas["fecha_corrida"], corridas["origen"], strict=True)]]
    cerradas = nuevas["origen"].map(lambda o: ventanas_cerradas(o, ultimo_origen)).astype(bool)
    return nuevas[cerradas], nuevas[~cerradas]


def rango_backtest(por_origen: pd.DataFrame) -> dict[str, float]:
    """Mínimo y máximo de cada métrica de alerta en los orígenes del backtest (`metrics.por_origen`), como
    `<métrica>_min` y `<métrica>_max`."""
    return {f"{m}_{f}": float(por_origen[m].agg(f)) for m in METRICAS_ALERTA for f in ("min", "max")}


def evaluar(ventanas: pd.DataFrame, rango: dict[str, float]) -> dict:
    """Métricas en vivo de un pronóstico (`metrics.a_ventanas` de un solo origen) y si se salen del rango.

    Un WAPE solo se sale por arriba del máximo; el sesgo, por arriba del máximo o por abajo del mínimo.
    """
    estandar = formato_estandar(ventanas)
    metricas = {m: estandar[m] for m in METRICAS_VIVO}
    fuera = [
        m for m in METRICAS_ALERTA
        if metricas[m] > rango[f"{m}_max"] or (m not in SOLO_ARRIBA and metricas[m] < rango[f"{m}_min"])
    ]
    return metricas | rango | {"fuera_de_rango": ",".join(fuera), "alerta": bool(fuera)}


def estados(medidas: pd.DataFrame) -> pd.Series:
    """Estado de cada corrida medida (columnas origen, fecha_corrida y alerta): en rango, alerta, o revisión manual si
    además tuvo alerta el origen medido anterior. De cada origen cuenta su última corrida, como en results.json."""
    ultima = medidas.sort_values("fecha_corrida").groupby("origen")["alerta"].last().sort_index()
    anterior = medidas["origen"].map(ultima.shift(1, fill_value=False)).astype(bool)
    alerta = medidas["alerta"].astype(bool)
    return pd.Series(np.select([alerta & anterior, alerta], ["revisión manual", "alerta"], "en rango"),
                     index=medidas.index)


def _porcentaje(metrica: str, valor: float) -> str:
    return f"{valor:+.1%}" if metrica == "sesgo_total" else f"{valor:.1%}"


def describir_rango(rango: dict[str, float]) -> str:
    """El rango del backtest en una línea, sin piezas."""
    return ", ".join(f"{m} {_porcentaje(m, rango[f'{m}_min'])} a {_porcentaje(m, rango[f'{m}_max'])}"
                     for m in METRICAS_ALERTA)


def describir_medida(medida) -> str:
    """Una medición en vivo en una línea, sin piezas."""
    partes = [f"{m} {_porcentaje(m, medida[m])} (backtest {_porcentaje(m, medida[f'{m}_min'])} a "
              f"{_porcentaje(m, medida[f'{m}_max'])})" for m in METRICAS_ALERTA]
    return f"origen {pd.Timestamp(medida['origen']).date()}: " + ", ".join(partes) + f": {medida['estado']}"
