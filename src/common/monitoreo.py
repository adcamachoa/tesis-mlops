"""Monitoreo del pronóstico: drift de los datos del origen y desempeño en vivo contra el rango del backtest.

Funciones puras en pandas. Los umbrales están documentados en CLAUDE.md.

- Drift de datos: compara las variables principales de las series en el origen con la ventana de entrenamiento del
  champion, con el PSI de cada variable sin nulos y, aparte, la parte de nulos contra los últimos 13 orígenes de la
  ventana. Revisa también lo que el PSI no ve: la cobertura de la semana del origen (SKU con stock y filas de venta), la
  del mes contra las semanas que ya revisó la corrida anterior (SKU con stock y venta entre salidas de stock) y que el
  stock llegue a la última semana con venta. Un nivel extremo detiene el job antes de entrenar, salvo que una persona
  lo acepte para ese origen.
- Desempeño: mide cada ventana de cada pronóstico en vivo apenas cierra (el mes 1, el mes 2 y, al cerrar el mes 3, el
  total), con el formato estándar, y la compara con el mínimo y el máximo por origen del backtest de la versión que
  pronosticó. Una alerta es una señal para revisar el modelo: no adelanta el reentrenamiento, que va cada 4 semanas.

Nada de esto lleva piezas: el PSI, la parte de nulos, las coberturas, el WAPE, el MAPE y el sesgo son proporciones.
"""
import numpy as np
import pandas as pd

from common.backtest import entrenamiento, prueba
from common.features import ultima_semana_completa
from common.metrics import fechas_ventanas, formato_estandar, sesgo

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
SEMANAS_REVISADAS = 4  # las semanas nuevas del mes se comparan con la más baja de las 4 que revisó la corrida anterior
COBERTURA_AVISO = 0.75  # los mismos para la semana y para el mes
COBERTURA_EXTREMO = 0.6
SEMANAS_ATRASO = 1  # semanas que el stock puede atrasar el origen antes de detener el job
COLUMNAS_DATOS = ["origen", "revision", "variable", "valor", "referencia", "umbral_aviso", "umbral_extremo", "nivel"]

METRICAS_VIVO = (  # el formato estándar, el sesgo de cada mes y el criterio del champion; sin MAE ni RMSE (en piezas)
    "wape_m1", "wape_m2", "wape_total", "mape_m1", "mape_m2", "mape_total", "sesgo_m1", "sesgo_m2", "sesgo_total",
    "wape_tipo_mes",
)
METRICAS_ALERTA = ("wape_m1", "wape_m2", "wape_total", "sesgo_total")
SOLO_ARRIBA = ("wape_m1", "wape_m2", "wape_total")  # un WAPE más bajo que el del mejor origen es buena noticia
# Cada medición se hace en la primera corrida después de que cierra su ventana. Alerta con sus métricas del formato
# estándar, contra el rango del backtest; el resto va solo como dato, también el sesgo de cada mes (sin rango fijado).
MEDICIONES = {
    "mes_1": {"ventana": 1, "alerta": ("wape_m1",), "dato": ("mape_m1", "sesgo_m1")},
    "mes_2": {"ventana": 2, "alerta": ("wape_m2",), "dato": ("mape_m2", "sesgo_m2")},
    "total": {"ventana": 3, "alerta": ("wape_total", "sesgo_total"), "dato": ("mape_total", "wape_tipo_mes")},
}
SEMANAS_REENTRENAMIENTO = 4  # un pronóstico cada 4 semanas; una alerta de desempeño no lo adelanta


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


def _lunes(fechas: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(fechas)
    return fechas - pd.to_timedelta(fechas.dt.dayofweek, unit="D")


def filas_por_semana(fechas: pd.Series) -> pd.Series:
    """Cuántas filas hay en cada semana, indexadas por su lunes: con las fechas de ventas_diarias, los SKU × día con
    venta."""
    return _lunes(fechas).value_counts().sort_index()


def unidades_por_semana(ventas: pd.DataFrame) -> pd.Series:
    """Unidades vendidas en cada semana (fecha y unidades de ventas_diarias), indexadas por su lunes."""
    return ventas["unidades"].groupby(_lunes(ventas["fecha"])).sum().astype(float)


def salidas_por_semana(stock: pd.DataFrame) -> pd.Series:
    """Salidas de stock de cada semana (fecha, sku y stock de stock_diario), indexadas por su lunes.

    Por SKU y día, lo que bajó el stock desde el día anterior: 0 si subió o si no hay dato del día anterior. El stock
    viene de los movimientos del almacén, independiente de la venta. src/monitoreo_datos.py lo calcula igual en Spark.
    """
    stock = stock.sort_values(["sku", "fecha"], kind="stable")
    por_sku = stock.groupby("sku", observed=True)
    consecutivo = stock["fecha"] - por_sku["fecha"].shift(1) == pd.Timedelta(days=1)
    salida = (por_sku["stock"].shift(1) - stock["stock"]).clip(lower=0).where(consecutivo, 0)
    return salida.groupby(_lunes(stock["fecha"])).sum().astype(float)


def venta_entre_salidas(unidades: pd.Series, salidas: pd.Series) -> pd.Series:
    """Unidades vendidas entre salidas de stock, por semana; nula si la semana no tiene salidas. Un evento mueve las dos
    cosas a la vez, pero una venta que llega incompleta solo baja la venta."""
    semanas = unidades.index.union(salidas.index)
    salidas = salidas.reindex(semanas)
    return unidades.reindex(semanas, fill_value=0) / salidas.where(salidas > 0)


def origen_revisado(origen: pd.Timestamp, revisiones: pd.DataFrame, origenes_pronosticados) -> pd.Timestamp:
    """El origen de la última corrida anterior a `origen` cuyas semanas ya se revisaron.

    Cuentan los orígenes de monitoreo_datos cuya corrida no tuvo extremos o los aceptó (`revisiones` trae fecha_corrida,
    origen, nivel y aceptado), y los de predicciones, porque una corrida que se detuvo no pronostica. Si no hay ninguno
    anterior, como en la primera corrida de un schema nuevo, origen − 4 semanas.
    """
    candidatos = set(pd.to_datetime(list(origenes_pronosticados)))
    if len(revisiones):
        por_corrida = revisiones.groupby(["fecha_corrida", "origen"]).agg(
            extremo=("nivel", lambda n: bool((n == "extremo").any())), aceptado=("aceptado", "any")
        ).reset_index()
        candidatos |= set(pd.to_datetime(por_corrida.loc[~por_corrida["extremo"] | por_corrida["aceptado"], "origen"]))
    anteriores = [o for o in candidatos if o < origen]
    return max(anteriores) if anteriores else origen - pd.Timedelta(weeks=SEMANAS_REVISADAS)


def cobertura_mes(serie: pd.Series, origen: pd.Timestamp, revisado: pd.Timestamp, nulo_es_cero: bool) -> float:
    """La semana nueva más baja (de `revisado` + 1 al origen) entre la más baja de las 4 semanas hasta `revisado`, que
    ya revisó la corrida anterior.

    Con el refresco mensual llegan 4 o 5 semanas a la vez: si todas vienen rotas, la cobertura semanal las compara entre
    sí y no lo ve. Con `nulo_es_cero`, una semana nueva sin dato cuenta 0; si no, la revisión queda nula. También es
    nula si a las semanas revisadas les falta dato o si la más baja es 0.
    """
    revisadas = serie.reindex(pd.date_range(end=revisado, periods=SEMANAS_REVISADAS, freq="7D"))
    nuevas = serie.reindex(pd.date_range(revisado + pd.Timedelta(weeks=1), origen, freq="7D"))
    if nuevas.empty or revisadas.isna().any() or revisadas.min() <= 0:
        return np.nan
    if nulo_es_cero:
        nuevas = nuevas.fillna(0)
    elif nuevas.isna().any():
        return np.nan
    return float(nuevas.min() / revisadas.min())


def revisar_cobertura_mes(origen: pd.Timestamp, revisado: pd.Timestamp, conteos: dict[str, pd.Series]) -> pd.DataFrame:
    """Revisiones de la cobertura del mes, una por serie semanal (`skus_con_stock` y `venta_salidas`); `referencia` es
    cuántas semanas nuevas se compararon, así que la corrida anterior es origen − referencia semanas.

    Sin SKU con stock, la semana cuenta 0. Sin salidas de stock, venta entre salidas queda `sin datos`: la venta del mes
    no se pudo verificar, y aceptar_drift no la cambia.
    """
    semanas = float((origen - revisado).days // 7)
    filas = []
    for variable, serie in conteos.items():
        valor = cobertura_mes(serie, origen, revisado, nulo_es_cero=variable == "skus_con_stock")
        filas.append((origen, "cobertura_mes", variable, valor, semanas, COBERTURA_AVISO, COBERTURA_EXTREMO,
                      nivel_cobertura(valor)))
    return pd.DataFrame(filas, columns=COLUMNAS_DATOS)


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
    if revision.revision == "cobertura_mes":
        revisado = (pd.Timestamp(revision.origen) - pd.Timedelta(weeks=int(revision.referencia))).date()
        if np.isnan(revision.valor) and revision.variable == "venta_salidas":
            return (f"la venta del mes no se verificó: faltan salidas de stock en las semanas nuevas desde el origen "
                    f"{revisado} ({revision.nivel})")
        valor = "sin datos" if np.isnan(revision.valor) else f"{revision.valor:.2f}"
        return (f"cobertura del mes de {revision.variable} {valor}: la más baja de las {int(revision.referencia)} "
                f"semanas nuevas contra la más baja de las {SEMANAS_REVISADAS} revisadas hasta el origen {revisado} "
                f"({revision.nivel})")
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


def cierre(origen: pd.Timestamp, medicion: str) -> pd.Timestamp:
    """El último día de la ventana que permite hacer `medicion` sobre el pronóstico de `origen`."""
    return fechas_ventanas(origen)[MEDICIONES[medicion]["ventana"] - 1][1]


def corrida_de_medicion(origen: pd.Timestamp, medicion: str) -> pd.Timestamp:
    """El lunes de la corrida semanal que hace `medicion`: la primera cuya venta (hasta el domingo anterior) llega al
    cierre de su ventana."""
    fin = cierre(origen, medicion)
    return fin + pd.Timedelta(days=7 - fin.dayofweek)


def por_medir(
    corridas: pd.DataFrame, medidas: pd.DataFrame, real_hasta: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Las mediciones de cada corrida de predicciones (`corridas`, una fila por fecha_corrida y origen) que todavía no
    están en `medidas`, con su `medicion` y su `cierre`, separadas en las que ya cerraron (`cierre` a lo más
    `real_hasta`, el último domingo con venta) y las que no."""
    filas = corridas.merge(pd.DataFrame({"medicion": list(MEDICIONES)}), how="cross")
    filas["cierre"] = [cierre(o, m) for o, m in zip(filas["origen"], filas["medicion"], strict=True)]
    claves = ["fecha_corrida", "origen", "medicion"]
    ya = set(medidas[claves].itertuples(index=False, name=None))
    nuevas = filas[[c not in ya for c in filas[claves].itertuples(index=False, name=None)]]
    cerradas = nuevas["cierre"] <= real_hasta
    return nuevas[cerradas].reset_index(drop=True), nuevas[~cerradas].reset_index(drop=True)


def rango_backtest(por_origen: pd.DataFrame) -> dict[str, float]:
    """Mínimo y máximo de cada métrica de alerta en los orígenes del backtest (`metrics.por_origen`), como
    `<métrica>_min` y `<métrica>_max`."""
    return {f"{m}_{f}": float(por_origen[m].agg(f)) for m in METRICAS_ALERTA for f in ("min", "max")}


def evaluar(ventanas: pd.DataFrame, rango: dict[str, float], medicion: str) -> dict:
    """Las métricas de `medicion` en un pronóstico (`metrics.a_ventanas` de un solo origen) y si se salen del rango.

    Solo cuentan las ventanas que ya cerraron: las de las otras mediciones quedan nulas. Un WAPE solo se sale por arriba
    del máximo; el sesgo total, por arriba del máximo o por abajo del mínimo.
    """
    calculadas = formato_estandar(ventanas) | {
        f"sesgo_m{k}": sesgo(v["real"], v["pronostico"]) for k, v in ventanas.groupby("ventana") if k < 3
    }
    propias = MEDICIONES[medicion]["alerta"] + MEDICIONES[medicion]["dato"]
    metricas = {m: calculadas[m] if m in propias else np.nan for m in METRICAS_VIVO}
    fuera = [
        m for m in MEDICIONES[medicion]["alerta"]
        if metricas[m] > rango[f"{m}_max"] or (m not in SOLO_ARRIBA and metricas[m] < rango[f"{m}_min"])
    ]
    return {"medicion": medicion} | metricas | rango | {"fuera_de_rango": ",".join(fuera), "alerta": bool(fuera)}


def _ultimas_corridas(medidas: pd.DataFrame) -> pd.DataFrame:
    """De cada origen y medición, la de su última corrida, como en results.json."""
    return medidas.sort_values("fecha_corrida").groupby(["origen", "medicion"]).tail(1)


def estados(medidas: pd.DataFrame) -> pd.Series:
    """Estado de cada medición (columnas origen, fecha_corrida, medicion y alerta): en rango, alerta, o revisión manual
    si la misma medición también tuvo alerta en el origen anterior medido. De cada origen cuenta su última corrida.

    Se compara la misma medición y no dos seguidas cualquiera: el mes 1 y el mes 2 de un pronóstico vienen del mismo
    modelo y el mismo origen, así que un solo pronóstico malo pediría revisión.
    """
    resultado = pd.Series("en rango", index=medidas.index, dtype=object)
    for _, del_tipo in medidas.groupby("medicion"):
        ultima = _ultimas_corridas(del_tipo).set_index("origen")["alerta"].sort_index()
        anterior = del_tipo["origen"].map(ultima.shift(1, fill_value=False)).astype(bool)
        alerta = del_tipo["alerta"].astype(bool)
        resultado[del_tipo.index] = np.select([alerta & anterior, alerta], ["revisión manual", "alerta"], "en rango")
    return resultado


def leer_reentrenar(texto: str) -> bool:
    """El parámetro reentrenar del job: vacío en cada corrida normal, o `si` para entrenar y pronosticar aunque no toque
    y repetir la semana aunque los datos no traigan un domingo nuevo."""
    if texto.strip() not in ("", "si"):
        raise ValueError(f"reentrenar tiene que ir vacío o con 'si', no {texto!r}")
    return texto.strip() == "si"


def decidir_reentrenamiento(
    origen: pd.Timestamp, ultimo_pronostico: pd.Timestamp | None, forzado: bool = False
) -> tuple[bool, str]:
    """Si la corrida de `origen` entrena y pronostica, y por qué: cuando pasaron 4 semanas desde el último origen
    pronosticado, o a mano con reentrenar=si.

    Una alerta de desempeño no lo adelanta (decisión del 29-oct-2026, en CLAUDE.md). Con orígenes cada 4 semanas, cada
    medición cae 1 semana después de un reentrenamiento programado, así que adelantar solo agregaría 1 semana de
    datos. En los 2 disparos de la demostración, el WAPE tipo bajó poco (1,0 y 3,4 puntos), el WAPE total no cambió y
    el sesgo empeoró. La alerta queda en el log y en la tabla como señal para revisar el modelo.
    """
    if forzado:
        return True, "reentrenar=si"
    if ultimo_pronostico is None:
        return True, "todavía no hay ningún pronóstico"
    semanas = (origen - ultimo_pronostico).days // 7
    if semanas >= SEMANAS_REENTRENAMIENTO:
        return True, f"pasaron {semanas} semanas desde el origen {ultimo_pronostico.date()}"
    return False, (f"van {semanas} semanas desde el origen {ultimo_pronostico.date()}; toca a las "
                   f"{SEMANAS_REENTRENAMIENTO}")


def _porcentaje(metrica: str, valor: float) -> str:
    return f"{valor:+.1%}" if metrica.startswith("sesgo") else f"{valor:.1%}"


def describir_rango(rango: dict[str, float]) -> str:
    """El rango del backtest en una línea, sin piezas."""
    return ", ".join(f"{m} {_porcentaje(m, rango[f'{m}_min'])} a {_porcentaje(m, rango[f'{m}_max'])}"
                     for m in METRICAS_ALERTA)


def describir_medida(medida) -> str:
    """Una medición en vivo en una línea, sin piezas."""
    medicion = MEDICIONES[medida["medicion"]]
    alerta = [f"{m} {_porcentaje(m, medida[m])} (backtest {_porcentaje(m, medida[f'{m}_min'])} a "
              f"{_porcentaje(m, medida[f'{m}_max'])})" for m in medicion["alerta"]]
    dato = [f"{m} {_porcentaje(m, medida[m])}" for m in medicion["dato"] if pd.notna(medida[m])]
    return (f"origen {pd.Timestamp(medida['origen']).date()}, {medida['medicion']} (cerró el "
            f"{pd.Timestamp(medida['cierre']).date()}): " + ", ".join(alerta) + "; dato: " + ", ".join(dato)
            + f": {medida['estado']}")
