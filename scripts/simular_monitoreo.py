"""Simulación local del monitoreo con los parquet de data/, sin Databricks.

    make calibrar-drift      # PSI, nulos y cobertura de cada origen histórico, sin entrenar nada
    make simular-monitoreo   # los 8 orígenes del backtest de dev y prod como si fueran corridas en vivo

Arma la base de features, el panel semanal y la venta diaria por serie con el mismo código del job: la agregación a
semana × SKU de src/features.py corre en Spark local y el resto con common.features. Los guarda en data/_local/ para no
rearmarlos, junto con las predicciones del backtest de la demostración. Solo imprime PSI, porcentajes, fechas y nombres
de variables: nunca piezas.

La demostración muestra primero las 8 corridas con la regla de la fase 8b (las tres ventanas juntas) y después con la
de la fase 9b: cada ventana medida apenas cierra, cuánto antes llega cada alerta, en qué lunes habría adelantado el
reentrenamiento y si el pronóstico adelantado habría tenido menos error que el vigente.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common import backtest, metrics, modelos, monitoreo
from common.features import construir_base, construir_panel
from features import agregar_semana_sku
from train import venta_diaria_serie

DATOS = Path("data")
GUARDADOS = DATOS / "_local"
PRIMER_ORIGEN = pd.Timestamp("2024-03-04")  # con menos de 2 meses de ventana el PSI no dice nada
INICIO_STOCK_COMPLETO = pd.Timestamp("2025-05-05")  # antes faltan los SKU que todavía no se movían
SEMANAS_EFECTO = 4  # un evento pesa en la historia del origen mientras está en las últimas 4 semanas (media_4)
EVENTOS = ("Hot Sale", "El Buen Fin")
CHAMPION = "lightgbm_tweedie_211"  # el algoritmo del champion en dev y prod
VIVOS = 8  # los orígenes del backtest de dev y prod, como si fueran corridas en vivo
PERMUTACIONES = 20_000  # para el valor p de la correlación cuando no caben todas las permutaciones
CONTROLADAS = [v for v in monitoreo.VARIABLES if v not in monitoreo.INFORMATIVAS]


def _spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def _armar() -> None:
    spark = _spark()
    ventas, stock = (spark.read.parquet(str(DATOS / f"{a}.parquet")) for a in ("ventas_diarias", "stock_diario"))
    productos = pd.read_parquet(DATOS / "productos.parquet")
    corte = pd.Timestamp(min(df.agg(F.max("fecha")).first()[0] for df in (ventas, stock)))
    ventas_sku, stock_sku = (df.toPandas() for df in agregar_semana_sku(ventas, stock))
    for df in (ventas_sku, stock_sku):
        df["semana"] = pd.to_datetime(df["semana"])
    panel = construir_panel(ventas_sku, stock_sku, productos, corte)
    base = construir_base(panel, pd.read_parquet(DATOS / "calendario_eventos.parquet"), corte)
    venta_diaria = venta_diaria_serie(ventas, spark.createDataFrame(productos)).toPandas()
    venta_diaria["fecha"] = pd.to_datetime(venta_diaria["fecha"])
    GUARDADOS.mkdir(exist_ok=True)
    base.to_parquet(GUARDADOS / "features.parquet")
    panel.to_parquet(GUARDADOS / "panel_semanal.parquet")
    venta_diaria.to_parquet(GUARDADOS / "venta_diaria.parquet")
    print(f"corte {corte.date()}: {base['origen'].nunique()} orígenes y {panel['serie'].nunique()} series")
    spark.stop()


def datos_locales() -> dict[str, pd.DataFrame]:
    """La base de features (con categóricas, como en train), el panel semanal y la venta diaria por serie."""
    nombres = ("features", "panel_semanal", "venta_diaria")
    if not all((GUARDADOS / f"{n}.parquet").exists() for n in nombres):
        _armar()
    datos = {n: pd.read_parquet(GUARDADOS / f"{n}.parquet") for n in nombres}
    for columna in ("origen", "semana_objetivo"):
        datos["features"][columna] = pd.to_datetime(datos["features"][columna])
    datos["features"] = modelos.a_categoricas(datos["features"])
    return datos


def conteos_cobertura(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Los conteos semanales de las revisiones de cobertura, como en src/monitoreo_datos.py."""
    fechas = pd.read_parquet(DATOS / "ventas_diarias.parquet", columns=["fecha"])["fecha"]
    return {
        "skus_con_stock": monitoreo.skus_con_stock_por_semana(panel),
        "filas_venta": monitoreo.filas_por_semana(fechas),
    }


def series_mes(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Las series semanales de la cobertura del mes, como en src/monitoreo_datos.py. Las salidas de stock salen del
    stock diario completo y quedan guardadas en data/_local/."""
    ruta = GUARDADOS / "salidas.parquet"
    if not ruta.exists():
        stock = pd.read_parquet(DATOS / "stock_diario.parquet")
        stock["sku"] = stock["sku"].astype("category")
        monitoreo.salidas_por_semana(stock).rename("salidas").to_frame().to_parquet(ruta)
    salidas = pd.read_parquet(ruta)["salidas"]
    unidades = monitoreo.unidades_por_semana(pd.read_parquet(DATOS / "ventas_diarias.parquet"))
    return {
        "skus_con_stock": monitoreo.skus_con_stock_por_semana(panel),
        "venta_salidas": monitoreo.venta_entre_salidas(unidades, salidas),
    }


def revisiones(datos: dict[str, pd.DataFrame], origenes) -> pd.DataFrame:
    """Las revisiones de datos de cada origen, como en src/monitoreo_datos.py salvo el corte del stock: en la historia,
    venta y stock siempre llegan juntos."""
    conteos = conteos_cobertura(datos["panel_semanal"])
    partes = [monitoreo.drift_datos(datos["features"], o) for o in origenes]
    partes += [monitoreo.revisar_cobertura(o, conteos) for o in origenes]
    return pd.concat(partes, ignore_index=True)


def _periodos_evento(calendario: pd.DataFrame) -> pd.DataFrame:
    """Hot Sale, Buen Fin y diciembre de cada año, con los orígenes cuya historia reciente los incluye: desde la semana
    en que empiezan hasta 4 semanas después de que terminan."""
    filas = [(f"{e['evento']} {e['fecha_inicio'].year}", e["fecha_inicio"], e["fecha_fin"])
             for _, e in calendario[calendario["evento"].isin(EVENTOS)].iterrows()]
    filas += [(f"Diciembre {a}", pd.Timestamp(a, 12, 1), pd.Timestamp(a, 12, 31)) for a in (2024, 2025)]
    periodos = pd.DataFrame(filas, columns=["evento", "inicio", "fin"])
    periodos["desde"] = periodos["inicio"] - pd.to_timedelta(periodos["inicio"].dt.dayofweek, unit="D")
    periodos["hasta"] = periodos["fin"] + pd.Timedelta(weeks=SEMANAS_EFECTO)
    return periodos


def _tabla(df: pd.DataFrame) -> str:
    filas = ["| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
    filas += ["| " + " | ".join(map(str, fila)) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(filas)


def _psi_maximo(revisadas: pd.DataFrame, variables) -> pd.DataFrame:
    """Por origen, el PSI más alto entre `variables` y de cuál es."""
    del_psi = revisadas[(revisadas["revision"] == "psi") & revisadas["variable"].isin(variables)]
    del_psi = del_psi.dropna(subset=["valor"])
    return del_psi.loc[del_psi.groupby("origen")["valor"].idxmax(), ["origen", "variable", "valor"]].set_index("origen")


def calibrar(datos: dict[str, pd.DataFrame]) -> None:
    """Revisiones de cada origen desde PRIMER_ORIGEN, sin entrenar: el PSI más alto de cada evento, la distribución del
    PSI por variable y cuántos orígenes habrían dado aviso o detención con los umbrales de common.monitoreo."""
    origenes = sorted(o for o in datos["features"]["origen"].unique() if o >= PRIMER_ORIGEN)
    revisadas = revisiones(datos, origenes)
    revisadas.to_parquet(GUARDADOS / "calibracion.parquet")
    todas, controladas = _psi_maximo(revisadas, monitoreo.VARIABLES), _psi_maximo(revisadas, CONTROLADAS)

    por_evento = []
    for _, p in _periodos_evento(pd.read_parquet(DATOS / "calendario_eventos.parquet")).iterrows():
        en_evento = (todas.index >= p["desde"]) & (todas.index <= p["hasta"])
        if not en_evento.any():
            continue
        pico, pico_c = todas[en_evento]["valor"].idxmax(), controladas[en_evento]["valor"].idxmax()
        por_evento.append({
            "evento": p["evento"],
            "orígenes": int(en_evento.sum()),
            "PSI máximo": f"{todas.loc[pico, 'valor']:.2f} ({todas.loc[pico, 'variable']}, {pico.date()})",
            "sin tendencia": f"{controladas.loc[pico_c, 'valor']:.2f} ({controladas.loc[pico_c, 'variable']})",
        })
    print(f"PSI de {len(origenes)} orígenes ({origenes[0].date()} a {origenes[-1].date()}), por evento:\n")
    print(_tabla(pd.DataFrame(por_evento).sort_values("evento", key=lambda s: s.str[-4:] + s, kind="stable")))

    del_psi = revisadas[revisadas["revision"] == "psi"].dropna(subset=["valor"])
    resumen = del_psi.groupby("variable")["valor"].describe(percentiles=[0.5, 0.95]).reindex(monitoreo.VARIABLES)
    cuenta = del_psi.groupby("variable")["valor"]
    print("\nPSI por variable en todos los orígenes:\n")
    print(_tabla(pd.DataFrame({
        "variable": resumen.index,
        "orígenes": resumen["count"].astype(int),
        "mediana": resumen["50%"].map("{:.2f}".format),
        "p95": resumen["95%"].map("{:.2f}".format),
        "máximo": resumen["max"].map("{:.2f}".format),
        "≥ 0,25": cuenta.apply(lambda v: (v >= monitoreo.PSI_AVISO).sum()).reindex(monitoreo.VARIABLES).to_numpy(),
        "≥ 1,0": cuenta.apply(lambda v: (v >= monitoreo.PSI_EXTREMO).sum()).reindex(monitoreo.VARIABLES).to_numpy(),
    })))

    nulos = revisadas[revisadas["revision"] == "nulos"]
    aumento = (nulos["valor"] - nulos["referencia"]).groupby(nulos["variable"]).max().reindex(monitoreo.VARIABLES)
    print("\nAumento máximo de la parte de nulos contra los 13 orígenes anteriores: "
          + ", ".join(f"{v} {a:+.1%}" for v, a in aumento.items()))
    cobertura = revisadas[revisadas["revision"] == "cobertura"].dropna(subset=["valor"])
    minimos = cobertura.loc[cobertura.groupby("variable")["valor"].idxmin()]
    print("Cobertura más baja: " + ", ".join(f"{f.variable} {f.valor:.2f} ({f.origen.date()})"
                                             for f in minimos.itertuples()))

    print("\nOrígenes con aviso o detención, con los umbrales de common.monitoreo:\n")
    niveles = revisadas[revisadas["nivel"].isin(["aviso", "extremo"])]
    conteo = niveles.groupby(["revision", "nivel"])["origen"].nunique().unstack(fill_value=0)
    print(_tabla(conteo.reindex(columns=["aviso", "extremo"], fill_value=0).reset_index()))
    extremos = niveles[niveles["nivel"] == "extremo"]
    print("\nDetenciones: " + ("; ".join(f"{f.origen.date()} {monitoreo.describir(f)}" for f in extremos.itertuples())
                              or "ninguna"))
    calibrar_mes(datos["panel_semanal"])


def calibrar_mes(panel: pd.DataFrame) -> None:
    """Cobertura del mes en cada origen semanal, con la corrida anterior 1, 4, 5 o 13 semanas antes (1 es la corrida
    semanal; 13, la primera corrida con monitoreo en prod), y qué tan seguido detecta un mes entero o una semana a
    medias."""
    print("\nCobertura del mes: la semana nueva más baja contra la más baja de las 4 que revisó la corrida anterior\n")
    filas, detecciones = [], []
    for variable, serie in series_mes(panel).items():
        nulo_es_cero = variable == "skus_con_stock"
        for salto in (1, 4, 5, 13):
            desde = INICIO_STOCK_COMPLETO + pd.Timedelta(weeks=salto + monitoreo.SEMANAS_REVISADAS - 1)
            origenes = serie.index[serie.index >= desde]
            valores = pd.Series({o: monitoreo.cobertura_mes(serie, o, o - pd.Timedelta(weeks=salto), nulo_es_cero)
                                 for o in origenes}).dropna()
            bajos = valores.nsmallest(2)
            filas.append({
                "revisión": variable, "corrida anterior": f"{salto} semanas antes", "orígenes": len(valores),
                "mínimo": f"{valores.min():.2f}", "p5": f"{valores.quantile(0.05):.2f}",
                "mediana": f"{valores.median():.2f}",
                "avisos": int((valores <= monitoreo.COBERTURA_AVISO).sum()),
                "detenciones": int((valores <= monitoreo.COBERTURA_EXTREMO).sum()),
                "más bajos": ", ".join(f"{o.date()} {v:.2f}" for o, v in bajos.items()),
            })
        origenes = serie.index[serie.index >= INICIO_STOCK_COMPLETO + pd.Timedelta(weeks=7)]
        for salto, rotura, semanas, f in (
            (4, "mes entero a la mitad", range(4), 0.5), (4, "mes entero al 70 %", range(4), 0.7),
            (4, "una semana a la mitad", [2], 0.5), (1, "la semana nueva a la mitad", [0], 0.5),
            (1, "la semana nueva al 70 %", [0], 0.7),
        ):
            valores = []
            for o in origenes:
                rota = serie.copy()
                for k in semanas:
                    rota.loc[o - pd.Timedelta(weeks=k)] *= f
                valores.append(monitoreo.cobertura_mes(rota, o, o - pd.Timedelta(weeks=salto), nulo_es_cero))
            valores = pd.Series(valores).dropna()
            detecciones.append({"revisión": variable, "corrida anterior": f"{salto} semanas antes", "rotura": rotura,
                                "orígenes": len(valores),
                                "avisos": int((valores <= monitoreo.COBERTURA_AVISO).sum()),
                                "detenciones": int((valores <= monitoreo.COBERTURA_EXTREMO).sum())})
    print(_tabla(pd.DataFrame(filas)))
    print("\nDetección de roturas:\n")
    print(_tabla(pd.DataFrame(detecciones)))


def backtest_local(base: pd.DataFrame, origenes: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.Series]:
    """Las predicciones del algoritmo del champion en `origenes`, entrenado hasta cada uno como en backtest.correr, y la
    ganancia de cada variable en el último origen. Quedan en data/_local/ para no volver a entrenar."""
    rutas = {n: GUARDADOS / f"{n}_{CHAMPION}.parquet" for n in ("backtest", "importancia")}
    if all(r.exists() for r in rutas.values()):
        guardado = pd.read_parquet(rutas["backtest"])
        if set(origenes) <= set(guardado["origen"]):
            importancia = pd.read_parquet(rutas["importancia"])["ganancia"]
            return guardado[guardado["origen"].isin(origenes)], importancia
    modelo = modelos.modelos()[CHAMPION]
    predicciones = backtest.correr({CHAMPION: modelo}, base, origenes)
    ganancia = np.mean([b.feature_importance("gain") for b in modelo.boosters], axis=0)
    importancia = pd.Series(ganancia, index=modelo.variables, name="ganancia").sort_values(ascending=False)
    predicciones.to_parquet(rutas["backtest"])
    importancia.to_frame().to_parquet(rutas["importancia"])
    return predicciones, importancia


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Correlación de Spearman y su valor p de dos colas por permutaciones: todas si caben, si no PERMUTACIONES al
    azar."""
    rx, ry = x.rank().to_numpy(), y.rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if len(ry) <= 9:
        permutadas = np.array(list(itertools.permutations(ry)))
    else:
        rng = np.random.default_rng(0)
        permutadas = np.array([rng.permutation(ry) for _ in range(PERMUTACIONES)])
    centradas = permutadas - permutadas.mean(axis=1, keepdims=True)
    rx_c = rx - rx.mean()
    nulas = centradas @ rx_c / (np.linalg.norm(centradas, axis=1) * np.linalg.norm(rx_c))
    return rho, float(np.mean(np.abs(nulas) >= abs(rho) - 1e-12))


def _con_tres_ventanas(ventanas: pd.DataFrame, rango: dict[str, float]) -> dict:
    """La regla de la fase 8b, con que se hizo esta demostración: el pronóstico se mide cuando cierran sus tres
    ventanas, con alerta si se sale cualquiera de las métricas de alerta. Junta las tres mediciones de
    common.monitoreo."""
    partes = [monitoreo.evaluar(ventanas, rango, m) for m in monitoreo.MEDICIONES]
    metricas = {m: next((p[m] for p in partes if pd.notna(p[m])), np.nan) for m in monitoreo.METRICAS_VIVO}
    fuera = [p["fuera_de_rango"] for p in partes if p["fuera_de_rango"]]
    return metricas | rango | {"fuera_de_rango": ",".join(fuera), "alerta": bool(fuera)}


def demo(datos: dict[str, pd.DataFrame]) -> None:
    """Los 8 orígenes del backtest de dev y prod como corridas en vivo: el drift de datos de cada uno, su error medido
    contra el rango del backtest que esa corrida habría hecho, y si los orígenes con más drift tuvieron más error."""
    base, venta_diaria = datos["features"], datos["venta_diaria"]
    ultima_semana = base.loc[base["y"].notna(), "semana_objetivo"].max()
    vivos = backtest.origenes(ultima_semana, VIVOS)
    del_rango = {o: backtest.origenes(o, backtest.ORIGENES) for o in vivos}
    todos = sorted(set(vivos).union(*del_rango.values()))
    print(f"en vivo: {vivos[0].date()} a {vivos[-1].date()}; con los de sus rangos, {len(todos)} orígenes "
          f"({todos[0].date()} a {todos[-1].date()})")

    predicciones, importancia = backtest_local(base, todos)
    ventanas = metrics.a_ventanas(predicciones, venta_diaria)
    por_origen = metrics.por_origen(ventanas)
    medidas = []
    for o in vivos:
        del_backtest = por_origen.loc[del_rango[o]]
        medida = _con_tres_ventanas(ventanas[ventanas["origen"] == o], monitoreo.rango_backtest(del_backtest))
        mediana = del_backtest[backtest.CRITERIO].median()
        medidas.append(medida | {"origen": o, "fecha_corrida": o, "relativo": medida[backtest.CRITERIO] / mediana})
    medidas = pd.DataFrame(medidas)
    medidas["estado"] = monitoreo.estados(medidas.assign(medicion="tres ventanas"))

    revisadas = revisiones(datos, todos)
    controladas = _psi_maximo(revisadas, CONTROLADAS)
    del_psi = revisadas[revisadas["revision"] == "psi"].pivot(index="origen", columns="variable", values="valor")
    drift = pd.DataFrame({
        "psi_max": controladas["valor"],
        "variable": controladas["variable"],
        "psi_medio": del_psi[CONTROLADAS].mean(axis=1),
        "psi_tendencia": del_psi["tendencia"],
        "datos": revisadas[revisadas["nivel"] != "informativa"].groupby("origen")["nivel"].agg(
            lambda n: "extremo" if (n == "extremo").any() else "aviso" if (n == "aviso").any() else "estable"),
    })
    pd.concat([medidas.set_index("origen"), drift.loc[vivos]], axis=1).to_parquet(GUARDADOS / "demo.parquet")

    print("\nLas 8 corridas en vivo (rango: mínimo y máximo por origen del backtest de esa corrida):\n")
    vivas = medidas.set_index("origen").join(drift)
    print(_tabla(pd.DataFrame({
        "origen": [f"{o.date()}" for o in vivas.index],
        "PSI máx.": [f"{p:.2f} ({v})" for p, v in zip(vivas["psi_max"], vivas["variable"], strict=True)],
        "PSI tendencia": vivas["psi_tendencia"].map("{:.2f}".format),
        "datos": vivas["datos"],
        **{nombre: [f"{m:.1%} ({a:.1%}–{b:.1%})" for m, a, b in vivas[[c, f"{c}_min", f"{c}_max"]].to_numpy()]
           for nombre, c in (("WAPE M1", "wape_m1"), ("WAPE M2", "wape_m2"), ("WAPE total", "wape_total"))},
        "sesgo": [f"{m:+.1%} ({a:+.1%} a {b:+.1%})"
                  for m, a, b in vivas[["sesgo_total", "sesgo_total_min", "sesgo_total_max"]].to_numpy()],
        "fuera": vivas["fuera_de_rango"].replace("", "-"),
        "estado": vivas["estado"],
    })))

    print("\nPSI de cada variable en las 8 corridas en vivo:\n")
    tabla_psi = del_psi.loc[vivos, list(monitoreo.VARIABLES)].map("{:.2f}".format)
    print(_tabla(tabla_psi.reset_index().assign(origen=lambda d: d["origen"].dt.date)))
    avisos = revisadas[revisadas["origen"].isin(vivos) & revisadas["nivel"].isin(["aviso", "extremo"])]
    print("\nAvisos y detenciones de datos en las 8: " + (
        "; ".join(f"{f.origen.date()} {monitoreo.describir(f)}" for f in avisos.itertuples()) or "ninguno"))

    errores = {
        "WAPE tipo × mes": vivas[backtest.CRITERIO],
        "WAPE total": vivas["wape_total"],
        "|sesgo|": vivas["sesgo_total"].abs(),
        "WAPE tipo × mes / mediana de su backtest": vivas["relativo"],
    }
    filas = []
    for medida_drift in ("psi_max", "psi_medio", "psi_tendencia"):
        for nombre, error in errores.items():
            rho, p = spearman(vivas[medida_drift], error)
            filas.append({"drift": medida_drift, "error": nombre, "Spearman": f"{rho:+.2f}", "p": f"{p:.2f}"})
    print("\n¿Más drift, más error? Spearman en las 8 corridas en vivo (p por permutaciones exactas):\n")
    print(_tabla(pd.DataFrame(filas)))

    todas = por_origen.join(drift)
    filas = []
    for medida_drift in ("psi_max", "psi_medio", "psi_tendencia"):
        for nombre, error in {"WAPE tipo × mes": todas[backtest.CRITERIO], "WAPE total": todas["wape_total"],
                              "|sesgo|": todas["sesgo_total"].abs()}.items():
            rho, p = spearman(todas[medida_drift], error)
            filas.append({"drift": medida_drift, "error": nombre, "Spearman": f"{rho:+.2f}", "p": f"{p:.2f}"})
    print(f"\nComplemento: los {len(todos)} orígenes, con ventanas que se traslapan "
          f"(p con {PERMUTACIONES:,} permutaciones):\n")
    print(_tabla(pd.DataFrame(filas)))

    print("\nLos 23 orígenes ordenados por PSI máximo (sin tendencia):\n")
    orden = todas.sort_values("psi_max", ascending=False)
    print(_tabla(pd.DataFrame({
        "origen": [f"{o.date()}" for o in orden.index],
        "en vivo": ["sí" if o in vivos else "" for o in orden.index],
        "PSI máx.": [f"{p:.2f} ({v})" for p, v in zip(orden["psi_max"], orden["variable"], strict=True)],
        "PSI tendencia": orden["psi_tendencia"].map("{:.2f}".format),
        "WAPE tipo × mes": orden[backtest.CRITERIO].map("{:.1%}".format),
        "puesto por error": orden[backtest.CRITERIO].rank(ascending=False).astype(int),
        "sesgo": orden["sesgo_total"].map("{:+.1%}".format),
    })))

    puestos = importancia.rank(ascending=False).astype(int)
    print(f"\nPuesto de las variables monitoreadas por ganancia del champion (de {len(importancia)} variables, "
          f"entrenado hasta {todos[-1].date()}): " + ", ".join(f"{v} {puestos[v]}" for v in monitoreo.VARIABLES))
    print("Las 10 primeras: " + ", ".join(importancia.index[:10]))

    demo_por_ventana(datos, ventanas, por_origen, vivos, del_rango, medidas)


def predicciones_adelantadas(base: pd.DataFrame, origenes: list[pd.Timestamp]) -> pd.DataFrame:
    """Las predicciones del algoritmo del champion en orígenes fuera de la rejilla de 4 semanas, entrenado hasta cada
    uno como en backtest.correr: nada con semana objetivo posterior al origen. Quedan en data/_local/, aparte de las del
    backtest, y solo se entrenan los orígenes que falten."""
    ruta = GUARDADOS / f"adelantados_{CHAMPION}.parquet"
    guardado = pd.read_parquet(ruta) if ruta.exists() else None
    faltan = [o for o in origenes if guardado is None or o not in set(guardado["origen"])]
    if faltan:
        nuevas = backtest.correr({CHAMPION: modelos.modelos()[CHAMPION]}, base, faltan)
        nuevas["serie"] = nuevas["serie"].astype(str)
        guardado = nuevas if guardado is None else pd.concat([guardado, nuevas], ignore_index=True)
        guardado.to_parquet(ruta)
    return guardado[guardado["origen"].isin(origenes)]


def medir_por_ventana(ventanas: pd.DataFrame, por_origen: pd.DataFrame, vivos, del_rango) -> pd.DataFrame:
    """Las 8 corridas en vivo con la regla de la fase 9b: una fila por origen × medición, con el lunes en que se habría
    medido (`mide`), sus métricas, la alerta y el estado, con el mismo código del job."""
    filas = []
    for o in vivos:
        rango = monitoreo.rango_backtest(por_origen.loc[del_rango[o]])
        del_origen = ventanas[ventanas["origen"] == o]
        for m in monitoreo.MEDICIONES:
            filas.append(monitoreo.evaluar(del_origen, rango, m) | {
                "origen": o, "fecha_corrida": o, "cierre": monitoreo.cierre(o, m),
                "mide": monitoreo.corrida_de_medicion(o, m),
            })
    medidas = pd.DataFrame(filas)
    medidas["estado"] = monitoreo.estados(medidas)
    return medidas


def repasar_decisiones(medidas: pd.DataFrame, rejilla: list[pd.Timestamp]) -> pd.DataFrame:
    """La regla del job (monitoreo.decidir_reentrenamiento), lunes a lunes, sobre la rejilla de 4 semanas de la 8b.

    La rejilla se mantiene: cada 4 semanas hay un pronóstico, como en la 8b. Un disparo agrega un pronóstico en su
    origen, vigente hasta que entra el del siguiente origen de la rejilla; desde ahí es el último pronóstico, así que la
    misma alerta no vuelve a disparar. Solo cuentan las mediciones de la rejilla: las de los pronósticos adelantados no
    se simulan. Se repasan los orígenes hasta el que sigue al último de la rejilla. Devuelve una fila por disparo, con
    su origen, su corrida, el motivo y los orígenes de la rejilla anterior y siguiente.
    """
    rejilla = sorted(rejilla)
    fin = rejilla[-1] + pd.Timedelta(weeks=backtest.CADA)
    entrenados, disparos = [rejilla[0]], []
    for origen in pd.date_range(rejilla[0] + pd.Timedelta(weeks=1), fin - pd.Timedelta(weeks=1), freq="7D"):
        if origen in rejilla:
            entrenados.append(origen)
            continue
        corrida = origen + pd.Timedelta(weeks=1)
        ultimo = max(o for o in entrenados if o < origen)
        reentrena, motivo = monitoreo.decidir_reentrenamiento(origen, ultimo, medidas[medidas["mide"] <= corrida])
        if reentrena:
            entrenados.append(origen)
            disparos.append({
                "origen": origen, "corrida": corrida, "motivo": motivo,
                "anterior": max(o for o in rejilla if o < origen),
                "siguiente": min((o for o in rejilla if o > origen), default=fin),
            })
    return pd.DataFrame(disparos, columns=["origen", "corrida", "motivo", "anterior", "siguiente"])


def error_semanal(panel: pd.DataFrame, predicciones: pd.DataFrame, semanas: pd.DatetimeIndex) -> dict[str, float]:
    """WAPE a nivel tipo de producto (serie × semana) y del total (por semana), y sesgo, en `semanas`. Una serie sin
    pronóstico cuenta con 0 y una sin venta con venta 0, como en la métrica del backtest."""
    real = panel[panel["semana"].isin(semanas)].groupby(["serie", "semana"])["venta"].sum()
    del_periodo = predicciones[predicciones["semana_objetivo"].isin(semanas)]
    pronostico = del_periodo.groupby([del_periodo["serie"].astype(str), "semana_objetivo"])["pronostico"].sum()
    pronostico.index.names = ["serie", "semana"]
    juntas = pd.concat([real.rename("real"), pronostico.rename("pronostico")], axis=1).fillna(0)
    total = juntas.groupby(level="semana").sum()
    return {
        "wape_tipo": metrics.wape(juntas["real"], juntas["pronostico"]),
        "wape_total": metrics.wape(total["real"], total["pronostico"]),
        "sesgo": metrics.sesgo(juntas["real"], juntas["pronostico"]),
    }


def _metricas_de(fila, nombres) -> str:
    return ", ".join(
        f"{m} {fila[m]:+.1%}" if m.startswith("sesgo") else f"{m} {fila[m]:.1%}" for m in nombres if pd.notna(fila[m])
    )


def demo_por_ventana(datos, ventanas, por_origen, vivos, del_rango, tres_ventanas: pd.DataFrame) -> None:
    """Las 8 corridas con la medición por ventana de la fase 9b, comparadas con la regla de las tres ventanas."""
    medidas = medir_por_ventana(ventanas, por_origen, vivos, del_rango)
    medidas.to_parquet(GUARDADOS / "demo_ventanas.parquet")
    print("\nMedición por ventana (fase 9b): cada ventana en la corrida siguiente a su cierre, contra el rango de su "
          "backtest\n")
    print(_tabla(pd.DataFrame({
        "origen": medidas["origen"].dt.date.astype(str),
        "medición": medidas["medicion"],
        "se mide": medidas["mide"].dt.date.astype(str),
        "alerta con": [
            ", ".join(f"{_metricas_de(f, [m])} (backtest {f[f'{m}_min']:+.1%} a {f[f'{m}_max']:+.1%})"
                      if m.startswith("sesgo") else f"{_metricas_de(f, [m])} (máx. {f[f'{m}_max']:.1%})"
                      for m in monitoreo.MEDICIONES[f["medicion"]]["alerta"])
            for _, f in medidas.iterrows()
        ],
        "dato": [_metricas_de(f, monitoreo.MEDICIONES[f["medicion"]]["dato"]) for _, f in medidas.iterrows()],
        "fuera": medidas["fuera_de_rango"].replace("", "-"),
        "estado": medidas["estado"],
    })))

    print("\n¿Cuánto antes llega cada alerta? Primera medición con alerta contra la regla de las tres ventanas:\n")
    con_alerta = medidas[medidas["alerta"]].sort_values("mide").drop_duplicates("origen")
    viejo = tres_ventanas.set_index("origen")
    print(_tabla(pd.DataFrame({
        "origen": con_alerta["origen"].dt.date.astype(str),
        "por ventana": [f"{f.medicion}, {f.mide.date()}" for f in con_alerta.itertuples()],
        "con las tres": [f"{monitoreo.corrida_de_medicion(o, 'total').date()}" for o in con_alerta["origen"]],
        "semanas antes": [(monitoreo.corrida_de_medicion(o, "total") - m).days // 7
                          for o, m in zip(con_alerta["origen"], con_alerta["mide"], strict=True)],
        "estado con las tres": [viejo.loc[o, "estado"] for o in con_alerta["origen"]],
        "revisión manual por ventana": [
            ", ".join(f"{m} ({d.date()})" for m, d in medidas.loc[
                (medidas["origen"] == o) & (medidas["estado"] == "revisión manual"), ["medicion", "mide"]
            ].itertuples(index=False, name=None)) or "-"
            for o in con_alerta["origen"]
        ],
    })))

    disparos = repasar_decisiones(medidas, list(vivos))
    disparos.to_parquet(GUARDADOS / "demo_disparos.parquet")
    print(f"\nReentrenamientos adelantados: la regla del job, lunes a lunes, del origen {vivos[0].date()} al "
          f"{(vivos[-1] + pd.Timedelta(weeks=backtest.CADA - 1)).date()}, sobre la rejilla de 4 semanas de la 8b\n")
    if disparos.empty:
        print("Ningún disparo: ninguna alerta habría adelantado el reentrenamiento, así que no hay pronóstico "
              "adelantado que comparar.")
        return
    adelantadas = predicciones_adelantadas(datos["features"], list(disparos["origen"]))
    de_la_rejilla = backtest_local(datos["features"], list(vivos))[0]
    filas = []
    for d in disparos.itertuples():
        # vigente desde la semana siguiente al disparo hasta la del siguiente origen de la rejilla, que entra después
        semanas = pd.date_range(d.origen + pd.Timedelta(weeks=1), d.siguiente, freq="7D")
        nuevo = error_semanal(datos["panel_semanal"], adelantadas[adelantadas["origen"] == d.origen], semanas)
        vigente = error_semanal(datos["panel_semanal"], de_la_rejilla[de_la_rejilla["origen"] == d.anterior], semanas)
        filas.append({
            "corrida": f"{d.corrida.date()}", "origen adelantado": f"{d.origen.date()}", "motivo": d.motivo,
            "semanas": f"{semanas[0].date()} a {semanas[-1].date()} ({len(semanas)})",
            "vigente (rejilla)": f"{d.anterior.date()}",
            **{f"{nombre} adelantado / vigente": f"{nuevo[m]:{formato}} / {vigente[m]:{formato}}"
               for nombre, m, formato in (("WAPE tipo", "wape_tipo", ".1%"), ("WAPE total", "wape_total", ".1%"),
                                          ("sesgo", "sesgo", "+.1%"))},
        })
    print(_tabla(pd.DataFrame(filas)))


def main():
    parser = argparse.ArgumentParser(description="Simulación local del monitoreo con los parquet de data/.")
    parser.add_argument("modo", choices=["calibrar", "demo"])
    args = parser.parse_args()
    datos = datos_locales()
    if args.modo == "calibrar":
        calibrar(datos)
    else:
        demo(datos)


if __name__ == "__main__":
    main()
