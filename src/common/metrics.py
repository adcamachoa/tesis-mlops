"""Métricas del backtest y formato estándar de comparación.

Funciones puras en pandas. El pronóstico es semanal y se evalúa a nivel tipo de producto (la serie) en tres ventanas de
31, 30 y 30 días desde el lunes siguiente al origen: los 91 días son exactamente las 13 semanas del horizonte. La venta
real de cada ventana sale de la venta diaria. El pronóstico de las dos semanas partidas entre dos ventanas (la 5 y la 9)
se reparte con el perfil por día de la semana de la venta total de las 52 semanas anteriores.
"""
import numpy as np
import pandas as pd

from common.features import HORIZONTE

VENTANAS_DIAS = (31, 30, 30)
SEMANAS_PERFIL = 52
METRICAS = (
    "wape_m1", "wape_m2", "wape_total",
    "mape_m1", "mape_m2", "mape_total",
    "sesgo_total",
    "wape_tipo_mes", "mae", "rmse",  # complemento
)


def _arreglos(real, pronostico) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)


def wape(real, pronostico) -> float:
    real, pronostico = _arreglos(real, pronostico)
    return float(np.abs(real - pronostico).sum() / real.sum()) if real.sum() > 0 else np.nan


def mape(real, pronostico) -> float:
    """Promedio del error relativo, solo en las celdas con venta."""
    real, pronostico = _arreglos(real, pronostico)
    con_venta = real > 0
    return float(np.mean(np.abs(real - pronostico)[con_venta] / real[con_venta])) if con_venta.any() else np.nan


def sesgo(real, pronostico) -> float:
    """Positivo si se pronosticó de más."""
    real, pronostico = _arreglos(real, pronostico)
    return float(pronostico.sum() / real.sum() - 1) if real.sum() > 0 else np.nan


def mae(real, pronostico) -> float:
    real, pronostico = _arreglos(real, pronostico)
    return float(np.mean(np.abs(real - pronostico)))


def rmse(real, pronostico) -> float:
    real, pronostico = _arreglos(real, pronostico)
    return float(np.sqrt(np.mean((real - pronostico) ** 2)))


def _ventana_de_cada_dia() -> np.ndarray:
    """Ventana (1, 2 o 3) de cada uno de los 91 días que empiezan el lunes siguiente al origen."""
    ventanas = np.repeat(np.arange(1, len(VENTANAS_DIAS) + 1), VENTANAS_DIAS)
    assert len(ventanas) == HORIZONTE * 7
    return ventanas


def perfil_semanal(venta_total: pd.Series, lunes: pd.Timestamp) -> np.ndarray:
    """Parte de la venta de las 52 semanas anteriores a `lunes` que cayó en cada día de la semana (0 = lunes).

    `venta_total` va indexada por fecha. Sin venta en esas semanas, cada día pesa 1/7.
    """
    desde = lunes - pd.Timedelta(weeks=SEMANAS_PERFIL)
    previa = venta_total[(venta_total.index >= desde) & (venta_total.index < lunes)]
    por_dia = previa.groupby(previa.index.dayofweek).sum().reindex(range(7), fill_value=0).to_numpy(dtype=float)
    return por_dia / por_dia.sum() if por_dia.sum() > 0 else np.full(7, 1 / 7)


def pesos_ventanas(perfil: np.ndarray) -> np.ndarray:
    """Matriz semana h × ventana: qué parte del pronóstico de la semana h cae en cada ventana, según el peso de cada día
    de la semana en `perfil`. Cada fila suma 1 y solo las semanas 5 y 9 se reparten entre dos ventanas."""
    dias = np.arange(HORIZONTE * 7)  # el día 0 es el lunes siguiente al origen
    pesos = np.zeros((HORIZONTE, len(VENTANAS_DIAS)))
    np.add.at(pesos, (dias // 7, _ventana_de_cada_dia() - 1), perfil[dias % 7])
    return pesos


def a_ventanas(predicciones: pd.DataFrame, venta_diaria: pd.DataFrame) -> pd.DataFrame:
    """Venta real y pronóstico por origen × serie × ventana.

    `predicciones` trae serie, origen, h y pronostico; `venta_diaria`, serie, fecha y unidades de todas las series. Una
    serie que no está en las predicciones de un origen (nueva, o sin venta en las 52 semanas previas) cuenta con
    pronóstico 0, y una serie pronosticada sin venta en una ventana cuenta con venta 0.
    """
    venta_diaria = venta_diaria.assign(serie=venta_diaria["serie"].astype(str))
    venta_total = venta_diaria.groupby("fecha")["unidades"].sum()
    ventana_del_dia = _ventana_de_cada_dia()
    numeros = np.arange(1, len(VENTANAS_DIAS) + 1)
    partes = []
    for origen, pred in predicciones.groupby("origen"):
        lunes = origen + pd.Timedelta(weeks=1)
        pesos = pesos_ventanas(perfil_semanal(venta_total, lunes))
        repartido = pred["pronostico"].to_numpy(dtype=float)[:, None] * pesos[pred["h"].to_numpy(dtype=int) - 1]
        pronostico = (
            pd.DataFrame(
                {
                    "serie": np.repeat(pred["serie"].astype(str).to_numpy(), len(numeros)),
                    "ventana": np.tile(numeros, len(pred)),
                    "pronostico": repartido.ravel(),
                }
            )
            .groupby(["serie", "ventana"])["pronostico"]
            .sum()
        )
        dias = venta_diaria[(venta_diaria["fecha"] >= lunes) & (venta_diaria["fecha"] < lunes + pd.Timedelta(days=91))]
        real = (
            dias.assign(ventana=ventana_del_dia[(dias["fecha"] - lunes).dt.days.to_numpy()])
            .groupby(["serie", "ventana"])["unidades"]
            .sum()
            .rename("real")
        )
        series = sorted(set(pronostico.index.get_level_values("serie")) | set(real.index.get_level_values("serie")))
        celdas = pd.MultiIndex.from_product([series, numeros], names=["serie", "ventana"])
        juntas = pd.concat([real, pronostico], axis=1).reindex(celdas).fillna(0)
        partes.append(juntas.reset_index().assign(origen=origen))
    return pd.concat(partes, ignore_index=True)[["origen", "serie", "ventana", "real", "pronostico"]]


def formato_estandar(ventanas: pd.DataFrame) -> dict[str, float]:
    """Métricas de comparación a nivel tipo de producto, con todos los orígenes juntos.

    WAPE y MAPE del mes 1, el mes 2 y el total (el trimestre: las tres ventanas sumadas por serie), y sesgo total. Como
    complemento, en las celdas serie × ventana: el WAPE con las tres ventanas juntas (`wape_tipo_mes`, el criterio para
    elegir el champion; a diferencia del total, ahí no se cancelan los errores de meses distintos), el MAE y el RMSE.
    """
    total = ventanas.groupby(["origen", "serie"])[["real", "pronostico"]].sum()
    niveles = {
        "m1": ventanas[ventanas["ventana"] == 1],
        "m2": ventanas[ventanas["ventana"] == 2],
        "total": total,
    }
    metricas = {f"wape_{n}": wape(c["real"], c["pronostico"]) for n, c in niveles.items()}
    metricas |= {f"mape_{n}": mape(c["real"], c["pronostico"]) for n, c in niveles.items()}
    metricas["sesgo_total"] = sesgo(total["real"], total["pronostico"])
    metricas["wape_tipo_mes"] = wape(ventanas["real"], ventanas["pronostico"])
    metricas["mae"] = mae(ventanas["real"], ventanas["pronostico"])
    metricas["rmse"] = rmse(ventanas["real"], ventanas["pronostico"])
    return {m: metricas[m] for m in METRICAS}


def por_origen(ventanas: pd.DataFrame) -> pd.DataFrame:
    """El formato estándar de cada origen por separado: una fila por origen y una columna por métrica."""
    return pd.DataFrame({origen: formato_estandar(v) for origen, v in ventanas.groupby("origen")}).T
