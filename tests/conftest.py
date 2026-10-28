"""Datos sintéticos de las pruebas de entrenamiento: venta diaria de cuatro series y la base que sale de ella con
common.features, igual que en src/features.py."""
import numpy as np
import pandas as pd
import pytest

from common.features import INICIO, INICIO_STOCK, clave_serie, construir_base, construir_panel
from common.modelos import a_categoricas

SEMANAS = 125  # hasta may-2026
SERIES = {  # submarca, edad, departamento, categoria, tipo_producto -> semanas con venta
    ("Submarca A", "Bebé", "Depto 1", "Cat 1", "Tipo 1"): (0, SEMANAS),
    ("Submarca A", "Bebé", "Depto 1", "Cat 1", "Tipo 2"): (0, SEMANAS),
    ("Submarca B", "Niño", "Depto 2", "Cat 2", "Tipo 3"): (105, SEMANAS),  # nueva a mitad del backtest
    ("Submarca B", "Niño", "Depto 2", "Cat 2", "Tipo 4"): (0, 40),  # deja de vender
}
FORMA_SEMANAL = np.array([1.0, 0.8, 0.8, 0.9, 1.1, 1.6, 1.4])  # de lunes a domingo


def calendario() -> pd.DataFrame:
    filas = [
        ("Hot Sale", "comercial", "2023-05-29", "2023-06-06"),
        ("Hot Sale", "comercial", "2024-05-15", "2024-05-23"),
        ("Hot Sale", "comercial", "2025-05-26", "2025-06-03"),
        ("Hot Sale", "comercial", "2026-05-25", "2026-06-02"),
        ("El Buen Fin", "comercial", "2023-11-17", "2023-11-20"),
        ("El Buen Fin", "comercial", "2024-11-15", "2024-11-18"),
        ("El Buen Fin", "comercial", "2025-11-13", "2025-11-17"),
        ("El Buen Fin", "comercial", "2026-11-13", "2026-11-17"),
    ]
    cal = pd.DataFrame(filas, columns=["evento", "tipo", "fecha_inicio", "fecha_fin"])
    cal[["fecha_inicio", "fecha_fin"]] = cal[["fecha_inicio", "fecha_fin"]].apply(pd.to_datetime)
    return cal


@pytest.fixture(scope="session")
def sinteticos() -> dict[str, pd.DataFrame]:
    """`base` (ya con categóricas), `panel` semanal, `venta_diaria` por serie y `ultima_semana`. Dos SKU por serie, con
    venta diaria Poisson con forma semanal y más venta en el Hot Sale."""
    rng = np.random.default_rng(0)
    dias = pd.date_range(INICIO, periods=SEMANAS * 7)
    hot_sale = np.zeros(len(dias))
    for _, evento in calendario().query("evento == 'Hot Sale'").iterrows():
        hot_sale[(dias >= evento["fecha_inicio"]) & (dias <= evento["fecha_fin"])] = 1
    productos, ventas, stock = [], [], []
    for s, (jerarquia, (desde, hasta)) in enumerate(SERIES.items()):
        activos = (dias >= dias[desde * 7]) & (dias < dias[0] + pd.Timedelta(weeks=hasta))
        for k in range(2):
            sku = f"SKU-{s}{k}"
            productos.append((sku, f"EC-{s}{k}", *jerarquia))
            media = (2 + s) * FORMA_SEMANAL[dias.dayofweek] * (1 + 2 * hot_sale) * activos
            unidades = rng.poisson(media)
            ventas += [(dia, sku, u) for dia, u in zip(dias, unidades, strict=True) if u > 0]
            semanas_stock = pd.date_range(INICIO_STOCK, dias[-1], freq="7D")
            stock += [(semana, sku, rng.integers(0, 20), rng.integers(0, 8)) for semana in semanas_stock]
    productos = pd.DataFrame(
        productos, columns=["sku", "estilo_color", "submarca", "edad", "departamento", "categoria", "tipo_producto"]
    )
    ventas = pd.DataFrame(ventas, columns=["fecha", "sku", "unidades"])
    stock_sku = pd.DataFrame(stock, columns=["semana", "sku", "stock_cierre", "dias_disponibles"])
    ventas_sku = (
        ventas.assign(semana=ventas["fecha"] - pd.to_timedelta(ventas["fecha"].dt.dayofweek, unit="D"))
        .groupby(["semana", "sku"], as_index=False)["unidades"]
        .sum()
    )
    corte = dias[-1]
    panel = construir_panel(ventas_sku, stock_sku, productos, corte)
    base = construir_base(panel, calendario(), corte)
    venta_diaria = (
        ventas.merge(productos.assign(serie=clave_serie)[["sku", "serie"]], on="sku")
        .groupby(["serie", "fecha"], as_index=False)["unidades"]
        .sum()
    )
    return {
        "base": a_categoricas(base),
        "panel": panel,
        "venta_diaria": venta_diaria,
        "ultima_semana": panel["semana"].max(),
    }
