"""Pruebas de common.features con datos sintéticos. La principal comprueba que ninguna variable usa información
posterior al origen: recorta los datos al domingo del origen, recalcula y compara."""
import numpy as np
import pandas as pd
import pytest

from common.features import (
    FIN_STOCK_SUBESTIMADO,
    INICIO,
    INICIO_STOCK,
    ErrorCalendario,
    construir_base,
    construir_panel,
    ultima_semana_completa,
    variables_calendario,
)

SEMANAS = pd.date_range("2024-01-01", periods=125, freq="7D")  # hasta may-2026
CORTE = SEMANAS[-1] + pd.Timedelta(days=6)
SERIES = [  # submarca, edad, departamento, categoria, tipo_producto
    ("Submarca A", "Bebé", "Depto 1", "Cat 1", "Tipo 1"),
    ("Submarca A", "Bebé", "Depto 1", "Cat 1", "Tipo 2"),
    ("Submarca B", "Niño", "Depto 2", "Cat 2", "Tipo 3"),  # empieza a vender en la semana 70
    ("Submarca B", "Niño", "Depto 2", "Cat 2", "Tipo 4"),  # deja de vender en la semana 40
]
COLUMNAS_PRODUCTO = ["sku", "estilo_color", "submarca", "edad", "departamento", "categoria", "tipo_producto"]


def _calendario(sin=()) -> pd.DataFrame:
    """Calendario sintético; `sin` quita pares (evento, año)."""
    filas = [
        ("Hot Sale", "comercial", "2023-05-29", "2023-06-06"),
        ("Hot Sale", "comercial", "2024-05-15", "2024-05-23"),
        ("Hot Sale", "comercial", "2025-05-26", "2025-06-03"),
        ("Hot Sale", "comercial", "2026-05-25", "2026-06-02"),
        ("El Buen Fin", "comercial", "2023-11-17", "2023-11-20"),
        ("El Buen Fin", "comercial", "2024-11-15", "2024-11-18"),
        ("El Buen Fin", "comercial", "2025-11-13", "2025-11-17"),
        ("El Buen Fin", "comercial", "2026-11-13", "2026-11-17"),
        *[("Navidad", "festividad", f"{anio}-12-25", f"{anio}-12-25") for anio in range(2023, 2027)],
    ]
    calendario = pd.DataFrame(filas, columns=["evento", "tipo", "fecha_inicio", "fecha_fin"])
    calendario[["fecha_inicio", "fecha_fin"]] = calendario[["fecha_inicio", "fecha_fin"]].apply(pd.to_datetime)
    anios = calendario["fecha_inicio"].dt.year
    return calendario[[par not in sin for par in zip(calendario["evento"], anios, strict=True)]]


@pytest.fixture(scope="module")
def datos():
    """Cuatro series con 3 estilos-color de 2 SKU; el tercer estilo-color de cada serie se lanza en la semana 80."""
    rng = np.random.default_rng(0)
    productos, ventas, stock = [], [], []
    for s, jerarquia in enumerate(SERIES):
        for e in range(3):
            for t in range(2):
                sku = f"SKU-{s}{e}{t}"
                productos.append((sku, f"EC-{s}{e}", *jerarquia))
                desde = 80 if e == 2 else {2: 70}.get(s, 0)
                hasta = 40 if s == 3 else len(SEMANAS)
                for i in range(desde, hasta):
                    if (unidades := rng.poisson(2)) > 0:
                        ventas.append((SEMANAS[i], sku, unidades))
                    if SEMANAS[i] >= INICIO_STOCK:
                        stock.append((SEMANAS[i], sku, rng.integers(0, 20), rng.integers(0, 8)))
    return {
        "ventas_sku": pd.DataFrame(ventas, columns=["semana", "sku", "unidades"]),
        "stock_sku": pd.DataFrame(stock, columns=["semana", "sku", "stock_cierre", "dias_disponibles"]),
        "productos": pd.DataFrame(productos, columns=COLUMNAS_PRODUCTO),
    }


def _calcular(datos, corte):
    """Panel y base usando solo los datos hasta `corte`."""
    hasta = {n: d[d["semana"] <= corte] if "semana" in d else d for n, d in datos.items()}
    panel = construir_panel(hasta["ventas_sku"], hasta["stock_sku"], hasta["productos"], corte)
    return panel, construir_base(panel, _calendario(), corte)


@pytest.fixture(scope="module")
def completos(datos):
    return _calcular(datos, CORTE)


@pytest.mark.parametrize("i", [30, 60, 85, 110])
def test_sin_fuga_de_informacion(datos, completos, i):
    # 30: sin stock ni lanzamientos; 60: stock subestimado; 85: serie nueva y serie que dejó de vender; 110: reciente
    origen = SEMANAS[i]
    _, base = completos
    _, recortada = _calcular(datos, origen + pd.Timedelta(days=6))

    def del_origen(b):
        return b[b["origen"] == origen].sort_values(["serie", "h"]).reset_index(drop=True)

    esperado, obtenido = del_origen(base), del_origen(recortada)
    assert len(esperado) > 0
    pd.testing.assert_frame_equal(obtenido.drop(columns="y"), esperado.drop(columns="y"))
    assert obtenido["y"].isna().all()


def test_ultima_semana_completa():
    assert ultima_semana_completa(pd.Timestamp("2026-10-18")) == pd.Timestamp("2026-10-12")  # domingo
    assert ultima_semana_completa(pd.Timestamp("2026-10-17")) == pd.Timestamp("2026-10-05")  # sábado


def test_panel_tiene_todas_las_semanas_con_venta_cero(datos, completos):
    panel, _ = completos
    assert panel.groupby("serie")["semana"].apply(lambda s: s.tolist() == list(SEMANAS)).all()
    assert panel["venta"].sum() == datos["ventas_sku"]["unidades"].sum()
    assert (panel["venta"] == 0).any()


def test_stock_y_lanzamientos_nulos_donde_no_se_pueden_saber(completos):
    panel, _ = completos
    antes = panel["semana"] < FIN_STOCK_SUBESTIMADO
    assert panel.loc[antes, "stock_cierre"].isna().all() and panel.loc[~antes, "stock_cierre"].notna().all()
    assert panel.loc[panel["semana"] < INICIO_STOCK, "disponibilidad"].isna().all()
    assert panel.loc[panel["semana"].between(INICIO_STOCK, FIN_STOCK_SUBESTIMADO), "disponibilidad"].notna().any()
    primer_anio = panel["semana"] < INICIO + pd.Timedelta(weeks=52)
    assert panel.loc[primer_anio, "lanzamientos_8"].isna().all()
    lanzados = panel[panel["semana"].between(SEMANAS[80], SEMANAS[87])]
    assert (lanzados.groupby("serie")["lanzamientos_8"].max() >= 1).sum() == 3  # la serie 4 ya no vende


def test_disponibilidad_pondera_por_la_venta_previa():
    semana = pd.Timestamp("2025-03-03")
    skus = ["SKU-1", "SKU-2"]
    productos = pd.DataFrame([(sku, f"EC-{sku}", *SERIES[0]) for sku in skus], columns=COLUMNAS_PRODUCTO)
    semanas_previas = [semana - pd.Timedelta(weeks=1), semana - pd.Timedelta(weeks=2)]
    ventas = pd.DataFrame({"semana": semanas_previas, "sku": skus, "unidades": [10, 30]})
    # SKU-1 disponible los 7 días, SKU-2 ninguno: (10 × 1 + 30 × 0) / 40
    stock = pd.DataFrame({"semana": semana, "sku": skus, "stock_cierre": [3, 0], "dias_disponibles": [7, 0]})
    panel = construir_panel(ventas, stock, productos, semana + pd.Timedelta(days=6))
    assert panel.loc[panel["semana"] == semana, "disponibilidad"].item() == pytest.approx(0.25)


def test_base_solo_con_series_activas_y_horizontes(completos):
    panel, base = completos
    nace, muere = SERIES[2][-1], SERIES[3][-1]
    assert base.loc[base["tipo_producto"] == nace, "origen"].min() == SEMANAS[70]
    assert base.loc[base["tipo_producto"] == muere, "origen"].max() == SEMANAS[39 + 51]  # última venta + 51
    assert base.groupby(["serie", "origen"])["h"].apply(lambda h: h.tolist() == list(range(1, 14))).all()
    assert (base["semana_objetivo"] == base["origen"] + pd.to_timedelta(7 * base["h"], unit="D")).all()
    assert base["y"].isna().equals(base["semana_objetivo"] > SEMANAS[-1])
    venta = panel.set_index(["serie", "semana"])["venta"]
    fila = base[base["semana_objetivo"] == SEMANAS[100]].iloc[0]
    assert fila["venta_anio_anterior"] == venta[(fila["serie"], SEMANAS[48])]
    assert fila["y"] == venta[(fila["serie"], SEMANAS[100])]


def test_variables_calendario():
    semanas = pd.to_datetime(["2025-01-06", "2025-03-03", "2025-05-26", "2025-06-02"])
    cal = variables_calendario(_calendario(), semanas).set_index("semana_objetivo")
    assert cal["dias_hot_sale"].tolist() == [0, 0, 7, 2]
    assert cal["semanas_a_hot_sale"].tolist() == [13, 12, 0, 0]  # tope de 13; en curso cuenta 0
    assert cal["dias_quincena"].tolist() == [0, 0, 1, 0]  # 31 de mayo
    assert cal["semana_iso"].tolist() == [2, 10, 22, 23]
    assert cal["mes"].tolist() == [1, 3, 5, 6]


def test_estima_la_fecha_que_falta_y_avisa():
    # sin Hot Sale 2026: se estima en la semana ISO 22 de 2026, el lunes 25 de mayo
    with pytest.warns(UserWarning, match="Hot Sale no está en el calendario.*2026-05-25"):
        cal = variables_calendario(_calendario(sin={("Hot Sale", 2026)}), [pd.Timestamp("2026-03-02")])
    assert cal["semanas_a_hot_sale"].item() == 12


def test_se_detiene_si_el_horizonte_llega_a_un_evento_que_falta():
    # el Hot Sale más temprano empezó un 15 de mayo: la semana del 11 al 17 de mayo de 2026 ya lo puede incluir
    with pytest.raises(ErrorCalendario, match="Hot Sale 2026"):
        variables_calendario(_calendario(sin={("Hot Sale", 2026)}), [pd.Timestamp("2026-05-11")])
