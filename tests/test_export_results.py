"""Pruebas de src/export_results.py y common.results_schema: un results.json por origen, con su última corrida y las
métricas del train de su versión; el árbol de la jerarquía, con las métricas del backtest sumadas a cada nodo; la venta
como índice; la venta real de las semanas y ventanas pronosticadas que ya cerraron, y el pronóstico vigente de los
orígenes anteriores; regenerar un origen viejo con datos posteriores solo agrega esa venta real; el manifest con las
reglas de la página, y no se escribe nada si algo falla."""
import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from common import backtest, metrics
from common.modelos import MediaMovil, NaiveEstacional
from common.results_schema import METRICAS_NODO, ErrorEsquema, validar, validar_carpeta, validar_manifest
from export_results import (
    DECIMALES_INDICE,
    SEMANAS_REALES,
    construir_publicacion,
    escribir,
    pronostico_vigente,
)
from predict import tabla_predicciones

ENTRENAMIENTO = pd.Timestamp("2026-05-20 08:00:00")
PREDICCION = pd.Timestamp("2026-05-20 08:30:00")
GENERADO = pd.Timestamp("2026-05-20 09:00:00")
# el origen viejo, su train y su predicción: con 8 semanas cerradas de su horizonte y su mes 1
ANTES = pd.Timedelta(weeks=8)
ESCALA_VIEJA = 1.5  # las métricas del train viejo, para distinguirlas de las del nuevo
REDONDEO = 0.5 * 10**-DECIMALES_INDICE + 1e-9
DESPUES_DEL_ORIGEN = ("generado", "real_hasta")  # lo que cambia cuando llegan semanas nuevas, además de la venta real
VENTA_DEL_HORIZONTE = ("real_horizonte", "ventanas_real")


def _prediccion(sinteticos, origen: pd.Timestamp, fecha: pd.Timestamp, version: int, escala: float = 1.0):
    """Una corrida de predict.py en `origen`, con la media móvil de 8 semanas como champion."""
    base = sinteticos["base"]
    filas = base[base["origen"] == origen]
    venta_total = sinteticos["venta_diaria"].groupby("fecha")["unidades"].sum()
    pronostico = MediaMovil(8).predecir(filas) * escala
    return tabla_predicciones(filas, pronostico, venta_total, fecha, "media_movil_8", version)


@pytest.fixture(scope="module")
def tablas(sinteticos) -> dict:
    """Las tablas como las lee export_results.py, con dos orígenes: el viejo lo pronosticó la versión 3, registrada en
    un train 8 semanas antes, y el nuevo la versión 4. `runs` es el run de cada versión en Unity Catalog, y `viejo` y
    `nuevo` las rutas de sus results.json."""
    ultima = sinteticos["ultima_semana"]
    origenes = backtest.origenes(ultima, n=2)
    candidatos = {"naive_estacional": NaiveEstacional(), "media_movil_8": MediaMovil(8)}
    del_backtest = backtest.correr(candidatos, sinteticos["base"], origenes)
    # el train viejo hizo su backtest con lo que se sabía en su origen: sus ventanas cierran antes
    del_backtest_viejo = backtest.correr(candidatos, sinteticos["base"], backtest.origenes(ultima - ANTES, n=2))
    resumen = pd.DataFrame(
        {n: metrics.formato_estandar(metrics.a_ventanas(p, sinteticos["venta_diaria"]))
         for n, p in del_backtest.groupby("modelo")}
    ).T

    def train(fecha: pd.Timestamp, resumen: pd.DataFrame, run: str) -> pd.DataFrame:
        """Las filas de una corrida de train.py en metricas_modelos, con un run hijo por modelo."""
        return resumen.reset_index(names="modelo").assign(
            fecha_corrida=fecha,
            run_id=run,
            run_id_modelo=lambda d: run + "/" + d["modelo"],
            origenes=len(origenes),
            primer_origen=origenes[0].date(),
            ultimo_origen=origenes[-1].date(),
            es_mejor=lambda d: d["modelo"] == backtest.elegir_mejor(resumen),
        )

    return {
        "metricas": pd.concat(
            [train(ENTRENAMIENTO - ANTES, resumen * ESCALA_VIEJA, "train-viejo"),
             train(ENTRENAMIENTO, resumen, "train-nuevo")],
            ignore_index=True,
        ),
        "runs": {3: "train-viejo/media_movil_8", 4: "train-nuevo/media_movil_8"},
        # el artefacto del backtest de cada modelo en cada train
        "backtests": {f"{run}/{modelo}": p for run, de_run in (("train-viejo", del_backtest_viejo),
                                                                ("train-nuevo", del_backtest))
                      for modelo, p in de_run.groupby("modelo")},
        "resumen": resumen,
        "predicciones": pd.concat(
            [_prediccion(sinteticos, ultima - ANTES, PREDICCION - ANTES, 3),
             _prediccion(sinteticos, ultima, PREDICCION, 4)],
            ignore_index=True,
        ),
        "panel": sinteticos["panel"],
        "venta_diaria": sinteticos["venta_diaria"],
        "viejo": f"{(ultima - ANTES).date()}/results.json",
        "nuevo": f"{ultima.date()}/results.json",
    }


def _publicar(tablas, generado=GENERADO, minimo=0.5, **cambios) -> tuple[dict, dict[str, dict]]:
    """construir_publicacion con las tablas de `tablas`, salvo las que vienen en `cambios`."""
    t = {k: tablas[k] for k in ("metricas", "predicciones", "panel", "venta_diaria", "runs", "backtests")} | cambios
    return construir_publicacion(t["metricas"], t["predicciones"], t["panel"], t["venta_diaria"], t["runs"],
                                 t["backtests"], generado, minimo)


@pytest.fixture(scope="module")
def publicacion(tablas) -> tuple[dict, dict[str, dict]]:
    return _publicar(tablas)


@pytest.fixture(scope="module")
def resultados(tablas, publicacion) -> dict:
    """El results.json del origen más reciente."""
    return publicacion[1][tablas["nuevo"]]


def _del_origen(df: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    return df[df["origen"] == pd.Timestamp(resultados["corrida"]["origen"])]


def _venta_real(tablas, resultados) -> pd.DataFrame:
    panel = tablas["panel"]
    return panel[panel["semana"].isin(pd.to_datetime(resultados["semanas"]["real"]))]


def _factor(tablas, resultados) -> float:
    return 100 / (_venta_real(tablas, resultados)["venta"].sum() / SEMANAS_REALES)


def _contenido(carpeta) -> dict[str, str]:
    return {p.relative_to(carpeta).as_posix(): p.read_text(encoding="utf-8") for p in carpeta.rglob("*") if p.is_file()}


def test_un_results_json_por_origen_y_el_manifest(tablas, publicacion, tmp_path):
    manifest, archivos = publicacion
    escribir(manifest, archivos, tmp_path)
    escritos = _contenido(tmp_path)
    assert sorted(escritos) == sorted(
        ["results.json", "datos/manifest.json", f"datos/{tablas['viejo']}", f"datos/{tablas['nuevo']}"]
    )
    assert validar_carpeta(tmp_path / "datos") == manifest
    for ruta, contenido in archivos.items():
        assert json.loads(escritos[f"datos/{ruta}"]) == contenido
    # el results.json de siempre es el mismo archivo del origen más reciente
    assert escritos["results.json"] == escritos[f"datos/{tablas['nuevo']}"]
    # los results.json van compactos, sin sangría; el manifest, legible
    assert "\n" not in escritos["results.json"] and "\n" in escritos["datos/manifest.json"]
    nuevo, viejo = (tablas[o].removesuffix("/results.json") for o in ("nuevo", "viejo"))
    assert manifest == {
        "version_manifest": 1,
        "generado": "2026-05-20T09:00:00",
        "pronosticos": [
            {"id": nuevo, "ruta": f"{nuevo}/results.json", "version_esquema": 2, "origen": nuevo,
             "fecha_prediccion": "2026-05-20T08:30:00"},
            {"id": viejo, "ruta": f"{viejo}/results.json", "version_esquema": 2, "origen": viejo,
             "fecha_prediccion": "2026-03-25T08:30:00"},
        ],
    }


def test_results_json_cumple_el_esquema(resultados):
    validar(resultados)
    assert resultados["version_esquema"] == 2 and resultados["indice_minimo"] == 0.5
    assert resultados["champion"] == {"modelo": "media_movil_8", "version": 4}
    assert resultados["corrida"]["fecha_prediccion"] == "2026-05-20T08:30:00"
    assert resultados["corrida"]["fecha_entrenamiento"] == "2026-05-20T08:00:00"
    assert resultados["real_hasta"] == resultados["corrida"]["datos_hasta"]  # nada del horizonte cerró todavía
    assert len(resultados["semanas"]["real"]) == SEMANAS_REALES and len(resultados["semanas"]["pronostico"]) == 13
    assert resultados["semanas"]["real"][-1] == resultados["corrida"]["origen"]
    assert [v["dias"] for v in resultados["ventanas"]] == [31, 30, 30]
    # el champion marcado aparte del mejor
    assert {m["modelo"]: m["es_champion"] for m in resultados["metricas"]} == {
        "naive_estacional": False,
        "media_movil_8": True,
    }
    assert sum(m["es_mejor"] for m in resultados["metricas"]) == 1


def test_cada_origen_con_las_metricas_del_train_de_su_version(tablas, publicacion):
    viejo, nuevo = (publicacion[1][tablas[o]] for o in ("viejo", "nuevo"))
    assert viejo["champion"] == {"modelo": "media_movil_8", "version": 3}
    # el origen viejo no toma el train posterior, aunque sea el último de metricas_modelos
    assert viejo["corrida"]["fecha_entrenamiento"] == "2026-03-25T08:00:00"
    assert nuevo["corrida"]["fecha_entrenamiento"] == "2026-05-20T08:00:00"
    for resultados, escala in ((viejo, ESCALA_VIEJA), (nuevo, 1.0)):
        for fila in resultados["metricas"]:
            esperado = tablas["resumen"].loc[fila["modelo"], "wape_total"] * escala
            assert fila["wape_total"] == pytest.approx(esperado, abs=1e-4)


def test_la_venta_real_de_lo_que_ya_cerro_del_horizonte(tablas, publicacion):
    """Al origen viejo le cerraron 8 de sus 13 semanas y su mes 1: esas traen su venta real, con el mismo factor."""
    viejo = publicacion[1][tablas["viejo"]]
    assert viejo["real_hasta"] == f"{(pd.Timestamp(viejo['corrida']['datos_hasta']) + ANTES).date()}"
    factor = _factor(tablas, viejo)
    total = viejo["series"][0]
    panel = tablas["panel"].groupby("semana")["venta"].sum()
    semanas = pd.to_datetime(viejo["semanas"]["pronostico"])
    np.testing.assert_allclose(total["real_horizonte"][:8], panel.loc[semanas[:8]] * factor, atol=REDONDEO)
    assert total["real_horizonte"][8:] == [None] * 5
    # la venta real por mes sale de la venta diaria, como el error; el mes 2 todavía no cierra
    por_ventana = metrics.real_por_ventana(tablas["venta_diaria"], pd.Timestamp(viejo["corrida"]["origen"]))
    assert total["ventanas_real"][0] == pytest.approx(por_ventana.xs(1, level="ventana").sum() * factor, abs=REDONDEO)
    assert total["ventanas_real"][1:] == [None, None]
    # el total y las submarcas siguen sumando
    submarcas = [s for s in viejo["series"] if s["nivel"] == "submarca"]
    for campo, cerradas in (("real_horizonte", 8), ("ventanas_real", 1)):
        suma = np.sum([s[campo][:cerradas] for s in submarcas], axis=0)
        np.testing.assert_allclose(total[campo][:cerradas], suma, atol=REDONDEO * (len(submarcas) + 1))


def test_las_semanas_pasadas_traen_el_pronostico_vigente(tablas, publicacion):
    """En el origen nuevo, las 8 semanas desde el origen viejo traen lo que ese origen había pronosticado, en el factor
    del archivo nuevo; las anteriores no tuvieron pronóstico."""
    viejo, nuevo = (publicacion[1][tablas[o]] for o in ("viejo", "nuevo"))
    assert all(v is None for s in viejo["series"] for v in s["pronostico_anterior"])
    predicciones = _del_origen(tablas["predicciones"], viejo)
    semanal = predicciones[predicciones["granularidad"] == "semana"].groupby("fecha_inicio")["pronostico"].sum()
    total = nuevo["series"][0]
    assert total["pronostico_anterior"][:-8] == [None] * (SEMANAS_REALES - 8)
    semanas = pd.to_datetime(nuevo["semanas"]["real"][-8:])
    np.testing.assert_allclose(total["pronostico_anterior"][-8:], semanal.loc[semanas] * _factor(tablas, nuevo),
                               atol=REDONDEO)


def test_el_pronostico_vigente_es_el_del_origen_mas_reciente_que_cubre_cada_semana():
    def corrida(origen: str, semanas: int) -> pd.DataFrame:
        inicio = pd.Timestamp(origen) + pd.Timedelta(weeks=1)
        return pd.DataFrame({"origen": pd.Timestamp(origen), "granularidad": "semana", "pronostico": 1.0,
                             "fecha_inicio": pd.date_range(inicio, periods=semanas, freq="7D")})

    vigente = pronostico_vigente(pd.concat([corrida("2026-10-12", 13), corrida("2026-11-09", 13)]))
    por_semana = vigente.set_index("fecha_inicio")["origen"].dt.date.astype(str)
    assert por_semana[:"2026-11-09"].unique().tolist() == ["2026-10-12"]  # 4 semanas del 12-oct
    assert por_semana["2026-11-16":].unique().tolist() == ["2026-11-09"]  # desde ahí, el más reciente
    assert len(por_semana) == 4 + 13


def test_regenerar_un_origen_viejo_con_datos_posteriores_solo_agrega_su_venta_real(sinteticos, tablas, publicacion):
    """Lo que había al exportar el origen viejo: la venta hasta su semana, su train y su predicción. Después llegan
    semanas de venta, un train y la predicción de un origen nuevo: su archivo sale igual, salvo `generado`, `real_hasta`
    y la venta real de lo que cerró de su horizonte."""
    origen = sinteticos["ultima_semana"] - ANTES
    metricas, predicciones, panel = tablas["metricas"], tablas["predicciones"], tablas["panel"]
    venta_diaria = tablas["venta_diaria"]
    assert panel["semana"].max() > origen
    _, entonces = _publicar(
        tablas,
        GENERADO - ANTES,
        metricas=metricas[metricas["run_id"] == "train-viejo"],
        predicciones=predicciones[predicciones["origen"] == origen],
        panel=panel[panel["semana"] <= origen],
        venta_diaria=venta_diaria[venta_diaria["fecha"] <= origen + pd.Timedelta(days=6)],
        runs={3: tablas["runs"][3]},
    )
    _, ahora = publicacion
    assert list(entonces) == [tablas["viejo"]] and len(ahora) == 2
    antes, despues = entonces[tablas["viejo"]], ahora[tablas["viejo"]]
    assert all(v is None for s in antes["series"] for c in VENTA_DEL_HORIZONTE for v in s[c])

    def sin_lo_posterior(resultados: dict) -> dict:
        series = [{k: v for k, v in s.items() if k not in VENTA_DEL_HORIZONTE} for s in resultados["series"]]
        return {k: v for k, v in resultados.items() if k not in DESPUES_DEL_ORIGEN} | {"series": series}

    assert sin_lo_posterior(antes) == sin_lo_posterior(despues)


def test_solo_la_ultima_corrida_de_cada_origen(sinteticos, tablas, publicacion):
    repetida = _prediccion(sinteticos, sinteticos["ultima_semana"], PREDICCION + pd.Timedelta(hours=1), 4, escala=2.0)
    _, archivos = _publicar(tablas, predicciones=pd.concat([tablas["predicciones"], repetida], ignore_index=True))
    del_viejo = _del_origen(tablas["predicciones"], publicacion[1][tablas["viejo"]])
    _, sola = _publicar(tablas, predicciones=pd.concat([del_viejo, repetida], ignore_index=True))
    assert archivos[tablas["nuevo"]] == sola[tablas["nuevo"]]
    assert archivos[tablas["nuevo"]]["corrida"]["fecha_prediccion"] == "2026-05-20T09:30:00"
    # la corrida nueva de otro origen no cambia el viejo
    assert archivos[tablas["viejo"]] == publicacion[1][tablas["viejo"]]


def test_sin_el_train_de_su_version_no_se_exporta(tablas):
    with pytest.raises(ValueError, match="registró la versión 3"):
        _publicar(tablas, runs={**tablas["runs"], 3: "un-run-que-no-esta"})


def test_venta_y_pronostico_como_indice(tablas, resultados):
    real, predicciones = _venta_real(tablas, resultados), _del_origen(tablas["predicciones"], resultados)
    factor = _factor(tablas, resultados)
    total = resultados["series"][0]
    assert np.mean(total["real"]) == pytest.approx(100, abs=REDONDEO)
    # el mismo factor para cada serie, en la venta real, el pronóstico y las ventanas
    semanas = pd.to_datetime(resultados["semanas"]["real"])
    for serie in (s for s in resultados["series"] if s["nivel"] == "serie"):
        venta = real[real["serie"] == serie["nombre"]].set_index("semana")["venta"].reindex(semanas, fill_value=0)
        np.testing.assert_allclose(serie["real"], venta * factor, atol=REDONDEO)
        de_la_serie = predicciones[predicciones["serie"] == serie["nombre"]]
        for granularidad, campo in (("semana", "pronostico"), ("ventana", "ventanas")):
            filas = de_la_serie[de_la_serie["granularidad"] == granularidad].sort_values("periodo")
            np.testing.assert_allclose(serie[campo], filas["pronostico"] * factor, atol=REDONDEO)
    # el MAE y el RMSE van en piezas y se escalan; los porcentajes no cambian
    for fila in resultados["metricas"]:
        esperado = tablas["resumen"].loc[fila["modelo"]]
        for metrica in ("mae", "rmse"):
            assert fila[metrica] == pytest.approx(esperado[metrica] * factor, abs=1e-4)
        for metrica in ("wape_m1", "wape_tipo_mes", "mape_total", "sesgo_total"):
            assert fila[metrica] == pytest.approx(esperado[metrica], abs=1e-4)


def _nodos(resultados: dict) -> dict[str, dict]:
    return {nodo["id"]: nodo for nodo in resultados["series"]}


def test_el_arbol_publica_lo_que_llega_al_minimo(tablas, resultados):
    """La serie que dejó de vender (Tipo 4) no llega al mínimo. Su resto tampoco, así que se junta con su única
    hermana publicada (Tipo 3): Cat 2 no se desglosa, y sus hijos nunca dejan un resto que se pueda derivar."""
    assert [(n["id"], n["padre"], n["nivel"]) for n in resultados["series"]] == [
        ("total", None, "total"),
        ("submarca-a", "total", "submarca"),
        ("submarca-a/depto-1", "submarca-a", "departamento"),
        ("submarca-a/depto-1/cat-1", "submarca-a/depto-1", "categoria"),
        ("submarca-a/depto-1/cat-1/tipo-2", "submarca-a/depto-1/cat-1", "serie"),
        ("submarca-a/depto-1/cat-1/tipo-1", "submarca-a/depto-1/cat-1", "serie"),
        ("submarca-b", "total", "submarca"),
        ("submarca-b/depto-2", "submarca-b", "departamento"),
        ("submarca-b/depto-2/cat-2", "submarca-b/depto-2", "categoria"),
    ]
    nodos = _nodos(resultados)
    assert (nodos["submarca-b/depto-2"]["nombre"], nodos["submarca-b/depto-2"]["etiqueta"]) == (
        "Submarca B | Depto 2", "Depto 2")
    # las series se llaman como su clave, igual que en la versión 1
    series = [n["nombre"] for n in resultados["series"] if n["nivel"] == "serie"]
    assert sorted(series) == ["Submarca A | Depto 1 | Cat 1 | Tipo 1", "Submarca A | Depto 1 | Cat 1 | Tipo 2"]


def test_cada_nodo_suma_a_sus_hijos(resultados):
    nodos, hijos = _nodos(resultados), {}
    for nodo in resultados["series"]:
        if nodo["padre"]:
            hijos.setdefault(nodo["padre"], []).append(nodo)
    for padre, de_padre in hijos.items():
        for campo in ("real", "pronostico", "ventanas"):
            suma = np.sum([h[campo] for h in de_padre], axis=0)
            np.testing.assert_allclose(nodos[padre][campo], suma, atol=REDONDEO * (len(de_padre) + 1))


def test_las_metricas_de_cada_nodo_salen_de_su_backtest_sumado_al_nodo(tablas, publicacion):
    """En una serie, el formato estándar de esa serie sola; en el total, el WAPE y el sesgo del total. Cada origen usa
    el backtest del train de su versión."""
    for archivo, train in (("nuevo", "train-nuevo"), ("viejo", "train-viejo")):
        resultados = publicacion[1][tablas[archivo]]
        nodos = _nodos(resultados)
        modelos = [m["modelo"] for m in resultados["metricas"]]
        assert all([m["modelo"] for m in n["metricas"]] == modelos for n in resultados["series"])
        for modelo in modelos:
            ventanas = metrics.a_ventanas(tablas["backtests"][f"{train}/{modelo}"], tablas["venta_diaria"])
            de = {m["modelo"]: m for m in nodos["submarca-a/depto-1/cat-1/tipo-1"]["metricas"]}[modelo]
            esperado = metrics.formato_estandar(ventanas[ventanas["serie"] == "Submarca A | Depto 1 | Cat 1 | Tipo 1"])
            for m in METRICAS_NODO:
                assert de[m] == pytest.approx(esperado[m], abs=1e-4), (archivo, modelo, m)
            total = {m["modelo"]: m for m in nodos["total"]["metricas"]}[modelo]
            mes_1 = ventanas[ventanas["ventana"] == 1].groupby("origen")[["real", "pronostico"]].sum()
            assert total["wape_m1"] == pytest.approx(
                (mes_1["real"] - mes_1["pronostico"]).abs().sum() / mes_1["real"].sum(), abs=1e-4)
            assert total["sesgo_total"] == pytest.approx(
                ventanas["pronostico"].sum() / ventanas["real"].sum() - 1, abs=1e-4)


def test_sin_el_backtest_de_un_modelo_no_se_exporta(tablas):
    sin_uno = {k: v for k, v in tablas["backtests"].items() if k != "train-nuevo/naive_estacional"}
    with pytest.raises(ValueError, match="falta el backtest del run train-nuevo/naive_estacional"):
        _publicar(tablas, backtests=sin_uno)


def test_el_error_de_la_venta_en_piezas_no_muestra_la_venta(resultados):
    """El error puede llegar al log público: no lleva el promedio, que sería la venta semanal del total en piezas."""
    malo = copy.deepcopy(resultados)
    malo["series"][0]["real"] = [v * 57.3 for v in malo["series"][0]["real"]]
    with pytest.raises(ErrorEsquema) as error:
        validar(malo)
    assert re.findall(r"\d+", str(error.value)) == ["1", "100"]  # 1 problema; no promedia 100


CAMBIOS = {
    "falta un campo": (lambda r: r.pop("champion"), "champion"),
    "tipo equivocado": (lambda r: r["champion"].update(version="4"), "champion/version"),
    "fecha que no existe": (lambda r: r["corrida"].update(datos_hasta="2026-02-30"), "corrida/datos_hasta"),
    "serie con otro largo": (lambda r: r["series"][1]["real"].pop(), "semanas en real"),
    "número no finito": (lambda r: r["series"][0]["pronostico"].__setitem__(0, float("nan")), "no es un número válido"),
    "dos totales": (lambda r: r["series"].append(copy.deepcopy(r["series"][0])), "una sola serie total"),
    "venta en piezas": (lambda r: r["series"][0].update(real=[v * 50 for v in r["series"][0]["real"]]), "como índice"),
    "venta real de una semana que no cerró": (lambda r: r["series"][0]["real_horizonte"].__setitem__(12, 1.0),
                                              "real_horizonte justo en lo que cerró"),
    "falta la venta real de una ventana cerrada": (lambda r: r["series"][0]["ventanas_real"].__setitem__(0, None),
                                                   "ventanas_real justo en lo que cerró"),
    "pronóstico anterior en otras semanas": (lambda r: r["series"][1]["pronostico_anterior"].__setitem__(0, 1.0),
                                             "pronostico_anterior en otras semanas"),
    "real_hasta antes del origen": (lambda r: r.update(real_hasta="2020-01-05"), "real_hasta es anterior"),
    # el árbol
    "id repetido": (lambda r: r["series"][2].update(id=r["series"][1]["id"]), "aparece dos veces"),
    "padre que no existe": (lambda r: r["series"][-1].update(padre="no-existe"), "no está en el árbol"),
    "total con padre": (lambda r: r["series"][0].update(padre="submarca-a"), "es el total y tiene padre"),
    "nivel que no le toca": (lambda r: r["series"][-1].update(nivel="departamento"), "no puede colgar"),
    "nodo bajo el mínimo": (lambda r: r["series"][-1].update(real=[0.1] * len(r["series"][-1]["real"]),
                                                            pronostico=[0.1] * 13), "no llega al índice mínimo"),
    "hijos que suman más": (lambda r: r["series"][-1].update(real=[v + 50 for v in r["series"][-1]["real"]]),
                            "no suman el nodo en real"),
    # restar los hijos al padre dejaría ver lo que no se publicó
    "hijos que suman menos": (lambda r: r["series"][4]["real"].__setitem__(0, 0.0), "no suman el nodo en real"),
    "venta del horizonte que no cuadra": (lambda r: r["series"][4]["real_horizonte"].__setitem__(0, 0.0),
                                          "no suman el nodo en real_horizonte"),
    "métricas de otros modelos": (lambda r: r["series"][3]["metricas"].pop(), "mismos modelos"),
}


@pytest.mark.parametrize(("cambio", "mensaje"), CAMBIOS.values(), ids=list(CAMBIOS))
def test_si_un_origen_no_cumple_el_esquema_no_se_escribe_nada(tablas, publicacion, tmp_path, cambio, mensaje):
    escribir(*publicacion, tmp_path)
    anterior = _contenido(tmp_path)
    manifest, archivos = copy.deepcopy(publicacion)
    for contenido in (manifest, *archivos.values()):
        contenido["generado"] = "2026-05-21T09:00:00"  # si se escribiera algún archivo, cambiaría
    cambio(archivos[tablas["viejo"]])
    with pytest.raises(ErrorEsquema, match=mensaje):
        escribir(manifest, archivos, tmp_path)
    assert _contenido(tmp_path) == anterior  # lo anterior queda intacto


def _ruta(m: dict, i: int) -> str:
    return m["pronosticos"][i]["ruta"]


CAMBIOS_MANIFEST = {
    # los mismos que rechaza la página (pruebas/datos.test.js de tesis-mlops-web)
    "sin pronósticos": (lambda m, a: m.update(pronosticos=[]), "pronosticos"),
    "ruta que sale de la carpeta": (lambda m, a: m["pronosticos"][0].update(ruta="../otro/results.json"),
                                    "pronosticos/0/ruta"),
    "ruta absoluta": (lambda m, a: m["pronosticos"][0].update(ruta="/results.json"), "pronosticos/0/ruta"),
    "ruta a otro sitio": (lambda m, a: m["pronosticos"][0].update(ruta="https://otro.sitio/results.json"),
                          "pronosticos/0/ruta"),
    "id con espacios": (lambda m, a: m["pronosticos"][1].update(id="otro id"), "pronosticos/1/id"),
    "fecha que no existe": (lambda m, a: m["pronosticos"][0].update(origen="2026-13-01"), "pronosticos/0/origen"),
    "dos entradas del mismo origen": (lambda m, a: m["pronosticos"].append({**m["pronosticos"][0], "id": "repetido"}),
                                      "el origen .+ aparece dos veces"),
    "id repetido": (lambda m, a: m["pronosticos"][1].update(id=m["pronosticos"][0]["id"]),
                    "el id .+ aparece dos veces"),
    "versión 2 del manifest": (lambda m, a: m.update(version_manifest=2), "version_manifest"),
    "sin versión del manifest": (lambda m, a: m.pop("version_manifest"), "version_manifest"),
    # y lo que revisa la publicación: cada entrada con su archivo y ninguno de más
    "falta un archivo": (lambda m, a: a.pop(_ruta(m, 1)), "falta el archivo"),
    "archivo de más": (lambda m, a: a.update({"2020-01-06/results.json": a[_ruta(m, 0)]}), "no está en el manifest"),
    "archivo de otra predicción": (lambda m, a: m["pronosticos"][0].update(fecha_prediccion="2026-05-21T08:30:00"),
                                   "no corresponde a su entrada"),
    "archivo que no cumple el esquema": (lambda m, a: a[_ruta(m, 1)].pop("champion"), r"results\.json: .*champion"),
}


@pytest.mark.parametrize(("cambio", "mensaje"), CAMBIOS_MANIFEST.values(), ids=list(CAMBIOS_MANIFEST))
def test_el_manifest_se_rechaza_como_en_la_pagina(publicacion, cambio, mensaje):
    validar_manifest(*publicacion)
    manifest, archivos = copy.deepcopy(publicacion)
    cambio(manifest, archivos)
    with pytest.raises(ErrorEsquema, match=mensaje):
        validar_manifest(manifest, archivos)


@pytest.mark.parametrize(
    ("nombre", "texto", "mensaje"),
    [("notas.txt", "sin formato", "notas.txt: no es un JSON válido"),
     ("2020-01-06/results.json", "{}", "2020-01-06/results.json: no está en el manifest")],
    ids=["un archivo que no es JSON", "un JSON fuera del manifest"],
)
def test_la_carpeta_no_publica_archivos_fuera_del_manifest(publicacion, tmp_path, nombre, texto, mensaje):
    escribir(*publicacion, tmp_path)
    datos = tmp_path / "datos"
    (datos / nombre).parent.mkdir(exist_ok=True)
    (datos / nombre).write_text(texto, encoding="utf-8")
    with pytest.raises(ErrorEsquema, match=mensaje):
        validar_carpeta(datos)
