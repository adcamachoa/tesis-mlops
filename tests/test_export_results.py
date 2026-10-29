"""Pruebas de src/export_results.py y common.results_schema: un results.json por origen, con su última corrida y las
métricas del train de su versión; la venta como índice; regenerar un origen viejo con datos posteriores da el mismo
archivo; el manifest con las reglas de la página, y no se escribe nada si algo falla."""
import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from common import backtest, metrics
from common.modelos import MediaMovil, NaiveEstacional
from common.results_schema import ErrorEsquema, validar, validar_carpeta, validar_manifest
from export_results import DECIMALES_INDICE, SEMANAS_REALES, construir_publicacion, escribir
from predict import tabla_predicciones

ENTRENAMIENTO = pd.Timestamp("2026-05-20 08:00:00")
PREDICCION = pd.Timestamp("2026-05-20 08:30:00")
GENERADO = pd.Timestamp("2026-05-20 09:00:00")
ANTES = pd.Timedelta(weeks=4)  # el origen viejo, su train y su predicción
ESCALA_VIEJA = 1.5  # las métricas del train viejo, para distinguirlas de las del nuevo
REDONDEO = 0.5 * 10**-DECIMALES_INDICE + 1e-9


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
    un train 4 semanas antes, y el nuevo la versión 4. `runs` es el run de cada versión en Unity Catalog, y `viejo` y
    `nuevo` las rutas de sus results.json."""
    ultima = sinteticos["ultima_semana"]
    origenes = backtest.origenes(ultima, n=2)
    candidatos = {"naive_estacional": NaiveEstacional(), "media_movil_8": MediaMovil(8)}
    del_backtest = backtest.correr(candidatos, sinteticos["base"], origenes)
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
        "resumen": resumen,
        "predicciones": pd.concat(
            [_prediccion(sinteticos, ultima - ANTES, PREDICCION - ANTES, 3),
             _prediccion(sinteticos, ultima, PREDICCION, 4)],
            ignore_index=True,
        ),
        "panel": sinteticos["panel"],
        "viejo": f"{(ultima - ANTES).date()}/results.json",
        "nuevo": f"{ultima.date()}/results.json",
    }


@pytest.fixture(scope="module")
def publicacion(tablas) -> tuple[dict, dict[str, dict]]:
    return construir_publicacion(tablas["metricas"], tablas["predicciones"], tablas["panel"], tablas["runs"], GENERADO)


@pytest.fixture(scope="module")
def resultados(tablas, publicacion) -> dict:
    """El results.json del origen más reciente."""
    return publicacion[1][tablas["nuevo"]]


def _del_origen(df: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    return df[df["origen"] == pd.Timestamp(resultados["corrida"]["origen"])]


def _venta_real(tablas, resultados) -> pd.DataFrame:
    panel = tablas["panel"]
    return panel[panel["semana"].isin(pd.to_datetime(resultados["semanas"]["real"]))]


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
    nuevo, viejo = (tablas[o].removesuffix("/results.json") for o in ("nuevo", "viejo"))
    assert manifest == {
        "version_manifest": 1,
        "generado": "2026-05-20T09:00:00",
        "pronosticos": [
            {"id": nuevo, "ruta": f"{nuevo}/results.json", "version_esquema": 1, "origen": nuevo,
             "fecha_prediccion": "2026-05-20T08:30:00"},
            {"id": viejo, "ruta": f"{viejo}/results.json", "version_esquema": 1, "origen": viejo,
             "fecha_prediccion": "2026-04-22T08:30:00"},
        ],
    }


def test_results_json_cumple_el_esquema(resultados):
    validar(resultados)
    assert resultados["champion"] == {"modelo": "media_movil_8", "version": 4}
    assert resultados["corrida"]["fecha_prediccion"] == "2026-05-20T08:30:00"
    assert resultados["corrida"]["fecha_entrenamiento"] == "2026-05-20T08:00:00"
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
    assert viejo["corrida"]["fecha_entrenamiento"] == "2026-04-22T08:00:00"
    assert nuevo["corrida"]["fecha_entrenamiento"] == "2026-05-20T08:00:00"
    for resultados, escala in ((viejo, ESCALA_VIEJA), (nuevo, 1.0)):
        for fila in resultados["metricas"]:
            esperado = tablas["resumen"].loc[fila["modelo"], "wape_total"] * escala
            assert fila["wape_total"] == pytest.approx(esperado, abs=1e-4)


def test_regenerar_un_origen_viejo_con_datos_posteriores_da_el_mismo_archivo(sinteticos, tablas, publicacion):
    """Lo que había al exportar el origen viejo: la venta hasta su semana, su train y su predicción. Después llegan
    semanas de venta, un train y la predicción de un origen nuevo, y su archivo sale igual salvo `generado`."""
    origen = sinteticos["ultima_semana"] - ANTES
    metricas, predicciones, panel = tablas["metricas"], tablas["predicciones"], tablas["panel"]
    assert panel["semana"].max() > origen
    _, entonces = construir_publicacion(
        metricas[metricas["run_id"] == "train-viejo"],
        predicciones[predicciones["origen"] == origen],
        panel[panel["semana"] <= origen],
        {3: tablas["runs"][3]},
        GENERADO - ANTES,
    )
    _, ahora = publicacion
    assert list(entonces) == [tablas["viejo"]] and len(ahora) == 2
    antes, despues = entonces[tablas["viejo"]], ahora[tablas["viejo"]]
    assert antes["generado"] != despues["generado"]
    assert {**antes, "generado": None} == {**despues, "generado": None}


def test_solo_la_ultima_corrida_de_cada_origen(sinteticos, tablas, publicacion):
    repetida = _prediccion(sinteticos, sinteticos["ultima_semana"], PREDICCION + pd.Timedelta(hours=1), 4, escala=2.0)
    todas = pd.concat([tablas["predicciones"], repetida], ignore_index=True)
    _, archivos = construir_publicacion(tablas["metricas"], todas, tablas["panel"], tablas["runs"], GENERADO)
    _, sola = construir_publicacion(tablas["metricas"], repetida, tablas["panel"], tablas["runs"], GENERADO)
    assert archivos[tablas["nuevo"]] == sola[tablas["nuevo"]]
    assert archivos[tablas["nuevo"]]["corrida"]["fecha_prediccion"] == "2026-05-20T09:30:00"
    # la corrida nueva de otro origen no cambia el viejo
    assert archivos[tablas["viejo"]] == publicacion[1][tablas["viejo"]]


def test_sin_el_train_de_su_version_no_se_exporta(tablas):
    runs = {**tablas["runs"], 3: "un-run-que-no-esta"}
    with pytest.raises(ValueError, match="registró la versión 3"):
        construir_publicacion(tablas["metricas"], tablas["predicciones"], tablas["panel"], runs, GENERADO)


def test_venta_y_pronostico_como_indice(tablas, resultados):
    real, predicciones = _venta_real(tablas, resultados), _del_origen(tablas["predicciones"], resultados)
    factor = 100 / (real["venta"].sum() / SEMANAS_REALES)
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


def test_el_total_es_la_suma_de_las_submarcas(resultados):
    total, *resto = resultados["series"]
    submarcas = [s for s in resto if s["nivel"] == "submarca"]
    assert [s["nombre"] for s in submarcas] == ["Submarca A", "Submarca B"]
    for campo in ("real", "pronostico", "ventanas"):
        suma = np.sum([s[campo] for s in submarcas], axis=0)
        np.testing.assert_allclose(total[campo], suma, atol=REDONDEO * (len(submarcas) + 1))


def test_las_series_de_mas_venta(tablas, resultados):
    venta = _venta_real(tablas, resultados).groupby("serie")["venta"].sum().sort_values(ascending=False)
    _, archivos = construir_publicacion(
        tablas["metricas"], tablas["predicciones"], tablas["panel"], tablas["runs"], GENERADO, top=2
    )
    pocas = archivos[tablas["nuevo"]]
    assert [s["nombre"] for s in pocas["series"] if s["nivel"] == "serie"] == venta.index[:2].tolist()
    # con el tope de 20 entran todas las que tienen pronóstico, y no la que dejó de vender
    todas = [s["nombre"] for s in resultados["series"] if s["nivel"] == "serie"]
    assert sorted(todas) == sorted(_del_origen(tablas["predicciones"], resultados)["serie"].unique())
    assert len(todas) == 3


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
    "serie con otro largo": (lambda r: r["series"][1]["real"].pop(), "semanas reales"),
    "número no finito": (lambda r: r["series"][0]["pronostico"].__setitem__(0, float("nan")), "no es un número válido"),
    "dos totales": (lambda r: r["series"].append(copy.deepcopy(r["series"][0])), "una sola serie total"),
    "venta en piezas": (lambda r: r["series"][0].update(real=[v * 50 for v in r["series"][0]["real"]]), "como índice"),
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
