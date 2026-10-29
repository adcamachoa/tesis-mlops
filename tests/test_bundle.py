"""Pruebas de databricks.yml: el job semanal corre sus tareas sobre el schema de cada target y reentrena solo si la
condición toca_reentrenar lo pide; export junta las dos ramas. La programación solo corre en prod, y el bundle no
declara nada más que el job, para que deploy y destroy nunca toquen tablas, volúmenes ni el modelo."""
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
TAREAS = {
    "ingest": "src/ingest.py",
    "features": "src/features.py",
    "monitoreo_datos": "src/monitoreo_datos.py",
    "monitoreo_desempeno": "src/monitoreo_desempeno.py",
    "toca_reentrenar": None,  # condition_task
    "train": "src/train.py",
    "predict": "src/predict.py",
    "export": "src/export_results.py",
}
DEPENDENCIAS = {
    "features": [{"task_key": "ingest"}],
    "monitoreo_datos": [{"task_key": "features"}],
    "monitoreo_desempeno": [{"task_key": "monitoreo_datos"}],
    "toca_reentrenar": [{"task_key": "monitoreo_desempeno"}],
    "train": [{"task_key": "toca_reentrenar", "outcome": "true"}],
    "predict": [{"task_key": "train"}],
    # las dos ramas: predict si reentrena y la rama false si no
    "export": [{"task_key": "predict"}, {"task_key": "toca_reentrenar", "outcome": "false"}],
}
PARAMETROS_EXTRA = {  # además del schema
    "ingest": ["--reentrenar", "{{job.parameters.reentrenar}}"],
    "monitoreo_datos": ["--aceptar-drift", "{{job.parameters.aceptar_drift}}"],
    "monitoreo_desempeno": ["--reentrenar", "{{job.parameters.reentrenar}}"],
}


@pytest.fixture(scope="module")
def bundle() -> dict:
    return yaml.safe_load((RAIZ / "databricks.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(bundle) -> dict:
    return bundle["resources"]["jobs"]["demanda_pipeline"]


@pytest.fixture(scope="module")
def tareas(job) -> dict[str, dict]:
    return {t["task_key"]: t for t in job["tasks"]}


def test_solo_declara_el_job(bundle):
    assert list(bundle["resources"]) == ["jobs"]
    assert list(bundle["resources"]["jobs"]) == ["demanda_pipeline"]
    for target in bundle["targets"].values():
        assert "resources" not in target


def test_las_tareas_y_sus_dependencias(job, tareas):
    assert [t["task_key"] for t in job["tasks"]] == list(TAREAS)
    assert "depends_on" not in tareas["ingest"]
    assert {k: t["depends_on"] for k, t in tareas.items() if "depends_on" in t} == DEPENDENCIAS
    # Databricks exige que toda dependencia de la condición diga su rama
    for tarea in job["tasks"]:
        for dependencia in tarea.get("depends_on", []):
            assert (dependencia["task_key"] == "toca_reentrenar") == ("outcome" in dependencia)
    # export corre si ninguna falló y alguna corrió: en las dos ramas, y no si train o predict fallan. Con
    # AT_LEAST_ONE_SUCCESS exportaría aunque predict falle, porque la rama excluida cuenta como éxito
    assert {k: t["run_if"] for k, t in tareas.items() if "run_if" in t} == {"export": "NONE_FAILED"}


def test_reentrena_solo_si_la_condicion_lo_pide(tareas):
    assert tareas["toca_reentrenar"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.monitoreo_desempeno.values.reentrenar}}",
        "right": "true",
    }
    orden = list(TAREAS)
    # el desempeño se mide antes de la condición, así que una ventana que cierra esta semana puede adelantar el
    # reentrenamiento; el drift de datos se revisa antes, y un extremo detiene el job sin llegar a la condición
    assert orden.index("monitoreo_datos") < orden.index("monitoreo_desempeno") < orden.index("toca_reentrenar")


def test_cada_tarea_corre_su_script_sobre_el_schema_del_target(job):
    entornos = {e["environment_key"] for e in job["environments"]}
    for tarea in job["tasks"]:
        script = TAREAS[tarea["task_key"]]
        if script is None:
            assert "spark_python_task" not in tarea and "environment_key" not in tarea
            continue
        assert tarea["spark_python_task"]["python_file"] == script and (RAIZ / script).is_file()
        extra = PARAMETROS_EXTRA.get(tarea["task_key"], [])
        assert tarea["spark_python_task"]["parameters"] == ["--schema", "${var.schema}", *extra]
        assert tarea["environment_key"] in entornos


def test_las_detenciones_a_proposito_no_se_reintentan(tareas):
    # serverless reintenta una tarea que falla: el freno y un drift extremo se repetirían, y monitoreo_datos
    # escribiría sus revisiones dos veces
    sin_reintento = {k for k, t in tareas.items() if t.get("disable_auto_optimization")}
    assert sin_reintento == {"ingest", "monitoreo_datos"}


def test_los_parametros_van_vacios_en_cada_corrida_normal(job):
    # existen para liberar un origen o forzar un reentrenamiento a mano; en la configuración siempre van vacíos
    assert job["parameters"] == [{"name": "aceptar_drift", "default": ""}, {"name": "reentrenar", "default": ""}]


def test_cada_target_fija_su_schema(bundle):
    assert "default" not in bundle["variables"]["schema"]
    targets = bundle["targets"]
    assert targets["dev"]["mode"] == "development" and targets["dev"]["default"] is True
    assert targets["dev"]["variables"] == {"schema": "tesis_dev"}
    assert targets["prod"]["mode"] == "production"
    assert targets["prod"]["variables"] == {"schema": "tesis", "programacion": "UNPAUSED"}


def test_programacion_semanal_solo_en_prod_y_una_corrida_a_la_vez(bundle, job):
    assert job["schedule"] == {
        "quartz_cron_expression": "0 0 9 ? * MON",
        "timezone_id": "America/Bogota",
        "pause_status": "${var.programacion}",
    }
    assert bundle["variables"]["programacion"]["default"] == "PAUSED"  # dev no la cambia
    assert job["max_concurrent_runs"] == 1


def test_sin_host_ni_token(bundle):
    for workspace in [bundle.get("workspace", {})] + [t.get("workspace", {}) for t in bundle["targets"].values()]:
        assert not {"host", "token", "profile"} & set(workspace)
    texto = (RAIZ / "databricks.yml").read_text(encoding="utf-8")
    assert "dapi" not in texto and "@" not in texto
