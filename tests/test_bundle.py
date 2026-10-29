"""Pruebas de databricks.yml: el job encadena las siete tareas sobre el schema de cada target, y el bundle no declara
nada más que el job, para que deploy y destroy nunca toquen tablas, volúmenes ni el modelo."""
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
TAREAS = {
    "ingest": "src/ingest.py",
    "features": "src/features.py",
    "monitoreo_datos": "src/monitoreo_datos.py",
    "train": "src/train.py",
    "predict": "src/predict.py",
    "export": "src/export_results.py",
    "monitoreo_desempeno": "src/monitoreo_desempeno.py",
}
ACEPTAR = ["--aceptar-drift", "{{job.parameters.aceptar_drift}}"]  # solo en monitoreo_datos


@pytest.fixture(scope="module")
def bundle() -> dict:
    return yaml.safe_load((RAIZ / "databricks.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(bundle) -> dict:
    return bundle["resources"]["jobs"]["demanda_pipeline"]


def test_solo_declara_el_job(bundle):
    assert list(bundle["resources"]) == ["jobs"]
    assert list(bundle["resources"]["jobs"]) == ["demanda_pipeline"]
    for target in bundle["targets"].values():
        assert "resources" not in target


def test_las_tareas_van_encadenadas(job):
    tareas = job["tasks"]
    assert [t["task_key"] for t in tareas] == list(TAREAS)
    assert "depends_on" not in tareas[0]
    for anterior, tarea in zip(tareas, tareas[1:], strict=False):
        assert tarea["depends_on"] == [{"task_key": anterior["task_key"]}]


def test_cada_tarea_corre_su_script_sobre_el_schema_del_target(job):
    entornos = {e["environment_key"] for e in job["environments"]}
    for tarea in job["tasks"]:
        script = tarea["spark_python_task"]["python_file"]
        assert script == TAREAS[tarea["task_key"]] and (RAIZ / script).is_file()
        extra = ACEPTAR if tarea["task_key"] == "monitoreo_datos" else []
        assert tarea["spark_python_task"]["parameters"] == ["--schema", "${var.schema}", *extra]
        assert tarea["environment_key"] in entornos


def test_el_drift_se_revisa_antes_de_entrenar_y_nadie_lo_deja_aceptado(job):
    orden = [t["task_key"] for t in job["tasks"]]
    assert orden.index("monitoreo_datos") < orden.index("train") < orden.index("predict")
    assert orden[-1] == "monitoreo_desempeno"
    # el parámetro existe para liberar un origen a mano; en la configuración siempre va vacío
    assert job["parameters"] == [{"name": "aceptar_drift", "default": ""}]


def test_cada_target_fija_su_schema(bundle):
    assert "default" not in bundle["variables"]["schema"]
    targets = bundle["targets"]
    assert targets["dev"]["mode"] == "development" and targets["dev"]["default"] is True
    assert targets["dev"]["variables"] == {"schema": "tesis_dev"}
    assert targets["prod"]["mode"] == "production"
    assert targets["prod"]["variables"] == {"schema": "tesis"}


def test_programacion_en_pausa_y_una_corrida_a_la_vez(job):
    assert job["schedule"]["pause_status"] == "PAUSED"
    assert job["max_concurrent_runs"] == 1


def test_sin_host_ni_token(bundle):
    for workspace in [bundle.get("workspace", {})] + [t.get("workspace", {}) for t in bundle["targets"].values()]:
        assert not {"host", "token", "profile"} & set(workspace)
    texto = (RAIZ / "databricks.yml").read_text(encoding="utf-8")
    assert "dapi" not in texto and "@" not in texto
