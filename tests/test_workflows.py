"""Pruebas de los workflows de GitHub. El repo es público: ninguno usa pull_request_target, las acciones van fijadas
por commit, los secretos solo llegan al env de los pasos que los usan y el usuario de Databricks se oculta antes de
cualquier comando del bundle. El CD solo despliega en prod desde main, nunca toca el schema de dev y solo corre el job
si se pide a mano, sin dejar en el log la salida de las tareas."""
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def leer(nombre: str) -> dict:
    return yaml.safe_load((WORKFLOWS / nombre).read_text(encoding="utf-8"))


def disparadores(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))  # YAML 1.1 lee la clave on como True


def pasos(workflow: dict) -> list[dict]:
    return [paso for job in workflow["jobs"].values() for paso in job["steps"]]


@pytest.fixture(scope="module")
def cd() -> dict:
    return leer("cd.yml")


@pytest.mark.parametrize("nombre", ["ci.yml", "cd.yml"])
def test_disparadores_permisos_y_acciones_fijadas(nombre):
    workflow = leer(nombre)
    assert "pull_request_target" not in disparadores(workflow)
    assert workflow["permissions"] == {"contents": "read"}
    for paso in pasos(workflow):
        if "uses" in paso:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", paso["uses"]), paso["uses"]


@pytest.mark.parametrize("nombre", ["ci.yml", "cd.yml"])
def test_secretos_solo_en_el_env_de_los_pasos(nombre):
    workflow = leer(nombre)
    assert "secrets." not in str(workflow.get("env", {}))
    for job in workflow["jobs"].values():
        assert "secrets." not in str(job.get("env", {}))
        for paso in job["steps"]:
            assert "secrets." not in paso.get("run", "") + str(paso.get("with", {}))


@pytest.mark.parametrize("nombre", ["ci.yml", "cd.yml"])
def test_el_usuario_se_oculta_antes_del_bundle(nombre):
    for job in leer(nombre)["jobs"].values():
        comandos = [paso.get("run", "") for paso in job["steps"]]
        del_bundle = [i for i, comando in enumerate(comandos) if "databricks bundle" in comando]
        if del_bundle:
            mascara = [i for i, comando in enumerate(comandos) if "::add-mask::" in comando]
            assert mascara and mascara[0] < del_bundle[0]


def test_cd_despues_del_merge_o_a_mano_y_solo_desde_main(cd):
    on = disparadores(cd)
    assert set(on) == {"push", "workflow_dispatch"}
    assert on["push"]["branches"] == ["main"]
    assert {"src/**", "databricks.yml", "requirements.txt"} <= set(on["push"]["paths"])
    assert on["workflow_dispatch"]["inputs"]["correr"]["default"] is False
    for job in cd["jobs"].values():
        assert job["if"] == "github.ref == 'refs/heads/main'"
        assert job["environment"] == "prod"
    assert cd["concurrency"]["cancel-in-progress"] is False


def test_cd_solo_toca_prod(cd):
    texto = (WORKFLOWS / "cd.yml").read_text(encoding="utf-8")
    assert "tesis_dev" not in texto and "-t dev" not in texto
    comandos = [paso["run"] for paso in pasos(cd) if "databricks bundle" in paso.get("run", "")]
    assert len(comandos) == 3  # validate, deploy y run
    for comando in comandos:
        assert re.match(r"databricks bundle (validate|deploy|run) -t prod\b", comando)
        assert "--var" not in comando  # el schema sale del target prod: tesis


def test_cd_corre_el_job_solo_a_mano_y_sin_su_salida(cd):
    corridas = [paso for paso in pasos(cd) if "databricks bundle run" in paso.get("run", "")]
    assert len(corridas) == 1
    assert corridas[0]["if"] == "inputs.correr"
    # al terminar, la CLI imprime en stdout lo que imprimió cada tarea, en piezas
    assert corridas[0]["run"].rstrip().endswith("> /dev/null")
