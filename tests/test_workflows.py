"""Pruebas de los workflows de GitHub. El repo es público: ninguno usa pull_request_target, las acciones van fijadas
por commit, los secretos solo llegan al env de los pasos que los usan y el usuario de Databricks se oculta antes de
cualquier otro comando de Databricks. El CD solo despliega en prod desde main, nunca toca el schema de dev y solo corre
el job o el export si se pide a mano, sin dejar en el log la salida de las tareas. Pages publica solo los datos de prod,
después de un CD lanzado a mano o cada lunes, los valida sin secretos y solo su job publicar puede escribir en Pages."""
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
TODOS = ["ci.yml", "cd.yml", "pages.yml"]


def leer(nombre: str) -> dict:
    return yaml.safe_load((WORKFLOWS / nombre).read_text(encoding="utf-8"))


def disparadores(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))  # YAML 1.1 lee la clave on como True


def pasos(workflow: dict) -> list[dict]:
    return [paso for job in workflow["jobs"].values() for paso in job["steps"]]


def posiciones(pasos: list[dict], condicion) -> list[int]:
    return [i for i, paso in enumerate(pasos) if condicion(paso)]


@pytest.fixture(scope="module")
def cd() -> dict:
    return leer("cd.yml")


@pytest.fixture(scope="module")
def pages() -> dict:
    return leer("pages.yml")


@pytest.mark.parametrize("nombre", TODOS)
def test_disparadores_permisos_y_acciones_fijadas(nombre):
    workflow = leer(nombre)
    assert "pull_request_target" not in disparadores(workflow)
    assert workflow["permissions"] == {"contents": "read"}
    for paso in pasos(workflow):
        if "uses" in paso:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", paso["uses"]), paso["uses"]


@pytest.mark.parametrize("nombre", TODOS)
def test_secretos_solo_en_el_env_de_los_pasos(nombre):
    workflow = leer(nombre)
    assert "secrets." not in str(workflow.get("env", {}))
    for job in workflow["jobs"].values():
        assert "secrets." not in str(job.get("env", {}))
        for paso in job["steps"]:
            assert "secrets." not in paso.get("run", "") + str(paso.get("with", {}))


@pytest.mark.parametrize("nombre", TODOS)
def test_el_usuario_se_oculta_antes_de_databricks(nombre):
    for job in leer(nombre)["jobs"].values():
        comandos = [paso.get("run", "") for paso in job["steps"]]
        de_databricks = [i for i, comando in enumerate(comandos) if re.search(r"\bdatabricks ", comando)]
        if de_databricks:  # el primero que llama a Databricks es el que oculta al usuario
            assert "::add-mask::" in comandos[de_databricks[0]]


def test_cd_despues_del_merge_o_a_mano_y_solo_desde_main(cd):
    on = disparadores(cd)
    assert set(on) == {"push", "workflow_dispatch"}
    assert on["push"]["branches"] == ["main"]
    assert {"src/**", "databricks.yml", "requirements.txt"} <= set(on["push"]["paths"])
    # una sola lista, así que no se pueden pedir dos cosas a la vez
    assert on["workflow_dispatch"]["inputs"] == {
        "accion": {
            "description": on["workflow_dispatch"]["inputs"]["accion"]["description"],
            "type": "choice",
            "options": ["desplegar", "correr", "reentrenar", "solo_export"],
            "default": "desplegar",
        }
    }
    for job in cd["jobs"].values():
        assert job["if"] == "github.ref == 'refs/heads/main'"
        assert job["environment"] == "prod"
    assert cd["concurrency"]["cancel-in-progress"] is False


def test_cd_solo_toca_prod(cd):
    texto = (WORKFLOWS / "cd.yml").read_text(encoding="utf-8")
    assert "tesis_dev" not in texto and "-t dev" not in texto
    comandos = [paso["run"] for paso in pasos(cd) if "databricks bundle" in paso.get("run", "")]
    assert len(comandos) == 5  # validate, deploy y los tres run
    for comando in comandos:
        assert re.match(r"databricks bundle (validate|deploy|run) -t prod\b", comando)
        assert "--var" not in comando  # el schema sale del target prod: tesis


def test_cd_corre_el_job_o_el_export_solo_a_mano_y_sin_su_salida(cd):
    corridas = {paso["if"]: paso["run"].strip() for paso in pasos(cd) if "databricks bundle run" in paso.get("run", "")}
    # al terminar, la CLI imprime en stdout lo que imprimió cada tarea, en piezas
    assert corridas == {
        "inputs.accion == 'correr'": "databricks bundle run -t prod demanda_pipeline > /dev/null",
        "inputs.accion == 'reentrenar'":
            "databricks bundle run -t prod demanda_pipeline --params reentrenar=si > /dev/null",
        "inputs.accion == 'solo_export'": "databricks bundle run -t prod demanda_pipeline --only export > /dev/null",
    }


def test_pages_despues_de_un_cd_a_mano_en_main_cada_lunes_o_a_mano(pages):
    on = disparadores(pages)
    assert set(on) == {"workflow_run", "schedule", "workflow_dispatch"}
    assert on["workflow_run"] == {"workflows": ["CD"], "types": ["completed"], "branches": ["main"]}
    # dos horas después de la corrida programada del job, que GitHub no ve: lunes 9:00 en Bogotá (UTC−5)
    assert on["schedule"] == [{"cron": "0 16 * * 1"}]
    # los merges solo despliegan: no publican
    assert " ".join(pages["jobs"]["descargar"]["if"].split()) == (
        "(github.event_name == 'workflow_run'"
        " && github.event.workflow_run.conclusion == 'success'"
        " && github.event.workflow_run.event == 'workflow_dispatch'"
        " && github.event.workflow_run.head_branch == 'main'"
        " && github.event.workflow_run.head_repository.full_name == github.repository)"
        " || (github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main')"
        " || github.event_name == 'schedule'"
    )
    assert pages["jobs"]["publicar"]["needs"] == "descargar"
    assert pages["concurrency"]["cancel-in-progress"] is False


def test_pages_solo_descargar_tiene_secretos_y_solo_publicar_escribe(pages):
    descargar, publicar = pages["jobs"]["descargar"], pages["jobs"]["publicar"]
    assert set(pages["jobs"]) == {"descargar", "publicar"}
    assert "secrets." not in str(publicar)
    assert "permissions" not in descargar  # se queda con contents: read
    assert publicar["permissions"] == {"pages": "write", "id-token": "write"}
    assert publicar["environment"]["name"] == "github-pages"
    assert [paso["uses"].split("@")[0] for paso in publicar["steps"]] == ["actions/deploy-pages"]


def test_pages_descarga_solo_prod_y_valida_sin_secretos_antes_de_subir(pages):
    texto = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert "tesis_dev" not in texto and "-t dev" not in texto
    lista = pages["jobs"]["descargar"]["steps"]
    descarga = posiciones(lista, lambda p: "databricks fs cp" in p.get("run", ""))
    assert len(descarga) == 1
    assert lista[descarga[0]]["run"] == "databricks fs cp -r dbfs:/Volumes/workspace/tesis/raw/output/datos sitio/datos"
    con_secretos = posiciones(lista, lambda p: "secrets." in str(p))
    python = posiciones(lista, lambda p: "setup-python" in p.get("uses", "") or "pip install" in p.get("run", ""))
    validar = posiciones(lista, lambda p: "common.results_schema" in p.get("run", ""))
    subir = posiciones(lista, lambda p: "upload-pages-artifact" in p.get("uses", ""))
    # nada instalado de PyPI corre mientras se usan los secretos
    assert max(con_secretos) == descarga[0] < min(python)
    assert len(validar) == 1 and min(python) < validar[0] < subir[0]
    assert lista[validar[0]]["run"] == "python -m common.results_schema sitio/datos"
    assert lista[subir[0]]["with"] == {"path": "sitio"}  # datos/ en la raíz del sitio
