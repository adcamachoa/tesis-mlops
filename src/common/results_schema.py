"""Esquemas de los datos de la página: results.json (un pronóstico) y manifest.json (la lista de pronósticos). Los
escribe src/export_results.py y los lee la página de tesis-mlops-web.

`ESQUEMA` es un JSON Schema (draft 2020-12). `validar` además revisa lo que JSON Schema no alcanza: que no haya números
no finitos, que cada serie traiga tantas semanas reales como `semanas.real`, que haya un solo total y que la venta real
del total promedie 100. También que la venta real del horizonte (`real_horizonte` y `ventanas_real`) traiga valor
exactamente en las semanas y ventanas que cerraron hasta `real_hasta`, y que el pronóstico anterior cubra las mismas
semanas en todas las series. La página es pública: la venta y el pronóstico van como índice, nunca en piezas.

`ESQUEMA_MANIFEST` y `validar_manifest` siguen las reglas de revisarManifest de tesis-mlops-web (docs/js/datos.js): si
la web cambia sus reglas, se cambian aquí también. `validar_manifest` revisa además que cada entrada tenga su
results.json, que no sobre ninguno y que cada uno sea el pronóstico de su entrada. Para validar una carpeta descargada:
    python -m common.results_schema <carpeta>

tesis-mlops-web (herramientas/generar_ejemplo.py) importa validar, VERSION, ESCALA y BASE_INDICE: no se cambian sus
nombres ni la firma de validar.
"""
import argparse
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from common.features import HORIZONTE
from common.metrics import METRICAS, VENTANAS_DIAS

VERSION = 1
VERSION_MANIFEST = 1
MANIFEST = "manifest.json"
BASE_INDICE = 100
ESCALA = "índice: 100 = venta semanal promedio del total en las semanas de venta real mostradas"
TOLERANCIA_BASE = 0.1  # el promedio del total se calcula con los valores ya redondeados

# además de su forma, las fechas tienen que existir (no "2026-13-01"), como en la página
_FORMATOS = FormatChecker(formats=())


@_FORMATOS.checks("fecha", raises=ValueError)
def _fecha_existe(valor) -> bool:
    if isinstance(valor, str):
        date.fromisoformat(valor)
    return True


@_FORMATOS.checks("fecha_hora", raises=ValueError)
def _fecha_hora_existe(valor) -> bool:
    if isinstance(valor, str):
        datetime.fromisoformat(valor)
    return True


_TEXTO = {"type": "string", "minLength": 1}
_FECHA = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "format": "fecha"}
_FECHA_HORA = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", "format": "fecha_hora"}
_INDICE = {"type": "number", "minimum": 0}
_INDICE_O_NULO = {"type": ["number", "null"], "minimum": 0}  # nulo donde todavía no se conoce
_METRICA = {"type": ["number", "null"]}  # nula si no se puede calcular, por ejemplo un MAPE sin celdas con venta
_ID = {"type": "string", "pattern": r"^[a-z0-9-]+$"}  # va en la dirección de la página: …/#2026-10-12
# relativa al manifest, con la expresión de la página: cada parte empieza con letra o número, así que no hay "..", "/"
# inicial ni "https:"
_RUTA = {"type": "string", "pattern": r"^([A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.json$"}


def _objeto(propiedades: dict) -> dict:
    return {"type": "object", "required": list(propiedades), "properties": propiedades, "additionalProperties": False}


def _lista(elemento: dict, largo: int | None = None) -> dict:
    lista = {"type": "array", "items": elemento, "minItems": largo or 1}
    return lista | {"maxItems": largo} if largo else lista


ESQUEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **_objeto(
        {
            "version_esquema": {"const": VERSION},
            "generado": _FECHA_HORA,
            "escala": {"const": ESCALA},
            "real_hasta": _FECHA,  # el último domingo con venta real: llega más allá del origen en las semanas cerradas
            "corrida": _objeto(
                {"fecha_prediccion": _FECHA_HORA, "fecha_entrenamiento": _FECHA_HORA, "origen": _FECHA,
                 "datos_hasta": _FECHA}
            ),
            "champion": _objeto({"modelo": _TEXTO, "version": {"type": "integer", "minimum": 1}}),
            "backtest": _objeto(
                {"origenes": {"type": "integer", "minimum": 1}, "primer_origen": _FECHA, "ultimo_origen": _FECHA}
            ),
            # el formato estándar; el MAE y el RMSE van en la misma escala que el índice
            "metricas": _lista(
                _objeto(
                    {"modelo": _TEXTO, **dict.fromkeys(METRICAS, _METRICA), "es_mejor": {"type": "boolean"},
                     "es_champion": {"type": "boolean"}}
                )
            ),
            "semanas": _objeto({"real": _lista(_FECHA), "pronostico": _lista(_FECHA, HORIZONTE)}),
            "ventanas": _lista(
                _objeto({"numero": {"type": "integer", "minimum": 1}, "desde": _FECHA, "hasta": _FECHA,
                         "dias": {"type": "integer", "minimum": 1}}),
                len(VENTANAS_DIAS),
            ),
            "series": _lista(
                _objeto(
                    {
                        "nivel": {"enum": ["total", "submarca", "serie"]},
                        "nombre": _TEXTO,
                        "real": _lista(_INDICE),
                        "pronostico": _lista(_INDICE, HORIZONTE),
                        "ventanas": _lista(_INDICE, len(VENTANAS_DIAS)),
                        # la venta real de cada semana y ventana pronosticada, nula mientras no cierra
                        "real_horizonte": _lista(_INDICE_O_NULO, HORIZONTE),
                        "ventanas_real": _lista(_INDICE_O_NULO, len(VENTANAS_DIAS)),
                        # en cada semana real, el pronóstico vigente de un origen anterior; nulo si no hubo
                        "pronostico_anterior": _lista(_INDICE_O_NULO),
                    }
                )
            ),
        }
    ),
}

# el formato 1 del README de tesis-mlops-web: una entrada por origen, con su última corrida
ESQUEMA_MANIFEST = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **_objeto(
        {
            "version_manifest": {"const": VERSION_MANIFEST},
            "generado": _FECHA_HORA,
            "pronosticos": _lista(
                _objeto(
                    {"id": _ID, "ruta": _RUTA, "version_esquema": {"type": "integer", "minimum": 1},
                     "origen": _FECHA, "fecha_prediccion": _FECHA_HORA}
                )
            ),
        }
    ),
}


class ErrorEsquema(ValueError):
    """Un archivo no cumple su esquema; `problemas` trae cada uno como texto y `archivo` dice cuál."""

    def __init__(self, problemas: list[str], archivo: str = "results.json"):
        self.problemas = problemas
        detalle = "\n".join(f"- {p}" for p in problemas)
        super().__init__(f"{archivo} no cumple el esquema ({len(problemas)} problema(s)):\n{detalle}")


def _contra_esquema(esquema: dict, valor) -> list[str]:
    validador = Draft202012Validator(esquema, format_checker=_FORMATOS)
    return [f"{'/'.join(map(str, e.absolute_path)) or '(raíz)'}: {e.message}" for e in validador.iter_errors(valor)]


def _no_finitos(valor, ruta: str = "") -> list[str]:
    if isinstance(valor, float) and not math.isfinite(valor):
        return [f"{ruta or '(raíz)'}: {valor} no es un número válido en JSON"]
    if isinstance(valor, dict):
        return [p for clave, v in valor.items() for p in _no_finitos(v, f"{ruta}/{clave}")]
    if isinstance(valor, list):
        return [p for i, v in enumerate(valor) for p in _no_finitos(v, f"{ruta}/{i}")]
    return []


def validar(resultados: dict) -> None:
    """Lanza ErrorEsquema con todos los problemas de `resultados`, ya leído del JSON."""
    problemas = _contra_esquema(ESQUEMA, resultados) + _no_finitos(resultados)
    if problemas:  # las revisiones de abajo suponen la estructura del esquema
        raise ErrorEsquema(problemas)

    semanas = len(resultados["semanas"]["real"])
    real_hasta = date.fromisoformat(resultados["real_hasta"])
    if real_hasta < date.fromisoformat(resultados["corrida"]["datos_hasta"]):
        problemas.append("real_hasta es anterior a datos_hasta")
    # dónde tiene que haber valor: las semanas y ventanas que cerraron hasta real_hasta
    conocidas = {
        "real_horizonte": [date.fromisoformat(s) + timedelta(days=6) <= real_hasta
                           for s in resultados["semanas"]["pronostico"]],
        "ventanas_real": [date.fromisoformat(v["hasta"]) <= real_hasta for v in resultados["ventanas"]],
    }
    cubiertas = None  # las semanas con pronóstico anterior, las mismas en todas las series
    for serie in resultados["series"]:
        nombre = serie["nombre"]
        for campo in ("real", "pronostico_anterior"):
            if len(serie[campo]) != semanas:
                problemas.append(f"la serie {nombre} trae {len(serie[campo])} semanas en {campo} y no {semanas}")
        for campo, esperadas in conocidas.items():
            if [v is not None for v in serie[campo]] != esperadas:
                problemas.append(f"la serie {nombre} no trae {campo} justo en lo que cerró hasta real_hasta")
        mascara = [v is not None for v in serie["pronostico_anterior"]]
        if cubiertas is not None and mascara != cubiertas:
            problemas.append(f"la serie {nombre} trae pronostico_anterior en otras semanas que la primera serie")
        cubiertas = mascara if cubiertas is None else cubiertas
    totales = [s for s in resultados["series"] if s["nivel"] == "total"]
    if len(totales) != 1:
        problemas.append(f"tiene que haber una sola serie total y hay {len(totales)}")
    elif abs(sum(totales[0]["real"]) / semanas - BASE_INDICE) > TOLERANCIA_BASE:
        # sin el promedio: si la venta viniera en piezas, sería la venta semanal del total, y el error puede llegar
        # al log público
        problemas.append(f"la venta real del total no promedia {BASE_INDICE}: tiene que ir como índice")
    if problemas:
        raise ErrorEsquema(problemas)


def validar_manifest(manifest: dict, archivos: dict[str, dict]) -> None:
    """Lanza ErrorEsquema con todos los problemas del manifest y de sus results.json, ya leídos del JSON.

    `archivos` trae cada results.json por su ruta relativa al manifest. Cada entrada tiene que tener su archivo, ningún
    archivo puede quedar fuera del manifest, y cada uno tiene que cumplir `validar` y traer el origen, la fecha de
    predicción y la versión del esquema de su entrada.
    """
    problemas = _contra_esquema(ESQUEMA_MANIFEST, manifest)
    if problemas:  # las revisiones de abajo suponen la estructura del esquema
        raise ErrorEsquema(problemas, MANIFEST)

    entradas = manifest["pronosticos"]
    for campo in ("id", "origen"):
        conteo = Counter(entrada[campo] for entrada in entradas)
        problemas += [f"pronosticos: el {campo} {valor} aparece dos veces" for valor, n in conteo.items() if n > 1]
    rutas = {entrada["ruta"] for entrada in entradas}
    problemas += [f"{ruta}: no está en el manifest" for ruta in sorted(set(archivos) - rutas)]
    for entrada in entradas:
        ruta = entrada["ruta"]
        if ruta not in archivos:
            problemas.append(f"{ruta}: falta el archivo")
            continue
        try:
            validar(archivos[ruta])
        except ErrorEsquema as error:
            problemas += [f"{ruta}: {problema}" for problema in error.problemas]
            continue
        corrida = archivos[ruta]["corrida"]
        trae = (corrida["origen"], corrida["fecha_prediccion"], archivos[ruta]["version_esquema"])
        dice = (entrada["origen"], entrada["fecha_prediccion"], entrada["version_esquema"])
        if trae != dice:
            problemas.append(
                f"{ruta}: no corresponde a su entrada: trae origen {trae[0]}, predicción {trae[1]} y versión "
                f"{trae[2]}, y el manifest dice {dice[0]}, {dice[1]} y {dice[2]}"
            )
    if problemas:
        raise ErrorEsquema(problemas, MANIFEST)


def validar_carpeta(carpeta: Path) -> dict:
    """Valida una carpeta como la que se publica, con manifest.json y sus results.json, y devuelve el manifest.
    Cualquier otro archivo es un problema: no se publica nada que el manifest no liste."""
    archivos, problemas = {}, []
    for ruta in sorted(p for p in carpeta.rglob("*") if p.is_file()):
        nombre = ruta.relative_to(carpeta).as_posix()
        try:
            archivos[nombre] = json.loads(ruta.read_text(encoding="utf-8"))
        except ValueError as error:  # JSON inválido o un archivo que no es texto
            problemas.append(f"{nombre}: no es un JSON válido ({error})")
    if not (carpeta / MANIFEST).is_file():
        problemas.append(f"{MANIFEST}: falta")
    if problemas:
        raise ErrorEsquema(problemas, f"la carpeta {carpeta}")
    manifest = archivos.pop(MANIFEST)
    validar_manifest(manifest, archivos)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Valida una carpeta de datos de la página: el manifest y sus results.")
    parser.add_argument("carpeta", type=Path)
    args = parser.parse_args()
    manifest = validar_carpeta(args.carpeta)
    origenes = sorted(entrada["origen"] for entrada in manifest["pronosticos"])
    # solo fechas y conteos: el log de Pages es público
    print(f"{args.carpeta}: {MANIFEST} y {len(origenes)} results.json válidos; orígenes {', '.join(origenes)}")


if __name__ == "__main__":
    main()
