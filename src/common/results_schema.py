"""Esquema de results.json: lo escribe src/export_results.py y de él se arma la página.

`ESQUEMA` es un JSON Schema (draft 2020-12). `validar` además revisa lo que JSON Schema no alcanza: que no haya números
no finitos, que cada serie traiga tantas semanas reales como `semanas.real`, que haya un solo total y que la venta real
del total promedie 100. La página es pública: la venta y el pronóstico van como índice, nunca en piezas.
"""
import math

from jsonschema import Draft202012Validator

from common.features import HORIZONTE
from common.metrics import METRICAS, VENTANAS_DIAS

VERSION = 1
BASE_INDICE = 100
ESCALA = "índice: 100 = venta semanal promedio del total en las semanas de venta real mostradas"
TOLERANCIA_BASE = 0.1  # el promedio del total se calcula con los valores ya redondeados

_TEXTO = {"type": "string", "minLength": 1}
_FECHA = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
_FECHA_HORA = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"}
_INDICE = {"type": "number", "minimum": 0}
_METRICA = {"type": ["number", "null"]}  # nula si no se puede calcular, por ejemplo un MAPE sin celdas con venta


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
                    }
                )
            ),
        }
    ),
}


class ErrorEsquema(ValueError):
    """results.json no cumple el esquema; `problemas` trae cada uno como texto."""

    def __init__(self, problemas: list[str]):
        self.problemas = problemas
        detalle = "\n".join(f"- {p}" for p in problemas)
        super().__init__(f"results.json no cumple el esquema ({len(problemas)} problema(s)):\n{detalle}")


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
    problemas = [
        f"{'/'.join(map(str, e.absolute_path)) or '(raíz)'}: {e.message}"
        for e in Draft202012Validator(ESQUEMA).iter_errors(resultados)
    ] + _no_finitos(resultados)
    if problemas:  # las revisiones de abajo suponen la estructura del esquema
        raise ErrorEsquema(problemas)

    semanas = len(resultados["semanas"]["real"])
    for serie in resultados["series"]:
        if len(serie["real"]) != semanas:
            problemas.append(f"la serie {serie['nombre']} trae {len(serie['real'])} semanas reales y no {semanas}")
    totales = [s for s in resultados["series"] if s["nivel"] == "total"]
    if len(totales) != 1:
        problemas.append(f"tiene que haber una sola serie total y hay {len(totales)}")
    else:
        promedio = sum(totales[0]["real"]) / semanas
        if abs(promedio - BASE_INDICE) > TOLERANCIA_BASE:
            problemas.append(
                f"la venta real del total promedia {promedio:,.2f} y no {BASE_INDICE}: tiene que ir como índice"
            )
    if problemas:
        raise ErrorEsquema(problemas)
