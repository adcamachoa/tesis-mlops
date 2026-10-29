# Lint, pruebas y build corren dentro de Docker: no hace falta Python local.
IMAGEN := tesis-mlops
comma := ,

# Certificado opcional para la inspección HTTPS del antivirus: solo se pasa si existe (en local sí, en CI no).
SECRETO_CA := $(if $(wildcard .certs/ca_bundle.pem),--secret id=ca$(comma)src=.certs/ca_bundle.pem)

# Si make usa el sh de Git en Windows, evita que convierta /app en una ruta de Windows.
export MSYS_NO_PATHCONV := 1

EJECUTAR := docker run --rm -v "$(CURDIR):/app" $(IMAGEN)
# los scripts de scripts/ importan de src/, como los del job en Databricks
EJECUTAR_SCRIPT := docker run --rm -e PYTHONPATH=src -v "$(CURDIR):/app" $(IMAGEN)

.PHONY: build lint test calibrar-drift simular-monitoreo

build:
	docker build $(SECRETO_CA) -t $(IMAGEN) .

lint: build
	$(EJECUTAR) ruff check .

test: build
	$(EJECUTAR) pytest

# monitoreo con los parquet de data/, sin Databricks (ver scripts/simular_monitoreo.py)
calibrar-drift: build
	$(EJECUTAR_SCRIPT) python scripts/simular_monitoreo.py calibrar

simular-monitoreo: build
	$(EJECUTAR_SCRIPT) python scripts/simular_monitoreo.py demo
