# Lint, pruebas y build corren dentro de Docker: no hace falta Python local.
IMAGEN := tesis-mlops
comma := ,

# Certificado opcional para la inspección HTTPS del antivirus: solo se pasa si existe (en local sí, en CI no).
SECRETO_CA := $(if $(wildcard .certs/ca_bundle.pem),--secret id=ca$(comma)src=.certs/ca_bundle.pem)

# Si make usa el sh de Git en Windows, evita que convierta /app en una ruta de Windows.
export MSYS_NO_PATHCONV := 1

EJECUTAR := docker run --rm -v "$(CURDIR):/app" $(IMAGEN)

.PHONY: build lint test

build:
	docker build $(SECRETO_CA) -t $(IMAGEN) .

lint: build
	$(EJECUTAR) ruff check .

test: build
	$(EJECUTAR) pytest
