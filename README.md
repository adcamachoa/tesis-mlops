# tesis-mlops

Pipeline MLOps de la tesis de maestría para pronosticar la demanda semanal de una marca de moda infantil que vende en e-commerce y marketplaces en México. Los datos y los jobs viven en Databricks Free Edition, el CI/CD en GitHub Actions y la página de resultados en GitHub Pages. El dataset está anonimizado y no se versiona.

La estructura del repositorio, los datos y las decisiones de modelado están en [CLAUDE.md](CLAUDE.md); el plan por fases, en [PLAN.md](PLAN.md).

## Requisitos

- Docker Desktop corriendo. Lint y pruebas corren dentro de Docker, así que no hace falta Python local.
- GNU Make. En Windows: `winget install ezwinports.make`.

## Comandos

```
make build   # construye la imagen tesis-mlops con requirements.txt y requirements-dev.txt
make lint    # ruff check sobre el repo
make test    # pytest
```

`lint` y `test` reconstruyen la imagen si cambiaron los requirements y montan el repo en `/app`, así que los cambios de código no requieren rebuild.

Si la red inspecciona el tráfico HTTPS, pon el certificado en `.certs/ca_bundle.pem` (ignorado por git). El Makefile lo pasa como secreto de BuildKit solo si existe, pip lo usa durante la instalación y no queda guardado en la imagen. En CI no se usa.
