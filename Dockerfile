# syntax=docker/dockerfile:1
# Entorno para lint y pruebas. El código no se copia: el Makefile monta el repo en /app al correr.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Java para PySpark (pruebas locales) y libgomp para LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./

# Certificado opcional como secreto de BuildKit: solo se pasa en local, por la inspección HTTPS del antivirus.
# Se monta únicamente durante este RUN, así que no queda en la imagen.
RUN --mount=type=secret,id=ca \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
