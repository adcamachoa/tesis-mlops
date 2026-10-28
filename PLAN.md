# Plan de ejecución con Claude Code

Plan para construir el MLOps de la tesis en Databricks Free Edition, con CI/CD en GitHub Actions y front en GitHub Pages, usando Claude Code en VS Code.

## Antes de empezar: decisión sobre GCP

El requisito original de la tesis dice que debe desarrollarse en GCP. Databricks Free Edition no corre en tu proyecto de GCP, así que confirma con tu director si esto es aceptable.

- Si es aceptable: sigue este plan tal cual (costo 0 USD).
- Si GCP es obligatorio: usa el free trial de Databricks en GCP (14 días de créditos) con el parquet en GCS. Cambian solo la Fase 0 (suscripción por Marketplace y bucket) y la Fase 2 (external location en lugar de Volume administrado). Actívalo pocos días antes de las pruebas para que el trial cubra también la sustentación, y cancélalo al terminar.

## Cómo usar este plan con Claude Code

1. Abre la carpeta del repo en VS Code e inicia Claude Code desde la raíz. `CLAUDE.md` se carga solo como contexto.
2. Haz una fase por sesión. Al terminar una fase, haz commit y empieza una conversación nueva para la siguiente.
3. Pega el prompt de la fase tal como está. Usa el modo plan para revisar qué va a hacer antes de que edite archivos.
4. Revisa cada diff antes de aceptarlo. Tú haces los pasos marcados como "Manual".
5. Nunca le pegues tokens en el chat.

Tiempo estimado total: 3 a 5 días de trabajo, dependiendo de cuánto reutilices del repo de experimentación.

---

## Fase 0. Cuentas e instalación (Manual)

- [x] Crear cuenta en Databricks Free Edition
- [x] Crear repo en GitHub `tesis-mlops` (público, para tener GitHub Pages y Actions gratis sin límites), con la cuenta personal
- [x] Instalar: Docker Desktop, Git, Databricks CLI (versión nueva, no `databricks-cli` de pip), GitHub CLI. Python 3.11 local no hace falta: lint y pruebas corren dentro de Docker
- [x] Generar token en Databricks: Settings → Developer → Access tokens
- [x] Configurar perfil local: `databricks configure` (host y token) y verificar con `databricks current-user me`
- [x] Repo local con `CLAUDE.md`, `PLAN.md`, `ingest.py` y el dataset anonimizado en `data/` (no se versiona), conectado a `origin`
- [x] Editar la sección "Datos" de `CLAUDE.md` con las columnas reales de los parquet

Terminado cuando: `databricks current-user me` responde con tu usuario y el repo está clonado con los dos .md.

---

## Fase 1. Esqueleto del repositorio

Prompt para Claude Code:

```
Lee CLAUDE.md y PLAN.md. Ejecuta la Fase 1: crea el esqueleto del repositorio
según la estructura de CLAUDE.md. Incluye:
- requirements.txt (pandas, numpy, scikit-learn, lightgbm, mlflow, pyarrow)
  y requirements-dev.txt (pytest, ruff, pyspark para pruebas locales)
- Dockerfile basado en python:3.11-slim que instale ambos requirements
- Makefile con targets lint, test, build (todos corriendo dentro de Docker)
- configuración de ruff y pytest
- .gitignore para Python, Databricks (.databricks/) y archivos de datos (*.parquet)
- mueve ingest.py a src/ y crea src/common/__init__.py
- tests/ con una prueba trivial para validar que pytest corre
- README.md corto con el propósito del repo y cómo correr lint y tests
Propón el plan primero y espera mi aprobación.
```

Terminado cuando:
- [x] `make lint` y `make test` pasan dentro de Docker
- [x] Commit `fase 1: esqueleto del repositorio`

---

## Fase 2. Datos en el Volume e ingesta

Manual:
- [x] En el SQL editor de Databricks: `CREATE SCHEMA IF NOT EXISTS workspace.tesis;` y `CREATE VOLUME IF NOT EXISTS workspace.tesis.raw;`
- [x] Repetir para `workspace.tesis_dev`
- [x] Subir los cinco parquet de `data/` desde Catalog → workspace → tesis → raw → Upload to this volume (y también en tesis_dev), o desde la terminal:
  `foreach ($f in Get-ChildItem data\*.parquet) { databricks fs cp $f.FullName "dbfs:/Volumes/workspace/tesis_dev/raw/$($f.Name)" --overwrite }`
- [x] Muestra pequeña local en `tests/data/muestra.parquet` (164 filas de `ventas_diarias`, ya anonimizada)

Prompt para Claude Code:

```
Ejecuta la Fase 2 de PLAN.md. Refactoriza src/ingest.py, que carga los cinco
parquet del Volume como tablas bronze (ver "Datos" en CLAUDE.md), para que la
lógica de validación viva en src/common/validation.py como funciones puras sobre
pandas: columnas obligatorias y tipos de cada tabla, nulos en columnas clave,
fechas válidas, unidades y stock no negativos, duplicados por fecha+sku (ventas,
stock y quiebres) y por sku (productos), SKU de ventas o stock que no existan en
productos, y fecha_inicio <= fecha_fin en el calendario. ingest.py valida en
pandas una muestra de cada tabla y luego escribe las tablas Delta completas con
Spark; si una validación falla, el job se detiene con un mensaje claro.
Crea pruebas en tests/test_validation.py usando tests/data/muestra.parquet
y casos sintéticos que fallen a propósito. Propón el plan primero.
```

Terminado cuando:
- [x] Pruebas de validación pasan en local
- [x] Commit `fase 2: ingesta y validación de datos`

---

## Fase 3. Features

Prompt para Claude Code:

```
Ejecuta la Fase 3 de PLAN.md. La unidad de pronóstico es la serie semanal
submarca × departamento × categoría × tipo de producto (ver "Decisiones de
modelado" en CLAUDE.md). Crea src/common/features.py con funciones puras en pandas:
- panel serie × semana (lunes a domingo) desde 2024-01-01, con venta 0 en las
  semanas sin venta, y la oferta de la semana: stock al cierre, SKU y
  estilos-color con stock y disponibilidad ponderada por la venta previa de cada SKU
- surtido: estilos-color con venta en las últimas 8 semanas y lanzamientos recientes
- base directa: una fila por serie × semana de origen × horizonte h = 1..13, con
  rezagos y promedios móviles (4, 8, 13, 26 y 52 semanas) conocidos al cierre del
  origen; calendario de la semana objetivo (días de cada evento, quincenas, semana
  del año, mes, semanas al próximo Hot Sale y Buen Fin); la misma semana del año
  anterior; y la jerarquía como categóricas
- y = venta de la semana origen + h (nula si todavía no ocurrió)
Crea src/features.py que lea las tablas bronze, agregue en Spark el stock diario
a semana × SKU antes de pasar a pandas, aplique las funciones y escriba las
tablas panel_semanal y features. Agrega pruebas que verifiquen especialmente que
ninguna variable usa información posterior al origen (por ejemplo, truncando los
datos en el origen y comprobando que las variables no cambian). Propón el plan primero.
```

Terminado cuando:
- [x] Prueba explícita de "sin fuga de información" pasa
- [x] Commit `fase 3: features`

---

## Fase 4. Entrenamiento, evaluación de algoritmos y registro

Esta fase es el corazón de la tesis (evaluación de algoritmos).

Prompt para Claude Code:

```
Ejecuta la Fase 4 de PLAN.md. Crea:
1. src/common/metrics.py con WAPE, MAPE (solo nodos con venta), MAE, RMSE y
   sesgo, y el formato estándar de comparación de CLAUDE.md: WAPE y MAPE del
   mes 1, el mes 2 y el total (ventanas de 31-30-30 días desde el lunes siguiente
   al origen, a nivel tipo de producto) y el sesgo total. Con pruebas.
2. src/common/backtest.py con validación temporal tipo rolling origin: orígenes
   cada 4 semanas, cada uno a 13 semanas, entrenando solo con semanas objetivo
   hasta el origen. Con pruebas que demuestren que no entra ninguna fecha futura.
   El número de orígenes y de semillas es configurable, para cuidar la cuota de
   cómputo de Free Edition.
3. src/train.py que:
   - lea la tabla features
   - compare al menos: naive estacional (misma semana del año anterior), media
     móvil de 8 semanas, RandomForest, LightGBM Tweedie y la mezcla de las dos
     configuraciones de LightGBM de CLAUDE.md (deja fácil agregar más modelos)
   - registre cada algoritmo como un run de MLflow con parámetros, las métricas
     del formato estándar más MAE y RMSE, la versión Delta de la tabla de entrada
     (task value bronze_version) y una gráfica real vs pronóstico del total como
     artefacto
   - al medir el error, sume la venta real de las series que no están en la base
     para ese origen (series nuevas o sin venta en las 52 semanas previas) con
     pronóstico 0, tomándola de panel_semanal: la tabla features solo trae las
     series activas en el origen, y sin esa venta el error del total queda corto
   - escriba un resumen comparativo en la tabla metricas_modelos
   - registre el mejor modelo (menor WAPE tipo de producto × mes, con las tres
     ventanas juntas) en Unity Catalog como workspace.<schema>.demanda_forecast
     y le asigne el alias champion solo si mejora al champion actual
Usa mlflow.set_registry_uri("databricks-uc"). Propón el plan primero.
```

Terminado cuando:
- [x] Métricas y backtest con pruebas
- [x] En MLflow aparece un run por algoritmo con sus métricas
- [x] El modelo existe en Unity Catalog con alias `champion`
- [x] Commit `fase 4: entrenamiento y evaluación de algoritmos`

Resultado (28-oct-2026):
- **Réplica de la referencia, en local** con los mismos 8 orígenes (11-ago-2025 a 23-feb-2026) y la misma métrica:

  | Modelo | WAPE M1 | WAPE M2 | WAPE Total | MAPE M1 | MAPE M2 | MAPE Total | Sesgo total |
  |---|---|---|---|---|---|---|---|
  | Naive estacional | 76,4 % | 78,3 % | 70,5 % | 231,3 % | 234,7 % | 362,8 % | +7,3 % |
  | Producción (mezcla) | 33,8 % | 44,0 % | 36,1 % | 80,3 % | 112,4 % | 112,6 % | +1,8 % |
  | Configuración 2 sola | 31,9 % | 42,6 % | 34,6 % | 66,3 % | 95,6 % | 80,4 % | −3,5 % |

  - El naive estacional da idéntico a la referencia, así que la métrica se calcula igual.
  - La mezcla queda en 36,1 % contra 35,8 % de la referencia, y el sesgo en +1,8 % contra +4,2 %. Como el naive coincide, la diferencia viene del modelo o de las variables, no de la métrica.
- **Corrida en dev** con los 8 orígenes más recientes (29-dic-2025 a 13-jul-2026): la configuración 2 (`lightgbm_tweedie_211`) tiene WAPE tipo × mes de 43,8 % contra 44,6 % de la mezcla, y sesgo total de +2,7 % contra +6,2 %. Quedó como champion en la versión 2 de `workspace.tesis_dev.demanda_forecast`. La versión 1 es la prueba de humo.
- **Por qué el champion no es la mezcla.**
  - En la experimentación se eligió la mezcla porque la auditoría vio que la configuración 2 ganaba pronosticando casi cero en las series chicas.
  - Aquí se respeta la regla fijada antes de ver los resultados: gana el menor WAPE tipo × mes.
  - La configuración 2 gana en las dos ventanas, y en cada una le gana a la mezcla en los 8 de 8 orígenes (WAPE tipo × mes de cada origen). En la ventana reciente, el origen del 20-abr-2026 es casi un empate (44,060 % contra 44,062 %); sin él, serían 7 de 8, que igual cumplen la regla.
  - Cada corrida vuelve a comparar todos los modelos, así que la mezcla recupera el alias si empieza a ganar.
- **Regla de reemplazo:**
  - un algoritmo distinto le quita el alias al del champion solo si mejora su WAPE tipo × mes y le gana en al menos 6 de los 8 orígenes, para que el champion no cambie por ruido;
  - si el retador no lo logra, se reentrena el algoritmo del champion.

---

## Fase 5. Predicción y export de resultados

Prompt para Claude Code:

```
Ejecuta la Fase 5 de PLAN.md. Crea:
1. src/predict.py que cargue models:/workspace.<schema>.demanda_forecast@champion,
   pronostique las 13 semanas siguientes a la última semana completa para cada
   serie y escriba la tabla predicciones con fecha de corrida, versión del modelo,
   la venta semanal y las ventanas mensuales de 31-30-30 días.
2. src/export_results.py que genere results.json con: fecha de la última corrida,
   versión y algoritmo del champion, tabla comparativa de metricas_modelos en el
   formato estándar, y la venta real reciente y el pronóstico del total, de cada
   submarca y de las 20 series de más venta.
   Lo escribe en /Volumes/workspace/<schema>/raw/output/results.json.
   Define el esquema del JSON en src/common/results_schema.py y prueba que el
   archivo generado lo cumple.
Propón el plan primero.
```

Terminado cuando:
- [x] `results.json` cumple el esquema en las pruebas
- [x] Commit `fase 5: predicción y export`

Resultado (28-oct-2026):
- **Corrida en dev.**
  - El champion (versión 2, `lightgbm_tweedie_211`, entrenado hasta el 12-oct-2026) pronosticó desde ese origen las 464 series con venta en las últimas 52 semanas.
  - Dejó 7.424 filas en `predicciones`: 13 semanas y 3 ventanas por serie. Las ventanas van del 19-oct al 18-nov, del 19-nov al 18-dic y del 19-dic al 17-ene.
- **`results.json`** en `/Volumes/workspace/tesis_dev/raw/output/` cumple el esquema.
  - Trae el total, las 3 submarcas y las 20 series de más venta.
  - Las métricas son las de la última corrida de train, iguales a las de la fase 4.
- **Lectura del pronóstico,** en índice (100 = la semana promedio del último año).
  - Arranca al nivel de las últimas semanas (98 contra cerca de 100) y repite la forma del año anterior, con el pico del Buen Fin en la semana del 9-nov.
  - El trimestre queda 13 % debajo de las mismas semanas del año anterior. El último año va −9,6 % y el último semestre −12,1 %, pero las últimas 4 semanas van +8,7 %.
  - La diferencia está en los picos: el Buen Fin queda en 275 contra 332, y diciembre también queda abajo. Se podrá medir cuando llegue la venta real, porque la corrida queda guardada en `predicciones`.
- **Privacidad.** Con 1 decimal de índice, una pieza vale menos de 0,1, así que la escala no se puede reconstruir. La serie 20 promedia 0,9 por semana, una resolución suficiente para la gráfica.

Pendiente:
- [ ] **Medir en enero de 2027,** cuando cierre la ventana 3 (17-ene-2027), el pronóstico del origen 12-oct-2026 contra la venta real.
  - Quedó 13 % debajo de las mismas semanas del año anterior, aunque las últimas 4 semanas antes del origen iban +8,7 %.
  - Hay que ver si el modelo subpronosticó el Buen Fin y diciembre: WAPE y sesgo por ventana, en el formato estándar.
  - La corrida está en `workspace.tesis_dev.predicciones`, con `fecha_corrida` 2026-10-28 16:28:09, y tiene un respaldo en `workspace.tesis_dev.predicciones_respaldo_20261028` (las mismas 7.424 filas).

---

## Fase 6. Asset Bundle y job

Prompt para Claude Code:

```
Ejecuta la Fase 6 de PLAN.md. Crea databricks.yml con:
- bundle name tesis-mlops
- targets dev (schema tesis_dev, mode development, default) y prod (schema tesis)
- variable schema usada por todas las tareas
- un job demanda_pipeline con tareas encadenadas ingest -> features -> train
  -> predict -> export, cada una como spark_python_task sobre serverless,
  con un environment que instale requirements.txt
- programación semanal en pausa (para no consumir cuota)
No incluyas host ni token en el archivo. Luego dime qué comandos debo correr
para validar y desplegar en dev, pero no los ejecutes.
```

Manual:
- [x] `databricks bundle validate -t dev`
- [x] `databricks bundle deploy -t dev`
- [x] `databricks bundle run -t dev demanda_pipeline` y revisar que todas las tareas terminen en verde

Terminado cuando:
- [x] El job corre completo en dev y deja `results.json` en el Volume
- [x] Commit `fase 6: asset bundle y job`

Resultado (28-oct-2026):
- **Bundle.**
  - `databricks.yml` solo declara el job; las tablas, los volúmenes y el modelo quedan fuera de su alcance. `tests/test_bundle.py` lo verifica.
  - Las tareas corren sobre serverless (`environment_version` 2, Python 3.11) con `requirements.txt`.
  - Tiene una corrida a la vez, un límite de 2 horas y la programación en pausa (lunes a las 7:00, hora de Ciudad de México).
  - `bundle validate -t prod` pasa sin desplegar.
- **Protección de `predicciones`, antes del primer deploy.**
  - Se respaldó la corrida del 2026-10-28 16:28:09 en `workspace.tesis_dev.predicciones_respaldo_20261028`: 7.424 filas, idénticas a las originales.
  - `predicciones` quedó con `delta.appendOnly = true`.
- **Corrida en dev:** 44 minutos, todas las tareas en verde. Train tardó 38 minutos.
  - Las métricas son idénticas a las de la fase 4.
  - `lightgbm_tweedie_211` sigue de champion, reentrenado como versión 3.
  - Predict agregó la corrida 2026-10-28 17:47:13 con la versión 3.
- **Verificación.**
  - El historial de `predicciones` solo muestra la creación, el cambio de propiedad y un `WRITE` en modo `Append`.
  - La corrida de las 16:28:09 sigue con 7.424 filas y coincide fila por fila con el respaldo.
  - La versión 3 pronostica lo mismo que la 2 (las dos suman lo mismo en las 13 semanas): con los mismos datos y semillas, el reentrenamiento es determinista.

---

## Fase 7. CI

Manual:
- [ ] En GitHub: Settings → Secrets and variables → Actions, crear `DATABRICKS_HOST` y `DATABRICKS_TOKEN`
- [ ] Activar protección de la rama main: requerir PR y que pase el check de CI

Prompt para Claude Code:

```
Ejecuta la Fase 7 de PLAN.md. Crea .github/workflows/ci.yml que corra en
cada pull request a main: build de la imagen Docker, make lint, make test y
databricks bundle validate -t prod (instalando el CLI con la acción oficial
databricks/setup-cli y usando los secrets DATABRICKS_HOST y DATABRICKS_TOKEN).
Propón el plan primero.
```

Terminado cuando:
- [ ] Un PR de prueba muestra el check de CI en verde
- [ ] Commit `fase 7: CI`

---

## Fase 8. CD

Prompt para Claude Code:

```
Ejecuta la Fase 8 de PLAN.md. Crea .github/workflows/cd.yml que corra en push
a main: databricks bundle deploy -t prod y luego databricks bundle run -t prod
demanda_pipeline, esperando a que termine y fallando si el job falla.
Agrega también workflow_dispatch para poder lanzarlo a mano.
Propón el plan primero.
```

Manual:
- [ ] Crear el schema `workspace.tesis` con su Volume y parquet si no lo hiciste
- [ ] Hacer merge del PR y verificar que el job corre en prod

Terminado cuando:
- [ ] Merge a main despliega y ejecuta el job en prod automáticamente
- [ ] Commit `fase 8: CD`

---

## Fase 9. Front en GitHub Pages

Manual:
- [ ] En GitHub: Settings → Pages → Source: GitHub Actions

Prompt para Claude Code:

```
Ejecuta la Fase 9 de PLAN.md. Crea:
1. front/template.html: página estática, limpia y responsive, que muestre
   el título de la tesis, fecha de la última corrida, el champion y su versión,
   una tabla comparativa de algoritmos en el formato estándar (WAPE y MAPE del
   mes 1, el mes 2 y el total, y sesgo total; MAE y RMSE como complemento) con el
   mejor resaltado, y un selector (total, submarca o serie) con un gráfico
   Chart.js de real vs pronóstico.
   Chart.js desde cdnjs. Todo el texto en español.
2. front/build.py que lea results.json, valide el esquema e inyecte los datos
   en el template generando dist/index.html. Con pruebas.
3. .github/workflows/pages.yml que se dispare al terminar con éxito el
   workflow de CD (workflow_run) y también con workflow_dispatch: descarga
   results.json del Volume con databricks fs cp, corre build.py y publica dist/
   con las acciones oficiales de GitHub Pages.
Para probar en local, agrega un results.json de ejemplo en tests/data/.
Propón el plan primero.
```

Terminado cuando:
- [ ] `python front/build.py` genera un HTML correcto con el JSON de ejemplo
- [ ] La página publicada en GitHub Pages muestra los resultados reales
- [ ] Commit `fase 9: front`

---

## Fase 10. Ensayo de sustentación (Manual, con ayuda de Claude Code)

- [ ] Hacer un cambio pequeño en una rama (por ejemplo, agregar un algoritmo o ajustar un parámetro) y recorrer el flujo completo: PR → CI verde → merge → CD → job → página actualizada
- [ ] Tomar capturas: checks de CI, historial del job, runs de MLflow, modelo con alias champion, página publicada
- [ ] Grabar un video corto del flujo como plan B
- [ ] Dejar el job ejecutado el día anterior a la sustentación para no depender de la cuota ese día
- [ ] Pedirle a Claude Code: "Genera un diagrama de arquitectura en Mermaid y un resumen técnico del repo para incluir en el documento de la tesis"

## Después de la sustentación

- [ ] Revocar el token de Databricks y borrar los secrets de GitHub
- [ ] Archivar el repo (queda como evidencia) o despublicar GitHub Pages
- [ ] Free Edition no genera cobros; si usaste el trial en GCP, borrar el workspace y cancelar la suscripción en el Marketplace
