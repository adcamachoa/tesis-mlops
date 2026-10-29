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
- [x] En GitHub: Settings → Secrets and variables → Actions, crear `DATABRICKS_HOST` y `DATABRICKS_TOKEN`
- [x] Activar protección de la rama main: requerir PR y que pase el check de CI

Prompt para Claude Code:

```
Ejecuta la Fase 7 de PLAN.md. Crea .github/workflows/ci.yml que corra en
cada pull request a main: build de la imagen Docker, make lint, make test y
databricks bundle validate -t prod (instalando el CLI con la acción oficial
databricks/setup-cli y usando los secrets DATABRICKS_HOST y DATABRICKS_TOKEN).
Propón el plan primero.
```

Terminado cuando:
- [x] Un PR de prueba muestra el check de CI en verde
- [x] Commit `fase 7: CI`

Resultado (28-oct-2026):
- **CI.** `.github/workflows/ci.yml` corre en cada PR a main y tiene dos jobs; los dos pasaron en el PR de la fase:
  - `pruebas`, sin secretos: build de la imagen, `make lint` y `make test`. Tardó cerca de 2 minutos. En GitHub la imagen se construye sin el certificado local.
  - `bundle`: `databricks bundle validate -t prod`, con la CLI 1.22.1. Solo valida la configuración: no despliega, no corre el job ni toca tablas.
- **Seguridad.**
  - Solo `pull_request`, nunca `pull_request_target`. Los PR de forks no reciben secretos y el job `bundle` no corre en ellos.
  - El host y el token solo llegan a los dos pasos que los usan. El usuario de Databricks se marca para que el log lo muestre como `***`: el log del PR no tiene el correo, el host ni el token.
  - Las acciones van fijadas por commit, el token del workflow solo lee y los PR de forks requieren aprobación.
- **Protección de main:** un ruleset sin bypass exige PR y los checks `pruebas` y `bundle`, y bloquea borrar la rama y forzar pushes.
- **Token de GitHub Actions.**
  - Es distinto del token local y vence a los 120 días, el **25-feb-2027**.
  - Para renovarlo: generar uno nuevo en Databricks (Settings → Developer → Access tokens), reemplazar el secreto `DATABRICKS_TOKEN` y revocar el anterior.
  - Si vence antes, el job `bundle` falla por autenticación.

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

Ajuste al prompt (28-oct-2026): el merge despliega, pero no corre el job. Correr el job completo gasta unos 45 minutos de cuota, así que solo corre cuando se pide a mano.

Manual:
- [x] Crear el schema `workspace.tesis` con su Volume y parquet si no lo hiciste (fase 2)
- [x] En GitHub, **antes del merge**: Settings → Environments → New environment `prod`
  - Required reviewers: tu usuario, con "Prevent self-review" sin marcar.
  - Deployment branches: solo `main`.
  - Si no existe al hacer el merge, GitHub lo crea sin reglas y despliega sin pedir aprobación.
- [x] Hacer merge del PR y aprobar el despliegue
- [x] Primera corrida en prod: `gh workflow run cd.yml --ref main -f correr=true`, y aprobarla

Terminado cuando:
- [x] El merge a main despliega en prod después de la aprobación, sin correr el job
- [x] La corrida manual termina en verde en prod
- [x] `workspace.tesis_dev.predicciones` sigue igual: mismo historial y la corrida del 2026-10-28 16:28:09 intacta
- [x] Commit `fase 8: CD`

Diseño:
- **Cuándo corre.** `.github/workflows/cd.yml` corre en push a main y con `workflow_dispatch`, nunca en un PR.
  - En push, solo si cambia algo que va al job: `src/`, `databricks.yml`, `requirements.txt` o el propio `cd.yml`.
- **Qué hace.** Es un solo job con el environment `prod`, que corre solo desde main.
  - Siempre hace `bundle validate` y `bundle deploy` con `-t prod`. El despliegue solo sube el código y la definición del job; la programación sigue en pausa.
  - Corre el job (`bundle run -t prod demanda_pipeline`) solo con `workflow_dispatch` y la casilla `correr`. Espera a que termine y falla si falla el job.
  - Una aprobación cubre el despliegue y la corrida.
- **Nunca toca `tesis_dev`.**
  - Todo va con `-t prod`, sin `--var`: el schema sale del target prod, que es `tesis`.
  - El modelo, el experimento (`/Shared/demanda_tesis`), las tablas y `results.json` quedan en `tesis`.
  - El despliegue de prod tiene su propia carpeta en el workspace, separada de la de dev.
- **Un despliegue a la vez.**
  - El que llega espera sin cancelar al que está en curso.
  - `deploy --fail-on-active-runs` falla si el job está corriendo, aunque lo hayan lanzado fuera de GitHub.
  - Así nunca se cambia el código a mitad de una corrida.
- **Log público sin piezas.**
  - Al terminar, `bundle run` imprime en stdout lo que imprimió cada tarea: métricas como MAE y RMSE en piezas, y el pronóstico total.
  - Por eso stdout se descarta. El progreso y el error de una tarea que falle van por stderr y sí quedan en el log.
  - El usuario de Databricks se oculta antes del despliegue, porque `deploy` imprime la ruta con el correo.
- **`predicciones` de solo agregar desde la primera corrida.** `predict.py` activa `delta.appendOnly` si la tabla no la tiene.
  - En prod lo hace al crearla.
  - En `tesis_dev`, que ya la tiene, solo lee la propiedad.
- **Pruebas.** `tests/test_workflows.py` revisa `ci.yml` y `cd.yml`:
  - que no haya `pull_request_target` y que las acciones vayan fijadas por commit;
  - que los secretos estén solo en el env de los pasos, y la máscara del usuario antes del bundle;
  - en el CD: solo main, solo `-t prod`, nada de `tesis_dev`, y la corrida solo con la casilla y sin su salida.

Resultado (28-oct-2026):
- **Merge (PR #2).**
  - El CD esperó la aprobación y desplegó en 1 minuto: creó el job `demanda_pipeline` en prod y subió 25 archivos.
  - El paso que corre el job se saltó, como debía.
- **Primera corrida en prod**, con `workflow_dispatch` y la casilla `correr`.
  - El despliegue no cambió nada (`1 unchanged`), y el job tardó 41 minutos con todas las tareas en verde.
  - Backtest: 8 orígenes, del 29-dic-2025 al 13-jul-2026, los mismos que en dev.
  - `lightgbm_tweedie_211` quedó de champion como versión 1 de `workspace.tesis.demanda_forecast`. Prod no tenía champion, así que el mejor modelo recibió el alias.
  - Pronosticó desde el origen del 12-oct-2026, con datos hasta el 18-oct-2026.

  | Modelo | WAPE M1 | WAPE M2 | WAPE Total | Sesgo total |
  |---|---|---|---|---|
  | `lightgbm_tweedie_211` (champion) | 35,5 % | 45,0 % | 36,4 % | +2,7 % |
  | Producción (mezcla) | 35,9 % | 45,7 % | 37,2 % | +6,2 % |
  | Naive estacional | 90,4 % | 90,7 % | 83,2 % | +15,9 % |

- **Comparación con dev, verificada contra MLflow.** Se compararon los runs de `backtest 2026-10-28 13:16` en `/Shared/demanda_tesis_dev` (versión 2) y `backtest 2026-10-28 23:18` en `/Shared/demanda_tesis` (versión 1).
  - A 1 decimal, los tres modelos de la tabla coinciden en WAPE del mes 1, del mes 2 y total, y en sesgo total. El champion y el orden de los modelos son los mismos.
  - Con todos los decimales, 4 de los 6 modelos dan idéntico en sus 10 métricas: naive estacional, media móvil, random forest y la configuración 1 de LightGBM.
  - La configuración 2 y la mezcla difieren como máximo 0,002 puntos en WAPE y 0,005 puntos en sesgo (2,6615 % contra 2,6571 %). El criterio del champion da 43,8259 % en los dos.
  - Probablemente LightGBM con varios hilos no da resultados idénticos con todos los decimales en máquinas distintas.
- **`predicciones` de prod.**
  - `workspace.tesis.predicciones` se creó con la corrida 2026-10-28 23:22:23.
  - Quedó con `delta.appendOnly = true` desde el principio: la versión 0 es la escritura y la versión 1 activa la propiedad.
- **`results.json`** quedó en `/Volumes/workspace/tesis/raw/output/`.
- **Dev intacto.** `workspace.tesis_dev.predicciones` sigue en la versión 2, con el último cambio a las 17:47:16 UTC y `appendOnly`, igual que antes de la corrida.
- **Log público.**
  - Muestra solo la URL de la corrida, con el host como `***`, y los estados `RUNNING` y `TERMINATED SUCCESS`.
  - No trae salida de las tareas ni piezas, y el usuario aparece como `***`.
- **Para la fase 9.** `pages.yml` se dispara cuando el CD termina con éxito, y eso incluye los merges que solo despliegan. En ese caso la página se reconstruye con el mismo `results.json`: no hace daño, pero se puede filtrar.

Pendiente (opcional):
- [ ] **Probar `deterministic=True` en LightGBM**, con `force_row_wise` o `force_col_wise` como pide su documentación, para que la configuración 2 dé idéntico entre corridas.
  - Entrena más lento y no garantiza resultados idénticos en máquinas distintas.
  - El código no se tocó.

---

## Fase 8b. Monitoreo y drift

Va antes del front para que el job ya mida el pronóstico en vivo cuando llegue la venta de enero de 2027. Se numera 8b para no cambiar las referencias a las fases 9 y 10.

Prompt para Claude Code:

```
Ejecuta la Fase 8b de PLAN.md: monitoreo de drift de datos y de desempeño.
1. Drift de datos: en cada corrida, PSI de las variables principales del origen
   contra la ventana de entrenamiento. Con un cambio extremo, el job se detiene
   antes de pronosticar. Se guarda en una tabla de solo agregar.
2. Drift de desempeño: para cada pronóstico de predicciones cuyas ventanas ya
   cerraron, WAPE y sesgo en vivo en el formato estándar, comparados con el rango
   por origen del backtest. Si se sale del rango, alerta; si se repite en 2
   orígenes seguidos, revisión manual. Se guarda en una tabla de solo agregar.
3. Los umbrales se fijan antes de ver resultados en vivo y quedan en CLAUDE.md.
4. Nada de reentrenamiento automático por drift: cada corrida ya reentrena y
   aplica la regla del champion.
5. No cambia results.json ni su esquema.
6. Demostración en local con Docker sobre los 8 orígenes del backtest, como si
   fueran en vivo: ¿los orígenes con más drift tuvieron más error?
7. Rama y PR. El log no muestra piezas.
Propón el plan primero.
```

Terminado cuando:
- [x] `make lint` y `make test` pasan, con las pruebas nuevas de monitoreo
- [x] La demostración local muestra el drift y el error en vivo de cada origen, y si van juntos
- [x] Los umbrales quedan en CLAUDE.md antes del merge
- [x] En dev, las dos tareas nuevas terminan en verde y `predicciones` sigue intacta
- [x] Commit `fase 8b: monitoreo y drift`, por PR

Manual:
- [x] Revisar la demostración antes del merge. Si el umbral de detención hubiera parado un origen sin datos rotos, se decide antes de enero.
  - Se revisó en el PR #7 (29-oct-2026), con la demostración por ventana, y cambió la regla de reentrenamiento: una alerta de desempeño ya no lo adelanta (ver "Decisión" en la fase 9b).
- [x] Hacer merge y aprobar el despliegue en prod, sin correr el job

Ajustes aprobados (28-oct-2026):
- Antes de fijar el 1,0 se calculó el PSI de los orígenes históricos, sin entrenar.
  - Varios orígenes normales lo pasaban, todos por `tendencia` después de un Hot Sale o diciembre. Por eso `tendencia` quedó solo como dato.
- Se agregaron revisiones de lo que el PSI no ve:
  - los nulos contra los 13 orígenes recientes;
  - la cobertura de SKU con stock y de filas de venta;
  - el corte del stock.
- La cobertura se compara contra la semana más baja de las 4 anteriores y no contra su promedio, porque el promedio queda inflado después de cada Hot Sale.
- Se agregó la cobertura del mes, contra las semanas que ya revisó la corrida anterior: SKU con stock y venta entre salidas de stock. Cubre el caso en que todo el mes llega roto.
- Los umbrales y su calibración están en "Monitoreo y drift" de CLAUDE.md.

Diseño:
- **Dónde va en el job.** Dos tareas nuevas: ingest → features → `monitoreo_datos` → train → predict → export → `monitoreo_desempeno`.
  - `monitoreo_datos` va antes de train, no solo antes de predict. Con datos rotos no se gastan los ~40 min de train ni se registra un champion entrenado con ellos.
  - `monitoreo_desempeno` va al final. No depende de la corrida, y si falla, el pronóstico y `results.json` ya quedaron.
  - La lógica va en `src/common/monitoreo.py`, en funciones puras de pandas, y cada tarea en su script: `src/monitoreo_datos.py` y `src/monitoreo_desempeno.py`.
- **Drift de datos.**
  - Variables principales, en una lista fija:
    - historia: `rezago_1`, `media_4`, `media_13`, `media_52`, `tendencia` y `crecimiento_anual`;
    - surtido: `estilos_color_con_venta_8` y `lanzamientos_8`;
    - oferta: `stock_cierre` y `disponibilidad`.
  - Sin calendario ni jerarquía. Las 13 semanas de un origen siempre tienen otro calendario que el historial completo, y la jerarquía no se mueve de una semana a otra.
  - Se compara el origen (una fila por serie activa) contra la ventana de entrenamiento: una fila por serie × origen en las filas con que se entrena el champion (`backtest.entrenamiento`).
  - El PSI usa 10 tramos en los deciles de la referencia y deja fuera los nulos. El entrenamiento tiene nulos por diseño: el stock antes de mayo de 2025 y `media_52` en 2024.
  - Umbrales: menos de 0,25 es estable; de 0,25 a 1,0 es aviso, que queda en la tabla y en el log; 1,0 o más es extremo. `tendencia` queda solo como dato.
  - **Nulos:** la parte de nulos del origen contra los últimos 13 orígenes. Da aviso si sube 10 puntos y es extremo si la variable llega 100 % nula y antes tenía datos.
  - **Cobertura:** SKU con stock al cierre y filas de venta de la semana del origen, contra la más baja de las 4 semanas anteriores. Da aviso en 0,75 y es extremo en 0,6.
  - **Cobertura del mes:** SKU con stock y venta entre salidas de stock de cada semana nueva, contra las 4 semanas que revisó la corrida anterior, con los mismos umbrales.
  - **Corte del stock:** es extremo si el stock atrasa el origen una semana o más.
  - Con un extremo, la tarea escribe la tabla y después falla, así que el job se detiene antes de train.
  - Para seguir con un cambio extremo que se revisó y es legítimo, el job tiene el parámetro `aceptar_drift`, con la fecha del origen. Solo vale para ese origen y queda anotado en la tabla.
    - Se pasa con `databricks bundle run -t <target> demanda_pipeline --params aceptar_drift=<fecha>` o desde "Run now with different parameters". El CD no lo usa.
  - Tabla `monitoreo_datos`: una fila por corrida × revisión (`psi`, `nulos`, `cobertura` o `corte`), con el valor, la referencia, los umbrales, el nivel y si se aceptó.
- **La semana duplicada en la ingesta.** `validation.py` revisa la venta completa, no por muestra, y se detiene si se repite una fecha + SKU, así que una semana cargada dos veces la para.
  - No ve una semana con las unidades dobladas en las mismas filas. Ahí el PSI de `rezago_1` da de 0,7 a 1,0: aviso, y detención solo si llega a 1,0.
- **Drift de desempeño.**
  - Mide cada corrida de `predicciones` cuya ventana 3 ya cerró (origen + 13 semanas ≤ el último origen de features) y que todavía no esté en la tabla.
  - Usa el mismo código del backtest: `metrics.a_ventanas` y `metrics.formato_estandar` sobre las filas semanales.
  - El rango es el mínimo y el máximo por origen en los 8 orígenes del backtest donde se registró la versión que pronosticó.
    - Sale del artefacto `predicciones_backtest.parquet` de ese run de MLflow, que se guardó al entrenar, así que quedó fijo antes de conocer la venta real.
    - Sirve también para las corridas del 12-oct-2026 que se miden en enero.
  - Alerta si el WAPE del mes 1, del mes 2 o el total queda arriba del máximo, o si el sesgo total queda fuera del rango, por arriba o por abajo. El MAPE va solo como referencia.
  - Revisión manual si el origen y el anterior medido tienen alerta. Si un origen tiene varias corridas, cuenta la última, como en `results.json`.
  - Ni la alerta ni la revisión detienen el job: quedan en la tabla y en el log.
  - Mientras no cierre ninguna ventana, la tarea imprime el rango que se usará para cada corrida pendiente.
  - Tabla `monitoreo_desempeno`: una fila por corrida medida, con las métricas del formato estándar y `wape_tipo_mes`, el rango de cada métrica de alerta, el run del backtest, las métricas fuera de rango y el estado. No guarda venta, pronóstico, MAE ni RMSE, que van en piezas.
- **Solo agregar.** Las dos tablas tienen `delta.appendOnly` desde que se crean, como `predicciones`.
- **Sin reentrenamiento por drift.** Cada corrida ya reentrena con todo lo conocido y aplica la regla del champion. Un drift extremo detiene el job, no reentrena.
- **Log sin piezas.** Solo PSI, porcentajes, fechas y nombres de variables, también en el error que detiene el job. Una prueba lo verifica.
- **No cambia** `results.json`, su esquema ni los workflows.
- **Pruebas** (`tests/test_monitoreo.py`):
  - PSI: 0 con distribuciones iguales, alto con un desplazamiento, y bien calculado con empates y nulos;
  - niveles, detención y `aceptar_drift` solo para su origen;
  - rango y alertas: el WAPE solo por arriba y el sesgo por los dos lados;
  - revisión con 2 seguidas, ventanas cerradas y log sin piezas;
  - cobertura del mes: un mes entero a medias que la cobertura de la semana no ve, la corrida anterior en la primera corrida (el último origen de `predicciones`) y la venta sin verificar cuando faltan salidas de stock;
  - `tests/test_monitoreo_spark.py`: las salidas de stock en Spark dan igual que en pandas;
  - `tests/test_bundle.py` con las 7 tareas.
- **Demostración local** (`scripts/simular_monitoreo.py`, con `make simular-monitoreo`).
  - Usa los 8 orígenes del backtest de dev y prod (29-dic-2025 a 13-jul-2026) como 8 corridas mensuales en vivo, con el algoritmo del champion actual (`lightgbm_tweedie_211`) entrenado hasta cada origen.
  - El rango de cada uno sale del backtest que esa corrida habría hecho: 8 orígenes que terminan 13 semanas antes. Son 15 orígenes más, del 17-mar-2025 al 13-abr-2026: 23 entrenamientos, unos 40 min.
  - Usa el mismo código que el job sobre `data/`, y arma features con las funciones de `src/features.py` en Spark local.
  - Por origen muestra el PSI máximo y su variable, WAPE M1, M2 y total, sesgo, rango y estado, y la correlación de Spearman entre el drift y el error.
  - Con 8 puntos solo se notaría una relación fuerte. Como complemento, la misma correlación en los 23 orígenes, cuyas ventanas se traslapan.
  - Revisa que las 10 variables estén entre las de más importancia del champion.
  - No escribe en Databricks ni en git: el resultado va a esta sección, en porcentajes.
- **Prueba en dev.** Solo las dos tareas nuevas (`databricks bundle run -t dev demanda_pipeline --only monitoreo_datos,monitoreo_desempeno`), unos minutos de cuota en vez de los ~45 del job completo.
  - `monitoreo_desempeno` no tendrá ventanas cerradas hasta enero, pero calcula e imprime el rango de las corridas abiertas. Así prueba desde ya la lectura del artefacto de MLflow.
- **Limitaciones conocidas.**
  - Con 8 orígenes, aunque el modelo siga igual de bueno, un origen nuevo queda arriba del peor de los 8 con probabilidad 1 en 9 en cada métrica. Por eso una alerta sola es aviso, y la revisión pide dos seguidas.
  - Los 8 orígenes del rango nunca caen en la misma temporada que el pronóstico en vivo. El del 12-oct-2026, con Buen Fin y diciembre, se compara con orígenes de diciembre a julio.
  - **El rango de enero es poco exigente.** Sus 8 orígenes incluyen la alerta del 20-abr-2026 y la revisión manual del 18-may-2026, así que va de 30,2 % a 45,4 % en WAPE total y de −18,0 % a +33,1 % en sesgo.
    - Se deja así porque se fijó antes de ver la venta real.
  - La cobertura de la semana compara contra las 4 semanas anteriores, que con el refresco mensual también son nuevas. Un mes entero roto lo ve la cobertura del mes, que compara contra las semanas que ya revisó la corrida anterior.
  - La cobertura del mes está calibrada con corridas cada 4, 5 o 13 semanas. Después de un hueco más largo, el bloque nuevo cruza más temporada y puede dar avisos falsos.
  - Con el stock atrasado, `aceptar_drift` solo repite el pronóstico desde el mismo origen. Ver el pendiente del modo sin stock.

Resultado (28-oct-2026):
- **Calibración** (`make calibrar-drift`, 137 orígenes semanales de mar-2024 a oct-2026, sin entrenar).
  - Con `tendencia` informativa, 28 orígenes darían aviso de PSI y 3 de cobertura. Solo se detendría el del 24-feb-2025, por el inicio del stock.
  - Ningún aviso ni detención por nulos. El detalle está en CLAUDE.md.
- **Demostración** (`make simular-monitoreo`): los 8 orígenes del backtest de dev y prod como corridas mensuales en vivo, con `lightgbm_tweedie_211` entrenado hasta cada uno. Cada uno se compara con el rango del backtest que esa corrida habría hecho.
  - **Reproduce dev.** Los 8 juntos dan WAPE M1 35,1 %, M2 44,9 %, total 36,5 %, sesgo +2,4 % y WAPE tipo × mes 43,6 %. En dev fueron 35,5 %, 45,0 %, 36,4 %, +2,7 % y 43,8 %: la misma diferencia entre máquinas de la fase 8.

  | Origen | PSI máx. (sin `tendencia`) | PSI `tendencia` | WAPE M1 | WAPE M2 | WAPE total | Sesgo total | Fuera del rango | Estado |
  |---|---|---|---|---|---|---|---|---|
  | 29-dic-2025 | 0,11 (`media_4`) | 0,49 | 35,0 % | 38,9 % | 36,2 % | +0,6 % | - | en rango |
  | 26-ene-2026 | 0,26 (`lanzamientos_8`), aviso | 0,26 | 31,9 % | 50,7 % | 41,2 % | −15,0 % | - | en rango |
  | 23-feb-2026 | 0,15 (`crecimiento_anual`) | 0,07 | 39,0 % | 47,0 % | 38,9 % | −11,6 % | - | en rango |
  | 23-mar-2026 | 0,11 (`media_13`) | 0,11 | 27,9 % | 30,1 % | 30,1 % | +9,3 % | - | en rango |
  | 20-abr-2026 | 0,10 (`crecimiento_anual`) | 0,22 | 23,5 % | 58,1 % | 35,1 % | +22,1 % | WAPE M2 (máx. 52,0 %), sesgo (máx. +13,7 %) | alerta |
  | 18-may-2026 | 0,11 (`media_13`) | 0,13 | 49,1 % | 58,9 % | 45,4 % | +32,5 % | las cuatro | revisión manual |
  | 15-jun-2026 | 0,20 (`rezago_1`) | 0,70 | 39,9 % | 43,0 % | 33,3 % | −3,8 % | WAPE M1 (máx. 39,2 %) | revisión manual |
  | 13-jul-2026 | 0,11 (`crecimiento_anual`) | 0,22 | 32,5 % | 40,3 % | 32,2 % | −18,4 % | - | en rango |

  - **Drift de datos bajo en las 8.** El PSI más alto fue 0,26 y dio un solo aviso, por `lanzamientos_8` en ene-2026. Nada se habría detenido.
  - **Drift de desempeño en abril y mayo de 2026.** Los orígenes del 20-abr y el 18-may sobrepronosticaron +22 % y +33 %, con ventanas que cubren el Hot Sale 2026 y el inicio del verano.
    - La alerta del 20-abr pasó a revisión manual el 18-may, y siguió el 15-jun por poco (WAPE M1 de 39,9 % contra un máximo de 39,2 %). Volvió al rango el 13-jul.
  - **¿Más drift, más error? No en este periodo.**
    - En las 8, Spearman entre el PSI máximo y el WAPE tipo × mes da +0,14 (p = 0,75, permutaciones exactas). La relación más alta es +0,62 (p = 0,11), entre el PSI medio y el WAPE relativo a la mediana de su backtest.
    - En los 23 orígenes da +0,41 (PSI medio contra WAPE tipo × mes, p = 0,06). Como las ventanas se traslapan, ese p es optimista. La relación la arrastran mar–abr 2025, con el stock recién iniciado: tuvieron el PSI más alto y errores entre los 4 más altos.
    - El segundo origen con más error de los 23 (18-may-2026, 51,8 %) tuvo uno de los PSI más bajos (0,11). Las variables de entrada se veían normales; lo que cambió fue la respuesta de la venta.
    - Por eso hacen falta los dos monitoreos: el de datos detecta datos rotos y el de desempeño, un modelo que dejó de acertar.
  - **Importancia.** De las 10 variables monitoreadas, 5 están entre las 10 de más ganancia del champion: `estilos_color_con_venta_8` (1.ª), `rezago_1` (2.ª), `media_4` (3.ª), `media_13` (6.ª) y `stock_cierre` (9.ª).
    - `disponibilidad` (27.ª) y `lanzamientos_8` (30.ª) pesan poco en el modelo, pero sirven para detectar roturas del stock y del surtido.
    - Entre las 10 primeras sin monitorear están `media_8`, `rezago_2` y `media_26`, que se mueven con las monitoreadas, y `tipo_producto` y `semana_iso`, que quedan fuera por diseño.
- **Para la medición de enero.** El rango del pronóstico del 12-oct-2026 sale de estos mismos 8 orígenes, así que es amplio porque incluye abril y mayo.
  - Con los números locales: WAPE M1 de 23,5 % a 49,1 %, M2 de 30,1 % a 58,9 %, total de 30,1 % a 45,4 % y sesgo de −18,4 % a +32,5 %.
  - El exacto lo calcula la tarea con el artefacto de MLflow de la versión que pronosticó.
- **Dev** (`bundle validate` y `deploy -t dev`, y `bundle run --only monitoreo_datos,monitoreo_desempeno`). Las dos tareas terminaron en verde en 2,5 minutos.
  - `monitoreo_datos` creó `workspace.tesis_dev.monitoreo_datos` con `appendOnly` y escribió 23 revisiones del origen 12-oct-2026.
    - Todas quedaron estables: el PSI más alto fue 0,09, la cobertura de SKU con stock 1,14 y la de filas de venta 1,18, y el stock no atrasa el origen.
  - `monitoreo_desempeno` no encontró ventanas cerradas, así que no creó su tabla.
    - Leyó desde MLflow el rango de las corridas de las versiones 2 y 3: WAPE M1 de 23,8 % a 50,0 %, M2 de 30,2 % a 58,8 %, total de 30,2 % a 45,4 % y sesgo de −18,0 % a +33,1 %. Coincide con la simulación local.
  - `predicciones` sigue intacta. La corrida solo ejecutó las dos tareas nuevas, que solo la leen, y la consulta de solo lectura de su historial lo confirma:
    - sigue en la versión 2, con el último cambio a las 17:47:16 UTC (un `WRITE` en modo `Append`);
    - `delta.appendOnly = true`;
    - la corrida del 2026-10-28 16:28:09 tiene 7.424 filas, igual que la de las 17:47:13.
- **Cobertura del mes** (aprobada el 28-oct-2026 e implementada en el mismo PR).
  - **Problema.** Con el refresco mensual llegan 4 o 5 semanas nuevas a la vez. Si todas llegan rotas, la cobertura de la semana las compara entre sí y queda en 1.
    - Simulado en 18 orígenes con la venta de todas las series × 0,5 en las últimas 4 o 5 semanas: el PSI más alto (`media_4` o `rezago_1`) tiene una mediana de 0,27. Da aviso en 9 de 18 y nunca llega a 1,0.
  - **Solución.** Cada semana nueva se compara contra las 4 semanas que ya revisó la corrida anterior, hasta su origen, con los mismos umbrales: aviso en 0,75 y detención en 0,6.
    - **SKU con stock del mes.**
    - **Venta entre salidas de stock del mes.** El conteo de filas de venta no sirve para el mes:
      - contra las semanas revisadas, la semana antes del Hot Sale cae sola a 0,50;
      - contra el año anterior, a 0,16, porque el Hot Sale cambia de fecha.
  - **Corrida anterior.** El último origen de `monitoreo_datos` cuya corrida no tuvo extremos o los aceptó, o el último de `predicciones`.
    - La primera corrida con monitoreo, como la de enero en prod, usa las 4 semanas hasta el origen de la última corrida en `predicciones`, el 12-oct-2026.
    - Si tampoco hay, las 4 semanas hasta el origen − 4 semanas.
  - **En Spark.** `monitoreo_datos` calcula las salidas de stock de `stock_bronze`. Una prueba con Spark local verifica que dan igual que la función de pandas con la que se calibró.
  - **Calibración** (`make calibrar-drift`): ninguna detención en la historia.

    | Revisión | Corrida anterior | Mínimo normal | Avisos | Mes entero a la mitad | Una semana a la mitad |
    |---|---|---|---|---|---|
    | SKU con stock | 4 / 5 / 13 semanas antes | 0,81 / 0,81 / 0,76 | 0 | se detiene en 64 de 69 | 62 de 69 |
    | Venta entre salidas | 4 / 5 / 13 semanas antes | 0,70 / 0,70 / 0,70 | 1 / 2 / 6 | se detiene en 60 de 69 | 41 de 69 |

    - Con 13 semanas, como la corrida de enero en prod, el bloque nuevo cruza Buen Fin y diciembre, y aun así ningún origen normal se detiene.
  - **Solo vale porque el stock viene de los movimientos del almacén, independiente de la venta.** Está documentado en CLAUDE.md.
  - Sin salidas de stock en una semana nueva, venta entre salidas queda `sin datos`. El log dice que la venta del mes no se verificó, no se detiene el job y `aceptar_drift` no la vuelve estable.
- **Merge (PR #4).**
  - El CD esperó la aprobación y desplegó: actualizó el job `demanda_pipeline` (`1 changed`) y subió 28 archivos. El paso que corre el job se saltó, como debía.
  - El job de prod quedó con las 7 tareas, todas con `--schema tesis`, el parámetro `aceptar_drift` y la programación en pausa.
  - Sigue con una sola corrida, la primera de la fase 8 (28-oct-2026, 22:42 UTC): el despliegue no corrió nada.
  - El log muestra el usuario como `***`.

Pendiente (propuesto, sin implementar): **modo sin stock reciente.**
- **Problema.** `features.py` corta en la fecha más temprana entre la venta y el stock. Si la venta llega a enero y el stock a octubre, el origen se queda en octubre: liberar el freno no produce un pronóstico nuevo.
- **Propuesta.** Un parámetro del job `exigir_stock`, `true` por defecto, que llega a `features.py`, como `exigir_stock=False` en la experimentación.
  - Con `false`, features corta por la fecha de la venta. En las semanas posteriores al último día de stock, `stock_cierre`, `skus_con_stock`, `estilos_color_con_stock` y `disponibilidad` quedan nulos, no en 0, igual que antes de mayo de 2025.
  - El modelo ya conoce el stock nulo: lo tuvo en todo 2024 y hasta abril de 2025 al entrenar. Con stock nulo se apoya en la historia de la venta.
- **Cómo se usa.** El job se detiene por el corte del stock y se corre con `exigir_stock=false` y `aceptar_drift=<origen de la venta>`.
  - En esa corrida, `monitoreo_datos` marca extremos que el mismo `aceptar_drift` libera:
    - el corte del stock;
    - los nulos de `stock_cierre` y `disponibilidad` (100 % contra 0 %);
    - la cobertura de SKU con stock, de la semana y del mes (0).
  - Así, liberar el freno sí da un pronóstico nuevo desde el origen de la venta.
  - **Venta entre salidas de stock** no se puede calcular en las semanas nuevas, porque no tienen stock.
    - Queda `sin datos`, y el log dice que la venta del mes no se verificó.
    - `aceptar_drift` no la marca como estable: el rastro dice que esa venta se usó sin revisar.
    - Ya funciona así desde esta fase. Una prueba lo cubre.
- **Rastro.**
  - `exigir_stock` queda como parámetro del run de MLflow y como columna nueva de `predicciones`. Agregar una columna está permitido con `appendOnly`.
  - `monitoreo_desempeno` mide esas corridas como cualquier otra.
- **Pruebas.**
  - Con el stock cortado antes que la venta, las variables de stock quedan nulas después del corte y no en 0.
  - El corte sale de la venta, y la prueba de "sin fuga de información" sigue pasando.
  - Con `exigir_stock=true` todo queda igual que hoy.

---

## Fase 9. Datos públicos para la página

Redefinida el 28-oct-2026. La página ya existe en el repo `tesis-mlops-web`: lee un `manifest.json` y un `results.json` por pronóstico desde una URL pública, con el contrato de su README. Esta fase ya no construye el front aquí. Genera esos archivos y los publica en el GitHub Pages de este repo, en `adcamachoa.github.io/tesis-mlops/datos/`.

Manual:
- [x] En GitHub: Settings → Pages → Source: GitHub Actions
- [x] En Settings → Environments, revisar que `github-pages` solo acepte `main`
- [x] Revisar la prueba en dev antes del merge
- [x] Hacer merge del PR y aprobar el despliegue en prod. El merge solo despliega: `pages.yml` no publica.
- [x] Exportar en prod: `gh workflow run cd.yml --ref main -f accion=solo_export`, y aprobarlo. Al terminar en verde, `pages.yml` publica solo.
- [x] Comprobar que `https://adcamachoa.github.io/tesis-mlops/datos/manifest.json` responde con el manifest real y que pasa la validación

Prompt para Claude Code:

```
Ejecuta la Fase 9 de PLAN.md: publicar los pronósticos para la página de
tesis-mlops-web, con el diseño de esa sección. Haz:
1. En src/common/results_schema.py, el esquema del manifest (versión 1) y una
   función que valide el manifest junto con sus results.json, con las mismas
   reglas que revisarManifest de tesis-mlops-web (docs/js/datos.js).
2. En src/export_results.py, además del results.json de siempre, un
   output/datos/<origen>/results.json por cada origen de predicciones (su
   última corrida) y output/datos/manifest.json. Cada origen se arma solo con
   datos hasta su datos_hasta y con las métricas del train que registró la
   versión que pronosticó. Valida todo antes de escribir.
3. En cd.yml, una casilla solo_export que corra solo la tarea export en prod.
4. .github/workflows/pages.yml, que publica output/datos/ de prod en GitHub
   Pages, con las condiciones y los dos jobs del diseño.
5. Pruebas del manifest, de regenerar un origen viejo con datos posteriores
   y de los workflows.
6. Actualiza CLAUDE.md: results.json, el manifest, la estructura sin front/ y
   que no se borran versiones del modelo en UC, porque el export las necesita
   para regenerar los orígenes.
La prueba en dev la corres tú, pidiéndome permiso en cada comando.
Rama fase-9-publicacion y PR. Propón el plan primero.
```

Terminado cuando:
- [x] `make lint` y `make test` pasan, con las pruebas nuevas
- [x] En dev, el export deja `output/datos/` con el manifest y un `results.json` por origen, y pasa la validación
- [x] El CD con `accion=solo_export` termina en verde en prod y `pages.yml` publica
- [x] `https://adcamachoa.github.io/tesis-mlops/datos/manifest.json` responde con el manifest real: el origen 12-oct-2026, con la corrida de prod de las 23:22:23 (revisado el 29-oct-2026)
- [x] Commit `fase 9: datos públicos para la página`, por PR

Ajustes aprobados (28-oct-2026):
- En el CD, la casilla `correr` y la nueva `solo_export` quedan en un solo input de lista, `accion`, así que no pueden chocar.
- `pages.yml` descarga de Databricks antes de instalar nada de PyPI, y valida después, sin secretos.
- `results_schema` no cambia los nombres que importa `tesis-mlops-web` (`validar` con un argumento, `VERSION`, `ESCALA` y `BASE_INDICE`). Agrega:
  - **Fechas que existan**, también en `results.json`. La web ya rechaza fechas como "2026-13-01".
  - **`ErrorEsquema` con el nombre del archivo,** como argumento opcional.
  - **El error de la venta en piezas no muestra el promedio,** porque sería la venta semanal del total y el error puede llegar a un log público.

Diseño:
- **Archivos.** `export_results.py` escribe en `/Volumes/workspace/<schema>/raw/output/`:
  - `datos/<origen>/results.json`: uno por origen de `predicciones`, con su última corrida. El esquema no cambia (`version_esquema` 1).
  - `datos/manifest.json`, en el formato 1 del README de la web: una entrada por origen, con `id` = el origen (`aaaa-mm-dd`), `ruta` = `<origen>/results.json`, `version_esquema`, `origen` y `fecha_prediccion`.
  - `results.json`, como hoy, pasa a ser el mismo archivo del origen más reciente.
  - En cada corrida vuelve a generar todos los orígenes desde las tablas, no solo el nuevo. Así lo publicado no depende de lo que haya quedado en el Volume.
  - Arma y valida todo en memoria antes de escribir. Si algo falla, no escribe nada y lo anterior queda intacto.
- **Cada origen se arma solo con datos hasta su `datos_hasta`.**
  - **Venta real:** las 52 semanas hasta el origen, de `panel_semanal`. Las semanas posteriores no entran, tampoco en el factor del índice ni en la elección de las 20 series.
  - **Pronóstico:** la última corrida de ese origen en `predicciones`. Las corridas de otros orígenes no entran.
  - **Métricas:** las de la corrida de `metricas_modelos` que registró la versión que pronosticó, es decir, cuyo `run_id_modelo` es el run de esa versión en Unity Catalog. `fecha_entrenamiento` y `backtest` salen de la misma corrida.
    - Hoy se toma la última corrida de `metricas_modelos`, que solo sirve para el origen más reciente.
    - Nunca entran las métricas de un train posterior. Si no encuentra la corrida de una versión, falla con un mensaje claro y no escribe nada.
  - **Prueba:** a las tablas de un origen viejo se les agregan semanas de venta posteriores, una corrida de train y una de predicción de un origen nuevo. El archivo del origen viejo sale idéntico, salvo `generado`.
  - **Limitación:** si un refresco corrige la venta de semanas ya pasadas, la venta real de un origen viejo cambia al regenerarlo. El pronóstico y las métricas no cambian.
- **Esquema del manifest** en `src/common/results_schema.py`, con las mismas reglas que `revisarManifest` de `tesis-mlops-web`:
  - `version_manifest` = 1 y `generado` con fecha y hora;
  - `pronosticos`: una lista no vacía;
  - `id` con `^[a-z0-9-]+$`, y `ruta` con la misma expresión de la web: `^([A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.json$`, es decir, relativa, sin `..`, `/` inicial ni `http`;
  - `origen` y `fecha_prediccion` con fechas que existen, y `version_esquema` entero;
  - `id` y `origen` únicos;
  - cada `results.json` de la lista cumple `validar` y trae en `corrida` el mismo `origen` y `fecha_prediccion` que su entrada.
  - Las pruebas rechazan los mismos casos que `pruebas/datos.test.js` de la web: sin pronósticos, ruta con `..`, absoluta o a otro sitio, `id` con espacios, fecha que no existe, origen o `id` repetido, y `version_manifest` 2 o ausente.
  - Si la web cambia sus reglas, se cambian aquí también.
- **Acción `solo_export` en el CD.**
  - La casilla `correr` pasa a ser un input de lista, `accion`: `desplegar` (por defecto), `correr` o `solo_export`. Como se elige una sola, no hay choque que revisar. La corrida completa ahora se pide con `-f accion=correr`.
  - Con `solo_export`, despliega como siempre y corre solo la tarea export en prod (`bundle run -t prod demanda_pipeline --only export`). Son unos 3 minutos en vez de 45, con la misma aprobación del environment `prod`.
  - Descarta stdout, igual que `correr`.
  - Sirve para publicar ya el pronóstico del 12-oct-2026 sin esperar a enero, y para volver a exportar si cambia el formato.
- **`pages.yml`.**
  - **Cuándo corre.** Con `workflow_run` del CD y con `workflow_dispatch`.
    - Con `workflow_run`, solo si el CD terminó en `success`, fue lanzado con `workflow_dispatch` y fue sobre `main`, desde este mismo repo. Los merges que solo despliegan no cambian datos, así que no publican.
    - `workflow_run` no trae la acción elegida en el CD: uno lanzado a mano con `accion=desplegar` también publica, con los mismos datos. No hace daño.
    - Con `workflow_dispatch`, solo desde `main`.
    - Nunca `pull_request` ni `pull_request_target`.
  - **Job `descargar`**, el único con secretos, y solo en los pasos que los usan.
    - Oculta el usuario de Databricks con `::add-mask::`.
    - Descarga `/Volumes/workspace/tesis/raw/output/datos/` con `databricks fs cp -r`: siempre `tesis`, nunca `tesis_dev`.
    - Después de descargar instala Python 3.11 y, de PyPI, solo pandas, numpy y jsonschema, con los rangos de `requirements.txt`. Así ningún paquete de PyPI está instalado mientras se usan los secretos.
    - Valida sin secretos, con `python -m common.results_schema`, que el manifest y los archivos coincidan: cada entrada con su archivo y ningún archivo fuera del manifest. Si algo falla, no publica y lo que ya estaba en Pages queda igual.
    - Sube la carpeta como artefacto de Pages, con `datos/` en la raíz del sitio.
  - **Job `publicar`**, sin secretos: el único con `pages: write` e `id-token: write`, con el environment `github-pages`.
  - Una publicación a la vez. Cada una reemplaza el sitio completo, y como el export regenera todos los orígenes, no se pierde ninguno.
  - El token del workflow solo lee, las acciones van fijadas por commit y el log no muestra piezas: solo rutas, fechas y conteos.
  - La raíz del sitio no tiene página. La página es la de `tesis-mlops-web`, en el mismo dominio, así que lee los datos sin configuración extra.
- **Pruebas.**
  - `tests/test_export_results.py`: un archivo por origen, con su última corrida y las métricas de su versión; regenerar un origen viejo con datos posteriores da el mismo archivo, salvo `generado`; si un origen no cumple el esquema, no se escribe nada.
  - El manifest válido y cada caso que rechaza la web.
  - `tests/test_workflows.py`:
    - `pages.yml`: las condiciones del disparo, los secretos solo en `descargar` y en los pasos que los usan, la máscara antes de Databricks, solo `tesis`, y los permisos de escritura solo en `publicar`;
    - `cd.yml`: `accion` con sus tres opciones, y `solo_export` con `--only export`, `-t prod` y sin su salida.
  - `tests/test_bundle.py` no cambia: el bundle sigue declarando solo el job.
- **Prueba en dev.** `bundle deploy -t dev` y `bundle run -t dev demanda_pipeline --only export`, unos minutos de cuota.
  - Se descarga `output/datos/` de `tesis_dev` y se valida en local con la misma función. Debe traer el origen del 12-oct-2026 con la corrida de las 17:47:13, la última de ese origen.
  - `pages.yml` nunca lee dev.

Resultado (28-oct-2026):
- **Pruebas.** `make lint` pasa y `make test` da 144 pruebas en verde, con las nuevas del export, del manifest y de los workflows.
  - Como control cruzado, el ejemplo sintético de `tesis-mlops-web` (`docs/datos/ejemplo/`, dos orígenes) pasa `python -m common.results_schema`, así que las reglas de los dos repos coinciden.
- **Dev.**
  - `bundle deploy -t dev` subió 12 archivos sin cambiar el job.
  - `bundle run -t dev demanda_pipeline --only export` terminó en verde en 1,5 minutos.
- **`output/datos/` de `tesis_dev`,** descargado con el mismo `databricks fs cp -r` de `pages.yml`. Copia el contenido de `datos/` en la carpeta de destino, y el log solo muestra rutas.
  - Trae `manifest.json` y `2026-10-12/results.json`, y pasa la validación.
  - El origen es el 12-oct-2026, con la corrida de las 17:47:13, la última de ese origen, y la versión 3.
  - Las métricas son las del train que registró la versión 3, el de las 17:43:07: backtest de 8 orígenes, del 29-dic-2025 al 13-jul-2026.
  - `lightgbm_tweedie_211` es el champion y el mejor, con WAPE tipo × mes de 43,8 %, igual que en la fase 6.
  - Trae 24 series (el total, 3 submarcas y 20 series), y la venta real del total promedia 100,0.
- **Tablas.** El export solo lee `predicciones`, `metricas_modelos` y `panel_semanal`, y busca las versiones en Unity Catalog: no escribe en ninguna tabla.
- **Horas en UTC.** `generado` y `fecha_prediccion` van en UTC, la hora de Databricks serverless, como `fecha_corrida` en las tablas.

Fuera de esta fase:
- **El PR de `docs/config.js` en `tesis-mlops-web`**: `url` a `https://adcamachoa.github.io/tesis-mlops/datos/` y `sinteticos: false`, y activar su Pages.
  - Va desde ese repo, como dice su CLAUDE.md.
  - **Va solo cuando el manifest real ya responde en la URL.** Antes, la página apuntaría a una carpeta vacía y mostraría un error en vez del ejemplo.
- El semáforo del monitoreo en la página.
- Las propuestas de la versión 2 de `results.json` del README de la web.

---

## Fase 9b. Cadencia semanal y monitoreo por ventana

La tesis decía "un pronóstico cada 4 semanas", y eso se mantiene. Desde esta fase, además:
- la venta real publicada se actualiza cada semana;
- cada ventana del pronóstico se mide apenas cierra.

La propuesta incluía que una alerta de desempeño adelantara el reentrenamiento. Se descartó el 29-oct-2026, con la evidencia de la demostración (ver "Decisión" al final de la fase).

Se numera 9b para no cambiar las referencias a la fase 10.

Prompt para Claude Code (resumido):

```
La cadencia de corridas y el monitoreo de desempeño. Propón el plan antes de
editar y no corras nada en prod.
1. Cada lunes se suben al Volume de prod los 5 parquet con corte al domingo, de
   una exportación nueva que reutiliza la llave.
2. Un solo job semanal, los lunes a las 9:00 (America/Bogota): siempre ingest →
   features → monitoreo_datos → monitoreo_desempeno → export, y una condition_task
   "¿toca reentrenar?" que agrega train → predict cada 4 semanas desde el último
   origen entrenado, o antes si la ventana medida más reciente dio alerta. Un
   drift extremo de datos sigue deteniendo el job.
3. Las semanas sin reentrenamiento solo se actualiza la venta real.
4. Calendario esperado: orígenes 9-nov, 7-dic y 4-ene.
5. En la página, las semanas pasadas conservan el pronóstico del origen anterior.
6. results.json necesita la venta real de las semanas ya pronosticadas.
7. Freno: sin un domingo nuevo, el job se detiene sin gastar cuota.
8. Medir cada ventana apenas cierre, contra el rango de esa ventana en el
   backtest, y revisar la regla de revisión manual por ventana.
9. Revisar la cuota de Free Edition.
10. Actualizar PLAN.md y CLAUDE.md.
```

Decisiones (29-oct-2026):
- El merge activa la programación en prod. Se apunta a hacerlo antes del lunes 2-nov; si no alcanza, la primera corrida se lanza a mano.
- **Campos nuevos en `results.json`,** en la versión 1: `real_hasta`, `real_horizonte`, `pronostico_anterior` y `ventanas_real`.
- **El sesgo de cada mes** va como dato, sin alerta.
- **La llave.** `ingest` solo se detiene si falta un SKU que ya estaba, o si cambia el estilo o `estilo_color` de más del 1 % de los existentes. Los casos sueltos van al log.
- **Antes del merge:** la calibración de 1 semana de `monitoreo_datos` y la prueba en dev, nunca un lunes.
- **Después del merge:** extender la demostración de la 8b a la medición por ventana, sin atrasar el merge.

Manual:
- [x] Revisar la prueba en dev antes del merge
- [x] Hacer merge del PR y aprobar el despliegue en prod, idealmente antes del lunes 2-nov. Desde ahí, el job corre cada lunes a las 9:00 de Bogotá.
  - Merge del PR #6 el 29-oct-2026.
  - Revisado con `jobs get` (solo lectura): el job de prod quedó `UNPAUSED`, los lunes a las 9:00 (`America/Bogota`), con las 8 tareas sobre `tesis`, los parámetros `aceptar_drift` y `reentrenar` vacíos y sin reintento en `ingest` y `monitoreo_datos`.
  - `jobs list` no muestra los parámetros; para verlos hace falta `jobs get`.
- [ ] Cada lunes (tarea recurrente): exportar y subir al Volume de prod (`/Volumes/workspace/tesis/raw/`) los 5 parquet con corte al domingo, con un solo comando, entre las 6:45 y las 9:00 de Bogotá.
  - BigQuery se actualiza a las 6:00 y el catálogo a las 6:30.
  - Si llegan después de las 9:00, el freno detiene la corrida programada; hay que lanzarla a mano con `gh workflow run cd.yml --ref main -f accion=correr`.
- [ ] **Hacia el 15-dic-2026, hacer un commit al repo**, por ejemplo con el resultado de la medición del 21-dic. En un repo público, GitHub desactiva el schedule de `pages.yml` después de 60 días sin actividad, y tiene que seguir activo hasta el 18-ene-2027.
- [x] PR en `tesis-mlops-web` para usar los campos nuevos: `generar_ejemplo.py` tiene que generarlos (la validación de aquí los exige), y la página puede dibujarlos.
  - PR #6 de la web, con merge el 29-oct-2026.

Terminado cuando:
- [x] `make lint` y `make test` pasan, con las pruebas nuevas
- [x] La calibración con la corrida anterior 1 semana antes no detiene ningún origen normal
- [x] En dev, la corrida normal va por la rama sin reentrenar y exporta, la repetida se detiene en el freno y la de `reentrenar=si` va por la rama que entrena
- [x] Commit `fase 9b: cadencia semanal y monitoreo por ventana`, por PR (#6)

Diseño:
- **El job** (`databricks.yml`): ingest → features → `monitoreo_datos` → `monitoreo_desempeno` → `toca_reentrenar` → (sí) train → predict → export; (no) export.
  - **La condición.** `toca_reentrenar` es un `condition_task` que compara el task value `reentrenar` que publica `monitoreo_desempeno`, como booleano.
  - **La unión de las dos ramas.** `export` depende de `predict` y de la rama `false` de la condición, con `run_if: NONE_FAILED`.
    - Databricks exige que toda dependencia de un `condition_task` diga su rama; el primer despliegue lo rechazó sin ella.
    - Las tareas excluidas cuentan como éxito, así que `export` corre en las dos ramas, y no corre si `train` o `predict` fallan.
    - `AT_LEAST_ONE_SUCCESS` exportaría aunque `predict` falle.
  - **Programación:** lunes a las 9:00, `America/Bogota`. La variable `programacion` vale `PAUSED` por defecto y `UNPAUSED` en prod.
  - **Sin reintento automático** (`disable_auto_optimization: true`) en `ingest` y `monitoreo_datos`, cuyas fallas son detenciones a propósito.
  - **Parámetros:** `aceptar_drift`, como antes, y `reentrenar` (vacío o `si`).
- **Cuándo toca reentrenar** (`monitoreo.decidir_reentrenamiento`):
  - cuando el origen queda 4 semanas o más después del último origen de `predicciones`;
  - o a mano, con `reentrenar=si`.
  - Hasta el 29-oct-2026 también reentrenaba antes si alguna medición con alerta cerraba después del `datos_hasta` del último pronóstico. Se descartó (ver "Decisión"). La alerta queda como señal para revisar el modelo.
- **Freno** (`validation.revisar_semana_nueva`, en `ingest`): el último domingo completo de venta y stock contra el `real_hasta` del último `results.json`.
  - Si `results.json` no trae `real_hasta`, como en la primera corrida, no hay freno.
  - `reentrenar=si` se lo salta.
- **Llave** (`validation.comparar_productos` y `revisar_llave`, en `ingest`): se detiene si faltan SKU o si más del 1 % cambia de `estilo` o `estilo_color`.
- **Medición por ventana** (`monitoreo.por_medir`, `evaluar` y `estados`).
  - El mes 1 y el mes 2 por separado, y el total cuando cierra el mes 3. Cada una contra el rango del backtest de su versión, sin umbrales nuevos.
  - La tabla `monitoreo_desempeno` queda con una fila por corrida × medición. No existía en dev ni en prod, así que se definió de nuevo.
  - Revisión manual con la misma medición en dos orígenes seguidos.
- **`results.json`** (`export_results.py` y `results_schema.py`):
  - `real_hasta`;
  - `real_horizonte` y `ventanas_real`, nulos mientras no cierran;
  - `pronostico_anterior`, el pronóstico vigente en cada semana real.
  - Todo con el mismo factor del índice, y `validar` revisa dónde va cada nulo.
- **Publicación:** `pages.yml` suma un `schedule` los lunes a las 16:00 UTC (11:00 en Bogotá).
- **CD:** la acción `reentrenar` (`--params reentrenar=si`).
- **Calibración** (`make calibrar-drift`): la cobertura del mes con la corrida anterior 1 semana antes, y la detección de una semana nueva rota.

Resultado (29-oct-2026):
- **Pruebas.** `make lint` pasa, y `make test` da 162 pruebas en verde.
- **Calibración** con la corrida anterior 1 semana antes, sin cambiar umbrales:

  | Revisión | Orígenes | Mínimo | Avisos | Detenciones | Semana nueva a la mitad se detiene en |
  |---|---|---|---|---|---|
  | SKU con stock | 72 | 0,88 | 0 | 0 | 66 de 69 |
  | Venta entre salidas | 72 | 0,77 | 0 | 0 | 38 de 69 |

  - Con la corrida semanal, la cobertura del mes da menos avisos que con 4 semanas (mínimo 0,70, 1 aviso).
  - El resto de la calibración sale igual que en la 8b.
- **Prueba en dev** (29-oct-2026, un jueves). El primer `bundle deploy -t dev` lo rechazó Databricks porque la dependencia de `export` en la condición no decía su rama. Con la rama `false` y `run_if: NONE_FAILED`, desplegó.

  | Corrida | Total | Lo que pasó |
  |---|---|---|
  | Normal | 5,3 min | Sin freno (el `results.json` de la fase 9 no traía `real_hasta`). `monitoreo_desempeno` publicó `reentrenar = false`: 0 semanas desde el 12-oct y ninguna ventana cerrada. `train` y `predict` quedaron excluidos y `export` corrió. |
  | La misma otra vez | 1,5 min | `ingest` se detuvo con `SinSemanaNueva`: venta y stock al 18-oct, y lo publicado ya llegaba al 18-oct. El resto se saltó. |
  | `reentrenar=si` | 41,1 min | Sin freno. `train` (35,9 min) registró la versión 4 de `tesis_dev.demanda_forecast` como champion, con las métricas de la fase 6 (WAPE tipo × mes 43,8 %). `predict` agregó la corrida 2026-10-29 06:12:58 (7.424 filas) y `export` la publicó. |

  - **Los task values funcionan en un `spark_python_task`.** La documentación solo los describe en notebooks; la rama elegida en las dos corridas lo confirma.
  - `monitoreo_desempeno` dejó 6 mediciones pendientes: las 2 corridas anteriores del 12-oct × 3. La primera, el mes 1, cierra el 18-nov.
  - `output/datos/` de `tesis_dev`, descargado, pasa la validación.
    - Trae el origen 12-oct, con la corrida de las 06:12:58 y la versión 4.
    - En dev, `real_horizonte`, `ventanas_real` y `pronostico_anterior` quedan nulos: los datos de dev no pasan del 18-oct y no hay un origen anterior al 12-oct. Los valores los cubren las pruebas.
  - **Serverless reintenta una tarea que falla, aunque la falla sea a propósito.** En la corrida frenada, `ingest` corrió dos veces (0,9 y 0,5 min).
    - Por eso `ingest` y `monitoreo_datos` llevan `disable_auto_optimization: true`. En `monitoreo_datos`, un drift extremo repetido escribiría sus revisiones dos veces.
    - Databricks lo aceptó en serverless. En la corrida frenada siguiente, `ingest` corrió una sola vez (intento 0) y la corrida tardó 1,3 min.
  - `predicciones` de dev tiene `appendOnly`, así que la corrida nueva solo se agregó. En la corrida normal, `monitoreo_desempeno` todavía leía las del 12-oct de las 16:28:09 (versión 2) y las 17:47:13 (versión 3). No se volvió a consultar la tabla después.

Demostración por ventana (después del merge, 29-oct-2026, `make simular-monitoreo`, solo en local):
- **Qué hace.** Mide los 8 orígenes de la demostración de la 8b (29-dic-2025 a 13-jul-2026) con la regla de la 9b, con los mismos pronósticos y rangos.
  - La tabla de la 8b, con las tres ventanas juntas, sale igual que antes.
  - Agrega `monitoreo.corrida_de_medicion`, el lunes en que se mide cada ventana. El log de `monitoreo_desempeno` ahora dice cuándo se mide la siguiente.
- **La alerta llega antes.**

  | Origen | Primera alerta por ventana | Con las tres ventanas | Semanas antes | Revisión manual por ventana |
  |---|---|---|---|---|
  | 20-abr-2026 | mes 2 (WAPE M2 58,1 % contra máx. 52,0 %), el 29-jun | 27-jul | 4 | - |
  | 18-may-2026 | mes 1 (WAPE M1 49,1 % contra máx. 39,2 %), el 29-jun | 24-ago | 8 | mes 2 el 27-jul y total el 24-ago |
  | 15-jun-2026 | mes 1 (WAPE M1 39,9 % contra máx. 39,2 %), el 27-jul | 21-sep | 8 | mes 1 el 27-jul |

  - **La primera revisión manual llega el 27-jul,** por el mes 2 del 18-may y el mes 1 del 15-jun. Con las tres ventanas llegaba el 24-ago.
  - **El sesgo por mes muestra el sobrepronóstico** del Hot Sale y el verano: +46,6 % en el mes 2 del 20-abr y +43,3 % en el mes 1 del 18-may. El total los mezclaba en +22,1 % y +32,5 %.
  - Los otros 5 orígenes quedan en rango en sus 3 mediciones.
- **Reentrenamientos adelantados.** Se repasó la regla que entonces tenía el job (reentrenar si una alerta cerraba después del `datos_hasta` del último pronóstico), lunes a lunes, del origen 29-dic-2025 al 3-ago-2026, manteniendo la rejilla de 4 semanas de la 8b. Hubo 2 disparos.
  - Cada pronóstico adelantado se entrenó con el mismo corte del backtest: nada después de su origen.
  - Se comparó con el vigente solo en las semanas en que habría estado vigente: desde el disparo hasta que entra el siguiente origen de la rejilla.

  | Corrida | Origen adelantado | Por | Semanas | Vigente | WAPE tipo, adelantado / vigente | WAPE total | Sesgo |
  |---|---|---|---|---|---|---|---|
  | 29-jun | 22-jun | mes 2 del 20-abr y mes 1 del 18-may | 29-jun a 13-jul (3) | 15-jun | 52,4 % / 53,4 % | 26,8 % / 27,0 % | +9,4 % / +7,5 % |
  | 27-jul | 20-jul | total del 20-abr, mes 2 del 18-may y mes 1 del 15-jun | 27-jul a 10-ago (3) | 13-jul | 40,2 % / 43,6 % | 8,4 % / 8,6 % | −8,4 % / −0,7 % |

- **Un hallazgo estructural: el adelanto siempre cae 1 semana después del reentrenamiento programado.**
  - Con orígenes cada 4 semanas y ventanas de 31, 30 y 30 días, el mes 1 se mide 6 semanas después de su origen, el mes 2 a las 10 y el total a las 14.
  - Todas caen 1 semana después de una corrida de la rejilla. Un reentrenamiento por alerta le agrega una sola semana de datos al modelo que se acaba de entrenar.
  - En los 2 casos, el WAPE tipo bajó 1,0 y 3,4 puntos, el total casi no cambió, y el sesgo empeoró 1,9 y 7,7 puntos.
  - **Con 2 casos de 3 semanas no hay evidencia de que adelantar ayude.** El valor de medir por ventana está en saber antes (4 a 8 semanas) que el modelo dejó de acertar, para revisarlo, más que en el reentrenamiento adelantado.
- **Limitaciones.**
  - En el job, después de un disparo las 4 semanas se cuentan desde ese origen, así que la rejilla se correría 1 semana. El repaso la mantiene fija.
  - No se miden los pronósticos adelantados, solo se comparan en sus semanas vigentes.
  - Un reentrenamiento cuesta unos 36 min de cuota.
- Entrenar los 2 orígenes adelantados tomó 465 y 289 s en local. Quedaron en `data/_local/`, aparte de los 23 de la 8b.

Decisión (29-oct-2026): **una alerta de desempeño ya no adelanta el reentrenamiento.**
- Se reentrena solo cada 4 semanas, o a mano con `reentrenar=si`.
- Se mantiene todo lo demás: la medición por ventana, la alerta, el sesgo por mes y la revisión manual. La alerta queda en el log y en la tabla `monitoreo_desempeno` como señal para revisar el modelo.
- **Evidencia, de la demostración por ventana:**
  - **Cada medición cae 1 semana después de un reentrenamiento programado** (mes 1 a las 6 semanas del origen, mes 2 a las 10 y total a las 14), así que adelantar solo agrega 1 semana de datos.
  - **No bajó el error.** En los 2 disparos, en sus 3 semanas vigentes, el WAPE tipo bajó 1,0 y 3,4 puntos, el WAPE total no cambió y el sesgo empeoró 1,9 y 7,7 puntos.
  - **Cada disparo cuesta unos 36 minutos de cuota**, lo que tarda train.
- **En el código.**
  - `monitoreo.decidir_reentrenamiento` ya no recibe las mediciones.
  - El log de `monitoreo_desempeno` avisa cuántas mediciones tuvieron alerta y que no adelantan el reentrenamiento.
  - La regla descartada quedó solo en `scripts/simular_monitoreo.py` (`alerta_posterior`), para reproducir la evidencia.
- **En prod.** Entra con el merge de este PR, antes de la primera medición real (el mes 1 del 12-oct se mide el 23-nov). Hasta entonces no cierra ninguna ventana, así que la regla anterior no habría disparado.

---

## Fase 9c. Jerarquía completa y métricas por nivel

`results.json` pasa de traer el total, 3 submarcas y 20 series a traer el árbol total → submarca → departamento → categoría → serie, con las métricas del backtest a nivel de cada nodo. Es la versión 2 del esquema.

Prompt para Claude Code (resumido):

```
Jerarquía completa y métricas por nivel en results.json. Propón el plan antes de
editar y no corras nada en prod.
1. Publicar los niveles total → submarca → departamento → categoría → tipo de
   producto, cada nodo con su padre; proponer un mínimo de venta para publicar una
   serie y qué pasa con el resto; los 4 campos de la 9b en cada nodo; el peso del
   archivo.
2. series[].metricas con el formato estándar sumado a cada nodo, desde las
   predicciones del backtest guardadas en MLflow, sin reentrenar.
3. Las mismas reglas: el mismo factor, 1 decimal, nada en piezas, y la versión 1
   si ningún campo existente cambia de significado.
4. Qué cambia para tesis-mlops-web.
5. Precisar el docstring de decidir_reentrenamiento.
```

Decisiones (29-oct-2026):
- **Mínimo de 0,5** de índice promedio semanal de venta real.
  - Se revisó publicar también por el pronóstico de 13 semanas, para lo nuevo o de temporada. Agregaba 32 nodos (16 series, 12 categorías y 4 "otras"), casi todos con una historia justo debajo del mínimo (0,2 a 0,49), y subía la exposición de 0,45 % a 2,24 %. No era poco, así que no se incluyó.
- **Versión 2**, porque la página de la versión 1 rechaza los niveles nuevos.
- **Métricas de los 6 modelos** por nodo.
- **Los hijos de un nodo desglosado siempre suman el nodo** (29-oct-2026, a pedido, después de la primera versión del PR).
  - Con el JSON público se puede restar los hijos a su padre semana a semana. Cuando un padre no tenía "otras", esa diferencia era justo lo chico que no se publicó: 26 restos que subían la exposición de 0,45 % a 3,12 %.
  - Ahora, cuando el resto no llega al mínimo, se junta con el hijo publicado más chico en "otras", con todo lo suyo. Si no queda ningún otro hijo, el nodo no se desglosa.
  - Se descartó saltarse los restos sin venta: el árbol de un origen podría cambiar si empezaban a vender, y ahorraba solo 8 nodos.
  - Costo: de 142 a 118 nodos. Se pierden 37 (24 series, 8 categorías, 3 "otras" y 2 departamentos) y entran 13 "otras". La venta visible en nodos de serie baja de 61 % a 35 %; en 12 categorías la única serie publicada se va a "otras", y la categoría, que vale casi lo mismo, sigue publicada.

Manual:
- [x] PR en `tesis-mlops-web` que acepte la versión 1 y la 2, y publicarlo. El resumen para la web va al final de esta fase.
  - PR #7 de la web, con merge el 29-oct-2026.
- [x] **Solo después**, revisar la prueba en dev, hacer merge del PR de la 9c y aprobar el despliegue en prod.
  - El merge no publica: el job del lunes exporta en versión 2, y Pages lo publica a las 11:00.
  - Para no esperar al lunes: `gh workflow run cd.yml --ref main -f accion=solo_export`.
  - Merge del PR #8 el 29-oct-2026, después del de la web, con el despliegue en prod. Después, `solo_export` y `pages.yml` terminaron en verde.
- [x] Revisar que `https://adcamachoa.github.io/tesis-mlops/datos/manifest.json` traiga `version_esquema` 2 y que la página lo muestre.
  - Publicado el 29-oct-2026: el origen 12-oct, con la corrida de prod de las 23:22:23, en versión 2 y con 118 nodos (1 total, 3 submarcas, 14 departamentos, 46 categorías, 25 series y 29 "otras"), los mismos que en dev.

Terminado cuando:
- [x] `make lint` y `make test` pasan, con las pruebas nuevas
- [x] En dev, el export publica la versión 2 con el árbol y las métricas por nodo, y pasa la validación
- [x] Commit `fase 9c: jerarquía completa y métricas por nivel`, por PR (#8), con el merge después del de la web

Diseño:
- **El árbol** (`src/common/arbol.py`, funciones puras).
  - Se publica un nodo si se publicó su padre y su índice promedio semanal de venta real en las 52 semanas hasta el origen llega a `indice_minimo` (0,5).
  - Los hijos que no llegan van juntos a un nodo `otras` del padre. Si su suma no llega al mínimo, se les junta el hijo publicado más chico, con todo lo suyo. Si no queda ningún otro hijo, el padre no se desglosa. Así los hijos siempre suman el padre.
  - **Campos de cada nodo:**
    - `id`: la ruta en minúsculas y guiones, por ejemplo `submarca-a/bebe-nino/conjuntos`, estable entre orígenes;
    - `padre`, `nivel`, `nombre` completo y `etiqueta`.
  - **Orden:** en profundidad, cada padre antes de sus hijos, y los hijos del más pesado al más liviano, con `otras` al final.
  - Si dos nodos dan el mismo `id`, el export falla.
- **Cada nodo** suma la venta real de todas sus series, también las que no se pronostican, y trae los campos de siempre y los de la 9b, con el mismo factor y 1 decimal.
- **Métricas por nodo** (`series[].metricas`), para los 6 modelos en el orden de `metricas`:
  - WAPE M1, M2 y total, MAPE M1, M2 y total como dato, y sesgo total;
  - salen del artefacto `predicciones_backtest.parquet` de cada modelo en el train de la versión que pronosticó, que train.py guarda desde la fase 4;
  - con `metrics.a_ventanas`, se suman al nodo por origen y ventana y se aplica el formato estándar;
  - sin MAE ni RMSE, que van en piezas;
  - si falta un artefacto, el export falla con un mensaje claro y no escribe nada.
- **Versión 2** (`results_schema.py`):
  - `indice_minimo` en la raíz;
  - en cada nodo, `id`, `padre`, `etiqueta`, `metricas` y los niveles `departamento`, `categoria` y `otras`.
  - `validar` revisa además los `id` únicos, que cada padre exista y sea del nivel anterior, el mínimo, que los hijos sumen su padre en cada valor (salvo el redondeo) y que cada nodo traiga los modelos de la raíz.
- **JSON compacto** en los `results.json`: con sangría pesaría el doble. El manifest queda igual.
- **Constante `ARTEFACTO_BACKTEST`** en train.py, que usan `monitoreo_desempeno` y el export.

Resultado (29-oct-2026):
- **Pruebas.** `make lint` pasa y `make test` da 181 pruebas en verde.
  - Las nuevas cubren la poda y los "otras", que los hijos tengan todas las series de su padre, que el export coincide con las series y su suma, las métricas de un nodo hoja y del total contra cálculos independientes, el backtest faltante y los casos que `validar` rechaza.
- **Análisis local** con el origen 12-oct-2026. El árbol completo sería de 3 submarcas, 20 departamentos, 173 categorías y 464 series; la serie mediana queda en 0,0 en el 63 % de las semanas.
  - La exposición son los valores semanales de 1 a 3 piezas, publicados o que se pueden derivar restando los hijos a su padre.
  - En dev todavía no cerró ninguna semana del horizonte, así que `real_horizonte` no suma valores.

  | Regla | Nodos | Valores de 1 a 3 piezas | Peso por origen, compacto |
  |---|---|---|---|
  | 20 series (versión 1) | 24 | 1,9 % | - |
  | Mínimo 0,2, restos sin juntar | 250 | 3,7 % sin contar lo derivable | ~0,5 MB |
  | Mínimo 0,5, restos sin juntar (primera versión del PR) | 142 | 0,45 % publicados; 3,12 % con los 26 restos derivables | 0,27 MB |
  | Mínimo 0,5 y también por el pronóstico | 170 | 2,24 % sin contar lo derivable | - |
  | **Mínimo 0,5, hijos que suman el padre (la regla final)** | **118** | **0,41 %, sin restos derivables** | **0,22 MB** |

- **Dev** (`bundle deploy -t dev` y `bundle run -t dev demanda_pipeline --only export`, unos 2,5 min cada vez).
  - Encontró los backtests de los 6 modelos de la versión 4 en MLflow.
  - Con la regla final publicó 118 nodos: 1 total, 3 submarcas, 14 departamentos, 46 categorías, 25 series y 29 "otras".
  - `output/datos/`, descargado, pasa la validación. Pesa 0,22 MB y no trae métricas nulas.
  - En los 37 padres desglosados, los hijos suman el padre en `real`, salvo el redondeo: la diferencia más grande es de 0,2.
  - **El WAPE baja al subir de nivel.** Mediana del WAPE total del champion: serie 31,3 %, categoría 26,0 %, departamento 21,3 %, submarca 16,9 % y total 14,7 %.

Para `tesis-mlops-web`:
- **Leer la versión 2**, y seguir aceptando la 1: `VERSIONES_ESQUEMA = [1, 2]`. Con solo la 1, la página muestra "formato versión 2" en todos los orígenes.
- **`series` es el árbol, en profundidad:**
  - cada nodo trae `id`, `padre` (nulo en el total), `nivel` (`total`, `submarca`, `departamento`, `categoria`, `serie` u `otras`), `nombre` completo y `etiqueta`;
  - ya no hace falta leer la submarca del nombre.
- **El desglose de un nodo son sus hijos por `padre`, y siempre suman el nodo,** salvo el redondeo a 1 decimal de cada uno.
  - No hay "resto sin desglose": lo chico va en el hijo `otras`, que puede incluir también un hijo grande, el más chico de los publicados.
  - Un nodo sin hijos no se desglosa.
- **`series[].metricas`:** la tabla de modelos puede seguir al nodo elegido.
  - Tiene la forma de `metricas`, sin `es_mejor`, `es_champion`, `wape_tipo_mes`, MAE ni RMSE; esos siguen en la `metricas` de la raíz, a nivel tipo de producto.
  - Conviene explicar que el WAPE baja al subir de nivel.
- **`indice_minimo`:** para una nota de qué no se publica.
- **El archivo es compacto, de unos 0,2 MB por origen**, y se lee igual.
- **`generar_ejemplo.py`** tiene que armar la versión 2. `validar` la exige, y le sirven `METRICAS_NODO`, `NIVELES` y `OTRAS`.

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
