# Tesis: pronóstico de demanda con MLOps

Repositorio productivo de la tesis de maestría. Es un pipeline MLOps para pronosticar la demanda semanal de una marca de moda infantil que vende en e-commerce y marketplaces en México:
- Datos y jobs en Databricks Free Edition.
- CI/CD en GitHub Actions.
- Página de resultados en GitHub Pages.

El plan por fases está en `PLAN.md`: una fase por sesión, y siempre proponer el plan antes de editar.

## Reglas

- **El dataset está 100 % anonimizado.**
  - No intentes re-identificar submarcas, productos ni códigos.
  - No cruces los datos con fuentes externas que lo permitan.
  - No escribas nombres de empresas ni de marcas en código, documentos, commits, MLflow ni en la página.
- **Los datos no van a git.** `data/` y `*.parquet` están en `.gitignore`; la única excepción es `tests/data/muestra.parquet`, que es un extracto anonimizado.
- **Nunca pongas tokens ni credenciales** en archivos ni en el chat. Databricks se configura con `databricks configure`; GitHub usa secrets.
- **Todo el texto va en español:** código, documentos, página y mensajes de commit.

## Datos

- **Local:** en `data/`, que no se versiona.
- **En Databricks:** en `/Volumes/workspace/<schema>/raw/`, con `<schema>` = `tesis_dev` o `tesis`.
- **Corte:** datos hasta el domingo 18-oct-2026 (`data/_extraccion.json`). Los volúmenes son reales; lo anonimizado son los identificadores.

| Archivo | Grano | Columnas | Filas |
|---|---|---|---|
| `ventas_diarias.parquet` | fecha × sku, solo días con venta | `fecha` (timestamp sin hora), `sku` (str), `unidades` (int) | 745.909 |
| `stock_diario.parquet` | fecha × sku | `fecha`, `sku`, `stock` (int, ≥ 0) | 12.650.230 |
| `quiebres_diarios.parquet` | fecha × sku | `fecha`, `sku`, `unidades` (int) | 5.028 |
| `productos.parquet` | sku | `sku`, `estilo`, `estilo_color`, `talla`, `orden_talla`, `grupo_talla`, `submarca`, `edad`, `departamento`, `categoria`, `tipo_producto`, `es_basico`, `temporada_tipo`, `temporada_anio` | 50.803 |
| `calendario_eventos.parquet` | evento | `evento`, `tipo`, `fecha_inicio`, `fecha_fin` (timestamp sin hora) | 63 |

Todas las fechas son timestamp a las 00:00, no `date`: Spark las lee como `timestamp_ntz` y pandas como `datetime64`.

**Definiciones**

- **ventas_diarias:** unidades vendidas válidas, con todos los canales sumados. No incluye órdenes canceladas, envíos ni ajustes. Arranca el 2024-01-01. Un día sin fila significa venta 0. Sus claves no se repiten (fecha + sku) y no hay unidades negativas.
- **stock_diario:** stock vendible al cierre del día, desde el 2025-02-17.
  - Cada SKU aparece desde su primer movimiento. Antes de eso no se sabe si tenía stock, así que feb–abr 2025 está subestimado.
  - El stock es al cierre: un SKU que vendió su última unidad aparece con 0 ese día.
- **quiebres_diarios:** unidades que un cliente pidió y no se pudieron surtir por falta de stock. Son cerca del 1 % de la venta, y más de dos tercios caen en el Hot Sale 2025.
- **productos:**
  - Relación de códigos: sku → `estilo_color` → `estilo`. Son códigos al azar sin orden: `SKU-000001`, `EC-000001`, `EST-00001`.
  - Jerarquía de navegación: `submarca` (Submarca A, B o C, de mayor a menor venta) → `edad` → `departamento` → `categoria` → `tipo_producto`.
  - "Sin dato" donde falta el dato. `talla` y `orden_talla` sirven para la curva de tallas.
- **calendario_eventos:** eventos comerciales (Hot Sale, El Buen Fin, Black Friday y Cyber Monday) y festividades, con sus fechas oficiales. Faltan el Hot Sale y el Buen Fin 2027: hay que agregarlos antes de que el horizonte llegue a mayo de 2027.

## Decisiones de modelado

Vienen del proyecto de experimentación, validadas con backtest. La tesis las replica sobre este dataset.

- **Unidad de pronóstico.**
  - Serie = submarca × departamento × categoría × tipo de producto: unas 540 series con venta.
  - Venta semanal, de lunes a domingo.
  - No se pronostica por SKU: solo ~17 % de la venta está en SKU pronosticables. La curva de tallas y colores se aplica después, de arriba hacia abajo.
- **Horizonte y formulación.**
  - Horizonte de 13 semanas, con un pronóstico cada 4 semanas.
  - Pronóstico directo: una fila por serie × semana de origen × horizonte, y el horizonte es una variable.
  - Las variables solo usan información hasta el cierre del origen.
- **Modelo.**
  - LightGBM global con objetivo Tweedie.
  - Producción = promedio de dos configuraciones:
    - potencia 1,2 y 63 hojas, con parada temprana;
    - potencia 1,76, 211 hojas y 449 rondas.
  - Cada una con 3 semillas promediadas (7, 8 y 9).
  - Parada temprana: una sola por origen, con la semilla 7, y mínimo 50 rondas.
    - El examen son las filas de los 13 orígenes internos de origen − 25 a origen − 13 semanas.
    - El ajuste usa las filas con semana objetivo hasta origen − 25.
    - Con esas rondas se entrenan las 3 semillas sobre todo lo conocido hasta el origen.
  - Los hiperparámetros completos están en `src/common/modelos.py`.
- **Variables:** historia de la serie (promedios, rezagos, tendencia, crecimiento contra el año anterior), surtido (estilos-color con venta, lanzamientos), jerarquía (categóricas), calendario (eventos, quincenas, semana del año), stock y disponibilidad, y la misma semana del año anterior.
- **Evaluación.**
  - Backtest con orígenes mensuales (rolling origin), sin fechas futuras en el entrenamiento.
  - Ventanas de 31, 30 y 30 días desde el lunes siguiente al origen.
    - La venta real de cada ventana sale de la venta diaria, con todas las series. Las que no están en la base cuentan con pronóstico 0.
    - El pronóstico de las semanas partidas (la 5 y la 9) se reparte con el perfil por día de la semana de la venta total de las 52 semanas previas al lunes siguiente al origen; 1/7 por día si no hubo venta.
  - **Formato obligatorio para comparar modelos:** WAPE mes 1, mes 2 y total (el trimestre); MAPE mes 1, mes 2 y total; y al final el sesgo total. Todo a nivel tipo de producto.
  - El MAPE va solo como referencia y se calcula sobre los nodos con venta.
- **Champion.**
  - El mejor modelo es el de menor WAPE tipo de producto × mes, con las tres ventanas juntas.
  - Para quitarle el alias al algoritmo del champion, un algoritmo distinto tiene que mejorar su WAPE tipo × mes en el mismo backtest y ganarle en al menos 6 de los 8 orígenes.
  - Si gana el mismo algoritmo, la versión reentrenada recibe el alias. Si el retador no lo logra, se reentrena y registra el algoritmo del champion, que conserva el alias.
- **Referencia a replicar:** desarrollo, 8 orígenes de ago-2025 a feb-2026 (del 11-ago-2025 al 23-feb-2026, cada 4 semanas: `--ultimo-origen 2026-02-23`).

  | Modelo | WAPE M1 | WAPE M2 | WAPE Total | Sesgo total |
  |---|---|---|---|---|
  | Producción | 34,2 % | 43,7 % | 35,8 % | +4,2 % |
  | Naive estacional | 76,4 % | 78,3 % | 70,5 % | +7,3 % |

- **Lo que no funcionó.** Ya se probó y no hace falta repetirlo:
  - variables de trayectoria del año anterior y de semanas desde el último evento (sobrepronostican);
  - un multiplicador sobre una base robusta de n días;
  - repartir las semanas partidas con la forma diaria del año anterior.

## Predicción y resultados

- **Tabla `predicciones`, en piezas.**
  - Cada fila es una serie en un periodo: `granularidad` = `semana` (periodos 1 a 13) o `ventana` (periodos 1 a 3, de 31, 30 y 30 días).
  - Cada corrida se agrega a la tabla con `fecha_corrida`, `origen`, `modelo` y `modelo_version`.
- **`results.json`.** Su esquema está en `src/common/results_schema.py`.
  - Usa solo la última corrida del origen más reciente, y la última corrida de `metricas_modelos`.
  - **La página es pública: nunca lleva piezas.** La venta real y el pronóstico van como índice, donde 100 = la venta semanal promedio del total en las 52 semanas mostradas. El factor es el mismo para todas las series, y el MAE y el RMSE se escalan con él.
  - El índice va con 1 decimal. Con más decimales, cada pieza se vería como un escalón fijo y se podría reconstruir la escala.

## Ajustes a PLAN.md (28-oct-2026)

Los prompts de las fases 2, 3, 4, 5 y 9 ya están ajustados a estas decisiones:
- cinco tablas bronze;
- serie semanal por tipo de producto;
- pronóstico directo a 13 semanas;
- el formato obligatorio de métricas;
- el champion se elige por el menor WAPE tipo de producto × mes.

## Entorno local

- **Sistema:** Windows con PowerShell. Docker Desktop tiene que estar corriendo para `make lint` y `make test`.
- **Antivirus:** inspecciona el tráfico HTTPS, así que las herramientas que no usan el almacén de certificados de Windows fallan con "unable to get local issuer certificate".
  - **git:** este repo ya tiene `http.sslBackend = schannel`.
  - **Docker:** `pip` dentro de la imagen falla igual. El `Dockerfile` tiene que aceptar un certificado opcional como secreto de BuildKit, que solo se pasa en local (`--secret id=ca,src=.certs/ca_bundle.pem`) y se usa como `PIP_CERT` al instalar. En CI no se pasa, y la imagen no debe guardar el certificado.
  - **`.certs/`** está en `.gitignore`.

## Estructura del repositorio

```
src/
  ingest.py            parquet del Volume → tablas bronze (Delta)
  features.py          bronze → tabla de features
  train.py             backtest, MLflow, registro del champion en Unity Catalog
  predict.py           pronóstico con el champion → tabla predicciones
  export_results.py    results.json para la página
  common/              funciones puras en pandas, con pruebas
    validation.py  features.py  metrics.py  backtest.py  modelos.py  results_schema.py
tests/                 pruebas (pytest) y tests/data/ (muestra.parquet, results.json de ejemplo)
front/                 template.html y build.py → dist/index.html
databricks.yml         Asset Bundle: job demanda_pipeline, targets dev y prod
.github/workflows/     ci.yml, cd.yml, pages.yml
data/                  dataset local (no se versiona)
```
