# Tesis: pronóstico de demanda con MLOps

Repositorio productivo de la tesis de maestría. Es un pipeline MLOps para pronosticar la demanda semanal de una marca de moda infantil que vende en e-commerce y marketplaces en México:
- Datos y jobs en Databricks Free Edition.
- CI/CD en GitHub Actions.
- Datos públicos de la página en GitHub Pages (`adcamachoa.github.io/tesis-mlops/datos/`). La página está en el repo `tesis-mlops-web`.

El plan por fases está en `PLAN.md`: una fase por sesión, y siempre proponer el plan antes de editar.

## Reglas

- **El dataset está 100 % anonimizado.**
  - No intentes re-identificar submarcas, productos ni códigos.
  - No cruces los datos con fuentes externas que lo permitan.
  - No escribas nombres de empresas ni de marcas en código, documentos, commits, MLflow ni en la página.
- **Los datos no van a git.** `data/` y `*.parquet` están en `.gitignore`; la única excepción es `tests/data/muestra.parquet`, que es un extracto anonimizado.
- **Nunca pongas tokens ni credenciales** en archivos ni en el chat. Databricks se configura con `databricks configure`; GitHub usa secrets.
- **Todo el texto va en español:** código, documentos, página y mensajes de commit.
- **Workflows de GitHub (el repo es público).**
  - Nunca uses `pull_request_target`.
  - Los secretos solo van en los pasos que los usan, y nunca en jobs que ejecuten código de un PR de fork.
  - El usuario de Databricks se oculta en el log con `::add-mask::`, y las acciones van fijadas por commit.
- **El bundle solo declara el job `demanda_pipeline`.** No agregues a `databricks.yml` schemas, volúmenes, tablas ni modelos como recursos, porque `deploy` y `destroy` podrían cambiarlos o borrarlos. `tests/test_bundle.py` lo verifica.
- **No se borran versiones del modelo en Unity Catalog.** El export las necesita para regenerar los orígenes: liga cada pronóstico con su train por el run de su versión. Sin la versión, el export falla y no publica nada.

## Datos

- **Local:** en `data/`, que no se versiona.
- **En Databricks:** en `/Volumes/workspace/<schema>/raw/`, con `<schema>` = `tesis_dev` o `tesis`.
- **Corte:** datos hasta el domingo 18-oct-2026 (`data/_extraccion.json`). Los volúmenes son reales; lo anonimizado son los identificadores.
- **Carga semanal en prod.** Cada lunes, antes de las 9:00 de Bogotá, se suben al Volume de prod los 5 parquet con corte al domingo.
  - Vienen de una exportación nueva que reutiliza la llave: los códigos existentes no cambian y solo se agregan los SKU nuevos.
  - `local` y `tesis_dev` se quedan con el corte del 18-oct.

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
  - Horizonte de 13 semanas, con un pronóstico nuevo cada 4 semanas, o antes si una alerta de desempeño lo adelanta. Cada semana se actualiza la venta real publicada (ver "Cadencia").
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

## Cadencia

- **Un solo job semanal.** `demanda_pipeline` corre los lunes a las 9:00 (America/Bogota), solo en prod; en dev la programación queda en pausa.
  - Siempre: ingest → features → `monitoreo_datos` → `monitoreo_desempeno` → `toca_reentrenar`.
  - `toca_reentrenar` es un `condition_task`. Si toca, sigue train → predict; si no, quedan excluidos y el pronóstico publicado no cambia.
  - `export` corre en las dos ramas, con `run_if: NONE_FAILED`: depende de `predict` y de la rama `false` de la condición. No exporta si `train` o `predict` fallan.
  - Un drift extremo de datos detiene el job en `monitoreo_datos`: no reentrena.
- **Cuándo toca reentrenar.** Lo decide `monitoreo_desempeno` y lo publica como task value booleano `reentrenar`.
  - Cuando el origen queda 4 semanas o más después del último origen de `predicciones`. Desde el 12-oct-2026 da los orígenes 9-nov, 7-dic y 4-ene, con corridas los lunes 16-nov, 14-dic y 11-ene.
  - O antes, si alguna medición con alerta cerró después del `datos_hasta` del último pronóstico, es decir, si trae información que el último entrenamiento no conocía.
    - Una alerta que ya conocía no vuelve a disparar, así que no reentrena cada semana mientras no cierre otra ventana.
    - Después de un reentrenamiento adelantado, las 4 semanas se cuentan desde ese origen.
- **Freno.** `ingest` se detiene antes de leer los datos completos si los parquet no traen un domingo nuevo.
  - Cuenta el último domingo completo de la venta y del stock, el que llegue menos, contra el `real_hasta` del último `results.json` exportado.
  - Se compara con el último export: una corrida que falló después de la ingesta no lo actualizó, así que repetirla pasa el freno.
  - Si los parquet llegan después de las 9:00, la corrida programada se detiene y hay que lanzarla a mano.
- **Llave.** `ingest` se detiene si falta algún SKU que ya estaba cargado, o si más del 1 % de los existentes cambia de `estilo` o `estilo_color`, como pasaría con una llave regenerada.
  - El catálogo se edita, así que los cambios sueltos de estilo y los de jerarquía solo van al log, con su conteo.
- **Parámetros del job**, vacíos en cada corrida normal:
  - `aceptar_drift=<origen>`: sigue con un drift extremo ya revisado;
  - `reentrenar=si`: entrena y pronostica aunque no toque, y se salta el freno.
- **CD a mano** (`gh workflow run cd.yml --ref main -f accion=<acción>`):
  - `desplegar`: solo despliega;
  - `correr`: el job como cada lunes, unos 5 min, o 41 si reentrena;
  - `reentrenar`: con `reentrenar=si`, unos 41 min;
  - `solo_export`: unos 3 min.
- **Cuota.** Free Edition no publica la cuota; si se pasa, el cómputo se apaga el resto del día.
  - En dev, el 29-oct-2026: una semana sin reentrenar tardó 5,3 min, una con reentrenamiento 41,1 min (train 35,9) y una frenada 1,5 min.
  - El 28-oct-2026 corrieron cerca de 2 horas de jobs sin llegar al límite.
  - Dev y prod comparten la cuota: ninguna prueba pesada en dev un lunes.
  - Si una corrida falla por cuota, se repite al día siguiente: el freno la deja pasar.

## Predicción y resultados

- **Tabla `predicciones`, en piezas.**
  - Cada fila es una serie en un periodo: `granularidad` = `semana` (periodos 1 a 13) o `ventana` (periodos 1 a 3, de 31, 30 y 30 días).
  - Cada corrida se agrega a la tabla con `fecha_corrida`, `origen`, `modelo` y `modelo_version`.
  - **Solo crece:** tiene `delta.appendOnly = true`, así que Delta rechaza reemplazarla, borrar filas o editarlas. Para borrar una corrida de prueba hay que quitar la propiedad a propósito.
  - La corrida del 2026-10-28 16:28:09 en `tesis_dev`, que se mide en enero de 2027, tiene además un respaldo en `workspace.tesis_dev.predicciones_respaldo_20261028`, también con `appendOnly`.
- **Datos de la página.** La tarea export escribe en `/Volumes/workspace/<schema>/raw/output/`:
  - `datos/<origen>/results.json`: uno por origen de `predicciones`, con su última corrida. Su esquema está en `src/common/results_schema.py` (`version_esquema` 1).
  - `datos/manifest.json`: una entrada por origen, en el formato 1 del README de `tesis-mlops-web`. `validar_manifest` sigue las reglas de `revisarManifest` de la web (`docs/js/datos.js`); si la web las cambia, se cambian aquí también.
  - `results.json`: el mismo archivo del origen más reciente.
  - En cada corrida vuelve a generar todos los orígenes desde las tablas. Valida todo en memoria antes de escribir: si algo falla, no escribe nada.
- **El pronóstico de cada origen se arma solo con datos hasta su `datos_hasta`.**
  - Venta real: las 52 semanas hasta el origen. Las posteriores no entran en el factor del índice ni en la elección de las 20 series.
  - Pronóstico: la última corrida de ese origen.
  - Métricas: las de la corrida de `metricas_modelos` que registró la versión que pronosticó, cuyo `run_id_modelo` es el run de esa versión en Unity Catalog. Nunca las de un train posterior.
  - `pronostico_anterior`: en cada una de las 52 semanas reales, el pronóstico que estaba vigente. Es el de la última corrida del origen anterior más reciente que la cubría, o nulo si no hubo.
- **Después del origen solo entra la evaluación,** hasta `real_hasta`, el último domingo con datos:
  - `real_horizonte`: la venta real de las semanas pronosticadas que ya cerraron, nula en las demás;
  - `ventanas_real`: la de cada mes cerrado, desde la venta diaria, como el error.
  - En las semanas sin reentrenamiento, el pronóstico publicado no cambia: solo se agregan esas semanas reales.
  - Al regenerar un origen viejo, el archivo sale igual salvo `generado`, `real_hasta`, `real_horizonte` y `ventanas_real`. Cada origen queda fijo cuando cierran sus 13 semanas.
  - Si un refresco corrige la venta de semanas pasadas, la venta real de un origen viejo cambia al regenerarlo. El pronóstico y las métricas no cambian.
- **Los campos nuevos van en la versión 1.** Son `real_hasta`, `real_horizonte`, `ventanas_real` y `pronostico_anterior`.
  - Ningún campo anterior cambia de significado, y la página ignora los que no conoce.
  - `validar` los exige, así que `generar_ejemplo.py` de la web tiene que agregarlos.
- **La página es pública: nunca lleva piezas.** La venta real y el pronóstico van como índice, donde 100 = la venta semanal promedio del total en las 52 semanas mostradas. El factor es el mismo para todas las series, y el MAE y el RMSE se escalan con él.
  - El índice va con 1 decimal. Con más decimales, cada pieza se vería como un escalón fijo y se podría reconstruir la escala.
- **Publicación.** `pages.yml` descarga `output/datos/` de prod (siempre `tesis`), lo valida con `python -m common.results_schema` y lo publica en GitHub Pages.
  - Corre cuando el CD lanzado a mano sobre main termina bien, los lunes a las 11:00 de Bogotá (dos horas después del job, que GitHub no ve), y a mano desde main. Los merges solo despliegan, así que no publican.
  - Para publicar sin correr el job completo: `gh workflow run cd.yml --ref main -f accion=solo_export` (unos 3 min de cuota).
  - **En un repo público, GitHub desactiva el schedule después de 60 días sin actividad.** Hace falta algún commit antes; por eso PLAN.md pide uno hacia el 15-dic.
- **La web depende de estos nombres.** `herramientas/generar_ejemplo.py` de `tesis-mlops-web` importa de este repo `results_schema.validar(resultados)` (con un solo argumento), `results_schema.VERSION`, `ESCALA` y `BASE_INDICE`, `features.HORIZONTE`, `metrics.METRICAS` y `metrics.VENTANAS_DIAS`. No se cambian sus nombres ni la firma de `validar`.

## Monitoreo y drift

Los umbrales se fijaron el 28-oct-2026, antes de ver resultados en vivo, con la calibración histórica de `make calibrar-drift` (137 orígenes semanales de mar-2024 a oct-2026, sin entrenar). **No se cambian después de ver la venta real.** Están en `src/common/monitoreo.py`.

- **Drift de datos.** Lo revisa la tarea `monitoreo_datos`, entre features y train, y lo agrega a la tabla `monitoreo_datos`, de solo agregar, con una fila por corrida × revisión.
  - **PSI** de cada variable en el origen contra la ventana de entrenamiento del champion, en los deciles de la referencia y sin nulos.
    - Variables: `rezago_1`, `media_4`, `media_13`, `media_52`, `tendencia`, `crecimiento_anual`, `estilos_color_con_venta_8`, `lanzamientos_8`, `stock_cierre` y `disponibilidad`.
    - Aviso desde 0,25 y detención desde 1,0.
    - **`tendencia` es solo un dato**, sin aviso ni detención. Sube después de cada evento y llega a 1,73 en orígenes normales. Sin ella, ningún origen normal pasa de 0,68.
    - El único origen que se habría detenido es el 24-feb-2025, con 1,72 en `disponibilidad`. Fue una semana después del inicio del stock, así que se comparó contra una sola semana.
    - Roturas simuladas: una semana sin venta da 5,5; el stock en cero, 8,6; la venta en docenas, de 1,5 a 2,4.
    - Una semana con las unidades dobladas da de 0,7 a 1,0: queda en aviso, y en detención solo si llega a 1,0.
    - El stock de solo la mitad de los SKU da de 0,1 a 0,2, así que el PSI no lo ve. Lo detectan las coberturas de SKU con stock.
  - **Nulos** de cada variable en el origen contra los últimos 13 orígenes de la ventana.
    - Aviso si la parte de nulos sube 10 puntos o más. El aumento más alto de la historia fue de +7,7 puntos.
    - Detención si la variable llega 100 % nula y en esos 13 orígenes tenía datos.
  - **Cobertura** de la semana del origen contra la semana más baja de las 4 anteriores: SKU con stock al cierre (de `panel_semanal`) y filas de venta, es decir, SKU × día con venta.
    - Aviso en 0,75 o menos y detención en 0,6 o menos.
    - Se compara con la semana más baja y no con el promedio. Después de un Hot Sale, contra el promedio, las filas de venta bajan hasta 0,25. Contra la más baja, el mínimo de la historia es 0,70 en filas de venta y 0,88 en SKU con stock.
  - **Cobertura del mes.** Con el refresco mensual llegan 4 o 5 semanas a la vez. Si todas vienen rotas, la cobertura de la semana las compara entre sí y no lo ve; el PSI solo da aviso, con una mediana de 0,27 si todo el mes llega a la mitad.
    - Por eso cada semana nueva se compara también contra las 4 semanas que ya revisó la corrida anterior, hasta su origen. Cuenta la semana nueva más baja contra la más baja de las revisadas.
    - Aviso en 0,75 o menos y detención en 0,6 o menos, como en la semana.
    - **Corrida anterior:** el último origen de `monitoreo_datos` cuya corrida no tuvo extremos o los aceptó, o el último origen de `predicciones`, porque una corrida que se detuvo no pronostica.
      - En la primera corrida con monitoreo, como la de enero en prod, cuenta el último origen de `predicciones`. Si no hay ninguno, las 4 semanas hasta el origen − 4 semanas.
    - **Con la corrida semanal, la corrida anterior es la de la semana pasada.** Se calibró antes de la primera corrida semanal (29-oct-2026), sin cambiar umbrales: con la corrida anterior 1 semana antes, en 72 orígenes, el mínimo es 0,88 en SKU con stock y 0,77 en venta entre salidas, sin avisos ni detenciones. Una semana nueva a la mitad se detiene en 66 y en 38 de 69 orígenes.
    - **SKU con stock del mes:** el mínimo de la historia es 0,81, y 0,76 con la corrida anterior 13 semanas antes, como en enero; en ningún caso hubo aviso.
    - **Venta entre salidas de stock del mes.** Por semana, las unidades vendidas entre las salidas de stock. Las salidas son, por SKU y día, lo que bajó el stock desde el día anterior.
      - El conteo de filas de venta no sirve para el mes: la semana previa al Hot Sale cae sola a 0,50, igual que un mes a medias, y contra el año anterior cae a 0,16 porque el Hot Sale cambia de fecha.
      - Un evento mueve la venta y las salidas a la vez, pero una venta incompleta solo baja la venta.
      - En la historia, el mínimo es 0,70, con 1 o 2 avisos y ninguna detención con la corrida anterior 4 o 5 semanas antes, y 6 avisos con 13.
      - Un mes entero a la mitad se detiene en 60 de 69 orígenes; una sola semana a la mitad, en 41.
      - **Solo vale porque el stock viene de los movimientos del almacén, independiente de la venta.** Si algún día el stock se calculara desde la venta, las dos fallarían juntas y la razón no cambiaría.
      - Si a una semana nueva le faltan salidas de stock, queda `sin datos` y el log dice que la venta del mes no se verificó. No detiene el job, y `aceptar_drift` no la vuelve estable.
  - **Corte del stock.** El job se detiene si el stock atrasa el origen una semana o más.
    - features corta en la última semana completa de las dos tablas. Sin stock reciente no quedan nulos: el origen se queda atrás.
  - **Liberar el freno.** El job tiene el parámetro `aceptar_drift`, que recibe la fecha del origen.
    - Solo vale para ese origen y queda anotado en la tabla.
    - Con el stock atrasado, liberarlo repite el pronóstico desde el mismo origen.
- **Drift de desempeño.** Lo revisa la tarea `monitoreo_desempeno`, antes de la condición `toca_reentrenar`, y lo agrega a la tabla `monitoreo_desempeno`, de solo agregar, con una fila por corrida de `predicciones` × medición.
  - **Cada ventana se mide apenas cierra,** en la primera corrida después de su último día, con el formato estándar. La regla por ventana se fijó el 29-oct-2026, antes de que cerrara ninguna ventana en vivo.

    | Medición | Cuándo | Alerta | Solo dato |
    |---|---|---|---|
    | `mes_1` | al cerrar la ventana 1 | WAPE M1 arriba del máximo | MAPE M1, sesgo del mes 1 |
    | `mes_2` | al cerrar la ventana 2 | WAPE M2 arriba del máximo | MAPE M2, sesgo del mes 2 |
    | `total` | al cerrar la ventana 3 | WAPE total arriba del máximo, o sesgo total fuera del rango | MAPE total, WAPE tipo × mes |

  - **Rango:** el mínimo y el máximo por origen en los 8 orígenes del backtest donde se registró la versión que pronosticó. Salen del artefacto `predicciones_backtest.parquet` de su run de MLflow. Son los mismos de antes: la medición por ventana no agregó umbrales.
  - El sesgo de cada mes va solo como dato: no tiene un rango fijado antes de ver resultados, y como alerta daría más falsas alarmas.
  - **Hacia el 18-ene-2027, 6 mediciones:**
    - 12-oct: mes 1 el 23-nov, mes 2 el 21-dic y total el 18-ene;
    - 9-nov: mes 1 el 21-dic y mes 2 el 18-ene;
    - 7-dic: mes 1 el 18-ene.
  - **Revisión manual** si la misma medición tiene alerta en el origen y en el origen anterior medido. De cada origen cuenta su última corrida.
    - Se compara la misma medición y no dos seguidas cualquiera: el mes 1 y el mes 2 de un pronóstico vienen del mismo modelo y el mismo origen, así que un solo pronóstico malo pediría revisión.
    - Si el modelo está igual de bien que en el backtest, una alerta de mes tiene probabilidad de cerca de 1/9 (quedar arriba del peor de 8 orígenes), y dos seguidas, de cerca de 1/81.
  - Las alertas no detienen el job, pero adelantan el reentrenamiento (ver "Cadencia").
- **Sin reentrenamiento por drift de datos.** Un drift extremo de datos detiene el job, no reentrena.
- **Las tablas de monitoreo no guardan piezas,** ni el log las muestra: solo PSI, proporciones, fechas y nombres de variables.

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
  ingest.py            parquet del Volume → tablas bronze (Delta); freno sin domingo nuevo y revisión de la llave
  features.py          bronze → tabla de features
  monitoreo_datos.py   drift de datos del origen → tabla monitoreo_datos; detiene el job si es extremo
  monitoreo_desempeno.py  cada ventana cerrada contra el rango del backtest → tabla monitoreo_desempeno; decide si
                       toca reentrenar (task value reentrenar)
  train.py             backtest, MLflow, registro del champion en Unity Catalog (solo si toca)
  predict.py           pronóstico con el champion → tabla predicciones (solo si toca)
  export_results.py    datos de la página: un results.json por origen y manifest.json
  common/              funciones puras en pandas, con pruebas
    validation.py  features.py  metrics.py  backtest.py  modelos.py  results_schema.py  monitoreo.py
scripts/               solo en local: simular_monitoreo.py (calibración y demostración del monitoreo)
tests/                 pruebas (pytest) y tests/data/ (muestra.parquet)
databricks.yml         Asset Bundle: job semanal demanda_pipeline con la condición toca_reentrenar, targets dev y prod
.github/workflows/     ci.yml (PR), cd.yml (despliegue y corridas en prod), pages.yml (publica los datos)
data/                  dataset local (no se versiona)
```
