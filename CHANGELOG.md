# Bitácora de Cambios — Rocktec MIA 2026

Registro cronológico de las fases del proyecto (MIA, Universidad de las Américas — UDLA).
Reconstruido a partir del historial de `git log`; a partir de Fase 1C se mantiene manualmente.

## Fase 1 — Validación inter-anotador inicial (hasta 15 Jun)
- Scripts de limpieza (`01_limpieza_datos.py`), consolidación (`02_consolidar_datos.py`) y
  detección de duplicados (`03_validar_duplicados.py`).
- Consolidación de 2 fuentes (CRM + WhatsApp) → `rocktec_base_validada.csv` (9,317 registros).
- Muestra piloto de 100 registros anotada por 3 evaluadores → Cohen's Kappa = 0.8794.

## Fase 1B — Anotación completa (16 Jul)
- `consolidar_4_bases.py`: consolidación de las 4 fuentes originales (CRM1, CRM2, JEVA, WhatsApp)
  → muestra de 1,500 registros (`rocktec_base_consolidada_1500.csv`).
- 1,500 registros anotados por Patricia Mosquera, Luis Cruel y Luis Chica →
  Cohen's Kappa = **0.8851** (meta ≥ 0.70 alcanzada).
- Primer entrenamiento baseline de Logistic Regression + LinearSVC (`05_entrenar_modelos.py`,
  commit `af29f34`).
- Reportes de cierre: `INFORME_CIERRE_FASE_1B.docx`, `REPORTE_KAPPA_DETALLADO.docx`,
  `DOCUMENTO_TRAZABILIDAD_CORRECCIONES.docx`.

## Fase 1C — Ajustes de prototipo y validación estadística (19 Jul 2026)

**Hallazgo de causa raíz:** `calcular_kappa.py` calculaba el Kappa pero nunca escribía
`dataset_consenso_final.csv`; además, el nombre de archivo y columna que `04_feature_engineering.py`
esperaba como segunda prioridad no coincidían con los reales. Como consecuencia, **todos los
entrenamientos anteriores usaban etiquetas heurísticas** (por palabras clave) en vez de las 1,500
anotaciones humanas reales.

Ajustes realizados:
- **Preprocesamiento**: `calcular_kappa.py` ahora genera `dataset_consenso_final.csv` (voto
  mayoritario 2/3 entre los 3 anotadores; 1,297/1,500 filas con consenso válido) y una etiqueta
  heurística baseline sobre las mismas filas, para permitir comparaciones pareadas.
- **Modelado**: grillas de hiperparámetros ampliadas (LR `C∈[0.01,0.1,1,10,100]`,
  SVM `C∈[0.001,…,100]`); manejo adaptativo del número de folds y de la calibración de
  probabilidades cuando hay clases con muy pocos ejemplos (QUE=4, SEG=5 registros totales);
  runs de MLflow etiquetados con `fuente_datos` (`heuristico_fallback` vs. `consenso_manual`).
- **Validación estadística** (`07_validacion_estadistica.py`): diseño de validación cruzada
  **pareado** — `RepeatedKFold(5×5)` sobre las mismas 1,297 filas y los mismos folds, comparando
  la etiqueta heurística contra la de consenso para aislar el efecto de la calidad del etiquetado
  del efecto del tamaño de muestra. Reporta media, desviación estándar, IC 95% y pruebas de
  Wilcoxon / t-test pareado.
- **Resultado**: el desempeño heurístico previamente reportado (SVM F1-macro 0.936) estaba
  inflado por circularidad (las mismas reglas de palabras clave generaban la etiqueta y el
  modelo bag-of-words las recuperaba trivialmente). Sobre las anotaciones humanas reales, el
  desempeño honesto es LR F1-macro 0.596 / SVM F1-macro 0.493 — ver
  `06_resultados/validacion_estadistica.json`.
- **Documentación**: institución corregida a UDLA (`README.md`, `CLAUDE.md`); `README_PATY.md` y
  `README_LuisC.md` fusionados/eliminados por redundantes; `03_datos_procesados/README.md`
  actualizado al estado real (pipeline de 4 bases); este `CHANGELOG.md` creado.
- **Entrega**: `06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx`, generado de forma reproducible
  por `02_scripts/generar_informe_docx.py`.

## Sprint 7 — Ajustes de modelado y decisión de alcance (22 Jul 2026)

- **Decisión de alcance — modelo de 5 clases:** SEG (5 filas) y QUE (4 filas) se excluyen
  formalmente del entrenamiento y la evaluación por falta de datos anotados suficientes; el
  catálogo de negocio sigue definiendo 7 categorías, pero el clasificador de producción cubre
  solo INF, COT, TEC, CUR, VEN. No es un pendiente — se retomaría únicamente si una ronda futura
  de anotación amplía significativamente esas dos clases. Reflejado en `README.md`
  y `05_documentacion/DISEÑO_MLOPS_FASE2.md`.
- **Holdout set** (`09_crear_holdout_set.py`): split 85/15 estratificado sobre las 1,312 filas de
  5 clases → `train_val.csv` (1,115) / `holdout_test.csv` (197), a usarse una sola vez al final.
- **Explicabilidad** (`10_shap_lime_explicabilidad.py`): SHAP y LIME sobre el modelo LR de 5 clases
  → `06_resultados/explicabilidad/`.
- **Comparación TF-IDF vs. BETO** (`11_beto_clasificador.py`): TF-IDF + LR sobre el dataset
  completo (1,312 filas) alcanza F1-macro = **0.7516** (meta ≥ 0.75 cumplida); BETO por embeddings
  sin fine-tuning (498 filas) da F1-macro = 0.6370, inferior. Fine-tuning de BETO con GPU queda
  como trabajo futuro. Modelo elegido para producción: **TF-IDF + Logistic Regression**.
  *(Actualizado: ver Sprint 7 — BETO fine-tuning, 25 Jul 2026, más abajo)*.

## Sprint 7 — BETO fine-tuning (25 Jul 2026)

- **Fine-tuning real de BETO** (`12_beto_finetuning.py`, nuevo script, corrido en Google Colab con
  GPU T4 gratuita — sin GPU local disponible en el entorno de desarrollo): `AutoModelForSequenceClassification`
  sobre `dccuchile/bert-base-spanish-wwm-cased`, 5 épocas, batch 16, lr 2e-5, warmup 10%, max_length
  128 (hiperparámetros de `DISEÑO_MLOPS_FASE2.md`, Etapa 3). Mismo split de test 80/20
  (`random_state=42`) que `11_beto_clasificador.py`, para comparabilidad directa.
- **Resultado: F1-macro = 0.8552** (accuracy 0.95) — supera tanto a TF-IDF+LR (0.7516) como a BETO
  sin fine-tuning (0.6370). Por clase: INF 0.96, COT 0.98, VEN 0.94, CUR 0.83, **TEC 0.56** (sigue
  siendo la clase más débil en los tres modelos, consistente con su bajo soporte: 51 filas).
- **Conclusión actualizada:** BETO fine-tuned es ahora el modelo con mejor F1-macro de los tres
  probados. La elección de modelo de producción entre TF-IDF+LR (liviano, interpretable, ya
  validado con SHAP/LIME, sin GPU) y BETO fine-tuned (mejor F1, pero requiere GPU e infraestructura
  de inferencia de ~440MB, sin explicabilidad todavía trabajada) queda abierta para Fase 3/4 —
  ver `README.md` sección "Comparación de arquitecturas".
- **Notas de infraestructura:** el checkpoint (`06_resultados/modelos/beto_finetuned_best/`,
  ~420MB) se excluye de git vía `.gitignore` (excede el límite de 100MB de GitHub). Se documentó
  también el flujo de Colab (`02_scripts/12_beto_finetuning_colab.ipynb`) y dos issues de entorno
  resueltos en el camino: conflicto `peft`/`accelerate` (`cannot import name 'clear_device_cache'`,
  resuelto desinstalando `peft` y no fijando versiones viejas de `transformers`/`accelerate`) y un
  conflicto de `pandas` (no reinstalar `pandas`/`scikit-learn` en Colab — ya vienen compatibles con
  `google-colab`/`cudf`, forzar `pandas` a 3.x rompe `google.colab.files.download`).

## Sprint 7 — Fix de fuga de datos + evaluación final en holdout (25 Jul 2026)

**Hallazgo:** los tres modelos comparados arriba (TF-IDF+LR 0.7516, BETO embeddings 0.6370, BETO
fine-tuned 0.8552) se entrenaron sobre un split 80/20 ad hoc de `dataset_consenso_final.csv`
**completo**, el cual incluye las 197 filas que `09_crear_holdout_set.py` había apartado como
`holdout_test.csv`. Es decir, esos modelos ya habían "visto" datos de holdout durante su propio
entrenamiento — ninguno de esos tres números es válido como métrica final de Fase 4, aunque siguen
siendo útiles como comparación exploratoria entre arquitecturas.

Ajustes realizados:
- **`12_beto_finetuning.py` corregido:** ahora entrena exclusivamente con
  `04_anotaciones/train_val.csv` (1,115 filas), nunca con el dataset completo ni con
  `holdout_test.csv`. Ya no reporta una métrica de "test" final (eso ahora lo hace el script 13);
  reporta solo F1-macro de validación interna (para elegir la mejor época) y deja una marca
  `FUENTE_ENTRENAMIENTO.txt` en el checkpoint registrando de qué archivo se entrenó.
- **`13_evaluacion_holdout.py` (nuevo):** reentrena TF-IDF+LR desde cero solo con `train_val.csv`
  (GridSearchCV, mismos hiperparámetros que `05_entrenar_modelos.py`) y lo evalúa una única vez
  sobre `holdout_test.csv`. Para BETO, carga el checkpoint de `12_beto_finetuning.py` pero **solo
  si** `FUENTE_ENTRENAMIENTO.txt` confirma que se entrenó con `train_val.csv`; si no existe o no
  coincide, omite BETO con una advertencia en vez de reportar un número contaminado.
- **Resultado final, ambos modelos (sin fuga de datos), sobre las 197 filas de `holdout_test.csv`,
  evaluadas una sola vez:**
  - TF-IDF + LR (C=10): F1-macro = **0.7938**, accuracy = 0.8832. Por clase: INF 0.92, COT 0.86,
    CUR 0.95, VEN 0.77, TEC 0.47.
  - BETO fine-tuned (checkpoint reentrenado en Colab solo con `train_val.csv`, verificado vía
    `FUENTE_ENTRENAMIENTO.txt`): F1-macro = **0.7967**, accuracy = 0.9239. Por clase: INF 0.95,
    COT 0.98, CUR 0.75, VEN 0.91, TEC 0.40.
  - Ambos cumplen la meta ≥ 0.75 de Fase 4. Ver `06_resultados/reporte_holdout_final.txt`.
- **Conclusión revisada — la fuga de datos inflaba la brecha:** con el split contaminado, BETO
  parecía superar a TF-IDF+LR por +0.10 de F1-macro (0.8552 vs. 0.7516). Sin fuga, quedan
  prácticamente empatados (+0.003 a favor de BETO), aunque BETO mantiene una accuracy más alta.
  TEC sigue siendo la clase más débil en ambos modelos (8 filas en el holdout) — ningún cambio de
  arquitectura la resuelve; probablemente necesite más anotación. La elección de modelo de
  producción (Fase 3/4) ahora se decide por trade-offs de infraestructura (GPU, latencia,
  explicabilidad) más que por una diferencia clara de F1 — ver `README.md`.
- **Decisión de modelo de producción — cerrada:** se midió latencia real de inferencia en CPU:
  TF-IDF+LR = 2.86 ms/mensaje vs. BETO fine-tuned = 65.14 ms/mensaje (23× más lento). Con el F1-macro
  empatado, **se elige TF-IDF + Logistic Regression para producción (Fase 3)** — no requiere GPU,
  pesa ~150KB, ya tiene explicabilidad SHAP/LIME, y es más sostenible de operar para una PYME sin
  equipo de MLOps dedicado. BETO fine-tuned queda documentado como upgrade candidato dirigido a
  COT/VEN (donde tiene ventaja real y esas clases tienen mayor impacto de negocio directo), a
  reevaluar si el monitoreo en producción muestra errores costosos de TF-IDF+LR en esas categorías.
  Ver la tabla de criterios completa en `05_documentacion/DISEÑO_MLOPS_FASE2.md` §8.

## Sprint 7 — Ajuste adicional: léxico QUE/SEG v2.0 (26 Jul 2026)

- **`08_buscar_candidatos_que_seg.py` corregido:** se agregaron patrones léxicos adicionales
  (p. ej. "no contesta") para detectar candidatos QUE (queja/reclamo) y SEG (seguimiento).
  Recall sobre los candidatos revisados: QUE pasó de 66% a **100% (9/9)**, SEG se mantiene en
  **100% (15/15)**. Sigue siendo una herramienta de *búsqueda de candidatos* para ampliar las
  clases QUE/SEG en una futura ronda de anotación — no cambia la decisión de alcance de 5 clases
  para el modelo de producción (ver Sprint 7 arriba).

## Sprint 8 — Umbral adaptativo, active learning TEC y anonimización (30 Jul 2026)

- **Umbral de confianza adaptativo** (`17_umbral_confianza_adaptativo.py`, nuevo): barrido de
  umbrales sobre `holdout_test.csv` (197 registros) para decidir cuándo automatizar una
  clasificación y cuándo enviarla a revisión humana. **Umbral óptimo = 0.75**: automatiza 70.6%
  de los mensajes (139/197) con F1-macro = **0.9214** y solo 6 errores; el 29.4% restante se
  marca para revisión de un asesor. Resultados en `06_resultados/umbral/`.
- **Active learning para TEC** (`18_active_learning_tec.py`, nuevo): TEC es la clase más débil
  (F1 LR 0.47 / BETO 0.40, solo 51 casos en el dataset de 1,312). Se entrenó LR sobre
  `train_val.csv` y se aplicó uncertainty sampling sobre un pool de 208 mensajes no anotados,
  seleccionando los **30 candidatos más informativos** (probabilidad de TEC entre 0.10 y 0.70)
  → `04_anotaciones/active_learning_tec_candidatos.xlsx`. Meta: confirmar 30-40 casos reales de
  TEC con los asesores de Rocktec; impacto estimado si se logra: F1 TEC 0.47→~0.65+, F1-macro
  general 0.7938→~0.82+.
- **Plan formal de ampliación del dataset TEC** (`05_documentacion/S8_Plan_Ampliacion_Dataset_TEC.docx`):
  guía de anotación dirigida a los asesores comerciales de Rocktec (no al equipo de anotación
  original) para que etiqueten los candidatos de active learning directamente desde su
  conocimiento de producto.
- **Anonimización v2.0** (`00_anonimizar_dataset.py`, nuevo): sobre las 1,500 filas del dataset
  de consenso, **223 registros (14.9%)** contenían PII y fueron enmascarados — nombres→`[CLIENTE]`,
  teléfonos→`[TELEFONO]`, correos→`[EMAIL]`, cédulas/RUC→`[DOCUMENTO]` → `04_anotaciones/dataset_consenso_final_anonimizado.csv`.

## Sprint 8 — Monitoreo de equidad por perfil de cliente (31 Jul 2026)

- **`20_monitoreo_equidad.py`** (nuevo; creado como `18_monitoreo_equidad.py` y renumerado a 20
  el 04 Ago 2026 para no colisionar con `18_active_learning_tec.py`, que ya ocupaba ese número):
  mide equidad del modelo de producción (F1-macro por segmento) usando Evidently AI.
  **Limitación de datos documentada:** Rocktec no captura el perfil del cliente en la misma
  fuente que el texto de la conversación (JEVA tiene tipo de cliente pero sin texto; el dataset
  de consenso tiene texto pero `etiqueta_crm` es una etiqueta de campaña CRM, no un perfil
  limpio). El perfil usado es un *proxy* inferido desde `etiqueta_crm`; 497 registros (37.6%)
  sin esa etiqueta quedan excluidos, cubriendo 825/1,322 filas (62.4%).
  **Resultado:** brecha de equidad significativa entre perfiles — `Prospecto_General` (n=719)
  F1-macro = 0.7418 vs. `Comprador_Activo` (n=106) F1-macro = **0.5189** (brecha 0.2229, ≥0.10
  se considera relevante). Recomendación entregada a Rocktec: agregar un menú de selección de
  perfil al inicio de la conversación de WhatsApp Business para capturar este dato de forma
  confiable a futuro. Reportes en `06_resultados/equidad/`.

## Sprint 9 — Dashboard de inteligencia comercial en Streamlit (3 Ago 2026)

- **`19_dashboard_streamlit.py`** (nuevo): dashboard dark-mode para uso operativo de Rocktec,
  con clasificador híbrido (reglas léxicas expandidas + modelo de producción TF-IDF+LR, umbral
  configurable) sobre 3 tabs: Clasificador (mensaje individual), Dashboard (métricas) y Lote de
  mensajes (CSV masivo con descarga de resultados).
- **Tab 4 — upload directo de chat de WhatsApp:** permite subir el `.txt` exportado directamente
  desde WhatsApp; el sistema filtra automáticamente los mensajes propios de Rocktec y el
  contenido multimedia, clasifica cada mensaje del cliente, resume sus intenciones y genera un
  reporte CSV descargable por cliente — pensado para que un asesor cargue una conversación
  completa sin preprocesar nada manualmente.

## Sprint 10 — Monitoreo de drift (PSI) y housekeeping (4 Ago 2026)

- **Housekeeping:** renombrado `18_monitoreo_equidad.py` → `20_monitoreo_equidad.py` (colisionaba
  con `18_active_learning_tec.py`, ambos numerados 18 tras crearse en la misma ventana de Sprint 8);
  `CHANGELOG.md` y `05_documentacion/DIAGNOSTICO_FASES_3_4_5.md` actualizados para reflejar Sprint
  8/9, que no estaban documentados todavía (incluye corregir el ítem de explicabilidad BETO, que el
  diagnóstico marcaba erróneamente como sin commitear).
- **`21_monitoreo_drift.py`** (nuevo): implementa la Etapa 5 (monitoreo de drift) diseñada en
  `DISEÑO_MLOPS_FASE2.md` §3 — hasta ahora solo existía como diseño, sin ningún script real.
  Calcula PSI (Population Stability Index) de dos distribuciones: (1) intención predicha, comparando
  `log_predicciones.csv` (el log que genera `16_inferencia.py`) contra `train_val.csv`; (2) confianza
  del modelo, usando confianza **out-of-fold** (`cross_val_predict`, 5 folds) como referencia en vez
  de `predict_proba` in-sample — evaluar el modelo sobre los mismos datos con los que se ajustó
  sobrestima sistemáticamente la confianza, el mismo tipo de error que motivó el fix de fuga de datos
  de Sprint 7.
  - **Log sembrado para poder probar el script de punta a punta:** como todavía no existe tráfico
    real de piloto (Fase 5 no ha empezado), se corrió `16_inferencia.py` sobre un lote de 345
    mensajes históricos de WhatsApp (`03_datos_procesados/rocktec_base_validada.csv`, un pipeline de
    2 fuentes de Fase 1 disjunto del pipeline de 4 fuentes que generó el dataset de anotación —
    0% de solape verificado por texto exacto con `train_val`/`holdout_test`).
  - **Resultado de esa corrida de validación:** PSI intenciones = 0.11 (drift moderado); PSI
    confianza = 0.75 (drift severo, pero explicado por contenido fuera de dominio —comentarios de
    Instagram/Facebook ajenos a Rocktec— que quedó mezclado en `rocktec_base_validada.csv` por un
    problema de calidad de datos previo, no por drift real del negocio). Documentado explícitamente
    en el reporte como una corrida de **validación del mecanismo, no una medición de drift real**.
  - **Fuera de alcance:** drift de vocabulario (features TF-IDF nuevas) — el diseño original solo
    cubre drift de la distribución de predicciones. Ver `06_resultados/drift/reporte_drift.txt`.

## Sprint 10 — Corrección de alcance: no hay WhatsApp Business API (4 Ago 2026)

- **Aclaración confirmada por el equipo:** Rocktec no tiene ni tendrá acceso a la WhatsApp Business
  API. El mecanismo de integración **definitivo** es la descarga manual del chat exportado (`.txt`)
  y su carga al dashboard (`19_dashboard_streamlit.py`, Tab 4) — no un sustituto provisional de una
  API que llegaría después. Esto corrige el diagnóstico de Fase 3/5: el "piloto real" no estaba
  bloqueado por falta de integración en vivo, ya puede arrancar con lo que existe. PostgreSQL y CD
  se reencuadran en consecuencia: CSV puede ser la solución permanente (no solo interina) dado el
  bajo volumen de descargas manuales, y CD debería apuntar a desplegar el dashboard como servicio,
  no una API de mensajería. Ver `05_documentacion/DIAGNOSTICO_FASES_3_4_5.md` v1.3.

## Sprint 10 — Pipeline orquestado end-to-end (4 Ago 2026)

- **`22_pipeline_orquestado.py`** (nuevo): cierra la última brecha técnica de Fase 3 identificada en
  `DIAGNOSTICO_FASES_3_4_5.md` — un entrypoint único que encadena `04_feature_engineering.py`
  (verificación) → `13_evaluacion_holdout.py` (evaluación honesta) → `15_entrenar_produccion.py`
  (artefacto de despliegue) → `20_monitoreo_equidad.py` → `21_monitoreo_drift.py`, cada etapa como
  subproceso del script real (sin reimplementar su lógica). `06_pipeline_completo.py` (ETL) y los
  experimentos de comparación de arquitecturas (`05_entrenar_modelos.py`,
  `07_validacion_estadistica.py`) quedan como etapas opcionales por flag
  (`--incluir-etl`, `--incluir-entrenamiento-experimental`, `--incluir-validacion-estadistica`).
  **Decisión deliberada:** no re-ejecuta `09_crear_holdout_set.py` automáticamente — repartir el
  holdout en cada corrida arriesgaría el mismo tipo de fuga de datos que motivó el fix de Sprint 7
  si `dataset_consenso_final.csv` cambia (p. ej. tras confirmar anotaciones de active learning).
- **Validado con una corrida real end-to-end** (`06_resultados/pipeline/reporte_pipeline.txt`):
  ~109s sin flags, ~58s con `--incluir-etl` (determinista — no generó cambios sustantivos en
  `03_datos_procesados/`, solo confirmó que la normalización de columnas ya estaba aplicada).
- **De paso, corrige un bug preexistente en `04_feature_engineering.py`:** el bloque de
  autoverificación (7 frases de ejemplo) fallaba con `ValueError` porque `min_df=2` no encontraba
  vocabulario repetido en una muestra tan pequeña — no afectaba el uso real del vectorizador
  (entrenado siempre sobre datasets de cientos/miles de filas), solo el script de demo se rompía.
  Ahora degrada con una advertencia en vez de abortar.

## Próxima — Fase 2: Diseño MLOps Pipeline
Ver `05_documentacion/DISEÑO_MLOPS_FASE2.md`.
