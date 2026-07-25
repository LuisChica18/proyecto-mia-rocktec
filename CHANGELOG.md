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

## Próxima — Fase 2: Diseño MLOps Pipeline
Ver `05_documentacion/DISEÑO_MLOPS_FASE2.md`.
