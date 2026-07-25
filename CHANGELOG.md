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

## Próxima — Fase 2: Diseño MLOps Pipeline
Ver `05_documentacion/DISEÑO_MLOPS_FASE2.md`.
