# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic NLP/ML project (MIA 2026, Universidad de las Américas — UDLA) building an intent classification platform for WhatsApp Business conversations from Rocktec, an Ecuadorian decorative concrete company. The pipeline transforms raw CRM/WhatsApp data into a labeled dataset, then trains classifiers to predict one of 7 intent codes: **INF, COT, TEC, CUR, VEN, SEG, QUE**.

## Environment Setup

```bash
pip install -r requirements.txt
```

Python packages: pandas, numpy, scikit-learn, spacy, pysentimiento, nltk, openpyxl, statsmodels.

## Running Scripts

**Important: scripts have different working directory requirements.**

Scripts in `02_scripts/` that use relative paths like `../01_datos_crudos` must be run from inside `02_scripts/`:
```bash
cd 02_scripts
python 01_limpieza_datos.py        # Clean raw data → 03_datos_procesados/
python 02_consolidar_datos.py      # Consolidate CRM + WhatsApp
python 03_validar_duplicados.py    # Detect/remove duplicates
python 06_pipeline_completo.py     # Full sequence: clean → consolidate → validate
```

`consolidar_4_bases.py` uses `Path('01_datos_crudos/')` so it must run from the **repo root**:
```bash
python 02_scripts/consolidar_4_bases.py
# Output: 04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx
```

Calculate Cohen's Kappa for inter-annotator agreement and generate the consensus dataset (run from repo root):
```bash
python 02_scripts/calcular_kappa.py "04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500 1ERA ETIQUETA.xlsx"
# Output: 04_anotaciones/dataset_consenso_final.csv (written automatically when Kappa >= 0.70)
```
`generar_dataset_consenso()` computes a 2/3 majority vote per row across `PATRICIA`/`LUIS_CRUEL`/`LUIS_CHICA`
(rows with 3-way disagreement are labeled `SIN_CONSENSO` and dropped downstream), and also attaches a
heuristic baseline label (`MapeoIntenciones.mapear_desde_texto`) on the *same* rows, so baseline vs.
adjusted models can be compared on identical data in `07_validacion_estadistica.py`.

Verify feature engineering pipeline (no data required):
```bash
python 02_scripts/04_feature_engineering.py
```
`cargar_dataset()` returns `(df, fuente)`; it prioritizes `04_anotaciones/dataset_consenso_final.csv`
(real annotations) over the heuristic fallback `03_datos_procesados/rocktec_base_validada.csv`.

Train LR + SVM (+ optionally BETO) with MLflow tracking (run from repo root):
```bash
python 02_scripts/05_entrenar_modelos.py             # LR + SVM + BETO
python 02_scripts/05_entrenar_modelos.py --skip-beto  # LR + SVM only (no GPU needed)
mlflow ui --backend-store-uri mlruns                  # View results at localhost:5000
```
If mlflow raises `filesystem tracking backend ... maintenance mode` (mlflow >= 2.11 with a local
`./mlruns` store), set `MLFLOW_ALLOW_FILE_STORE=true` in the environment before running.

Run the paired statistical validation (baseline heurístico vs. ajustado consenso, same folds — run from repo root):
```bash
python 02_scripts/07_validacion_estadistica.py
# Output: 06_resultados/validacion_estadistica.json + 06_resultados/modelos/validacion_cv_boxplot.png
```

Regenerate the Word deliverable from the JSON results + git log (run from repo root):
```bash
python 02_scripts/generar_informe_docx.py
# Output: 06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx
```

## Data Architecture

Raw Excel files in `01_datos_crudos/` are **gitignored** (too large). Processed CSVs in `03_datos_procesados/` are tracked.

**4 source files → 1,500-record annotation dataset:**

| Source | Records | Content |
|--------|---------|---------|
| `Copia de clienty-prospectos 1.xlsx` | ~5,000 | CRM contacts/metadata |
| `Copia de clienty-prospectos 2.xlsx` | ~3,143 | CRM contacts/metadata |
| `ROCKTEC - JEVA base datos.xlsx` | ~1,155 | Master company data |
| `base_maestra_raw_total_rocktec.xlsx` | ~5,676 | **Primary**: WhatsApp + Instagram conversations (sheet: `BASE_TOTAL_RAW`) |

The CRM files use sheet name `'Worksheet'`. The base_maestra file uses sheet `'BASE_TOTAL_RAW'`.

**Data flow:**
1. `01_datos_crudos/` → raw Excel → `03_datos_procesados/crm_limpio.csv` + `whatsapp_limpio.csv`
2. Cleaned CSVs → `rocktec_base_consolidada.csv` → deduplication → `rocktec_base_validada.csv`
3. `consolidar_4_bases.py` creates `04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx` with blank columns `PATRICIA`, `LUIS_CRUEL`, `LUIS_CHICA`, `NOTAS` ready for manual annotation

## Pipeline Logic

`06_pipeline_completo.py` contains three inline classes that mirror the standalone scripts:
- `LimpiadoDatos` – normalizes column names (lowercase/underscores), drops all-null rows, applies regex text cleaning
- `MapeoIntenciones` – keyword-based heuristic to assign intent codes from text or CRM fields (used as initial labels before manual annotation)
- `ConsolidadorDatos` – merges CRM and WhatsApp records into unified schema; shuffles with `random_state=42`; adds empty annotator columns
- `DetectorDuplicados` – exact deduplication on `(texto_conversacion, nombre_cliente, canal)`

## Intent Codes

| Code | Meaning |
|------|---------|
| INF | General information request |
| COT | Quote / budget |
| TEC | Technical consultation |
| CUR | Course / training inquiry |
| VEN | Sale confirmation |
| SEG | Follow-up on prior request |
| QUE | Complaint / claim |

## Key Metrics

Minimum acceptable inter-annotator Cohen's Kappa: **≥ 0.70**. Achieved: 0.8794 (Fase 1, 100 records) → 0.8851 (Fase 1B, 1,500 records).
Target model F1-score: **≥ 0.75** (Fase 4).

Early runs trained LR/SVM on heuristic keyword-based labels (`rocktec_base_validada.csv`) because
`dataset_consenso_final.csv` did not yet exist — those runs are preserved in `mlruns/` as the
**baseline** (`fuente_datos=heuristico_fallback`; F1-macro: LR 0.716, SVM 0.936). Once the real
1,297-row human-consensus dataset was wired in (**ajustado**, `fuente_datos=consenso_manual`),
F1-macro dropped to LR 0.596 / SVM 0.493 on a single train/test split — see
`06_resultados/validacion_estadistica.json` for the paired cross-validation comparison and
`06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx` for the full analysis. The heuristic numbers were
inflated because the same keyword rules used to generate those labels are trivially recoverable by
a bag-of-words model; the consensus numbers are the honest current baseline for Fase 2+.

## Team Roles

- **Patricia Mosquera** – data annotation, EDA, preprocessing, PostgreSQL
- **Luis Cruel** – ML models, MLflow experiments, metrics
- **Luis Chica** – architecture, UML, CI/CD, drift monitoring
