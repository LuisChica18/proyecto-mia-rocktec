# Diseño MLOps — Fase 2
## Plataforma de Clasificación de Intenciones — Rocktec MIA 2026

**Versión:** 1.0  
**Fecha:** Julio 2026  
**Responsable de arquitectura:** Luis Chica (A3)  
**Estado:** Diseño aprobado — implementación en progreso

---

## 1. Contexto y Objetivos

### Problema
Rocktec recibe mensajes por WhatsApp Business e Instagram sin clasificar. El equipo de ventas atiende sin priorización, perdiendo oportunidades de cotización y seguimiento.

### Solución
Pipeline MLOps que clasifica automáticamente la intención de cada mensaje entrante en una de 5 categorías modeladas (de un catálogo de 7 definidas), con trazabilidad completa de experimentos y monitoreo de drift.

### Métricas de éxito
| Métrica | Umbral |
|---------|--------|
| F1-macro (test, 5 clases modeladas) | ≥ 0.75 |
| Cohen's Kappa (inter-anotador) | ≥ 0.70 |
| Cobertura de clases (F1 por clase) | ≥ 0.60 en todas las clases modeladas |

---

## 2. Catálogo de Intenciones

**Clases modeladas (5)** — el clasificador se entrena y evalúa solo sobre estas:

| Código | Nombre | Ejemplo real |
|--------|--------|--------------|
| **INF** | Información General | "¿Qué productos tienen para pisos?" |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta el microcemento para 30 m²?" |
| **TEC** | Consulta Técnica | "¿Cómo se aplica sobre cerámica existente?" |
| **CUR** | Consulta de Cursos | "¿Cuándo es el próximo taller de concreto decorativo?" |
| **VEN** | Venta / Confirmación | "Confirmo la compra, envíenme la factura" |

**Clases fuera de alcance (2)** — definidas en el catálogo de negocio pero excluidas del modelo por
falta de datos anotados (`dataset_consenso_final.csv`: SEG = 5 filas, QUE = 4 filas de 1,297 válidas).
Decisión de alcance para esta fase, no un pendiente:

| Código | Nombre | Ejemplo real |
|--------|--------|--------------|
| **SEG** | Seguimiento | "¿En qué estado está mi cotización del 15 de junio?" |
| **QUE** | Queja / Reclamo | "El producto llegó dañado, quiero hacer un reclamo" |

---

## 3. Arquitectura del Pipeline (5 Etapas)

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE MLOPS ROCKTEC                      │
└─────────────────────────────────────────────────────────────────┘

 ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
 │ ETAPA 1  │ → │   ETAPA 2    │ → │  ETAPA 3   │ → │ ETAPA 4  │ → │ ETAPA 5  │
 │  Ingesta │   │   Feature    │   │Entrenamiento│  │Evaluación│   │Monitoreo │
 │   & ETL  │   │ Engineering  │   │  + MLflow  │   │          │   │  Drift   │
 └──────────┘   └──────────────┘   └────────────┘   └──────────┘   └──────────┘
```

### Etapa 1 — Ingesta & ETL

**Entradas:** 4 fuentes heterogéneas (CRM × 2, JEVA, WhatsApp/Instagram)  
**Proceso:** Deduplicación por email/teléfono → filtrado de registros con texto → limpieza → 1,500 registros  
**Salida:** `03_datos_procesados/rocktec_base_validada.csv`  
**Scripts:** `01_limpieza_datos.py` → `02_consolidar_datos.py` → `03_validar_duplicados.py`  
**Orquestación:** `06_pipeline_completo.py` ejecuta la secuencia completa

```
01_datos_crudos/          →   03_datos_procesados/
  clienty 1+2 (8,143)         crm_limpio.csv
  JEVA (1,155)           →    whatsapp_limpio.csv
  base_maestra (5,676)        rocktec_base_consolidada.csv (13,413)
                              rocktec_base_validada.csv (9,317)
```

### Etapa 2 — Feature Engineering

**Script:** `02_scripts/04_feature_engineering.py`

Dos representaciones del texto generadas en paralelo:

#### 2a. TF-IDF + Features Manuales (para LR y SVM)

```
Texto raw
   │
   ▼  PreprocessadorTexto
   │   • lowercase
   │   • URLs → token URL
   │   • precios → token PRECIO
   │   • normalización puntuación WhatsApp (!!, ??, ...)
   │
   ▼  VectorizadorTFIDF
   │   • TF-IDF unigrams + bigrams (max 15,000 features)
   │   • sublinear_tf=True, min_df=2
   │   • stop words español
   │
   ▼  Features Manuales (8 features)
   │   • longitud en caracteres
   │   • número de palabras
   │   • es pregunta (contiene ? ¿)
   │   • menciona precio ($ o "dólares")
   │   • tiene saludo ("hola", "buenos días")
   │   • indica queja ("problema", "reclamo", "dañado")
   │   • menciona cursos ("taller", "capacitación")
   │   • indica venta ("confirmo", "pago", "factura")
   │
   ▼  hstack(TF-IDF, features_manuales)
      Matriz sparse: (n_muestras × ~15,008)
```

#### 2b. Tokenización BERT (para BETO)

```
Texto raw
   │
   ▼  BertTokenizerFast (dccuchile/bert-base-spanish-wwm-cased)
      • max_length = 128 tokens
      • padding + truncation
      • input_ids, attention_mask, token_type_ids
```

### Etapa 3 — Entrenamiento con MLflow

**Script:** `02_scripts/05_entrenar_modelos.py`  
**Experimento MLflow:** `rocktec-intent-classification`  
**Split:** Stratified 80/20 (seed=42)

#### Modelo 1: Logistic Regression (baseline)

| Parámetro | Valor |
|-----------|-------|
| Solver | lbfgs (multinomial) |
| class_weight | balanced |
| C (GridSearch) | {0.1, 1, 10, 100} |
| Validación | Stratified K-Fold (k=5) |
| Scoring | F1-macro |

#### Modelo 2: LinearSVC

| Parámetro | Valor |
|-----------|-------|
| Calibración | CalibratedClassifierCV (probabilidades) |
| class_weight | balanced |
| C (GridSearch) | {0.01, 0.1, 1, 10} |
| Validación | Stratified K-Fold (k=5) |
| Scoring | F1-macro |

> **Decisión de diseño:** LinearSVC en lugar de SVC con kernel RBF. Para representaciones TF-IDF de alta dimensión, el kernel lineal es más eficiente y suele igualar o superar al RBF.

#### Modelo 3: BETO Fine-tuned

| Parámetro | Valor |
|-----------|-------|
| Base model | `dccuchile/bert-base-spanish-wwm-cased` |
| Épocas | 5 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup | 10% de pasos totales |
| Optimizer | AdamW (weight_decay=0.01) |
| Max tokens | 128 |
| Checkpoint | Mejor época por F1-macro en validación |

> **Justificación BETO:** Es el BERT entrenado sobre corpus en español (Wikipedia ES + OPUS). Captura morfología del español y expresiones informales de WhatsApp mejor que modelos multilingüe.

### Etapa 4 — Evaluación

**Métricas registradas por modelo en MLflow:**

| Métrica | Descripción |
|---------|-------------|
| `f1_macro` | Métrica principal — penaliza desbalance de clases |
| `f1_weighted` | F1 ponderado por soporte de clase |
| `precision_macro` | Precisión macro |
| `recall_macro` | Recall macro |
| `accuracy` | Exactitud global |
| `f1_INF` … `f1_QUE` | F1 individual por cada intención |

**Artefactos guardados por run:**
- `confusion_matrix_<modelo>.png`
- `reporte_<modelo>.txt` (classification_report completo)
- Modelo serializado (`pkl` o directorio BETO)

**Acceso a resultados:**
```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

### Etapa 5 — Monitoreo de Drift

**Alcance para el proyecto académico:** monitoreo offline sobre datos del piloto (Fase 5).

**Método:** PSI (Population Stability Index) sobre la distribución de predicciones.

```
PSI = Σ (P_actual - P_esperado) × ln(P_actual / P_esperado)
```

| PSI | Interpretación |
|-----|----------------|
| < 0.10 | Sin drift significativo |
| 0.10 – 0.25 | Drift moderado — revisar |
| > 0.25 | Drift severo — reentrenar |

La distribución de referencia es la del conjunto de entrenamiento final.

---

## 4. Flujo de Datos Completo

```
01_datos_crudos/           (4 xlsx, ~15K registros)
         │
         ▼  Fase 1 ETL
04_anotaciones/
  ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx   (1,500 × 3 anotadores)
         │
         ▼  calcular_kappa.py  →  Kappa ≥ 0.70
  dataset_consenso_final.csv              (1,500 con etiqueta consenso)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  TF-IDF features                    Tokenización BERT
  (LR, SVM)                          (BETO)
         │                                  │
         └──────────┬───────────────────────┘
                    ▼
             MLflow Experiment
             ┌─────────────────────────────────┐
             │  Run: logistic_regression       │
             │  Run: linear_svc                │
             │  Run: beto_finetuned            │
             └─────────────────────────────────┘
                    │
                    ▼
         06_resultados/
           comparacion_modelos.json
           confusion_matrix_*.png
           modelos/  (pkl + checkpoint BETO)
```

---

## 5. Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|------------|---------|
| Lenguaje | Python | 3.11 |
| Manipulación de datos | pandas / numpy | 1.5.3 / 1.24.3 |
| ML clásico | scikit-learn | 1.3.0 |
| Deep learning | PyTorch | ≥ 2.0 |
| Transformers | HuggingFace transformers | 4.36.0 |
| Modelo base NLP | BETO (BERT español) | bert-base-spanish-wwm-cased |
| Experiment tracking | MLflow | 2.9.0 |
| NLP español | spaCy / pysentimiento | 3.6.0 / 0.2.1 |
| Versionamiento | Git + GitHub | — |

---

## 6. Estructura de Archivos (Fase 2)

```
02_scripts/
  calcular_kappa.py          →  Valida Fase 1B, genera consenso
  04_feature_engineering.py  →  Preprocessing + TF-IDF + features manuales
  05_entrenar_modelos.py     →  Pipeline completo LR + SVM + BETO + MLflow

04_anotaciones/
  ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx  (entrada)
  dataset_consenso_final.csv             (generado por calcular_kappa.py)

06_resultados/
  modelos/
    vectorizador_tfidf.pkl
    modelo_lr.pkl
    modelo_svm.pkl
    beto_best/                (checkpoint HuggingFace)
  comparacion_modelos.json
  confusion_matrix_*.png
  reporte_*.txt

mlruns/                       (generado por MLflow — no versionar)
```

---

## 7. Comandos de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Validar anotaciones y generar consenso (Fase 1B)
python 02_scripts/calcular_kappa.py 04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx

# 3. Verificar feature engineering
python 02_scripts/04_feature_engineering.py

# 4a. Entrenamiento completo (requiere GPU para BETO)
python 02_scripts/05_entrenar_modelos.py

# 4b. Solo LR + SVM (sin GPU)
python 02_scripts/05_entrenar_modelos.py --skip-beto

# 5. Visualizar experimentos
mlflow ui --backend-store-uri mlruns
```

---

## 8. Decisiones de Diseño Relevantes

**¿Por qué LinearSVC y no SVC kernel RBF?**  
Para matrices TF-IDF de alta dimensión (>10K features), el kernel lineal tiene complejidad O(n) en inferencia vs O(n²) del RBF, sin pérdida significativa de F1 en clasificación de texto.

**¿Por qué F1-macro como métrica principal?**  
El dataset tiene desbalance natural (COT e INF son más frecuentes; QUE y VEN son minoría). F1-macro trata todas las clases por igual, penalizando modelos que ignoren las minorías.

**¿Por qué `class_weight='balanced'` en LR y SVM?**  
Compensa el desbalance sin necesidad de oversampling, manteniendo el dataset original intacto para la evaluación.

**¿Por qué BETO y no mBERT o XLM-R?**  
BETO fue entrenado exclusivamente sobre corpus en español, capturando mejor la morfología flexional del español y las expresiones coloquiales de WhatsApp. XLM-R es superior en configuraciones multilingüe pero agrega latencia sin beneficio aquí.

---

*Documento generado: Julio 2026 — Equipo MIA Rocktec*
