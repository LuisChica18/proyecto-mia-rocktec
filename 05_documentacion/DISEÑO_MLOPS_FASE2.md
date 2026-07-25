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

> **Estado (25 Jul 2026):** implementado en `02_scripts/12_beto_finetuning.py`, ejecutado en Google
> Colab (GPU T4 gratuita — no hay GPU disponible en el entorno de desarrollo local). Primera corrida
> exploratoria sobre un split 80/20 del dataset completo dio F1-macro = 0.8552, pero ese split incluía
> filas de `holdout_test.csv` (fuga de datos) — no era válido como métrica final. Se corrigió
> `12_beto_finetuning.py` para entrenar solo con `train_val.csv`, y `02_scripts/13_evaluacion_holdout.py`
> evaluó ambos modelos una sola vez sobre `holdout_test.csv`, ya sin fuga:
>
> | Modelo | F1-macro (holdout) | Accuracy |
> |---|---:|---:|
> | TF-IDF + LR | 0.7938 | 0.8832 |
> | BETO fine-tuned | 0.7967 | 0.9239 |
>
> Con la fuga corregida, ambos quedan prácticamente empatados en F1-macro (antes la brecha
> "aparente" era de +0.10 a favor de BETO). Ver `06_resultados/reporte_holdout_final.txt` y
> `CHANGELOG.md` Sprint 7 (25 Jul 2026) para el detalle completo del hallazgo y el fix.

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

### Etapa 4.5 — Inferencia (Fase 3, implementada 25 Jul 2026)

**Estado:** cierra el gap identificado en `DIAGNOSTICO_FASES_3_4_5.md` (Fase 3) — hasta ahora todo
el proyecto era entrenamiento/evaluación offline sobre CSVs históricos, sin ninguna forma de tomar
un mensaje nuevo y devolver una predicción.

**Diseño: inferencia por lotes (batch), no una API en vivo.** Decisión deliberada, no una limitación
temporal: los mensajes de WhatsApp de Rocktec se descargan manualmente (no hay integración en vivo
con la WhatsApp Business API — ver `PROPUESTA_REVISADA_FASE2.md` §2, ajuste #8, misma razón por la
que PostgreSQL queda pospuesto). El flujo real es: alguien exporta un Excel/CSV nuevo de
conversaciones → corre `16_inferencia.py` → recibe el mismo archivo con columnas de predicción
añadidas, listo para priorizar al equipo de ventas. Una API viva no tendría ningún tráfico real que
atender todavía.

**Scripts:**
- `02_scripts/15_entrenar_produccion.py` — reentrena el modelo sobre el **100% de los datos
  etiquetados** (`train_val.csv` + `holdout_test.csv`, 1,312 filas). A diferencia de
  `13_evaluacion_holdout.py`, no reserva holdout: ese F1 honesto (0.7938) ya se midió y quedó
  reportado como cierre de Fase 4, así que seguir reservando esas 197 filas solo le restaría datos
  al modelo que realmente sirve predicciones. Guarda un artefacto versionado y autocontenido en
  `06_resultados/modelos/produccion/` (`vectorizador_tfidf.pkl`, `modelo_lr.pkl`, `metadata.json`
  con versión, fecha, hiperparámetro ganador y la nota de que 0.7938 sigue siendo la referencia
  honesta de desempeño esperado) — deliberadamente separado de `06_resultados/modelos/` (el
  experimento de `05_entrenar_modelos.py`, en formato MLflow pensado para tracking, no para servir).
- `02_scripts/16_inferencia.py` — componente de inferencia en sí. Expone `predecir(textos) -> DataFrame`
  (columnas: `texto`, `intencion_predicha`, `confianza`, `revisar_manual`) y un CLI
  (`--input`/`--output`/`--columna-texto`) para correr sobre un archivo completo.

**Umbral de confianza y revisión manual:** mensajes con `confianza < 0.50` se marcan
`revisar_manual=True` en vez de enrutarse automáticamente — mitigación directa de los riesgos **R1**
(TEC débil) y **R2** (SEG/QUE fuera del alcance del modelo) de `ANALISIS_RIESGOS_FASE2.md`: evita
forzar en silencio esos mensajes a una de las 5 clases modeladas. **Limitación observada al validar
el script:** texto sin señal real (p. ej. una sola palabra sin relación al dominio) puede recibir de
todos modos una confianza > 0.50 — el `intercept_` del modelo domina cuando no hay features de
TF-IDF activas. El umbral de 0.50 es un punto de partida para el piloto de Fase 5, a calibrar con
datos reales, no un valor validado estadísticamente.

**Log de predicciones:** cada corrida agrega filas a `06_resultados/predicciones/log_predicciones.csv`
(timestamp, texto, predicción, confianza, flag de revisión) — sustituto directo de la base de datos
pospuesta (PostgreSQL, ajuste #8) y el insumo que alimentará el cálculo de PSI de la Etapa 5.

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
El dataset tiene desbalance natural (INF y COT son más frecuentes; TEC y VEN son minoría dentro de las 5 clases modeladas). F1-macro trata todas las clases por igual, penalizando modelos que ignoren las minorías.

**¿Por qué `class_weight='balanced'` en LR y SVM?**  
Compensa el desbalance sin necesidad de oversampling, manteniendo el dataset original intacto para la evaluación.

**¿Por qué BETO y no mBERT o XLM-R?**  
BETO fue entrenado exclusivamente sobre corpus en español, capturando mejor la morfología flexional del español y las expresiones coloquiales de WhatsApp. XLM-R es superior en configuraciones multilingüe pero agrega latencia sin beneficio aquí.

**TF-IDF+LR vs. BETO fine-tuned — decisión de modelo de producción (cerrada 25 Jul 2026)**

La primera corrida de fine-tuning (F1-macro 0.8552) tenía fuga de datos hacia el holdout — no era
comparable como métrica final (ver Sprint 7 en `CHANGELOG.md`). Sobre el holdout real, sin fuga,
TF-IDF+LR = 0.7938 y BETO fine-tuned = 0.7967 — prácticamente empatados (la brecha de +0.10 que
sugería la corrida contaminada era en buena parte artefacto de la fuga, no una ventaja real de BETO).
Con el F1 ya no siendo un diferenciador, la decisión se resolvió por los demás criterios relevantes
para el contexto real de despliegue (Rocktec, PYME ecuatoriana, sin equipo de MLOps dedicado):

| Criterio | TF-IDF + LR | BETO fine-tuned | Gana |
|---|---|---|---|
| F1-macro (holdout) | 0.7938 | 0.7967 | Empate (Δ=0.003, no significativo) |
| Accuracy (holdout) | 0.8832 | 0.9239 | BETO |
| F1 en COT (cotización — oportunidad de venta) | 0.86 | 0.98 | BETO |
| F1 en VEN (venta confirmada — ingreso directo) | 0.77 | 0.91 | BETO |
| F1 en CUR (cursos) | 0.95 | 0.75 | TF-IDF+LR |
| F1 en TEC (consulta técnica) | 0.47 | 0.40 | Ninguno — débil en ambos, cuello de botella es dato (8 filas en holdout), no arquitectura |
| Latencia de inferencia (CPU, medida) | **2.86 ms/mensaje** | 65.14 ms/mensaje | TF-IDF+LR (23×), pero ambos son instantáneos al volumen real de mensajes de Rocktec (decenas/día) |
| Tamaño del artefacto | ~150KB | ~420MB | TF-IDF+LR |
| Requiere GPU para (re)entrenar | No | Sí (Colab T4 gratuita funciona, ~2 min) | TF-IDF+LR |
| Explicabilidad ya construida | Sí (SHAP/LIME, `06_resultados/explicabilidad/`) | No (pendiente) | TF-IDF+LR |
| Facilidad de mantenimiento post-tesis (equipo sin MLOps dedicado) | Alta | Media-baja | TF-IDF+LR |

**Decisión: TF-IDF + Logistic Regression pasa a producción para Fase 3.** Con el F1-macro
estadísticamente empatado, el criterio decisivo es operativo: Rocktec es una PYME sin
infraestructura GPU ni equipo de MLOps, y un modelo de 150KB con inferencia en milisegundos,
sin dependencias pesadas, y ya explicado con SHAP/LIME es más sostenible de operar y defender
ante el negocio que un modelo de 420MB que requiere GPU para reentrenarse.

**BETO fine-tuned queda documentado como upgrade candidato, no descartado.** Su ventaja real y
medible está en COT y VEN — las dos categorías con mayor impacto de negocio directo (cotizaciones
y ventas). Si en producción se observa que TF-IDF+LR genera falsos negativos costosos en esas dos
clases (p. ej. cotizaciones o confirmaciones de venta mal etiquetadas y perdidas), reevaluar el swap
a BETO fine-tuned como una mejora dirigida, no como reemplazo total. Ambos modelos quedan
documentados y reproducibles (`05_entrenar_modelos.py`, `12_beto_finetuning.py`,
`13_evaluacion_holdout.py`), así que el cambio es de bajo costo si se decide más adelante.

**Pendiente para que esta decisión sea completamente defendible:** ninguno de los dos modelos
resuelve TEC (F1 ≤ 0.47) — antes de Fase 5 conviene anotar más ejemplos de esa clase (solo 51 en
todo el dataset) en vez de seguir iterando sobre arquitectura.

---

## 9. Diagramas UML (Mermaid)

Entregable de Fase 2 (`README.md` §"Próxima Fase (FASE 2)", ítem 1: "Diagrama UML — arquitectura
modular"). Cuatro vistas complementarias del mismo sistema: componentes, clases, secuencia y flujo
de datos.

### 9.1 Diagrama de componentes — arquitectura del pipeline (5 etapas)

Versión Mermaid del diagrama ASCII de la §3.

```mermaid
flowchart LR
    subgraph E1["Etapa 1 · Ingesta & ETL"]
        direction TB
        C1[LimpiadoDatos]
        C2[ConsolidadorDatos]
        C3[DetectorDuplicados]
        C1 --> C2 --> C3
    end

    subgraph E2["Etapa 2 · Feature Engineering"]
        direction TB
        C4[PreprocessadorTexto]
        C5[VectorizadorTFIDF]
        C6[Tokenización BERT]
        C4 --> C5
        C4 --> C6
    end

    subgraph E3["Etapa 3 · Entrenamiento + MLflow"]
        direction TB
        C7[Logistic Regression]
        C8[LinearSVC]
        C9[BETO fine-tuned]
    end

    subgraph E4["Etapa 4 · Evaluación"]
        direction TB
        C10[Métricas F1-macro / accuracy]
        C11[Matrices de confusión]
        C12[SHAP / LIME]
    end

    subgraph E5["Etapa 5 · Monitoreo de Drift"]
        direction TB
        C13[PSI sobre predicciones]
    end

    E1 --> E2 --> E3 --> E4 --> E5

    style E1 fill:#e8f0fe,stroke:#4285f4
    style E2 fill:#fef7e0,stroke:#f9ab00
    style E3 fill:#e6f4ea,stroke:#34a853
    style E4 fill:#fce8e6,stroke:#ea4335
    style E5 fill:#f3e8fd,stroke:#a142f4
```

### 9.2 Diagrama de clases

Basado en las clases reales definidas en `02_scripts/06_pipeline_completo.py` (Etapas 1–3) y
`02_scripts/04_feature_engineering.py` (Etapa 2). `MapeoIntenciones` solo produce las **etiquetas
heurísticas iniciales**; el consenso humano (`calcular_kappa.py`) las reemplaza en
`dataset_consenso_final.csv` para el modelado real.

```mermaid
classDiagram
    class LimpiadoDatos {
        -Path ruta_crudos
        -Path ruta_salida
        -dict estadisticas
        +__init__(ruta_crudos, ruta_salida)
        +cargar_crm(archivo_1, archivo_2) DataFrame
        +cargar_whatsapp(archivo) DataFrame
        +normalizar_columnas(df) DataFrame
        +limpiar_crm(crm) DataFrame
        +limpiar_whatsapp(whatsapp) DataFrame
        +guardar_datos(crm, whatsapp)
    }

    class MapeoIntenciones {
        <<static>>
        +mapear_desde_texto(texto)$ str
        +mapear_desde_crm(row)$ str
    }

    class ConsolidadorDatos {
        -Path ruta_limpios
        -Path ruta_salida
        -dict estadisticas
        +__init__(ruta_datos_limpios, ruta_salida)
        +consolidar_crm(crm) list
        +consolidar_whatsapp(whatsapp) list
        +crear_base_final(crm_consolidado, wa_consolidado) DataFrame
        +guardar_base(df)
    }

    class DetectorDuplicados {
        -Path ruta_entrada
        -Path ruta_salida
        -list duplicados_similares
        +__init__(ruta_entrada, ruta_salida)
        +detectar_duplicados_exactos(df) DataFrame
        +eliminar_duplicados(df) DataFrame
        +guardar_datos(df_limpio)
    }

    class PreprocessadorTexto {
        +limpiar(texto) str
        +limpiar_serie(serie) Series
        +features_manuales(serie) DataFrame
    }

    class VectorizadorTFIDF {
        -int max_features
        -tuple ngram_range
        +__init__(max_features, ngram_range)
        +fit_transform(textos) sparse
        +transform(textos) sparse
        +guardar(ruta)
        +cargar(ruta)$ VectorizadorTFIDF
    }

    class CodificadorIntenciones {
        -LabelEncoder encoder
        +__init__()
        +encode(labels) ndarray
        +decode(indices) list
        +clases() ndarray
        +n_clases() int
    }

    ConsolidadorDatos ..> MapeoIntenciones : usa (etiqueta heurística inicial)
    LimpiadoDatos --> ConsolidadorDatos : crm_limpio.csv / whatsapp_limpio.csv
    ConsolidadorDatos --> DetectorDuplicados : rocktec_base_consolidada.csv
    DetectorDuplicados ..> PreprocessadorTexto : rocktec_base_validada.csv
    PreprocessadorTexto --> VectorizadorTFIDF : texto limpio
    VectorizadorTFIDF --> CodificadorIntenciones : features + labels
```

### 9.3 Diagrama de secuencia — mensaje de WhatsApp → predicción de intención

Flujo de inferencia en producción con el modelo elegido (TF-IDF + LR, §8).

```mermaid
sequenceDiagram
    actor Cliente
    participant WA as WhatsApp Business API
    participant Prep as PreprocessadorTexto
    participant Vec as VectorizadorTFIDF
    participant Model as Modelo LR (modelo_lr.pkl)
    participant Explic as SHAP/LIME
    participant Vendedor as Equipo de ventas

    Cliente ->> WA: Envía mensaje ("¿Cuánto cuesta el microcemento para 30 m²?")
    WA ->> Prep: texto_conversacion
    Prep ->> Prep: limpiar() + features_manuales()
    Prep ->> Vec: texto normalizado
    Vec ->> Vec: transform() → TF-IDF + 8 features manuales
    Vec ->> Model: vector disperso (1 × ~15,008)
    Model ->> Model: predict_proba()
    Model -->> Explic: (opcional) explicar predicción
    Explic -->> Vendedor: top features que explican la etiqueta
    Model -->> WA: intención predicha = COT
    WA -->> Vendedor: mensaje + etiqueta COT (prioridad: cotización)
    Vendedor -->> Cliente: respuesta priorizada
```

### 9.4 Diagrama de flujo de datos — trazabilidad de archivos

Versión Mermaid del flujo ASCII de la §4, extendido con la corrección de fuga de holdout (Sprint 7).

```mermaid
flowchart TD
    A1["01_datos_crudos/<br/>clienty-prospectos 1+2 (~8,143)"]
    A2["01_datos_crudos/<br/>JEVA base datos (~1,155)"]
    A3["01_datos_crudos/<br/>base_maestra_raw (~5,676)"]

    A1 & A2 & A3 --> B["03_datos_procesados/<br/>crm_limpio.csv + whatsapp_limpio.csv"]
    B --> C["rocktec_base_consolidada.csv (13,413)"]
    C --> D["rocktec_base_validada.csv (9,317, sin duplicados)"]

    D --> E["04_anotaciones/<br/>ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx<br/>(PATRICIA / LUIS_CRUEL / LUIS_CHICA)"]
    E --> F["calcular_kappa.py<br/>Kappa = 0.8851 ✅"]
    F --> G["dataset_consenso_final.csv<br/>(1,297 filas, voto mayoritario 2/3)"]

    G --> H["09_crear_holdout_set.py<br/>split estratificado 85/15"]
    H --> I["train_val.csv (1,115 filas)"]
    H --> J["holdout_test.csv (197 filas) — nunca visto en entrenamiento"]

    I --> K1["05_entrenar_modelos.py<br/>TF-IDF + LR / SVM"]
    I --> K2["12_beto_finetuning.py<br/>BETO fine-tuned (Colab GPU T4)"]
    K2 --> K3["FUENTE_ENTRENAMIENTO.txt<br/>marca: train_val.csv"]

    K1 --> L["13_evaluacion_holdout.py"]
    K3 --> L
    J --> L
    L --> M["06_resultados/<br/>reporte_holdout_final.txt<br/>TF-IDF+LR F1=0.7938 · BETO F1=0.7967"]

    style G fill:#e6f4ea,stroke:#34a853
    style J fill:#fce8e6,stroke:#ea4335
    style M fill:#e8f0fe,stroke:#4285f4
```

---

*Documento generado: Julio 2026 — Equipo MIA Rocktec*
