# Plataforma de Inteligencia Comercial para Rocktec — MIA 2026

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%20Cloud-red)](https://proyecto-mia-rocktec-rrxptjq8pvgqg68vpvc4ff.streamlit.app/)
[![F1-macro](https://img.shields.io/badge/F1--macro-0.75%20±%200.04-green)](https://github.com/LuisChica18/proyecto-mia-rocktec)
[![Kappa](https://img.shields.io/badge/Cohen%27s%20Kappa-0.8854-green)](https://github.com/LuisChica18/proyecto-mia-rocktec)
[![License](https://img.shields.io/badge/Licencia-Académica%20MIA%202026-lightgrey)](https://github.com/LuisChica18/proyecto-mia-rocktec)

---

## Descripción del problema y solución

**Problema:** Rocktec, empresa ecuatoriana de concreto decorativo ([www.rocktec.com.ec](https://www.rocktec.com.ec)), gestiona cientos de conversaciones mensuales de WhatsApp Business sin poder priorizarlas comercialmente. Los asesores no tienen visibilidad de cuáles clientes tienen mayor interés, cuáles están en riesgo de pérdida o cuáles ya confirmaron una venta.

**Solución:** Pipeline MLOps de 5 capas que clasifica automáticamente la intención de cada mensaje de cliente (cotización, consulta técnica, queja, venta, etc.) y alimenta un panel de inteligencia comercial accesible desde cualquier navegador — sin instalación, sin costo de infraestructura.

🔗 **Dashboard en producción:** https://proyecto-mia-rocktec-rrxptjq8pvgqg68vpvc4ff.streamlit.app/

---

## Stack tecnológico

Python 3.11 · scikit-learn · spaCy · pandas · MLflow · Evidently AI · GitHub Actions · Streamlit Community Cloud · Google Sheets API · fpdf2 · openpyxl · BETO (transformers)

---

## Equipo

| Miembro | Rol | Responsabilidad |
|---------|-----|-----------------|
| **Patricia Mosquera (A1)** | Análisis y Datos | Anotación, EDA, limpieza, consolidación, dashboard |
| **Luis Cruel (A2)** | Modelos y ML | Algoritmos, experimentos, métricas, SHAP/LIME |
| **Luis Chica (A3)** | Arquitectura | UML, pipeline, CI/CD, despliegue |

**Institución:** Universidad de Las Américas (UDLA) — Ecuador  
**Programa:** Maestría en Inteligencia Artificial Aplicada (MIA 2026)  
**Empresa auspiciante:** Rocktec — www.rocktec.com.ec

---

## Estado actual — Fase 5 ✅ Completada

### Resumen ejecutivo

- ✅ **1,312 registros etiquetados** (consenso de 3 anotadores, Cohen's Kappa = 0.8851)
- ✅ **Modelo de producción:** TF-IDF + Logistic Regression — F1-macro = **0.75**, Accuracy = **85%**
- ✅ **Validación cruzada:** 25 folds pareados — LR consenso: 0.75 ± 0.04 / SVM consenso: 0.69 ± 0.06
- ✅ **Interpretabilidad:** SHAP global y local implementados (Script 10)
- ✅ **Monitoreo de equidad:** Evidently AI — brecha F1 entre perfiles documentada (0.2229)
- ✅ **Aprendizaje activo:** 30 candidatos TEC priorizados para anotación adicional
- ✅ **Dashboard operativo:** Panel de Inteligencia Comercial en Streamlit Cloud
- ✅ **Informe final S8** entregado con figuras, métricas con σ y ajustes del profesor aplicados

---

## Métricas finales del modelo de producción

| Métrica | Validación Cruzada (25 folds) | Holdout (263 registros) |
|---------|:---:|:---:|
| F1-macro | **0.75 ± 0.04** | 0.72 |
| Accuracy | **85%** | 85% |
| Precision (aprox.) | — | ~0.76 |
| Recall (aprox.) | — | ~0.74 |

### Por clase (holdout)

| Clase | Precision | Recall | F1-score | Soporte |
|-------|:---------:|:------:|:--------:|:-------:|
| INF | 0.89 | 0.92 | 0.90 | 175 |
| COT | 0.82 | 0.86 | 0.84 | 58 |
| CUR | 0.79 | 0.92 | 0.85 | 12 |
| VEN | 0.70 | 0.88 | 0.78 | 8 |
| TEC | 0.33 | 0.50 | 0.40 | 10 |

---

## Requisitos técnicos

```bash
Python 3.11+
pip install -r requirements.txt
```

Dependencias principales: `scikit-learn`, `spacy`, `mlflow`, `evidently`, `streamlit`, `gspread`, `transformers`, `fpdf2`, `openpyxl`

---

## Instrucciones de ejecución paso a paso

### Opción A — Dashboard en producción (recomendado)

1. Abrir: https://proyecto-mia-rocktec-rrxptjq8pvgqg68vpvc4ff.streamlit.app/
2. Ingresar contraseña (configurada en Streamlit Secrets)
3. Subir archivos `.txt` o `.zip` exportados desde WhatsApp Business
4. Ver el Panel Comercial con leads, pérdidas y ventas por asesor

### Opción B — Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/LuisChica18/proyecto-mia-rocktec.git
cd proyecto-mia-rocktec

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Pipeline completo
python 02_scripts/06_pipeline_completo.py

# 5. Solo dashboard
streamlit run 02_scripts/19_dashboard_streamlit.py
```

Abrir en el navegador: http://localhost:8501

---

## Explicación del pipeline (5 capas MLOps)

```
[WhatsApp .txt exportado por asesor]
    ↓
Capa 1 — Ingesta de datos
    Parser Python: regex + corrección unicode (p.m., guion largo)
    Google Sheets: base de datos persistente con deduplicación automática
    (clave única: remitente + fecha + primeros 50 chars del texto)
    ↓
Capa 2 — Preprocesamiento NLP
    spaCy es_core_news_md: limpieza y lematización
    TF-IDF: vectorización del texto → vector numérico
    ↓
Capa 3 — Clasificación de intención
    Regresión Logística: 5 clases supervisadas (INF, COT, TEC, CUR, VEN)
    Reglas léxicas: QUE y SEG con 100% recall
    Sentimiento inferido: VEN=Positivo · QUE/SEG=Negativo · resto=Neutro
    Umbral adaptativo 0.75: probabilidad < 0.75 → revisión humana
    ↓
Capa 4 — Despliegue
    Streamlit Community Cloud: gratuito, accesible sin instalación
    Panel Comercial: leads · pérdidas · ventas · filtros por período
    Descargas: CSV · Excel · PDF
    ↓
Capa 5 — Monitoreo
    Evidently AI: drift de datos y equidad por perfil de cliente
    MLflow: tracking de experimentos y registro de modelos
    GitHub Actions: CI/CD automático en cada push a main
    ↓
[Panel Comercial Rocktec — Martha Andrade, Gerente General]
```

---

## Estructura del repositorio

```
proyecto-mia-rocktec/
├── 01_datos_crudos/                    → Datos originales sin procesar
│
├── 02_scripts/                         → Scripts Python del pipeline
│   ├── 01_limpieza_datos_CORREGIDO.py
│   ├── 02_consolidar_datos_CORREGIDO.py
│   ├── 05_entrenar_modelos.py
│   ├── 06_pipeline_completo.py
│   ├── 07_validacion_estadistica.py
│   ├── 09_crear_holdout_set.py
│   ├── 10_shap_lime_explicabilidad.py
│   ├── 13_evaluacion_holdout.py
│   ├── 15_entrenar_produccion.py
│   ├── 17_umbral_confianza_adaptativo.py
│   ├── 18_active_learning_tec.py
│   ├── 19_dashboard_streamlit.py       ← Entry point del dashboard
│   ├── 20_monitoreo_equidad.py
│   └── tab5_panel_comercial.py         ← Panel de inteligencia comercial
│
├── 03_datos_procesados/                → Datos limpios y consolidados
│
├── 04_anotaciones/                     → Dataset etiquetado
│   ├── dataset_consenso_final.csv      ← 1,312 registros consenso
│   ├── candidatos_QUE.xlsx            ← Candidatos ground truth QUE
│   ├── candidatos_SEG.xlsx            ← Candidatos ground truth SEG
│   └── active_learning_tec_candidatos.xlsx
│
├── 05_documentacion/                   → Documentación técnica
│   ├── DIARIO_ITERACIONES_PILOTO.md
│   ├── REPORTE_PRUEBAS_ESTRES.md
│   └── CATALOGO_INTENCIONES_ROCKTEC_EJECUTIVO.docx
│
├── 06_resultados/                      → Resultados y métricas
│   ├── equidad/                        → Reporte Evidently AI
│   ├── explicabilidad/                 → Figuras SHAP y LIME
│   ├── modelos/                        → Modelo serializado (.pkl)
│   └── pipeline/                       → Métricas y logs
│
├── .github/workflows/                  → GitHub Actions CI/CD
├── requirements.txt                    → Dependencias Python
└── README.md                           → Este archivo
```

---

## Catálogo de intenciones (7 definidas · 5 modeladas)

| Código | Definición | Ejemplo | ¿En el modelo? |
|--------|-----------|---------|:---:|
| **INF** | Información General | "¿Qué colores tienen?" | ✅ |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta?" | ✅ |
| **TEC** | Consulta Técnica | "¿Cómo se aplica?" | ✅ |
| **CUR** | Consulta de Cursos | "¿Cuándo es el curso?" | ✅ |
| **VEN** | Venta / Confirmación | "Confirmo compra" | ✅ |
| **SEG** | Seguimiento | "¿Estado mi cotización?" | ❌ regla léxica |
| **QUE** | Queja / Reclamo | "Llegó dañado" | ❌ regla léxica |

> SEG y QUE se manejan por reglas léxicas (100% recall) dado que tienen solo 5 y 4 registros respectivamente — insuficiente para entrenar un clasificador supervisado. Refleja la realidad operativa de Rocktec: pocas quejas formales y seguimiento mayoritariamente por teléfono.

---

## Scripts principales (02_scripts/)

| Script | Descripción |
|--------|-------------|
| `01_limpieza_datos_CORREGIDO.py` | Limpieza y normalización de datos crudos |
| `02_consolidar_datos_CORREGIDO.py` | Consolidación de 3 fuentes (WhatsApp, CRM, JEVA) |
| `05_entrenar_modelos.py` | Entrenamiento LR + SVM con TF-IDF |
| `06_pipeline_completo.py` | Pipeline MLOps end-to-end |
| `07_validacion_estadistica.py` | Validación cruzada pareada, Wilcoxon/t-test |
| `09_crear_holdout_set.py` | Separación holdout sin fuga de datos |
| `10_shap_lime_explicabilidad.py` | SHAP global y local, LIME |
| `13_evaluacion_holdout.py` | Evaluación final honesta sobre holdout |
| `15_entrenar_produccion.py` | Entrenamiento modelo de producción |
| `17_umbral_confianza_adaptativo.py` | Umbral óptimo 0.75 → F1-auto = 0.9214 |
| `18_active_learning_tec.py` | 30 candidatos TEC priorizados por incertidumbre |
| `19_dashboard_streamlit.py` | Dashboard principal (entry point) |
| `20_monitoreo_equidad.py` | Reporte Evidently AI por perfil de cliente |
| `tab5_panel_comercial.py` | Panel comercial: leads, ventas, pérdidas por asesor |

---

## Fases del proyecto

| Fase | Descripción | Estado | Métrica | Fecha |
|------|-------------|:------:|---------|-------|
| **Fase 1** | Validación inter-anotador (100 registros) | ✅ | Kappa = 0.8794 | Jun 2026 |
| **Fase 1B** | Anotación completa (1,500 registros) | ✅ | Kappa = 0.8851 | Jul 2026 |
| **Fase 1C** | Ajustes prototipo + validación estadística | ✅ | F1 holdout = 0.75 | Jul 2026 |
| **Fase 2** | Diseño MLOps Pipeline | ✅ | UML + arquitectura 5 capas | Jul 2026 |
| **Fase 3** | Desarrollo e implementación | ✅ | 22 scripts, pipeline funcional | Jul 2026 |
| **Fase 4** | Evaluación del modelo | ✅ | F1-macro = 0.75 ≥ meta | Ago 2026 |
| **Fase 5** | Piloto y dashboard | ✅ | Dashboard en producción | Ago 2026 |

---

## Comparación de modelos

| Modelo | Datos | F1-macro | Accuracy |
|--------|-------|:--------:|:--------:|
| TF-IDF + LR | Heurístico (baseline) | 0.716 | 0.792 |
| TF-IDF + **LR** | **Consenso humano** | **0.75** | **0.85** |
| TF-IDF + SVM | Consenso humano | 0.69 | 0.85 |
| BETO fine-tuned | Holdout (exploratoria) | 0.7967 | 0.9239 |

**Decisión de producción: TF-IDF + LR** — F1 equivalente a BETO, 23× más rápido (2.86 ms vs. 65 ms), sin GPU, ~150KB vs. ~420MB. BETO queda como upgrade candidato si se necesita mayor precisión en COT/VEN.

---

## Decisiones de diseño

| Decisión | Alternativa descartada | Justificación |
|----------|----------------------|---------------|
| Google Sheets como BD | PostgreSQL | PYME sin equipo técnico; Sheets es gratuito y accesible |
| Streamlit Cloud | Power BI | Deploy gratuito, sin instalación para el usuario final |
| LR en producción | BETO | Diferencia mínima (0.0029); LR no requiere GPU |
| Sentimiento inferido | pysentimiento | Sin GPU disponible en producción |
| Batch manual | API WhatsApp | Rocktec no tiene API Business Platform activa |

---

## Limitaciones documentadas

1. **TEC F1=0.40** — Pocos ejemplos técnicos; active learning identifica 30 candidatos prioritarios
2. **Equidad brecha 0.2229** — Perfil de cliente no capturado en WhatsApp; se usa proxy desde CRM
3. **Exportación manual** — Sin API WhatsApp Business; usuario exporta `.txt` y sube al dashboard
4. **QUE/SEG escasos** — Realidad del negocio, no fallo metodológico

Ver reporte completo: `05_documentacion/REPORTE_PRUEBAS_ESTRES.md`

---

## Evidencias de validación

- **Holdout set:** F1-macro = 0.72 (datos nunca vistos durante entrenamiento)
- **RepeatedKFold 5×5:** F1-macro = 0.75 ± 0.04 (25 particiones independientes)
- **Interpretabilidad:** SHAP global + local, LIME por clase → `06_resultados/explicabilidad/`
- **Equidad:** Brecha F1 = 0.2229 entre perfiles → `06_resultados/equidad/`
- **Pruebas de estrés:** → `05_documentacion/REPORTE_PRUEBAS_ESTRES.md`
- **Piloto real:** Chats reales de Rocktec procesados por Martha Andrade (Gerente General)

---

## Contacto

| Nombre | GitHub |
|--------|--------|
| Patricia Mosquera | @PatriciaMC |
| Luis Cruel | @LuisCruel |
| Luis Chica | @LuisChica18 |

---

## Licencia

Proyecto académico MIA 2026. Datos confidenciales de Rocktec (anonimizados).

---

**Última actualización:** Agosto 2026  
**Estado:** FASE 5 COMPLETADA ✅ — Dashboard en producción
