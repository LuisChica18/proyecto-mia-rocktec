# Plataforma de Inteligencia Comercial para Rocktec — MIA 2026

## 📋 Descripción

Proyecto de titulación para la **Maestría en Inteligencia Artificial Aplicada (MIA)** — **Universidad de las Américas (UDLA)**.

Plataforma basada en **NLP y Machine Learning** para clasificar automáticamente las intenciones de clientes en conversaciones de WhatsApp Business de **Rocktec** (empresa ecuatoriana especializada en concreto decorativo y acabados arquitectónicos).

🔗 **Dashboard en producción:** https://proyecto-mia-rocktec-rrxptjq8pvgqg68vpvc4ff.streamlit.app/

---

## 🧰 Stack Tecnológico

Python 3.11 · pandas · scikit-learn · spaCy · pysentimiento · nltk · statsmodels · Evidently AI · FastAPI · Streamlit · PostgreSQL · Git

---

## 👥 Equipo

| Miembro | Rol | Responsabilidad |
|---------|-----|-----------------|
| **Patricia Mosquera (A1)** | Análisis y Datos | Anotación, EDA, limpieza, consolidación, dashboard |
| **Luis Cruel (A2)** | Modelos y ML | Algoritmos, experimentos, métricas, SHAP/LIME |
| **Luis Chica (A3)** | Arquitectura | UML, pipeline, CI/CD, Docker |

---

## 🚀 ESTADO ACTUAL — FASE 5 ✅ COMPLETADA

### Resumen Ejecutivo

- ✅ **1,312 registros etiquetados** (consenso de 3 anotadores, Cohen's Kappa = 0.8851)
- ✅ **Modelo de producción:** TF-IDF + Logistic Regression — F1-macro = **0.75**, Accuracy = **85%**
- ✅ **Validación cruzada:** 25 folds pareados — LR consenso: 0.75 ± 0.04 / SVM consenso: 0.69 ± 0.06
- ✅ **Interpretabilidad:** SHAP global y local implementados (Script 10)
- ✅ **Monitoreo de equidad:** Evidently AI — brecha F1 entre perfiles documentada (0.22)
- ✅ **Aprendizaje activo:** 30 candidatos TEC priorizados para anotación adicional
- ✅ **Dashboard operativo:** Panel de Inteligencia Comercial en Streamlit Cloud
- ✅ **Informe final S8** entregado con 4 figuras, métricas con σ, ajustes del profesor aplicados

---

## 📊 Métricas Finales del Modelo de Producción

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

## 🏷️ Catálogo de Intenciones (7 definidas · 5 modeladas)

| Código | Definición | Ejemplo | ¿En el modelo? |
|--------|-----------|---------|:---:|
| **INF** | Información General | "¿Qué colores tienen?" | ✅ |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta?" | ✅ |
| **TEC** | Consulta Técnica | "¿Cómo se aplica?" | ✅ |
| **CUR** | Consulta de Cursos | "¿Cuándo es el curso?" | ✅ |
| **VEN** | Venta / Confirmación | "Confirmo compra" | ✅ |
| **SEG** | Seguimiento | "¿Estado mi cotización?" | ❌ regla léxica |
| **QUE** | Queja / Reclamo | "Llegó dañado" | ❌ regla léxica |

> SEG y QUE se manejan por reglas léxicas (100% recall) dado que tienen solo 5 y 4 registros respectivamente — insuficiente para entrenar un clasificador supervisado.

---

## 📂 Scripts Principales (02_scripts/)

| Script | Descripción |
|--------|-------------|
| `01_limpieza_datos_CORREGIDO.py` | Limpieza y normalización de datos crudos |
| `02_consolidar_datos_CORREGIDO.py` | Consolidación de 4 fuentes (CRM, JEVA, WhatsApp, consolidado) |
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

## 🔧 Cómo Reproducir

```bash
git clone https://github.com/LuisChica18/proyecto-mia-rocktec.git
cd proyecto-mia-rocktec
pip install -r requirements.txt

# Pipeline completo
python 02_scripts/06_pipeline_completo.py

# Solo dashboard
python3 -m streamlit run 02_scripts/19_dashboard_streamlit.py
```

---

## 📊 Fases del Proyecto

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

## 📈 Comparación de Modelos

| Modelo | Datos | F1-macro | Accuracy |
|--------|-------|:--------:|:--------:|
| TF-IDF + LR | Heurístico (baseline) | 0.716 | 0.792 |
| TF-IDF + **LR** | **Consenso humano** | **0.75** | **0.85** |
| TF-IDF + SVM | Consenso humano | 0.69 | 0.85 |
| BETO fine-tuned | Holdout (exploratoria) | 0.7967 | 0.9239 |

**Decisión de producción: TF-IDF + LR** — F1 equivalente a BETO, 23× más rápido (2.86 ms vs. 65 ms), sin GPU, ~150KB vs. ~420MB. BETO queda como upgrade candidato si se necesita mayor precisión en COT/VEN.

---

## 📧 Contacto

| Nombre | GitHub |
|--------|--------|
| Patricia Mosquera | @PatriciaMC |
| Luis Cruel | @LuisCruel |
| Luis Chica | @LuisChica18 |

---

## 📄 Licencia

Proyecto académico MIA 2026. Datos confidenciales de Rocktec (anonimizados).

---

**Última actualización:** Agosto 2026
**Estado:** FASE 5 COMPLETADA ✅ — Dashboard en producción
