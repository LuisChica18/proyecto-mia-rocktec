# Plataforma de Inteligencia Comercial para Rocktec - MIA 2026

## 📋 Descripción

Proyecto de titulación para la **Maestría en Inteligencia Artificial Aplicada (MIA)** — **Universidad de las Américas (UDLA)**.

Plataforma basada en **NLP y Machine Learning** para clasificar automáticamente las intenciones de clientes en conversaciones de WhatsApp Business de **Rocktec** (empresa ecuatoriana especializada en concreto decorativo y acabados arquitectónicos).

## 🧰 Stack Tecnológico

Python 3.11 · pandas · scikit-learn · spaCy · pysentimiento · nltk · statsmodels · MLflow · python-docx · Git

---

## 👥 Equipo

| Miembro | Rol | Responsabilidad |
|---------|-----|-----------------|
| **Patricia Mosquera (A1)** | Análisis y Datos | Anotación, EDA, limpieza y consolidación de múltiples fuentes |
| **Luis Cruel (A2)** | Modelos y ML | Algoritmos, experimentos, métricas |
| **Luis Chica (A3)** | Arquitectura | UML, pipeline, CI/CD |

---

## 🚀 ESTADO ACTUAL - FASE 1B ✅ COMPLETADA

### Resumen Ejecutivo

- ✅ **1,500 registros anotados** (3 anotadores independientes)
- ✅ **Cohen's Kappa: 0.8851** (META >= 0.70 ALCANZADA)
- ✅ **Scripts ejecutados y reproducibles** (GitHub)
- ✅ **Reportes técnicos completos** (Kappa, Trazabilidad, Cierre)
- ✅ **TODO documentado y versionado**

### Métricas Finales
## 📂 Estructura del Repositorio
## 🏷️ Las 7 Categorías de Intención

| Código | Definición | Ejemplo |
|--------|-----------|---------|
| **INF** | Información General | "¿Qué colores tienen?" |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta?" |
| **TEC** | Consulta Técnica | "¿Cómo se aplica?" |
| **CUR** | Consulta de Cursos | "¿Cuándo es el curso?" |
| **SEG** | Seguimiento | "¿Estado mi cotización?" |
| **VEN** | Venta / Confirmación | "Confirmo compra" |
| **QUE** | Queja / Reclamo | "Llegó dañado" |

---

## 🔧 Cómo Reproducir TODO (Paso a Paso)

### Requisitos

```bash
git clone https://github.com/LuisChica18/proyecto-mia-rocktec.git
cd proyecto-mia-rocktec
pip install -r requirements.txt
```

### Ejecutar Pipeline Completo

```bash
# 1. Limpiar datos (14,974 → CSV normalizados)
python 02_scripts/01_limpieza_datos_CORREGIDO.py

# 2. Consolidar (4 CSV → 1,500)
python 02_scripts/02_consolidar_datos_CORREGIDO.py

# 3. Validar (sin duplicados)
python 02_scripts/03_validar_duplicados_CORREGIDO.py

# 4. Calcular Cohen's Kappa + generar dataset de consenso real
python 02_scripts/calcular_kappa.py "04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500 1ERA ETIQUETA.xlsx"

# 5. Entrenar LR + SVM sobre el consenso real (sin GPU)
python 02_scripts/05_entrenar_modelos.py --skip-beto

# 6. Validación estadística pareada baseline (heurístico) vs. ajustado (consenso)
python 02_scripts/07_validacion_estadistica.py

# Resultado: 1,500 anotados + Kappa 0.8851 ✅ + dataset_consenso_final.csv (1,297 filas válidas)
```

---

## 📊 Fases del Proyecto

| Fase | Descripción | Estado | Métrica | Hito |
|------|-------------|--------|---------|------|
| **Fase 1** | Validación inter-anotador (100 registros) | ✅ COMPLETA | Kappa = 0.8794 | 15 Jun |
| **Fase 1B** | Anotación completa (1,500 registros) | ✅ COMPLETA | Kappa = **0.8851** | 16 Jul |
| **Fase 1C** | Ajustes de prototipo + validación estadística | ✅ **COMPLETA** | Ver [resultados](#-resultados-baseline-vs-ajustado) | **19 Jul** |
| **Fase 2** | Diseño MLOps Pipeline | 🔜 Próxima | UML + Arquitectura | 20 Ago |
| **Fase 3** | Desarrollo e implementación | 🔜 Por hacer | Pipeline funcional | 31 Ago |
| **Fase 4** | Evaluación del modelo | 🔜 Por hacer | F1 >= 0.75 | 14 Sep |
| **Fase 5** | Piloto y defensa final | 🔜 Por hacer | Presentación exitosa | 21 Sep |

---

## 📈 Resultados: Baseline vs. Ajustado

Se detectó que los modelos originales se entrenaron con **etiquetas heurísticas** (por palabras
clave) porque el dataset de consenso humano nunca se generó — un bug corregido en Fase 1C. Al
reentrenar con las 1,297 anotaciones reales de consenso (voto mayoritario de 3 anotadores), el
desempeño real es más bajo que el heurístico, que estaba inflado por circularidad (las mismas
reglas de palabras clave generaban la etiqueta y eran fácilmente recuperables por el modelo bag-of-words):

| Modelo | Datos | F1-macro (test) | Accuracy |
|--------|-------|-----------------:|---------:|
| Logistic Regression | Heurístico (baseline, ~9,317 filas) | 0.716 | 0.792 |
| LinearSVC | Heurístico (baseline, ~9,317 filas) | 0.936 | 0.974 |
| **Logistic Regression** | **Consenso humano (ajustado, 1,297 filas)** | **0.596** | **0.854** |
| **LinearSVC** | **Consenso humano (ajustado, 1,297 filas)** | **0.493** | **0.835** |

Ver el análisis estadístico completo (validación cruzada pareada, intervalos de confianza,
Wilcoxon/t-test) en `06_resultados/validacion_estadistica.json` y en
[`06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx`](06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx).
Bitácora de cambios en [`CHANGELOG.md`](CHANGELOG.md).

---

## 📝 Documentación Técnica (FASE 1B)

### Reportes Disponibles

- **INFORME_CIERRE_FASE_1B.docx** - Resumen ejecutivo + métricas finales
- **REPORTE_KAPPA_DETALLADO.docx** - Análisis completo inter-anotador (desacuerdos, patrones)
- **DOCUMENTO_TRAZABILIDAD_CORRECCIONES.docx** - Errores encontrados y cómo se corrigieron

### Guías de Referencia

- **GUIA_MEJORADA_ANOTACION_FASE1.txt** - Regla de Oro para clasificar + ejemplos reales
- **CATÁLOGO_INTENCIONES_ROCKTEC.docx** - Definiciones formales de 7 categorías

---

## ✅ FASE 1B: Checklist de Cierre

- ✅ 1,500 registros anotados por 3 anotadores
- ✅ Cohen's Kappa validado: 0.8851
- ✅ Scripts Python ejecutados: 01, 02, 03, calcular_kappa
- ✅ 4 CSV procesados (crm, jeva, whatsapp, consolidado)
- ✅ Excel anotado versionado en GitHub
- ✅ Reportes técnicos: Kappa, Trazabilidad, Cierre
- ✅ Documentación: Guías, Cronograma, Catálogo
- ✅ Reproducibilidad 100%: anyone can clone & run
- ✅ GitHub actualizado con historial completo

**FASE 1B: LISTA PARA HANDOFF A FASE 2** 🚀

---

## 🎯 Próxima Fase (FASE 2)

**Objetivo:** Diseño MLOps Pipeline

**Plazo:** Semanas 6-7 (20 Julio - 3 Agosto)

**Entregables:**
1. Diagrama UML (arquitectura modular)
2. Pipeline detallado (5 etapas)
3. Propuesta revisada (ajustes obligatorios)
4. Cronograma semanal con responsables
5. Análisis de riesgos

---

## 📧 Contacto

| Rol | Nombre | GitHub |
|-----|--------|--------|
| Datos & Anotación | Patricia Mosquera | @PatriciaMC |
| Modelos & ML | Luis Cruel | @LuisCruel |
| Arquitectura | Luis Chica | @LuisChica18 |

---

## 📄 Licencia

Proyecto académico MIA 2026. Datos confidenciales de Rocktec (anonimizados).

---

**Última actualización:** 16 JULIO 2026
**Estado:** FASE 1B COMPLETADA ✅
**Próxima revisión:** Fase 2 (20 Julio 2026)

*Proyecto demuestra NLP, ML, y metodología rigurosa en contexto real PYME ecuatoriana.*
