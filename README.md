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
## 🏷️ Catálogo de Intención (7 definidas · 5 modeladas)

| Código | Definición | Ejemplo | ¿En el modelo? |
|--------|-----------|---------|:---:|
| **INF** | Información General | "¿Qué colores tienen?" | ✅ |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta?" | ✅ |
| **TEC** | Consulta Técnica | "¿Cómo se aplica?" | ✅ |
| **CUR** | Consulta de Cursos | "¿Cuándo es el curso?" | ✅ |
| **VEN** | Venta / Confirmación | "Confirmo compra" | ✅ |
| **SEG** | Seguimiento | "¿Estado mi cotización?" | ❌ excluida |
| **QUE** | Queja / Reclamo | "Llegó dañado" | ❌ excluida |

> **Decisión de alcance:** SEG y QUE se excluyen por completo del entrenamiento y evaluación del
> modelo. En `dataset_consenso_final.csv` (1,297 filas válidas) solo hay 5 registros SEG y 4 QUE —
> insuficiente para entrenar o medir de forma confiable. El clasificador de producción cubre las
> 5 clases restantes. No es un pendiente a resolver en esta fase; se retomaría solo si una ronda
> futura de anotación amplía significativamente esas dos clases.

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
| **Fase 4** | Evaluación del modelo | 🔜 Por hacer (meta ya validada en holdout, falta cerrar el resto de entregables de fase) | F1 >= 0.75 — **alcanzado: 0.7938–0.7967** | 14 Sep |
| **Fase 5** | Piloto y defensa final | 🔜 Por hacer | Presentación exitosa | 21 Sep |

> Diagnóstico detallado de estado (hecho vs. pendiente) para Fases 3, 4 y 5: ver
> [`05_documentacion/DIAGNOSTICO_FASES_3_4_5.md`](05_documentacion/DIAGNOSTICO_FASES_3_4_5.md).

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

### Comparación de arquitecturas — exploratoria (⚠️ con fuga de datos hacia el holdout)

Primera comparación de tres arquitecturas, todas entrenadas sobre un split 80/20 ad hoc del
**dataset completo** (1,312 filas de las 5 clases modeladas). Sirve para elegir arquitectura, pero
**no es una métrica final válida**: ese split 80/20 incluye filas que también están en
`holdout_test.csv`, así que estos modelos ya "vieron" datos de holdout durante el entrenamiento
(ver sección siguiente para el número correcto, sin fuga):

| Modelo | Requiere GPU | F1-macro (split ad hoc, con fuga) |
|--------|:---:|-----------------:|
| TF-IDF + Logistic Regression (`05_entrenar_modelos.py`) | No | 0.7516 |
| BETO embeddings (sin ajustar) + LR (`11_beto_clasificador.py`) | No | 0.6370 |
| BETO fine-tuned, 5 épocas (corrida original, ya no reproducible con el script actual) | Sí | 0.8552 |

Por clase (BETO fine-tuned, corrida exploratoria): INF F1=0.96, COT F1=0.98, VEN F1=0.94,
CUR F1=0.83, **TEC F1=0.56** (la clase más difícil en los tres modelos — soporte bajo, 51 filas).
BETO fine-tuned superó claramente a los otros dos, confirmando que el ajuste de pesos sobre el
dominio de construcción/concreto decorativo ecuatoriano aporta sobre representaciones genéricas —
pero el número exacto no es fiable como métrica final.

### Evaluación final honesta sobre holdout (una sola vez, sin fuga de datos)

`02_scripts/13_evaluacion_holdout.py` reentrena TF-IDF+LR usando **solo** `train_val.csv`
(1,115 filas) y evalúa ambos modelos una única vez sobre `holdout_test.csv` (197 filas, nunca antes
tocadas). El checkpoint de BETO se verificó contra su marca `FUENTE_ENTRENAMIENTO.txt` antes de
evaluarlo (entrenado exclusivamente con `train_val.csv`, sin ver el holdout):

| Modelo | F1-macro (holdout) | Accuracy (holdout) |
|--------|-----------------:|---------:|
| TF-IDF + Logistic Regression (C=10) | 0.7938 ✅ | 0.8832 |
| **BETO fine-tuned** | **0.7967** ✅ | **0.9239** |

**Con la fuga de datos corregida, los dos modelos quedan prácticamente empatados en F1-macro**
(diferencia de 0.003) — muy lejos de la brecha de +0.10 que sugería la comparación exploratoria
contaminada (0.7516 vs. 0.8552). BETO sí mantiene una accuracy más alta (0.92 vs. 0.88). Por clase:

| Clase | TF-IDF+LR F1 | BETO fine-tuned F1 |
|---|---:|---:|
| INF | 0.92 | 0.95 |
| COT | 0.86 | 0.98 |
| CUR | 0.95 | 0.75 |
| VEN | 0.77 | 0.91 |
| TEC | 0.47 | 0.40 |

TEC sigue siendo la clase más débil en ambos modelos (8 ejemplos en el holdout) — ninguna
arquitectura la resuelve bien con los datos actuales; probablemente necesite más anotación antes
de que cualquier modelo mejore ahí. Fuera de eso, no hay un ganador claro: TF-IDF+LR es mejor en
CUR, BETO es mejor en COT/VEN/INF, ambos son débiles en TEC.

**Decisión de modelo de producción (cerrada 25 Jul 2026): TF-IDF + Logistic Regression.** Con el
F1-macro empatado, medimos también latencia real en CPU: TF-IDF+LR responde en **2.86 ms/mensaje**
vs. **65.14 ms/mensaje** de BETO (23× más lento, aunque ambos son instantáneos al volumen real de
mensajes de Rocktec). Sumado a que TF-IDF+LR no necesita GPU, pesa ~150KB (vs. ~420MB) y ya tiene
explicabilidad SHAP/LIME construida, es el modelo más sostenible de operar para una PYME sin equipo
de MLOps dedicado. **BETO fine-tuned queda documentado como upgrade candidato** — su ventaja real
está en COT y VEN (cotizaciones y ventas, las clases de mayor impacto de negocio directo); si en
producción TF-IDF+LR genera errores costosos ahí, vale reevaluar el swap. Ver la tabla de criterios
completa y el razonamiento en `05_documentacion/DISEÑO_MLOPS_FASE2.md` §8.

Reportes: `06_resultados/reporte_holdout_final.txt` (final, ambos modelos),
`06_resultados/beto/reporte_beto_finetuned_val.txt` (validación interna de BETO, no final).
Checkpoint del modelo: `06_resultados/modelos/beto_finetuned_best/` (no versionado en git — ver
`.gitignore`, ~420MB).

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
1. Diagrama UML (arquitectura modular) — ver [`05_documentacion/DISEÑO_MLOPS_FASE2.md` §9](05_documentacion/DISEÑO_MLOPS_FASE2.md#9-diagramas-uml-mermaid) (componentes, clases, secuencia, flujo de datos)
2. Pipeline detallado (5 etapas)
3. Propuesta revisada (ajustes obligatorios) — ver [`05_documentacion/PROPUESTA_REVISADA_FASE2.md`](05_documentacion/PROPUESTA_REVISADA_FASE2.md)
4. Cronograma semanal con responsables
5. Análisis de riesgos — ver [`05_documentacion/ANALISIS_RIESGOS_FASE2.md`](05_documentacion/ANALISIS_RIESGOS_FASE2.md)

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
