# Propuesta Revisada — Ajustes Obligatorios (Fase 2)
## Plataforma de Clasificación de Intenciones — Rocktec MIA 2026

**Versión:** 1.0
**Fecha:** Julio 2026
**Responsable:** Luis Chica (A3) — Arquitectura
**Entregable de:** README.md §"Próxima Fase (FASE 2)", ítem 3

---

## 1. Propósito

Este documento consolida, con evidencia y justificación, los cambios **obligatorios** de alcance,
metodología y arquitectura que el proyecto sufrió entre la propuesta inicial y el estado actual
(25 Jul 2026). No es un documento nuevo de diseño — es la trazabilidad de *por qué* lo entregado
difiere de lo planeado originalmente, construida a partir de `CHANGELOG.md`, el historial de git y
`README.md`.

Se distingue entre dos tipos de cambio:
- **§2 — Ajustes obligatorios ya resueltos:** decisiones activas, documentadas y cerradas.
- **§3 — Divergencias de alcance aún no formalizadas:** diferencias entre el plan original y el
  estado actual que nadie decidió explícitamente; quedan señaladas para que el equipo las resuelva
  (aceptarlas formalmente o actuar sobre ellas) antes del cierre de Fase 2.

---

## 2. Ajustes Obligatorios Ya Resueltos

| #     | Área | Plan / alcance original | Ajuste aplicado | Causa | Impacto | Evidencia |
|-------|------|--------------------------|------------------|-------|---------|-----------|
| **1** | Fuente de etiquetas de entrenamiento | Entrenar sobre las 1,500 anotaciones humanas reales (consenso de 3 anotadores) | Se descubrió que `calcular_kappa.py` calculaba el Kappa pero **nunca escribía** `dataset_consenso_final.csv`; todos los entrenamientos hasta Fase 1B usaban en realidad etiquetas heurísticas (por palabras clave), no las anotaciones humanas. Corregido en Fase 1C: el script ahora genera el consenso real (voto mayoritario 2/3). | Bug de pipeline detectado en auditoría de Fase 1C, no un cambio de plan deliberado. | **Crítico** — invalidó las métricas reportadas hasta entonces (SVM heurístico F1=0.936 vs. SVM sobre consenso real F1=0.493). | `CHANGELOG.md` Fase 1C, `04_anotaciones/dataset_consenso_final.csv`, `06_resultados/validacion_estadistica.json` |
| **2** | Alcance del catálogo de intenciones | 7 clases catalogadas (INF, COT, TEC, CUR, VEN, SEG, QUE) planeadas para modelar | Reducido a **5 clases modeladas**; SEG y QUE quedan definidas en el catálogo de negocio pero excluidas del entrenamiento/evaluación. | Solo 5 filas SEG y 4 QUE de 1,297 válidas — insuficiente para entrenar o medir con confianza. | Medio — decisión de alcance formal, no un pendiente técnico. | `CHANGELOG.md` Sprint 7, `README.md` §Catálogo de Intención |
| **3** | Metodología de validación estadística | Comparar modelos con un único split train/test | Se adoptó **validación cruzada pareada** (`RepeatedKFold` 5×5) sobre las mismas filas y folds, comparando etiqueta heurística vs. consenso, con intervalos de confianza y pruebas Wilcoxon/t-test. | Aislar el efecto de la *calidad del etiquetado* del efecto del *tamaño de muestra* — un split único no permitía esa separación. | Medio — mejora el rigor estadístico de las comparaciones. | `02_scripts/07_validacion_estadistica.py`, `06_resultados/validacion_estadistica.json` |
| **4** | Arquitecturas de modelo comparadas | Plan original: solo Logistic Regression + LinearSVC sobre TF-IDF | Se amplió a comparar **3 arquitecturas**: TF-IDF+LR, BETO por embeddings (sin ajustar) y BETO fine-tuned. | Explorar si un modelo de lenguaje en español supera a bag-of-words; viabilidad de usar GPU gratuita de Google Colab para el fine-tuning. | Alto — cambió por completo la discusión de modelo final de producción. | `02_scripts/11_beto_clasificador.py`, `02_scripts/12_beto_finetuning.py` |
| **5** | Metodología de evaluación final | Reportar F1-macro de la arquitectura ganadora como métrica final | La primera comparación de las 3 arquitecturas (TF-IDF+LR 0.7516, BETO embeddings 0.6370, BETO fine-tuned 0.8552) se hizo sobre un split 80/20 ad hoc del dataset **completo**, que incluía filas que después se apartaron como `holdout_test.csv`. Se corrigió `12_beto_finetuning.py` para entrenar solo con `train_val.csv`, y se creó `13_evaluacion_holdout.py` para evaluar una única vez sobre el holdout real. | Se detectó fuga de datos: esos 3 modelos ya habían "visto" las filas de holdout durante su propio entrenamiento — ninguno de esos 3 números era válido como métrica final. | **Crítico** — la brecha reportada entre TF-IDF+LR y BETO pasó de +0.10 F1-macro (con fuga) a +0.003 F1-macro (sin fuga, prácticamente empatados). | `02_scripts/13_evaluacion_holdout.py`, `06_resultados/reporte_holdout_final.txt`, `CHANGELOG.md` Sprint 7 (fix de fuga) |
| **6** | Decisión de modelo de producción | Elegir el modelo con mejor F1-macro | Con la fuga corregida y el F1-macro prácticamente empatado, el criterio de selección pasó de "mejor F1" a un **análisis multicriterio**: latencia de inferencia (2.86 ms vs. 65.14 ms), tamaño de artefacto (~150KB vs. ~420MB), necesidad de GPU para reentrenar, y explicabilidad ya construida. | Rocktec es una PYME ecuatoriana sin infraestructura GPU ni equipo de MLOps dedicado — el criterio técnico puro (F1) ya no discriminaba entre opciones. | Alto — decisión formal de arquitectura de producción, con BETO documentado como upgrade candidato (no descartado). | `05_documentacion/DISEÑO_MLOPS_FASE2.md` §8 |
| **7** | Explicabilidad del modelo | No estaba contemplada como entregable en las fases originales | Se añadió SHAP/LIME sobre el modelo LR de producción, y (pendiente de commit) el equivalente para BETO fine-tuned. | Necesidad de justificar las predicciones del modelo ante Rocktec y ante el comité, no solo reportar una métrica agregada. | Medio — mejora la defendibilidad de la decisión de modelo. | `02_scripts/10_shap_lime_explicabilidad.py`, `02_scripts/14_explicabilidad_beto.py` |
| **8** | Base de datos (PostgreSQL) | El rol de Patricia incluye explícitamente "PostgreSQL" como parte del stack de datos | **Pospuesto a trabajo futuro**, no descartado: no se implementa mientras la ingesta de mensajes de WhatsApp siga siendo manual (descarga periódica de exportaciones Excel, sin integración en vivo). Se retomará cuando exista el componente de inferencia/serving de Fase 3 que reciba mensajes en tiempo real y necesite loguear conversación + predicción + confianza para el monitoreo de drift. | Con archivos planos (CSV/Excel) alcanza para 1,297 filas anotadas manualmente; una base de datos relacional solo aporta valor real frente a un flujo de mensajes en vivo que loguear, no para el pipeline de entrenamiento offline actual. | Bajo hoy (no bloquea el alcance académico actual); pasa a Alto en cuanto exista integración en vivo con WhatsApp, ya que ahí sí sería necesaria para el log de mensajes y el drift monitoring. | `05_documentacion/DIAGNOSTICO_FASES_3_4_5.md` §2 (Fase 3, componente de inferencia pendiente), `CLAUDE.md` §Team Roles |
| **9** | CI/CD | El rol de Luis Chica incluye "CI/CD" desde el primer commit de `README.md` | **CI implementado** (`.github/workflows/ci.yml`, commit `19f05d6`): compila todos los scripts de `02_scripts/` y reentrena+evalúa TF-IDF+LR sobre `holdout_test.csv` en cada push a `main`, fallando el build si el F1-macro cae por debajo de la meta de Fase 4 (0.75). **CD queda pospuesto**, por la misma razón que PostgreSQL (ajuste #8): no hay ningún artefacto/servicio de inferencia todavía que desplegar — se retoma junto con el componente de inferencia de Fase 3. | No tenía sentido automatizar el despliegue de un servicio que aún no existe; CI sí podía aportar valor inmediato como gate de regresión sobre la métrica central del proyecto. | Medio — cierra la mitad de la divergencia original; la otra mitad (CD) queda ligada al mismo bloqueante de Fase 3 que PostgreSQL. | `.github/workflows/ci.yml`, `05_documentacion/DIAGNOSTICO_FASES_3_4_5.md` §2 |

---

## 3. Divergencias de Alcance Aún No Formalizadas

Esta diferencia existe entre lo planeado en la estructura inicial del repositorio
(commit `ab79b72`, roles de equipo en `README.md`) y el estado actual, pero **nadie la
ha decidido ni documentado explícitamente** como los ajustes de la §2. Se lista aquí para que el
equipo la cierre (aceptarla formalmente como fuera de alcance, o retomarla) antes de cerrar
Fase 2 — queda también reflejada como riesgo **R9** en `ANALISIS_RIESGOS_FASE2.md`.

| Área | Plan original | Estado actual | Nota |
|------|----------------|----------------|------|
| Documentos planeados en la estructura inicial | `CRONOGRAMA_EJECUTIVO_FASE1.txt`, `METODOLOGIA_ANOTACION.md`, `DOCUMENTO_DISEÑO_S4.docx` (listados en el primer README, commit `ab79b72`) | Ninguno de los tres existe con ese nombre; su contenido parece haberse cubierto con otros documentos (`README.md`, `DISEÑO_MLOPS_FASE2.md`, este mismo documento) | Confirmar explícitamente que estos tres se consideran reemplazados y no pendientes, para que no aparezcan como "entregable faltante" en una revisión externa. |

---

## 4. Cómo Usar Este Documento

Este documento es el entregable **"Propuesta revisada (ajustes obligatorios)"** de Fase 2. Su
función es que cualquier persona que compare la propuesta de tesis original contra el repositorio
actual encuentre aquí, en un solo lugar, la justificación de cada desviación — en vez de tener que
reconstruirla leyendo commit por commit. Se recomienda revisarlo junto con
`05_documentacion/DISEÑO_MLOPS_FASE2.md` (arquitectura resultante) y `05_documentacion/ANALISIS_RIESGOS_FASE2.md`
(riesgos derivados de estos mismos ajustes).

---

*Documento generado: Julio 2026 — Equipo MIA Rocktec*
