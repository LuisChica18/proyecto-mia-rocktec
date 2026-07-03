# Plataforma de Inteligencia Comercial para Rocktec - MIA 2026

## 📋 Descripción

Proyecto de titulación para la **Maestría en Inteligencia Artificial Aplicada (MIA)**.

Plataforma basada en **NLP y Machine Learning** para clasificar automáticamente las intenciones de clientes en conversaciones de WhatsApp Business de **Rocktec** (empresa ecuatoriana especializada en concreto decorativo y acabados arquitectónicos).

---

## 👥 Equipo

| Miembro | Rol | Responsabilidad |
|---------|-----|-----------------|
| **Patricia Mosquera (A1)** | Análisis y Datos | Anotación, EDA, preprocesamiento, PostgreSQL |
| **Luis Cruel (A2)** | Modelos y ML | Algoritmos, experimentos, MLflow, métricas |
| **Luis Chica (A3)** | Arquitectura | UML, pipeline, CI/CD, drift monitoring |

---

## 📂 Estructura del Repositorio

```
proyecto-mia-rocktec/
├── 01_datos_crudos/                    → Datos originales sin procesar
│   ├── Copia_de_clienty-prospectos_1.xlsx
│   ├── Copia_de_clienty-prospectos_2.xlsx
│   ├── ROCKTEC_-_JEVA_base_datos.xlsx
│   ├── base_maestra_raw_total_rocktec.xlsx
│   └── README_DATOS.txt
│
├── 02_scripts/                         → Scripts Python reutilizables
│   ├── 01_limpieza_datos.py
│   ├── 02_consolidar_datos.py
│   ├── 03_validar_duplicados.py
│   ├── 06_pipeline_completo.py
│   ├── consolidar_4_bases.py           ← NUEVO (consolidación de 4 fuentes)
│   └── calcular_kappa.py               (Cohen's Kappa)
│
├── 03_datos_procesados/                → Datos limpios y normalizados
│   ├── README.md
│   ├── crm_limpio.csv
│   ├── whatsapp_limpio.csv
│   └── rocktec_base_consolidada.csv
│
├── 04_anotaciones/                     → Dataset etiquetado con intenciones
│   └── ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx  (1,500 registros anotados)
│
├── 05_documentacion/                   → Guías, metodología, cronogramas
│   ├── METODOLOGIA_ANOTACION.md
│   ├── REPORTE_CONSOLIDACION_4_BASES.txt  ← NUEVO (trazabilidad)
│   ├── GUIA_MEJORADA_ANOTACION_FASE1.txt
│   ├── CRONOGRAMA_EJECUTIVO_FASE1.txt
│   └── CATÁLOGO_INTENCIONES_ROCKTEC.docx
│
├── 06_resultados/                      → Informes y documentos de diseño
│   ├── INFORME_RESULTADOS_PRELIMINARES.docx
│   ├── DOCUMENTO_DISEÑO_S4.docx
│   └── DOCUMENTO_DISEÑO_SOLUCION_VALIDACION_INICIAL.docx
│
├── .gitignore                          → Archivos a excluir de Git
├── README.md                           → Este archivo
└── requirements.txt                    → Dependencias Python
```

---

## 🚀 Fases del Proyecto

| Fase | Descripción | Estado | Métrica | Hito |
|------|-------------|--------|---------|------|
| **Fase 1** | Validación inter-anotador (100 registros) | ✅ **COMPLETA** | Kappa = **0.8794** | Alineación exitosa |
| **Fase 1B** | Anotación completa (1,500 registros) | ⏳ **EN PROGRESO** | Kappa ≥ 0.70 | Target: 20 Jul 2026 |
| **Fase 2** | Diseño MLOps Pipeline | 📋 Planificada | UML + Arquitectura | Target: 10 Ago 2026 |
| **Fase 3** | Desarrollo e implementación | 🔜 Por hacer | Pipeline funcional | Target: 31 Ago 2026 |
| **Fase 4** | Evaluación del modelo baseline | 🔜 Por hacer | F1-score ≥ 0.75 | Target: 14 Sep 2026 |
| **Fase 5** | Piloto y defensa final | 🔜 Por hacer | Presentación exitosa | Target: 21 Sep 2026 |

---

## 📊 Métricas Actuales (Fase 1)

### Cohen's Kappa Post-Alineación ✅

Después de sesión de alineación entre anotadores:

```
PATRICIA vs LUIS CRUEL:        0.9360  (95.8% acuerdo)  ✅
PATRICIA vs LUIS CHICA:        0.8735  (91.7% acuerdo)  ✅
LUIS CRUEL vs LUIS CHICA:      0.8286  (88.9% acuerdo)  ✅
─────────────────────────────────────────────────────────
PROMEDIO FINAL:                0.8794  (META: ≥ 0.70)   ✅
```

**Conclusión:** Equipo ALINEADO. Proceder a Fase 1B autorizado.

---

## 📥 Trazabilidad de Consolidación

El dataset final de 1,500 registros fue consolidado desde 4 fuentes independientes:

### Composición de fuentes

| Fuente | Registros Brutos | Registros Válidos | Rol en Consolidación |
|--------|------------------|------------------|----------------------|
| **clienty-prospectos 1** | 5,000 | 5,000 (100%) | CRM con metadata de cliente |
| **clienty-prospectos 2** | 3,143 | 3,143 (100%) | CRM complementario (clientes adicionales) |
| **JEVA base datos** | 1,155 | 1,150 (99.6%) | Datos maestros de empresa |
| **base_maestra_consolidada** | 5,676 | 5,676 (100%) | **FUENTE PRINCIPAL: Conversaciones textuales (WhatsApp + Instagram)** |
| **TOTAL BRUTO** | **14,974** | **14,969** | - |

### Proceso de consolidación

El dataset se construyó seleccionando 1,500 conversaciones textuales válidas de las 14,974 disponibles:

- **Deduplicación:** Eliminación de ~2,000 registros duplicados entre bases (mismo cliente/email/teléfono)
- **Filtrado por conversación:** Selección de registros con texto completo (~8,000 CRM puros descartados)
- **Limpieza final:** Eliminación de ~200 registros vacíos, corruptos o duplicados exactos
- **Retención global:** 10% (1,500 / 14,974) — La mayoría de datos eran metadata de contacto

**Detalles completos:** Ver `05_documentacion/REPORTE_CONSOLIDACION_4_BASES.txt`

---

## 🏷️ Categorías de Intención (7)

| Código | Definición | Ejemplo |
|--------|-----------|---------|
| **INF** | Información General | "¿Qué colores tienen?" |
| **COT** | Cotización / Presupuesto | "¿Cuánto cuesta el microcemento?" |
| **TEC** | Consulta Técnica | "¿Cómo se aplica el producto?" |
| **CUR** | Consulta de Cursos | "¿Cuándo es el próximo curso?" |
| **SEG** | Seguimiento | "¿En qué estado está mi cotización?" |
| **VEN** | Venta / Confirmación | "Confirmo la compra del producto" |
| **QUE** | Queja / Reclamo | "El producto llegó dañado" |

**Guía completa:** Ver `05_documentacion/GUIA_MEJORADA_ANOTACION_FASE1.txt`

---

## 🔧 Cómo Reproducir

### Requisitos

```bash
pip install -r requirements.txt
```

Incluye: pandas, numpy, scikit-learn, openpyxl, spacy

### Consolidar 4 bases → 1,500 registros

```bash
python 02_scripts/consolidar_4_bases.py
```

**Salida:** `04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx`

### Calcular Cohen's Kappa (Validación Inter-Anotador)

```bash
python 02_scripts/calcular_kappa.py 04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx
```

**Salida esperada:**
- Kappa por pares
- Kappa promedio
- Matriz de confusión
- Análisis de desacuerdos

### Ejecutar Pipeline Completo

```bash
python 02_scripts/06_pipeline_completo.py
```

Ejecuta: limpieza → consolidación → validación → versionado con DVC

---

## 📖 Documentación

### Fase 1: Validación Inter-Anotador

- **Metodología:** `05_documentacion/METODOLOGIA_ANOTACION.md`
- **Guía de Anotación v2.0:** `05_documentacion/GUIA_MEJORADA_ANOTACION_FASE1.txt`
  - Incluye "Regla de Oro" para resolver ambigüedades
  - Ejemplos reales de categorización
  - Resolución de casos límite
- **Cronograma Ejecutivo:** `05_documentacion/CRONOGRAMA_EJECUTIVO_FASE1.txt`

### Fase 1B: Anotación Completa (en progreso)

- **Dataset:** `04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx`
  - 1,500 registros reales de WhatsApp Business
  - Anotados por 3 anotadores independientes
  - Validación con Cohen's Kappa

### Trazabilidad de Consolidación

- **Reporte completo:** `05_documentacion/REPORTE_CONSOLIDACION_4_BASES.txt`
  - Estado inicial de 4 bases
  - Proceso de consolidación paso a paso
  - Detalles de limpieza y filtrado
  - Explicación de retención (10%)

### Resultados Preliminares

- **Informe Fase 1:** `06_resultados/INFORME_RESULTADOS_PRELIMINARES.docx`
  - Análisis de Cohen's Kappa
  - Distribución de intenciones
  - Próximos pasos

### Diseño Técnico

- **Documento S4:** `06_resultados/DOCUMENTO_DISEÑO_S4.docx`
  - Capítulo 7.1: Diseño detallado de la solución
  - Diagrama UML
  - MLOps Pipeline (5 etapas)
  - Baseline: Logistic Regression
  - Estrategias de validación
  - Ingeniería de características
  - Tratamiento de anomalías

---

## 🎯 Próximos Pasos (Roadmap)

### Semana 3-4 (Junio 29 - Julio 13)
- [ ] Continuar anotación de 1,500 registros
- [ ] Cada anotador: ~750 registros/semana
- [ ] Monitoreo de avance en Google Sheets

### Semana 5 (Julio 13-20)
- [ ] Finalizar anotación de 1,500
- [ ] Calcular Cohen's Kappa en 1,500
- [ ] Confirmar Kappa ≥ 0.70
- [ ] Generar informe final Fase 1B

### Semana 6-7 (Julio 20 - Agosto 3)
- [ ] **Fase 2: Diseño MLOps**
- [ ] Crear diagramas UML
- [ ] Diseñar arquitectura modular
- [ ] Definir esquema de experimentos

### Semana 8-11 (Agosto 3-31)
- [ ] **Fase 3: Desarrollo**
- [ ] Implementar ETL
- [ ] Entrenar modelos (Logistic Regression + SVM)
- [ ] Configurar MLflow
- [ ] CI/CD con GitHub Actions

### Semana 12-13 (Septiembre 1-14)
- [ ] **Fase 4: Evaluación**
- [ ] Calcular métricas (Precision, Recall, F1)
- [ ] Validar monitoreo de drift
- [ ] Ajustes finales

### Semana 14 (Septiembre 14-21)
- [ ] **Fase 5: Piloto y Defensa**
- [ ] Prueba piloto con datos Rocktec
- [ ] Generación de dashboards
- [ ] Defensa final

---

## 🔍 Validación de Calidad

### Ground Truth (Verdad Absoluta)

El dataset anotado es la "verdad absoluta" para entrenar modelos supervisados:

- ✅ Anotado por 3 anotadores independientes
- ✅ Validado con Cohen's Kappa ≥ 0.70
- ✅ Sesiones de alineación cuando hay desacuerdos
- ✅ Guía v2.0 mejorada con Regla de Oro
- ✅ Documentación de decisiones para casos ambiguos

### Garantías de Confiabilidad

| Aspecto | Garantía |
|---------|----------|
| **Inter-anotador** | Kappa ≥ 0.70 (riguroso) |
| **Reproducibilidad** | Scripts Python versionados |
| **Trazabilidad** | GitHub con historial completo |
| **Documentación** | Cada decisión registrada |

---

## 📝 Cómo Contribuir

1. **Clone el repositorio:**
   ```bash
   git clone https://github.com/LuisChica18/proyecto-mia-rocktec.git
   cd proyecto-mia-rocktec
   ```

2. **Cree una rama para su trabajo:**
   ```bash
   git checkout -b feature/nombre-descriptivo
   ```

3. **Haga cambios y commit:**
   ```bash
   git add .
   git commit -m "Descripción clara de cambios"
   ```

4. **Push y Pull Request:**
   ```bash
   git push origin feature/nombre-descriptivo
   ```

---

## 📧 Contacto y Coordinación

| Rol | Nombre | Responsable |
|-----|--------|-------------|
| Datos y Anotación | Patricia Mosquera | `@PatriciaMC` |
| Modelos y ML | Luis Cruel | `@luis-cruel` |
| Arquitectura | Luis Chica | `@luis-chica` |

**Reuniones:** Semanal (Zoom), coordinación en GitHub Issues

---

## 📄 Licencia

Proyecto académico de la Maestría en Inteligencia Artificial Aplicada (MIA).  
Propiedad intelectual compartida entre el equipo, MIA y Rocktec.

**Confidencialidad:** Los datos de conversaciones de Rocktec están anonimizados según acuerdo de confidencialidad.

---

## 🔗 Enlaces Importantes

- **Universidad:** Universidad de las Fuerzas Armadas (ESPE)
- **Programa:** Maestría en Inteligencia Artificial Aplicada
- **Empresa:** Rocktec (Soluciones de Concreto Decorativo)
- **GitHub:** https://github.com/LuisChica18/proyecto-mia-rocktec

---

## 📅 Fechas Clave

| Evento | Fecha |
|--------|-------|
| Fase 1 Completada | 15 Junio 2026 ✅ |
| Fase 1B Target | 20 Julio 2026 ⏳ |
| Fase 2 Target | 10 Agosto 2026 |
| Fase 3 Target | 31 Agosto 2026 |
| Fase 4 Target | 14 Septiembre 2026 |
| Defensa Final | 21 Septiembre 2026 |

---

## 📞 Soporte

Preguntas o problemas:
1. Revisar documentación en `05_documentacion/`
2. Abrir un **Issue** en GitHub
3. Contactar al equipo por email

---

**Última actualización:** Junio 2026  
**Estado:** Fase 1B en progreso  
**Próxima revisión:** Julio 20, 2026  

---

*Este proyecto demuestra aplicación de NLP, Machine Learning y MLOps en un contexto real de PYME ecuatoriana.*
