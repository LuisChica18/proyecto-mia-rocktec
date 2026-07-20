# Datos Procesados - ROCKTEC MIA 2026

## Descripción
Esta carpeta contiene los datos generados por dos generaciones del pipeline de limpieza,
consolidación y validación: el pipeline original de 2 fuentes (CRM + WhatsApp) y el pipeline
de 4 bases (`consolidar_4_bases.py`) que produjo la muestra de 1,500 registros para Fase 1B.

## Pipeline de 4 bases (actual — usado para la anotación de Fase 1B)

### rocktec_base_consolidada_1500.csv / .xlsx ⭐ MUESTRA ANOTADA (FASE 1B)
- Registros: 1,500
- Contenido: muestra final consolidada de las 4 fuentes (CRM1, CRM2, JEVA, WhatsApp), enviada a
  `04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx` para anotación inter-anotador.

### crm_limpio_1.csv
- Registros: 5,000 — fuente: `Copia de clienty-prospectos 1.xlsx`

### crm_limpio_2.csv
- Registros: 3,143 — fuente: `Copia de clienty-prospectos 2.xlsx`

### jeva_limpio.csv
- Registros: 1,155 — fuente: `ROCKTEC - JEVA base datos.xlsx`

## Pipeline original de 2 fuentes (histórico)

### rocktec_base_validada.csv
- Registros: 9,317
- Contenido: CRM + WhatsApp limpios, consolidados y sin duplicados exactos.
- Uso actual: **fallback heurístico** en `04_feature_engineering.py::cargar_dataset()` cuando no
  existe `04_anotaciones/dataset_consenso_final.csv` — sus etiquetas (`intencion_catalogo`) son
  heurísticas por palabras clave, no anotaciones humanas.

### rocktec_base_consolidada.csv
- Registros: 13,413 — antes de eliminar duplicados (para referencia).

### crm_limpio.csv
- Registros: 8,143 — fuente: archivos CRM originales (pipeline de 2 fuentes).

### whatsapp_limpio.csv
- Registros: 5,676 — fuente: `base_maestra_raw_total_rocktec.xlsx`.

## Estado (Fase 1B completada)

- 1,500 registros anotados por 3 evaluadores → Cohen's Kappa = 0.8851
- Dataset de consenso real: `04_anotaciones/dataset_consenso_final.csv` (1,297 filas con
  consenso válido, generado por `02_scripts/calcular_kappa.py`)
- Ver `CHANGELOG.md` (raíz del repo) y `06_resultados/INFORME_AJUSTES_Y_VALIDACION.docx` para el
  detalle de los ajustes de Fase 1C y la comparación baseline (heurístico) vs. ajustado (consenso).

## Próximos Pasos
1. Fase 2: diseño del pipeline MLOps (ver `05_documentacion/DISEÑO_MLOPS_FASE2.md`)
2. Ampliar la anotación humana para reducir el desbalance de clases raras (QUE, SEG)
