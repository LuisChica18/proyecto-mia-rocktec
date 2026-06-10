# Datos Procesados - ROCKTEC MIA 2026

## Descripción
Esta carpeta contiene los datos generados por el pipeline de limpieza, consolidación y validación.

## Archivos Principales

### rocktec_base_validada.csv ⭐ USAR PARA FASE 1
- Registros: 9,317
- Contenido: Datos limpios, consolidados y sin duplicados exactos
- Uso: Anotación inter-anotador (Cohen Kappa)

### rocktec_base_consolidada.csv
- Registros: 13,413
- Contenido: Antes de eliminar duplicados (para referencia)

### crm_limpio.csv
- Registros: 8,143
- Fuente: Archivos CRM originales

### whatsapp_limpio.csv
- Registros: 5,676
- Fuente: base_maestra_raw_total_rocktec.xlsx

## Estadísticas

Datos cargados:         13,819
Consolidados:           13,413
Duplicados eliminados:   4,096
Base final:              9,317 registros

## Próximos Pasos
1. Seleccionar 50 conversaciones de rocktec_base_validada.csv
2. Anotación inter-anotador (Fase 1)
3. Calcular Cohen's Kappa >= 0.70
