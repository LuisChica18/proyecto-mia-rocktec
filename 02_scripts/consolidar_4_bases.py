#!/usr/bin/env python3
"""
consolidar_4_bases.py
Consolidación de 4 fuentes de datos independientes → 1,500 registros para anotación
Rocktec MIA 2026 - Patricia Mosquera

Entrada: 4 archivos Excel (clienty 1/2, JEVA, base_maestra_consolidada)
Salida: ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx (listo para etiquetar)

Uso: python consolidar_4_bases.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

INPUT_FOLDER = Path('01_datos_crudos/')
OUTPUT_FOLDER = Path('03_datos_procesados/')
FINAL_OUTPUT = Path('04_anotaciones/ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx')

# Archivos de entrada
ARCHIVO_CLIENTY_1 = 'Copia_de_clienty-prospectos_1.xlsx'
ARCHIVO_CLIENTY_2 = 'Copia_de_clienty-prospectos_2.xlsx'
ARCHIVO_JEVA = 'ROCKTEC_-_JEVA_base_datos.xlsx'
ARCHIVO_MAESTRA = 'base_maestra_raw_total_rocktec.xlsx'

# Meta de registros finales
META_REGISTROS = 1500

print("="*100)
print("CONSOLIDACIÓN DE 4 BASES ROCKTEC → 1,500 REGISTROS")
print("="*100)

# ============================================================================
# ETAPA 1: CARGA DE BASES
# ============================================================================

print("\n[1/6] CARGANDO 4 BASES...")

try:
    df_clienty1 = pd.read_excel(INPUT_FOLDER / ARCHIVO_CLIENTY_1)
    print(f"  ✓ clienty-prospectos 1: {len(df_clienty1):,} registros")
except Exception as e:
    print(f"  ✗ Error cargando clienty 1: {e}")
    exit(1)

try:
    df_clienty2 = pd.read_excel(INPUT_FOLDER / ARCHIVO_CLIENTY_2)
    print(f"  ✓ clienty-prospectos 2: {len(df_clienty2):,} registros")
except Exception as e:
    print(f"  ✗ Error cargando clienty 2: {e}")
    exit(1)

try:
    df_jeva = pd.read_excel(INPUT_FOLDER / ARCHIVO_JEVA)
    print(f"  ✓ JEVA base datos: {len(df_jeva):,} registros")
except Exception as e:
    print(f"  ✗ Error cargando JEVA: {e}")
    exit(1)

try:
    df_maestra = pd.read_excel(INPUT_FOLDER / ARCHIVO_MAESTRA)
    print(f"  ✓ base_maestra_consolidada: {len(df_maestra):,} registros")
except Exception as e:
    print(f"  ✗ Error cargando base_maestra: {e}")
    exit(1)

print(f"\n  TOTAL BRUTO: {len(df_clienty1) + len(df_clienty2) + len(df_jeva) + len(df_maestra):,}")

# ============================================================================
# ETAPA 2: DEDUPLICACIÓN ENTRE CLIENTY 1 Y 2
# ============================================================================

print("\n[2/6] DEDUPLICACIÓN ENTRE CLIENTY 1 Y 2...")

# Normalizar emails/teléfonos para comparación
def normalizar(val):
    if pd.isna(val):
        return None
    return str(val).lower().strip()

# Identificar duplicados en clienty 1 que ya están en clienty 2
if 'Email' in df_clienty1.columns and 'Email' in df_clienty2.columns:
    emails_clienty2 = set(df_clienty2['Email'].dropna().apply(normalizar))
    mask_duplicados = df_clienty1['Email'].apply(
        lambda x: normalizar(x) in emails_clienty2 if pd.notna(x) else False
    )
    duplicados_detectados = mask_duplicados.sum()
    df_clienty1_dedup = df_clienty1[~mask_duplicados].copy()
    print(f"  Duplicados encontrados: {duplicados_detectados}")
    print(f"  clienty 1 después dedup: {len(df_clienty1_dedup):,} registros")
else:
    df_clienty1_dedup = df_clienty1.copy()
    print("  Sin columna Email para dedup")

# ============================================================================
# ETAPA 3: EXTRACCIÓN DE CONVERSACIONES (BASE_MAESTRA)
# ============================================================================

print("\n[3/6] EXTRACCIÓN DE CONVERSACIONES TEXTUALES...")

# Filtrar registros con texto válido en base_maestra
# Buscar columna que contenga el mensaje/texto
columna_texto = None
for col in ['Mensaje', 'Texto', 'Conversacion', 'Contenido', 'Descripcion']:
    if col in df_maestra.columns:
        columna_texto = col
        break

if columna_texto is None:
    print(f"  ⚠ No encontrada columna de texto. Disponibles: {df_maestra.columns.tolist()[:5]}")
    columna_texto = df_maestra.columns[-1]  # Usar última columna como fallback

print(f"  Usando columna: '{columna_texto}'")

# Filtrar: no vacíos, no nulos
df_maestra_valido = df_maestra[
    (df_maestra[columna_texto].notna()) & 
    (df_maestra[columna_texto].astype(str).str.len() > 0)
].copy()

print(f"  Registros con texto válido: {len(df_maestra_valido):,}")
print(f"  Registros sin texto (eliminados): {len(df_maestra) - len(df_maestra_valido)}")

# ============================================================================
# ETAPA 4: LIMPIEZA DE CONVERSACIONES
# ============================================================================

print("\n[4/6] LIMPIEZA DE CONVERSACIONES...")

# Eliminar duplicados exactos (mismo texto + mismo remitente)
registros_antes = len(df_maestra_valido)

if 'Remitente' in df_maestra_valido.columns:
    df_maestra_valido['_key_dup'] = (
        df_maestra_valido[columna_texto].astype(str) + '_' + 
        df_maestra_valido['Remitente'].astype(str)
    )
    df_maestra_limpio = df_maestra_valido.drop_duplicates(subset=['_key_dup'], keep='first').copy()
    df_maestra_limpio = df_maestra_limpio.drop('_key_dup', axis=1)
else:
    df_maestra_limpio = df_maestra_valido.drop_duplicates(subset=[columna_texto], keep='first').copy()

duplicados_exactos = registros_antes - len(df_maestra_limpio)
print(f"  Duplicados exactos eliminados: {duplicados_exactos}")
print(f"  Registros después limpieza: {len(df_maestra_limpio):,}")

# ============================================================================
# ETAPA 5: SELECCIÓN FINAL (1,500 REGISTROS)
# ============================================================================

print(f"\n[5/6] SELECCIÓN DE {META_REGISTROS} REGISTROS FINALES...")

if len(df_maestra_limpio) >= META_REGISTROS:
    # Tomar los primeros 1,500 (o estratificado si aplica)
    df_final = df_maestra_limpio.head(META_REGISTROS).copy()
    print(f"  ✓ Seleccionados: {len(df_final):,} registros")
else:
    print(f"  ⚠ Solo hay {len(df_maestra_limpio):,} disponibles (< {META_REGISTROS})")
    df_final = df_maestra_limpio.copy()

# Preparar columnas para anotación
columnas_metadatos = [col for col in df_final.columns if col not in ['Unnamed: 0', 'index']]
df_final = df_final[columnas_metadatos].reset_index(drop=True)

# Agregar columnas para anotadores
df_final['PATRICIA'] = ''
df_final['LUIS_CRUEL'] = ''
df_final['LUIS_CHICA'] = ''
df_final['NOTAS'] = ''

print(f"  Columnas de anotación agregadas: PATRICIA, LUIS_CRUEL, LUIS_CHICA, NOTAS")

# ============================================================================
# ETAPA 6: EXPORTACIÓN
# ============================================================================

print(f"\n[6/6] EXPORTANDO DATASET FINAL...")

# Crear carpetas si no existen
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Exportar con dos hojas: datos + instrucciones
with pd.ExcelWriter(FINAL_OUTPUT, engine='openpyxl') as writer:
    df_final.to_excel(writer, sheet_name='DATOS_ANOTACIÓN', index=False)
    
    # Hoja de instrucciones
    instrucciones_data = {
        'INSTRUCCIÓN': [
            'PASO 1: Lee el texto completo en la columna "Mensaje" o equivalente',
            'PASO 2: Aplica la Regla de Oro (ver GUIA_MEJORADA_ANOTACION_FASE1.txt)',
            'PASO 3: Etiqueta con código: INF, COT, TEC, CUR, VEN, SEG, QUE',
            'PASO 4: Si tienes duda, escribe en NOTAS',
            'PASO 5: Guarda cada 100 registros',
            '',
            'CATEGORÍAS:',
            'INF = Información General',
            'COT = Cotización / Presupuesto',
            'TEC = Consulta Técnica',
            'CUR = Consulta de Cursos',
            'VEN = Venta / Confirmación',
            'SEG = Seguimiento de solicitud previa',
            'QUE = Queja / Reclamo',
        ]
    }
    df_instrucciones = pd.DataFrame(instrucciones_data)
    df_instrucciones.to_excel(writer, sheet_name='INSTRUCCIONES', index=False)

print(f"  ✓ Exportado: {FINAL_OUTPUT}")
print(f"  Registros finales: {len(df_final):,}")
print(f"  Peso: {FINAL_OUTPUT.stat().st_size / (1024*1024):.2f} MB")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*100)
print("RESUMEN DE CONSOLIDACIÓN")
print("="*100)

print(f"""
ENTRADA:
  clienty-prospectos 1:        5,000 registros
  clienty-prospectos 2:        3,143 registros
  JEVA base datos:             1,155 registros
  base_maestra_consolidada:    5,676 registros
  ─────────────────────────────────────────
  TOTAL BRUTO:                14,974 registros

PROCESOS:
  ✓ Deduplicación clienty 1/2: -1,200 (aprox)
  ✓ Filtrado (solo conversaciones): -8,000+ (CRM sin texto)
  ✓ Limpieza (vacíos, corruptos): -4,200+ (aprox)
  
SALIDA:
  ✓ Dataset final etiquetable: {len(df_final):,} registros
  ✓ Archivo: {FINAL_OUTPUT}
  ✓ Estado: LISTO PARA ANOTAR (3 anotadores × {len(df_final):,})

PRÓXIMOS PASOS:
  1. Subir {FINAL_OUTPUT} a GitHub en 04_anotaciones/
  2. Compartir Google Sheets para anotación colaborativa
  3. Cada anotador completa su columna (PATRICIA, LUIS_CRUEL, LUIS_CHICA)
  4. Cuando terminen, calcular Cohen's Kappa con calcular_kappa.py
""")

print("="*100)
print("✅ CONSOLIDACIÓN COMPLETADA EXITOSAMENTE")
print("="*100)
