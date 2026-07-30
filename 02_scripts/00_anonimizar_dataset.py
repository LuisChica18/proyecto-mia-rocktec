"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 00_anonimizar_dataset.py  v2.0
Anonimización de datos sensibles — versión corregida
================================================================================
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime

RUTA_ENTRADA = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_SALIDA  = Path('04_anotaciones/dataset_consenso_final_anonimizado.csv')
RUTA_REPORTE = Path('06_resultados/reporte_anonimizacion.txt')

# Teléfonos ecuatorianos
RE_TELEFONO = re.compile(r'\b(?:\+593\s?|0)9\d{8}\b|\b09\d{8}\b|\b\+593\d{9}\b', re.IGNORECASE)

# Emails
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')

# RUC/cédula
RE_DOCUMENTO = re.compile(r'\b\d{10,13}\b')

# Nombre después de "Soy X" o "Mi nombre es X" — captura 1-3 palabras con mayúscula
RE_SOY = re.compile(
    r'\b(soy|mi nombre es|me llamo|llámame|llamame)\s+'
    r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})',
    re.IGNORECASE
)

# Nombre después de "Cliente: X" seguido de "Asesor:" o "|" o fin
# Captura nombre propio (1-3 palabras con mayúscula inicial)
RE_CLIENTE_TAG = re.compile(
    r'(Cliente:\s*)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})'
    r'(?=\s*(?:Asesor:|Cliente:|\||$))',
    re.IGNORECASE
)

# Nombre repetido por el asesor: "Excelente Marcelino" / "Gracias Maria"
RE_NOMBRE_ASESOR = re.compile(
    r'\b(Excelente|Gracias|Hola|Claro|Perfecto|Bienvenid[ao])\s+'
    r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b'
)


def anonimizar_texto(texto):
    if not isinstance(texto, str):
        return texto

    # 1. Teléfonos
    texto = RE_TELEFONO.sub('[TELEFONO]', texto)

    # 2. Emails
    texto = RE_EMAIL.sub('[EMAIL]', texto)

    # 3. Documentos
    texto = RE_DOCUMENTO.sub('[DOCUMENTO]', texto)

    # 4. "Soy X" / "Mi nombre es X"
    texto = RE_SOY.sub(lambda m: m.group(1) + ' [CLIENTE]', texto)

    # 5. "Cliente: Nombre Apellido" antes de Asesor o |
    texto = RE_CLIENTE_TAG.sub(lambda m: m.group(1) + '[CLIENTE]', texto)

    # 6. Asesor repite nombre: "Excelente Marcelino"
    texto = RE_NOMBRE_ASESOR.sub(lambda m: m.group(1) + ' [CLIENTE]', texto)

    return texto


def main():
    print("=" * 70)
    print("ANONIMIZACIÓN DE DATOS v2.0 — ROCKTEC MIA 2026")
    print("=" * 70)

    df = pd.read_csv(RUTA_ENTRADA)
    print(f"  Total registros: {len(df)}")

    df_anon = df.copy()
    df_anon['texto_conversacion'] = df['texto_conversacion'].apply(anonimizar_texto)

    cambios = (df_anon['texto_conversacion'] != df['texto_conversacion']).sum()
    print(f"  Registros modificados: {cambios}")

    # Verificar ejemplos
    print("\n  Verificación:")
    diff = df[df_anon['texto_conversacion'] != df['texto_conversacion']]
    for i, row in diff.head(4).iterrows():
        print(f"\n  ORIG: {df.loc[i,'texto_conversacion'][200:350]}")
        print(f"  ANON: {df_anon.loc[i,'texto_conversacion'][200:350]}")

    df_anon.to_csv(RUTA_SALIDA, index=False, encoding='utf-8')
    print(f"\n✅ Guardado: {RUTA_SALIDA}")
    print(f"   {cambios} registros anonimizados de {len(df)}")

    reporte = f"""
================================================================================
REPORTE ANONIMIZACIÓN v2.0 — ROCKTEC MIA 2026
================================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Registros procesados: {len(df)}
Registros modificados: {cambios} ({cambios/len(df)*100:.1f}%)

REEMPLAZOS:
  [CLIENTE]   → nombres de clientes
  [TELEFONO]  → números telefónicos 09XXXXXXXX
  [EMAIL]     → correos electrónicos
  [DOCUMENTO] → RUC/cédulas

Archivo: {RUTA_SALIDA}
================================================================================
"""
    Path('06_resultados').mkdir(exist_ok=True)
    RUTA_REPORTE.write_text(reporte, encoding='utf-8')
    print(f"  ✓ Reporte: {RUTA_REPORTE}")


if __name__ == '__main__':
    main()
