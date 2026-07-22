"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 09_crear_holdout_set.py
Ajuste S7 #2 — Validación externa con holdout set
================================================================================

Objetivo:
    Separar UN ÚNICO conjunto de test fijo (15%) ANTES de cualquier
    experimento de entrenamiento. Este holdout set:
    - NUNCA se usa para entrenar ni para ajustar hiperparámetros
    - Es la validación externa definitiva del modelo final
    - Garantiza que el F1-macro reportado es sobre datos no vistos

Entrada:
    04_anotaciones/dataset_consenso_final.csv

Salida:
    04_anotaciones/holdout_test.csv       ← 15% — NO TOCAR durante entrenamiento
    04_anotaciones/train_val.csv          ← 85% — para entrenar y validar
    06_resultados/reporte_holdout.txt     ← distribución y estadísticas

Uso:
    python 02_scripts/09_crear_holdout_set.py

IMPORTANTE: Correr UNA SOLA VEZ. El random_state=42 garantiza
reproducibilidad — siempre genera el mismo split.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

INTENCIONES_5 = ['INF', 'COT', 'TEC', 'CUR', 'VEN']
RANDOM_STATE  = 42
HOLDOUT_SIZE  = 0.15

RUTA_CONSENSO = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_HOLDOUT  = Path('04_anotaciones/holdout_test.csv')
RUTA_TRAIN    = Path('04_anotaciones/train_val.csv')
RUTA_REPORTE  = Path('06_resultados/reporte_holdout.txt')

def main():
    print("=" * 70)
    print("CREACIÓN DE HOLDOUT SET — ROCKTEC MIA 2026")
    print("=" * 70)

    # Verificar que no existe ya
    if RUTA_HOLDOUT.exists():
        print("⚠ ADVERTENCIA: holdout_test.csv ya existe.")
        print("  El holdout set debe crearse UNA SOLA VEZ.")
        print("  Si lo regeneras, los experimentos anteriores quedan invalidados.")
        resp = input("  ¿Deseas sobreescribir? (escribe 'SI' para confirmar): ")
        if resp.strip().upper() != 'SI':
            print("  Operación cancelada. El holdout existente se mantiene.")
            return

    # Cargar dataset
    print(f"\n[1/3] Cargando dataset...")
    df = pd.read_csv(RUTA_CONSENSO)
    df = df[df['intencion_consenso'].isin(INTENCIONES_5)].reset_index(drop=True)
    print(f"  Total registros (5 clases): {len(df)}")

    print("\n  Distribución completa:")
    for cls, n in df['intencion_consenso'].value_counts().items():
        print(f"    {cls}: {n:4d}  ({n/len(df)*100:.1f}%)")

    # Split estratificado
    print(f"\n[2/3] Separando holdout ({int(HOLDOUT_SIZE*100)}%) estratificado...")
    train_val, holdout = train_test_split(
        df,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['intencion_consenso']
    )

    print(f"  Train+Val: {len(train_val)} registros")
    print(f"  Holdout:   {len(holdout)} registros")

    print("\n  Distribución holdout:")
    for cls, n in holdout['intencion_consenso'].value_counts().items():
        pct_hold = n / len(holdout) * 100
        pct_orig = df['intencion_consenso'].value_counts()[cls] / len(df) * 100
        print(f"    {cls}: {n:3d}  ({pct_hold:.1f}%)  ← original: {pct_orig:.1f}%")

    # Guardar
    print(f"\n[3/3] Guardando...")
    RUTA_HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)

    train_val.to_csv(RUTA_TRAIN, index=False, encoding='utf-8')
    holdout.to_csv(RUTA_HOLDOUT, index=False, encoding='utf-8')

    print(f"  ✓ train_val.csv:    {len(train_val)} registros")
    print(f"  ✓ holdout_test.csv: {len(holdout)} registros")

    # Reporte
    reporte = f"""
================================================================================
REPORTE HOLDOUT SET — ROCKTEC MIA 2026
================================================================================
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Random state: {RANDOM_STATE} (reproducible)
Clases: {INTENCIONES_5}

SPLIT:
  Dataset completo (5 clases): {len(df)} registros
  Train+Val (85%):             {len(train_val)} registros
  Holdout test (15%):          {len(holdout)} registros

DISTRIBUCIÓN HOLDOUT:
"""
    for cls, n in holdout['intencion_consenso'].value_counts().items():
        reporte += f"  {cls}: {n:3d}  ({n/len(holdout)*100:.1f}%)\n"

    reporte += """
INSTRUCCIONES:
  - train_val.csv   → usar para entrenar y validar modelos (cross-validation)
  - holdout_test.csv → usar UNA SOLA VEZ al final para reportar métricas finales
  - NUNCA usar holdout_test.csv para ajustar hiperparámetros o seleccionar modelos

================================================================================
"""
    RUTA_REPORTE.write_text(reporte, encoding='utf-8')
    print(f"  ✓ reporte_holdout.txt guardado")

    print("\n" + "=" * 70)
    print("✅ HOLDOUT SET CREADO EXITOSAMENTE")
    print("=" * 70)
    print("  IMPORTANTE: NO volver a correr este script.")
    print("  El holdout es sagrado — solo se usa al final para reportar F1 final.")
    print("=" * 70)

if __name__ == '__main__':
    main()
