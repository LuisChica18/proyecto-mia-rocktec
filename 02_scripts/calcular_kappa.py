"""
calcular_kappa.py
Validación inter-anotador con Cohen's Kappa — Rocktec MIA 2026

Carga el archivo de anotaciones, calcula Kappa por pares y promedio,
muestra acuerdo por categoría, y si Kappa >= 0.70 genera el CSV de consenso
listo para entrenamiento.

Uso: python calcular_kappa.py [ruta_archivo]
     python calcular_kappa.py 04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx
"""

import sys
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, classification_report

ANOTADORES = {
    'PATRICIA':   'Patricia Mosquera',
    'LUIS_CRUEL': 'Luis Cruel',
    'LUIS_CHICA': 'Luis Chica',
}
INTENCIONES  = ['INF', 'COT', 'TEC', 'CUR', 'VEN', 'SEG', 'QUE']
KAPPA_META   = 0.70
RUTA_DEFAULT = Path('04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────────────────

def cargar_anotaciones(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        print(f"✗ Archivo no encontrado: {ruta}")
        sys.exit(1)

    df = (pd.read_excel(ruta, sheet_name='DATOS_ANOTACIÓN')
          if ruta.suffix == '.xlsx'
          else pd.read_csv(ruta))

    print(f"✓ Registros cargados: {len(df):,}")

    for col in ANOTADORES:
        if col not in df.columns:
            print(f"✗ Columna faltante: '{col}'")
            sys.exit(1)

    # Conservar solo registros completamente anotados
    df_ok = df.dropna(subset=list(ANOTADORES)).copy()
    df_ok = df_ok[
        df_ok[list(ANOTADORES)].apply(
            lambda r: all(str(v).strip() != '' for v in r), axis=1
        )
    ]

    # Normalizar a mayúsculas
    for col in ANOTADORES:
        df_ok[col] = df_ok[col].str.strip().str.upper()

    # Solo etiquetas válidas
    mascara = df_ok[list(ANOTADORES)].apply(
        lambda c: c.isin(INTENCIONES)
    ).all(axis=1)
    df_ok = df_ok[mascara]

    print(f"✓ Registros con anotación completa y válida: {len(df_ok):,}")
    if len(df_ok) == 0:
        print("✗ Sin registros válidos para calcular Kappa.")
        sys.exit(1)

    return df_ok


# ─────────────────────────────────────────────────────────────────────────────
# Kappa
# ─────────────────────────────────────────────────────────────────────────────

def calcular_kappa_pares(df):
    print("\n" + "=" * 70)
    print("COHEN'S KAPPA — VALIDACIÓN INTER-ANOTADOR")
    print("=" * 70)

    resultados = {}
    for a1, a2 in combinations(ANOTADORES, 2):
        kappa   = cohen_kappa_score(df[a1], df[a2])
        acuerdo = (df[a1] == df[a2]).mean() * 100
        nivel   = (
            "EXCELENTE"    if kappa >= 0.80 else
            "SUSTANCIAL"   if kappa >= 0.70 else
            "MODERADO"     if kappa >= 0.60 else
            "INSUFICIENTE"
        )
        marca = "✅" if kappa >= KAPPA_META else "❌"
        print(f"\n  {ANOTADORES[a1]}  vs  {ANOTADORES[a2]}")
        print(f"    Kappa:   {kappa:.4f}  [{nivel}]  {marca}")
        print(f"    Acuerdo: {acuerdo:.1f}%")
        resultados[f"{a1}_vs_{a2}"] = kappa

    promedio = np.mean(list(resultados.values()))
    marca    = "✅" if promedio >= KAPPA_META else "❌"
    print(f"\n{'─' * 70}")
    print(f"  PROMEDIO FINAL: {promedio:.4f}   (META ≥ {KAPPA_META})   {marca}")
    print("=" * 70)
    return resultados, promedio


def acuerdo_por_intencion(df):
    print("\n" + "=" * 70)
    print("ACUERDO POR CATEGORÍA")
    print("=" * 70)
    cols = list(ANOTADORES)
    for cod in INTENCIONES:
        # Registros donde al menos un anotador usó este código
        mask = df[cols].apply(lambda r: cod in r.values, axis=1)
        sub  = df[mask]
        if sub.empty:
            continue
        acuerdo = sub.apply(lambda r: len(set(r[cols].values)) == 1, axis=1).sum()
        print(f"  {cod}: {acuerdo:3d}/{len(sub):3d}  ({acuerdo/len(sub)*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Consenso
# ─────────────────────────────────────────────────────────────────────────────

def _consenso_fila(row, cols):
    votos = [row[c] for c in cols]
    for v in votos:
        if votos.count(v) >= 2:
            return v
    return votos[0]   # triple desacuerdo → primer anotador


def generar_consenso(df):
    cols = list(ANOTADORES)
    df   = df.copy()
    df['intencion_consenso'] = df.apply(_consenso_fila, cols=cols, axis=1)

    triple = df.apply(lambda r: len(set(r[cols].values)) == 3, axis=1).sum()
    print(f"\n  Registros con triple desacuerdo (usa {list(ANOTADORES)[0]}): "
          f"{triple} ({triple/len(df)*100:.1f}%)")

    dist = df['intencion_consenso'].value_counts()
    print("\n  Distribución final (consenso):")
    for cod, n in dist.items():
        print(f"    {cod}: {n:4d}  ({n/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RUTA_DEFAULT
    print(f"Archivo: {ruta}\n")

    df                    = cargar_anotaciones(ruta)
    resultados, promedio  = calcular_kappa_pares(df)
    acuerdo_por_intencion(df)

    if promedio >= KAPPA_META:
        print(f"\n✅ FASE 1B APROBADA — Kappa promedio: {promedio:.4f}")
        df_consenso = generar_consenso(df)

        ruta_salida = Path('04_anotaciones/dataset_consenso_final.csv')
        ruta_salida.parent.mkdir(exist_ok=True)
        df_consenso.to_csv(ruta_salida, index=False, encoding='utf-8')
        print(f"\n✓ Dataset de consenso guardado: {ruta_salida}")
        print(  "  → Listo para usar en 05_entrenar_modelos.py")
    else:
        print(f"\n❌ KAPPA INSUFICIENTE ({promedio:.4f} < {KAPPA_META})")
        print(  "   Se requiere sesión de alineación antes de continuar.")


if __name__ == '__main__':
    main()
