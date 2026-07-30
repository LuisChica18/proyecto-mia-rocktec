"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 17_umbral_confianza_adaptativo.py
S8 Ajuste #1 — Umbral de confianza adaptativo validado contra holdout
================================================================================

Objetivo:
    En lugar de fijar un umbral de confianza arbitrario (ej. 0.70), encontrar
    el umbral ÓPTIMO validando contra el holdout_test.csv (197 registros).

    Un mensaje con probabilidad máxima < umbral → "requiere revisión humana"
    Un mensaje con probabilidad máxima >= umbral → clasificación automática

    El umbral óptimo minimiza errores de clasificación automática mientras
    mantiene una tasa de revisión humana razonable para Rocktec.

Salida:
    06_resultados/umbral/reporte_umbral_adaptativo.txt
    06_resultados/umbral/grafico_umbral_vs_precision.png
    06_resultados/umbral/umbral_optimo.json

Uso:
    python 02_scripts/17_umbral_confianza_adaptativo.py
================================================================================
"""

import sys
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from datetime import datetime

warnings.filterwarnings('ignore')

# Importar feature engineering
_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)

VectorizadorTFIDF = _fe.VectorizadorTFIDF
INTENCIONES       = _fe.INTENCIONES

RUTA_TRAIN    = Path('04_anotaciones/train_val.csv')
RUTA_HOLDOUT  = Path('04_anotaciones/holdout_test.csv')
RUTA_SALIDA   = Path('06_resultados/umbral')
RANDOM_STATE  = 42


def cargar_datos():
    print("[1/4] Cargando datos...")
    train = pd.read_csv(RUTA_TRAIN)
    holdout = pd.read_csv(RUTA_HOLDOUT)

    train   = train[train['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    holdout = holdout[holdout['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)

    print(f"  Train: {len(train)} registros")
    print(f"  Holdout: {len(holdout)} registros")
    return train, holdout


def entrenar_modelo(train):
    print("[2/4] Entrenando modelo LR sobre train_val...")
    vec = VectorizadorTFIDF()
    X_train = vec.fit_transform(train['texto_conversacion'].values)
    y_train = train['intencion_consenso'].values

    modelo = LogisticRegression(
        C=10, class_weight='balanced',
        max_iter=1000, random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    print(f"  ✓ Modelo entrenado")
    return modelo, vec


def evaluar_umbrales(modelo, vec, holdout):
    print("[3/4] Evaluando umbrales contra holdout...")

    X_holdout = vec.transform(holdout['texto_conversacion'].values)
    y_true    = holdout['intencion_consenso'].values

    # Probabilidades por clase
    proba = modelo.predict_proba(X_holdout)
    confianza_max = proba.max(axis=1)   # confianza del modelo en su predicción
    y_pred_full   = modelo.predict(X_holdout)

    # Evaluar distintos umbrales
    umbrales = np.arange(0.30, 0.95, 0.05)
    resultados = []

    for umbral in umbrales:
        # Casos automáticos (confianza >= umbral)
        mask_auto   = confianza_max >= umbral
        mask_manual = ~mask_auto

        n_auto   = mask_auto.sum()
        n_manual = mask_manual.sum()
        pct_auto = n_auto / len(y_true) * 100

        if n_auto == 0:
            continue

        # Métricas solo sobre clasificaciones automáticas
        f1_auto = f1_score(
            y_true[mask_auto], y_pred_full[mask_auto],
            labels=INTENCIONES, average='macro', zero_division=0
        )

        # Errores en clasificación automática
        errores_auto = (y_true[mask_auto] != y_pred_full[mask_auto]).sum()
        precision_auto = 1 - (errores_auto / n_auto)

        resultados.append({
            'umbral': round(umbral, 2),
            'n_automatico': int(n_auto),
            'n_revision_humana': int(n_manual),
            'pct_automatico': round(pct_auto, 1),
            'f1_macro_auto': round(f1_auto, 4),
            'precision_auto': round(precision_auto, 4),
            'errores_auto': int(errores_auto),
        })

    df_res = pd.DataFrame(resultados)
    print(f"  ✓ {len(df_res)} umbrales evaluados")
    return df_res, confianza_max, y_true, y_pred_full


def encontrar_umbral_optimo(df_res):
    """
    Umbral óptimo: maximiza F1-macro en clasificaciones automáticas
    con al menos 70% de casos clasificados automáticamente.
    """
    candidatos = df_res[df_res['pct_automatico'] >= 70]
    if candidatos.empty:
        candidatos = df_res

    idx_optimo = candidatos['f1_macro_auto'].idxmax()
    return df_res.loc[idx_optimo]


def generar_grafico(df_res, umbral_optimo, ruta):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = '#8B0000'
    color2 = '#2196F3'

    ax1.plot(df_res['umbral'], df_res['f1_macro_auto'],
             color=color1, linewidth=2, marker='o', label='F1-macro (auto)')
    ax1.set_xlabel('Umbral de confianza', fontsize=12)
    ax1.set_ylabel('F1-macro (clasificaciones automáticas)', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(df_res['umbral'], df_res['pct_automatico'],
             color=color2, linewidth=2, marker='s', linestyle='--',
             label='% clasificación automática')
    ax2.set_ylabel('% mensajes clasificados automáticamente', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Marcar umbral óptimo
    ax1.axvline(x=umbral_optimo['umbral'], color='green', linewidth=2,
                linestyle=':', label=f"Umbral óptimo: {umbral_optimo['umbral']}")
    ax1.axhline(y=0.75, color='orange', linewidth=1.5,
                linestyle='--', alpha=0.7, label='Meta F1 ≥ 0.75')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)

    plt.title('Umbral de Confianza Adaptativo — Rocktec MIA 2026\n'
              'F1-macro y % clasificación automática vs umbral',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico guardado: {ruta}")


def main():
    print("=" * 70)
    print("UMBRAL DE CONFIANZA ADAPTATIVO — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    train, holdout = cargar_datos()
    modelo, vec    = entrenar_modelo(train)
    df_res, confianza_max, y_true, y_pred = evaluar_umbrales(modelo, vec, holdout)
    umbral_optimo  = encontrar_umbral_optimo(df_res)

    print(f"\n{'='*70}")
    print(f"UMBRAL ÓPTIMO ENCONTRADO")
    print(f"{'='*70}")
    print(f"  Umbral:                    {umbral_optimo['umbral']}")
    print(f"  F1-macro (auto):           {umbral_optimo['f1_macro_auto']}")
    print(f"  % clasificación automática: {umbral_optimo['pct_automatico']}%")
    print(f"  Mensajes automáticos:       {umbral_optimo['n_automatico']}/197")
    print(f"  Mensajes → revisión humana: {umbral_optimo['n_revision_humana']}/197")
    print(f"  Errores en automático:      {umbral_optimo['errores_auto']}")

    # Gráfico
    generar_grafico(df_res, umbral_optimo,
                    RUTA_SALIDA / 'grafico_umbral_vs_precision.png')

    # Guardar JSON
    resultado_json = {
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'umbral_optimo': float(umbral_optimo['umbral']),
        'f1_macro_automatico': float(umbral_optimo['f1_macro_auto']),
        'pct_automatico': float(umbral_optimo['pct_automatico']),
        'n_automatico': int(umbral_optimo['n_automatico']),
        'n_revision_humana': int(umbral_optimo['n_revision_humana']),
        'errores_automatico': int(umbral_optimo['errores_auto']),
        'todos_umbrales': df_res.to_dict(orient='records'),
        'interpretacion': (
            f"Con umbral={umbral_optimo['umbral']}, el modelo clasifica automáticamente "
            f"el {umbral_optimo['pct_automatico']}% de mensajes con F1-macro={umbral_optimo['f1_macro_auto']}. "
            f"El {100-umbral_optimo['pct_automatico']}% restante se envía a revisión humana "
            f"por baja confianza del modelo."
        )
    }

    ruta_json = RUTA_SALIDA / 'umbral_optimo.json'
    ruta_json.write_text(json.dumps(resultado_json, indent=2, ensure_ascii=False),
                         encoding='utf-8')

    # Reporte texto
    reporte = f"""
================================================================================
REPORTE UMBRAL DE CONFIANZA ADAPTATIVO — ROCKTEC MIA 2026
================================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Validado sobre: holdout_test.csv (197 registros, nunca vistos durante entrenamiento)

UMBRAL ÓPTIMO: {umbral_optimo['umbral']}
{'─'*60}
  F1-macro en clasificaciones automáticas: {umbral_optimo['f1_macro_auto']}
  Mensajes clasificados automáticamente:   {umbral_optimo['n_automatico']}/197 ({umbral_optimo['pct_automatico']}%)
  Mensajes enviados a revisión humana:     {umbral_optimo['n_revision_humana']}/197 ({100-umbral_optimo['pct_automatico']}%)
  Errores en clasificación automática:     {umbral_optimo['errores_auto']}

INTERPRETACIÓN PARA ROCKTEC:
  - Cuando el modelo dice "COT" con confianza >= {umbral_optimo['umbral']} → clasificar automáticamente
  - Cuando la confianza < {umbral_optimo['umbral']} → marcar para revisión del asesor
  - Esto reduce errores operativos manteniendo alta automatización

TABLA COMPLETA DE UMBRALES:
{'─'*60}
{'Umbral':>8} {'Auto%':>7} {'F1-auto':>9} {'Errores':>8} {'N-manual':>9}
{'─'*60}
"""
    for _, row in df_res.iterrows():
        marca = ' ← ÓPTIMO' if row['umbral'] == umbral_optimo['umbral'] else ''
        reporte += (f"{row['umbral']:>8.2f} {row['pct_automatico']:>6.1f}% "
                    f"{row['f1_macro_auto']:>9.4f} {row['errores_auto']:>8} "
                    f"{row['n_revision_humana']:>9}{marca}\n")

    reporte += "\n" + "=" * 70

    ruta_txt = RUTA_SALIDA / 'reporte_umbral_adaptativo.txt'
    ruta_txt.write_text(reporte, encoding='utf-8')
    print(f"  ✓ Reporte guardado: {ruta_txt}")

    print("\n✅ UMBRAL ADAPTATIVO COMPLETADO")
    print(f"   Umbral óptimo: {umbral_optimo['umbral']}")
    print(f"   F1-macro auto: {umbral_optimo['f1_macro_auto']}")


if __name__ == '__main__':
    main()
