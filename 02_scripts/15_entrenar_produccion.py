"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 15_entrenar_produccion.py
Fase 3 — Entrenamiento del artefacto final de producción (TF-IDF + LR)
================================================================================

13_evaluacion_holdout.py mide el F1 honesto reservando holdout_test.csv (nunca
usado para entrenar). Ese número (F1-macro = 0.7938) ya quedó reportado como el
cierre de Fase 4 — una vez medido, seguir reservando esas 197 filas solo le resta
datos al modelo que realmente va a predecir en producción.

Este script reentrena el modelo de producción usando el 100% de los datos
etiquetados (train_val.csv + holdout_test.csv), con los mismos hiperparámetros
y metodología de selección que 13_evaluacion_holdout.py, y lo guarda como un
artefacto versionado explícito — separado de 06_resultados/modelos/ (ese es el
experimento de 05_entrenar_modelos.py, entrenado antes de que existiera el split
train_val/holdout, y en formato MLflow pensado para tracking de experimentos,
no para servir).

Salida:
    06_resultados/modelos/produccion/vectorizador_tfidf.pkl
    06_resultados/modelos/produccion/modelo_lr.pkl
    06_resultados/modelos/produccion/metadata.json

Uso (desde la raíz del repo, no requiere GPU):
    python 02_scripts/15_entrenar_produccion.py
================================================================================
"""

import sys
import json
import pickle
import importlib.util
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

warnings.filterwarnings('ignore')

INTENCIONES  = ['INF', 'COT', 'TEC', 'CUR', 'VEN']
RANDOM_STATE = 42
VERSION      = 'v1.0'

RUTA_TRAIN_VAL      = Path('04_anotaciones/train_val.csv')
RUTA_HOLDOUT        = Path('04_anotaciones/holdout_test.csv')
RUTA_SALIDA         = Path('06_resultados/modelos/produccion')
F1_REFERENCIA_HONESTO = 0.7938  # de 06_resultados/reporte_holdout_final.txt (13_evaluacion_holdout.py)

# ── Reutiliza el vectorizador TF-IDF + features manuales del pipeline oficial ──
_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec    = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe      = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)
VectorizadorTFIDF = _fe.VectorizadorTFIDF


def cargar_todo_lo_etiquetado():
    train_val = pd.read_csv(RUTA_TRAIN_VAL)
    holdout   = pd.read_csv(RUTA_HOLDOUT)
    df = pd.concat([train_val, holdout], ignore_index=True)
    df = df[df['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    return df


def main():
    print("=" * 70)
    print(f"ENTRENAMIENTO DE PRODUCCIÓN ({VERSION}) — TF-IDF + Logistic Regression")
    print("=" * 70)

    df = cargar_todo_lo_etiquetado()
    print(f"  Entrenando con el 100% de los datos etiquetados: {len(df)} filas "
          f"(train_val.csv + holdout_test.csv — Fase 4 ya midió y reportó el F1 honesto)")

    vec = VectorizadorTFIDF()
    X = vec.fit_transform(df['texto_conversacion'].values)
    y = df['intencion_consenso'].values

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [1000]}
    base_lr = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(base_lr, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
    gs.fit(X, y)
    modelo = gs.best_estimator_

    print(f"  Mejor C: {gs.best_params_['C']}")
    print(f"  F1-macro en CV interna (100% de los datos — optimista, NO es la métrica final): {gs.best_score_:.4f}")

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    vec.guardar(RUTA_SALIDA / 'vectorizador_tfidf.pkl')
    with open(RUTA_SALIDA / 'modelo_lr.pkl', 'wb') as f:
        pickle.dump(modelo, f)

    metadata = {
        'version': VERSION,
        'fecha_entrenamiento': datetime.now().isoformat(timespec='seconds'),
        'n_filas_entrenamiento': int(len(df)),
        'fuente_datos': [str(RUTA_TRAIN_VAL), str(RUTA_HOLDOUT)],
        'clases': INTENCIONES,
        'mejor_C': gs.best_params_['C'],
        'f1_macro_cv_interna_100pct_datos': round(float(gs.best_score_), 4),
        'f1_macro_referencia_honesto_holdout': F1_REFERENCIA_HONESTO,
        'nota_f1_referencia': (
            "El F1=0.7938 viene de 13_evaluacion_holdout.py, medido sobre un modelo "
            "entrenado SOLO con train_val.csv (sin las filas de holdout). Este artefacto "
            "de producción se reentrena incluyendo esas mismas filas de holdout, por lo que "
            "ya no queda ningún conjunto reservado para remedir F1 sin fuga — 0.7938 sigue "
            "siendo la mejor estimación honesta disponible del desempeño esperado en producción."
        ),
    }
    with open(RUTA_SALIDA / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Artefactos de producción guardados en {RUTA_SALIDA}/")
    print("✅ ENTRENAMIENTO DE PRODUCCIÓN COMPLETADO")


if __name__ == '__main__':
    main()
