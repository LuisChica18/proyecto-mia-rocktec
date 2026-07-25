"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 13_evaluacion_holdout.py
Ajuste S7 #2b — Evaluación final sobre holdout (una sola vez)
================================================================================

Calcula la métrica final y honesta de Fase 4: F1-macro sobre
`04_anotaciones/holdout_test.csv` (197 filas, nunca usadas para entrenar ni
para elegir hiperparámetros — ver 09_crear_holdout_set.py).

Dos modelos evaluados:
  1. TF-IDF + Logistic Regression — se REENTRENA aquí mismo desde cero usando
     solo `04_anotaciones/train_val.csv` (GridSearchCV sobre esas 1,115 filas),
     y se evalúa una única vez sobre el holdout. No reutiliza
     06_resultados/modelos/modelo_lr.pkl porque ese modelo fue entrenado sobre
     el dataset completo (incluye las filas de holdout — fuga de datos).
  2. BETO fine-tuned — carga el checkpoint de
     06_resultados/modelos/beto_finetuned_best/ (generado por
     12_beto_finetuning.py) y lo evalúa sobre el holdout, PERO SOLO SI el
     archivo FUENTE_ENTRENAMIENTO.txt del checkpoint confirma que se entrenó
     exclusivamente con train_val.csv. Si el checkpoint no existe o tiene una
     fuente distinta (p.ej. una versión vieja entrenada sobre el dataset
     completo), este script lo omite con una advertencia en vez de reportar
     un número contaminado.

Salida:
    06_resultados/reporte_holdout_final.txt

Uso (desde la raíz del repo):
    python 02_scripts/13_evaluacion_holdout.py
================================================================================
"""

import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

warnings.filterwarnings('ignore')

INTENCIONES     = ['INF', 'COT', 'TEC', 'CUR', 'VEN']
RANDOM_STATE    = 42
RUTA_TRAIN_VAL  = Path('04_anotaciones/train_val.csv')
RUTA_HOLDOUT    = Path('04_anotaciones/holdout_test.csv')
RUTA_CHECKPOINT = Path('06_resultados/modelos/beto_finetuned_best')
RUTA_SALIDA     = Path('06_resultados/reporte_holdout_final.txt')

# ── Reutiliza el vectorizador TF-IDF + features manuales del pipeline oficial ──
_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec    = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe      = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)
VectorizadorTFIDF = _fe.VectorizadorTFIDF


def cargar_split():
    train_val = pd.read_csv(RUTA_TRAIN_VAL)
    holdout   = pd.read_csv(RUTA_HOLDOUT)
    train_val = train_val[train_val['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    holdout   = holdout[holdout['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    print(f"  train_val.csv: {len(train_val)} filas  |  holdout_test.csv: {len(holdout)} filas")
    return train_val, holdout


def evaluar_tfidf_lr(train_val, holdout):
    print("\n[1/2] TF-IDF + Logistic Regression — reentrenando solo con train_val.csv")
    print("─" * 70)

    vec = VectorizadorTFIDF()
    X_train = vec.fit_transform(train_val['texto_conversacion'].values)
    X_holdout = vec.transform(holdout['texto_conversacion'].values)
    y_train = train_val['intencion_consenso'].values
    y_holdout = holdout['intencion_consenso'].values

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [1000]}
    base_lr = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(base_lr, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    modelo = gs.best_estimator_
    y_pred = modelo.predict(X_holdout)

    f1 = f1_score(y_holdout, y_pred, labels=INTENCIONES, average='macro', zero_division=0)
    acc = accuracy_score(y_holdout, y_pred)
    reporte = classification_report(y_holdout, y_pred, labels=INTENCIONES, zero_division=0)

    print(f"  Mejor C (CV sobre train_val): {gs.best_params_['C']}")
    print(f"  F1-macro (holdout, evaluación única): {f1:.4f}")
    print(f"  Accuracy (holdout): {acc:.4f}")
    print(reporte)

    return {'f1_macro': f1, 'accuracy': acc, 'reporte': reporte, 'mejor_C': gs.best_params_['C']}


def evaluar_beto_finetuned(holdout):
    print("\n[2/2] BETO fine-tuned — verificando checkpoint antes de evaluar")
    print("─" * 70)

    marca = RUTA_CHECKPOINT / 'FUENTE_ENTRENAMIENTO.txt'
    if not RUTA_CHECKPOINT.exists() or not marca.exists():
        print(f"  ⚠ No se encontró {marca} — se omite BETO fine-tuned.")
        print("    Corre `python 02_scripts/12_beto_finetuning.py` (en Colab/GPU) primero.")
        return None

    contenido = marca.read_text(encoding='utf-8')
    if str(RUTA_TRAIN_VAL) not in contenido:
        print(f"  ⚠ El checkpoint en {RUTA_CHECKPOINT} NO se entrenó con {RUTA_TRAIN_VAL}")
        print("    (fuente registrada distinta — probablemente entrenado sobre el dataset")
        print("    completo, lo que contaminaría el holdout). Se omite para evitar reportar")
        print("    un número inválido. Vuelve a correr 12_beto_finetuning.py y repite este paso.")
        print(f"    Contenido de la marca:\n{contenido}")
        return None

    print(f"  ✓ Checkpoint verificado — entrenado solo con {RUTA_TRAIN_VAL}")
    print(f"    {contenido.strip()}")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(RUTA_CHECKPOINT))
    model = AutoModelForSequenceClassification.from_pretrained(str(RUTA_CHECKPOINT))
    model.eval()

    textos = holdout['texto_conversacion'].astype(str).tolist()
    y_holdout = holdout['intencion_consenso'].tolist()
    id2label = model.config.id2label

    preds = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            batch = textos[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            logits = model(**encoded).logits
            preds.extend(logits.argmax(dim=-1).tolist())
    y_pred = [id2label[p] for p in preds]

    f1 = f1_score(y_holdout, y_pred, labels=INTENCIONES, average='macro', zero_division=0)
    acc = accuracy_score(y_holdout, y_pred)
    reporte = classification_report(y_holdout, y_pred, labels=INTENCIONES, zero_division=0)

    print(f"  F1-macro (holdout, evaluación única): {f1:.4f}")
    print(f"  Accuracy (holdout): {acc:.4f}")
    print(reporte)

    return {'f1_macro': f1, 'accuracy': acc, 'reporte': reporte}


def main():
    print("=" * 70)
    print("EVALUACIÓN FINAL SOBRE HOLDOUT (una sola vez) — ROCKTEC MIA 2026")
    print("=" * 70)

    train_val, holdout = cargar_split()

    res_lr = evaluar_tfidf_lr(train_val, holdout)
    res_beto = evaluar_beto_finetuned(holdout)

    lineas = [
        "=" * 80,
        "REPORTE FINAL — EVALUACIÓN ÚNICA SOBRE HOLDOUT — ROCKTEC MIA 2026",
        "=" * 80,
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Holdout: {RUTA_HOLDOUT} ({len(holdout)} filas, 5 clases, nunca antes evaluadas)",
        f"Entrenamiento: {RUTA_TRAIN_VAL} ({len(train_val)} filas) — sin fuga de datos",
        "",
        "RESULTADOS:",
        f"  TF-IDF + LR (C={res_lr['mejor_C']}):  F1-macro = {res_lr['f1_macro']:.4f}  "
        f"Accuracy = {res_lr['accuracy']:.4f}" + ("  ✅ META ≥ 0.75" if res_lr['f1_macro'] >= 0.75 else ""),
    ]
    if res_beto is not None:
        lineas.append(
            f"  BETO fine-tuned:            F1-macro = {res_beto['f1_macro']:.4f}  "
            f"Accuracy = {res_beto['accuracy']:.4f}"
            + ("  ✅ META ≥ 0.75" if res_beto['f1_macro'] >= 0.75 else "")
        )
    else:
        lineas.append("  BETO fine-tuned:            NO EVALUADO (checkpoint ausente o con fuga de datos — ver log)")

    lineas += [
        "",
        "REPORTE DETALLADO — TF-IDF + LR:",
        res_lr['reporte'],
    ]
    if res_beto is not None:
        lineas += [
            "REPORTE DETALLADO — BETO fine-tuned:",
            res_beto['reporte'],
        ]
    lineas.append("=" * 80)

    reporte_final = "\n".join(lineas)
    print("\n" + reporte_final)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA.write_text(reporte_final, encoding='utf-8')
    print(f"\n✓ Guardado: {RUTA_SALIDA}")
    print("\n✅ EVALUACIÓN HOLDOUT COMPLETADA")


if __name__ == '__main__':
    main()
