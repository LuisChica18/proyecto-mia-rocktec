"""
05_entrenar_modelos.py
Pipeline de entrenamiento con 3 modelos + MLflow — Rocktec MIA 2026

Modelos:
  1. Logistic Regression   (TF-IDF + features manuales)
  2. LinearSVC             (TF-IDF + features manuales)
  3. BETO fine-tuned       (dccuchile/bert-base-spanish-wwm-cased)

Registra cada experimento en MLflow (./mlruns/).
Genera reporte comparativo en 06_resultados/.

Uso: python 05_entrenar_modelos.py [--skip-beto]
     --skip-beto  omite BETO (útil sin GPU o para prueba rápida del pipeline)
"""

import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, accuracy_score,
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Importar módulo de feature engineering (04_feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────

_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec    = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe      = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe  # requerido por pickle al guardar VectorizadorTFIDF
_spec.loader.exec_module(_fe)

PreprocessadorTexto   = _fe.PreprocessadorTexto
VectorizadorTFIDF     = _fe.VectorizadorTFIDF
CodificadorIntenciones = _fe.CodificadorIntenciones
cargar_dataset        = _fe.cargar_dataset
INTENCIONES           = _fe.INTENCIONES

# ─────────────────────────────────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────────────────────────────────

SEED             = 42
TEST_SIZE        = 0.20
N_FOLDS          = 5
EXPERIMENTO_MLFLOW = 'rocktec-intent-classification'
RUTA_MODELOS     = Path('06_resultados/modelos')
RUTA_REPORTES    = Path('06_resultados')

BETO_MODEL_NAME  = 'dccuchile/bert-base-spanish-wwm-cased'
BETO_MAX_LEN     = 128
BETO_BATCH_SIZE  = 16
BETO_EPOCHS      = 5
BETO_LR          = 2e-5


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de evaluación
# ─────────────────────────────────────────────────────────────────────────────

def metricas(y_true, y_pred, clases):
    return {
        'f1_macro':        f1_score(y_true, y_pred, average='macro',    zero_division=0),
        'f1_weighted':     f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro':    recall_score(y_true, y_pred, average='macro',    zero_division=0),
        'accuracy':        accuracy_score(y_true, y_pred),
        **{f'f1_{c}': f1_score(y_true, y_pred, labels=[c], average='micro', zero_division=0)
           for c in clases},
    }


def guardar_confusion_matrix(y_true, y_pred, clases, nombre, ruta_dir):
    cm   = confusion_matrix(y_true, y_pred, labels=clases)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(clases)), yticks=range(len(clases)),
           xticklabels=clases, yticklabels=clases,
           xlabel='Predicción', ylabel='Real', title=f'Confusion Matrix — {nombre}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    ruta = ruta_dir / f'confusion_matrix_{nombre}.png'
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


# ─────────────────────────────────────────────────────────────────────────────
# Modelo 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_lr(X_train, y_train, X_test, y_test, clases, ruta_artefactos):
    print("\n[1/3] Logistic Regression")
    print("─" * 50)

    param_grid = {'C': [0.1, 1, 10, 100], 'max_iter': [1000]}
    base_lr    = LogisticRegression(class_weight='balanced', solver='lbfgs',
                                    random_state=SEED)
    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gs         = GridSearchCV(base_lr, param_grid, cv=cv, scoring='f1_macro',
                              n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    modelo    = gs.best_estimator_
    y_pred    = modelo.predict(X_test)
    clases_arr = np.array(clases)
    y_test_lbl, y_pred_lbl = clases_arr[y_test], clases_arr[y_pred]
    m         = metricas(y_test_lbl, y_pred_lbl, clases)
    reporte   = classification_report(y_test_lbl, y_pred_lbl, labels=clases, zero_division=0)
    ruta_cm   = guardar_confusion_matrix(y_test_lbl, y_pred_lbl, clases, 'logistic_regression', ruta_artefactos)

    print(f"  Mejor C: {gs.best_params_['C']}")
    print(f"  F1-macro (test): {m['f1_macro']:.4f}")
    print(f"  Accuracy (test): {m['accuracy']:.4f}")

    with mlflow.start_run(run_name='logistic_regression'):
        mlflow.log_params({'modelo': 'LogisticRegression', **gs.best_params_,
                           'class_weight': 'balanced', 'cv_folds': N_FOLDS})
        mlflow.log_metrics(m)
        mlflow.log_artifact(str(ruta_cm))

        ruta_reporte = ruta_artefactos / 'reporte_lr.txt'
        ruta_reporte.write_text(reporte, encoding='utf-8')
        mlflow.log_artifact(str(ruta_reporte))

        ruta_modelo = ruta_artefactos / 'modelo_lr.pkl'
        mlflow.sklearn.save_model(modelo, str(ruta_modelo))
        mlflow.log_artifact(str(ruta_modelo))

    return modelo, m


# ─────────────────────────────────────────────────────────────────────────────
# Modelo 2: LinearSVC
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_svm(X_train, y_train, X_test, y_test, clases, ruta_artefactos):
    print("\n[2/3] LinearSVC")
    print("─" * 50)

    param_grid = {'estimator__C': [0.01, 0.1, 1, 10]}
    base_svm   = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, random_state=SEED)
    )
    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gs         = GridSearchCV(base_svm, param_grid, cv=cv, scoring='f1_macro',
                              n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    modelo  = gs.best_estimator_
    y_pred  = modelo.predict(X_test)
    clases_arr = np.array(clases)
    y_test_lbl, y_pred_lbl = clases_arr[y_test], clases_arr[y_pred]
    m       = metricas(y_test_lbl, y_pred_lbl, clases)
    reporte = classification_report(y_test_lbl, y_pred_lbl, labels=clases, zero_division=0)
    ruta_cm = guardar_confusion_matrix(y_test_lbl, y_pred_lbl, clases, 'svm', ruta_artefactos)

    print(f"  Mejor C: {gs.best_params_['estimator__C']}")
    print(f"  F1-macro (test): {m['f1_macro']:.4f}")
    print(f"  Accuracy (test): {m['accuracy']:.4f}")

    with mlflow.start_run(run_name='linear_svc'):
        mlflow.log_params({'modelo': 'LinearSVC', **gs.best_params_,
                           'class_weight': 'balanced', 'cv_folds': N_FOLDS})
        mlflow.log_metrics(m)
        mlflow.log_artifact(str(ruta_cm))

        ruta_reporte = ruta_artefactos / 'reporte_svm.txt'
        ruta_reporte.write_text(reporte, encoding='utf-8')
        mlflow.log_artifact(str(ruta_reporte))

        ruta_modelo = ruta_artefactos / 'modelo_svm.pkl'
        mlflow.sklearn.save_model(modelo, str(ruta_modelo))
        mlflow.log_artifact(str(ruta_modelo))

    return modelo, m


# ─────────────────────────────────────────────────────────────────────────────
# Modelo 3: BETO fine-tuned
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_beto(train_textos, train_labels, test_textos, test_labels,
                  clases, cod, ruta_artefactos):
    print("\n[3/3] BETO fine-tuned (dccuchile/bert-base-spanish-wwm-cased)")
    print("─" * 50)

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            BertTokenizerFast, BertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
    except ImportError:
        print("  ✗ torch/transformers no instalados — omitiendo BETO")
        print("    Ejecutar: pip install torch transformers")
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Dispositivo: {device}")

    label2id = {c: i for i, c in enumerate(clases)}
    id2label = {i: c for c, i in label2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained(BETO_MODEL_NAME)

    class IntentDataset(Dataset):
        def __init__(self, textos, labels):
            self.enc    = tokenizer(list(textos), truncation=True, padding=True,
                                    max_length=BETO_MAX_LEN, return_tensors='pt')
            self.labels = torch.tensor([label2id[l] for l in labels], dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]

    train_ds = IntentDataset(train_textos, train_labels)
    test_ds  = IntentDataset(test_textos,  test_labels)
    train_dl = DataLoader(train_ds, batch_size=BETO_BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BETO_BATCH_SIZE)

    modelo = BertForSequenceClassification.from_pretrained(
        BETO_MODEL_NAME,
        num_labels=len(clases),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer   = AdamW(modelo.parameters(), lr=BETO_LR, weight_decay=0.01)
    total_steps = len(train_dl) * BETO_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    mejor_f1        = 0.0
    historial_loss  = []

    for epoch in range(1, BETO_EPOCHS + 1):
        # ── Entrenamiento ──
        modelo.train()
        total_loss = 0.0
        for batch_enc, batch_labels in train_dl:
            batch_enc    = {k: v.to(device) for k, v in batch_enc.items()}
            batch_labels = batch_labels.to(device)
            outputs      = modelo(**batch_enc, labels=batch_labels)
            loss         = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss  += loss.item()

        avg_loss = total_loss / len(train_dl)
        historial_loss.append(avg_loss)

        # ── Validación ──
        modelo.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch_enc, batch_labels in test_dl:
                batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
                logits    = modelo(**batch_enc).logits
                preds_all.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                labels_all.extend(batch_labels.numpy())

        y_pred_lbl  = [id2label[p] for p in preds_all]
        y_true_lbl  = [id2label[l] for l in labels_all]
        f1_ep       = f1_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
        print(f"  Época {epoch}/{BETO_EPOCHS} — loss: {avg_loss:.4f}  F1-macro: {f1_ep:.4f}")

        if f1_ep > mejor_f1:
            mejor_f1 = f1_ep
            ruta_ckpt = ruta_artefactos / 'beto_best'
            modelo.save_pretrained(str(ruta_ckpt))
            tokenizer.save_pretrained(str(ruta_ckpt))

    # ── Evaluación final con mejor checkpoint ──
    modelo_final = BertForSequenceClassification.from_pretrained(
        str(ruta_artefactos / 'beto_best')
    ).to(device)
    modelo_final.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch_enc, batch_labels in test_dl:
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            logits    = modelo_final(**batch_enc).logits
            preds_all.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels_all.extend(batch_labels.numpy())

    y_pred_lbl = [id2label[p] for p in preds_all]
    y_true_lbl = [id2label[l] for l in labels_all]
    m          = metricas(y_true_lbl, y_pred_lbl, clases)
    reporte    = classification_report(y_true_lbl, y_pred_lbl, labels=clases, zero_division=0)
    ruta_cm    = guardar_confusion_matrix(y_true_lbl, y_pred_lbl, clases, 'beto', ruta_artefactos)

    print(f"\n  F1-macro final (test): {m['f1_macro']:.4f}")
    print(f"  Accuracy final (test): {m['accuracy']:.4f}")

    # Curva de loss
    fig, ax = plt.subplots()
    ax.plot(range(1, BETO_EPOCHS + 1), historial_loss, marker='o')
    ax.set(xlabel='Época', ylabel='Loss', title='BETO — Curva de entrenamiento')
    ruta_loss = ruta_artefactos / 'beto_training_loss.png'
    plt.savefig(ruta_loss, dpi=150)
    plt.close()

    with mlflow.start_run(run_name='beto_finetuned'):
        mlflow.log_params({
            'modelo': 'BETO', 'base_model': BETO_MODEL_NAME,
            'epochs': BETO_EPOCHS, 'batch_size': BETO_BATCH_SIZE,
            'lr': BETO_LR, 'max_len': BETO_MAX_LEN, 'device': str(device),
        })
        mlflow.log_metrics(m)
        for art in [ruta_cm, ruta_loss]:
            mlflow.log_artifact(str(art))

        ruta_reporte = ruta_artefactos / 'reporte_beto.txt'
        ruta_reporte.write_text(reporte, encoding='utf-8')
        mlflow.log_artifact(str(ruta_reporte))
        mlflow.log_artifact(str(ruta_artefactos / 'beto_best'))

    return modelo_final, m


# ─────────────────────────────────────────────────────────────────────────────
# Reporte comparativo
# ─────────────────────────────────────────────────────────────────────────────

def reporte_comparativo(resultados, ruta_dir):
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(f"  {'Modelo':<22} {'F1-macro':>10} {'Accuracy':>10} {'F1-weighted':>12}")
    print("  " + "─" * 56)

    mejor_modelo, mejor_f1 = None, 0.0
    for nombre, m in resultados.items():
        if m is None:
            continue
        marca = " ← mejor" if m['f1_macro'] == max(
            r['f1_macro'] for r in resultados.values() if r
        ) else ""
        print(f"  {nombre:<22} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} "
              f"{m['f1_weighted']:>12.4f}{marca}")
        if m['f1_macro'] > mejor_f1:
            mejor_f1, mejor_modelo = m['f1_macro'], nombre

    meta = "✅" if mejor_f1 >= 0.75 else "❌"
    print(f"\n  Mejor modelo: {mejor_modelo}  —  F1-macro: {mejor_f1:.4f}  "
          f"(META ≥ 0.75) {meta}")

    # Guardar JSON
    ruta_json = ruta_dir / 'comparacion_modelos.json'
    ruta_json.write_text(
        json.dumps({k: v for k, v in resultados.items() if v}, indent=2),
        encoding='utf-8'
    )
    print(f"\n✓ Resultados guardados: {ruta_json}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    skip_beto = '--skip-beto' in sys.argv

    print("=" * 70)
    print("PIPELINE DE ENTRENAMIENTO — ROCKTEC MIA 2026")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Directorios ──
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    df = cargar_dataset()

    X_texto    = df['texto_conversacion'].values
    y_raw      = df['label'].values

    X_train_txt, X_test_txt, y_train_raw, y_test_raw = train_test_split(
        X_texto, y_raw, test_size=TEST_SIZE, stratify=y_raw, random_state=SEED
    )

    # ── Codificación ──
    cod      = CodificadorIntenciones()
    clases   = cod.clases

    # ── Features TF-IDF para LR y SVM ──
    print("\nGenerando features TF-IDF...")
    vec        = VectorizadorTFIDF()
    X_train    = vec.fit_transform(X_train_txt)
    X_test     = vec.transform(X_test_txt)
    y_train    = cod.encode(y_train_raw)
    y_test     = cod.encode(y_test_raw)

    vec.guardar(RUTA_MODELOS / 'vectorizador_tfidf.pkl')
    print(f"✓ Train: {X_train.shape}  Test: {X_test.shape}")

    # ── MLflow ──
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)

    resultados = {}

    # ── Modelo 1: LR ──
    _, resultados['logistic_regression'] = entrenar_lr(
        X_train, y_train, X_test, y_test, clases, RUTA_MODELOS
    )

    # ── Modelo 2: SVM ──
    _, resultados['linear_svc'] = entrenar_svm(
        X_train, y_train, X_test, y_test, clases, RUTA_MODELOS
    )

    # ── Modelo 3: BETO ──
    if skip_beto:
        print("\n[3/3] BETO — omitido por --skip-beto")
        resultados['beto_finetuned'] = None
    else:
        _, resultados['beto_finetuned'] = entrenar_beto(
            X_train_txt, y_train_raw,
            X_test_txt,  y_test_raw,
            clases, cod, RUTA_MODELOS
        )

    # ── Comparativa final ──
    reporte_comparativo(resultados, RUTA_REPORTES)

    print(f"\n✅ Pipeline completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Resultados MLflow: mlflow ui --backend-store-uri mlruns")


if __name__ == '__main__':
    main()
