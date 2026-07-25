"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 12_beto_finetuning.py
Ajuste S7 #1b — Fine-tuning real de BETO (requiere GPU)
================================================================================

A diferencia de 11_beto_clasificador.py (BETO como extractor de embeddings fijos
+ Logistic Regression, sin GPU), este script hace fine-tuning real de BETO
(AutoModelForSequenceClassification) sobre las 5 clases modeladas, con los
hiperparámetros documentados en 05_documentacion/DISEÑO_MLOPS_FASE2.md
(Etapa 3 — Modelo 3: BETO Fine-tuned).

Pensado para correr en un entorno con GPU (Google Colab / Kaggle Notebooks con
runtime T4 gratuito, o una máquina con CUDA). En CPU también corre, pero muy
lento (no recomendado).

Salida:
    06_resultados/beto/reporte_beto_finetuned.txt
    06_resultados/beto/comparacion_tfidf_vs_beto_finetuned.txt
    06_resultados/modelos/beto_finetuned_best/   (checkpoint HuggingFace)

Uso:
    python 02_scripts/12_beto_finetuning.py

Requiere torch + transformers + accelerate (ya en requirements.txt, sección
Fase 2: MLOps Pipeline). En Colab/Kaggle: pip install -r requirements.txt
o al menos `pip install torch transformers accelerate`.

NOTA: Primera ejecución descarga BETO (~440MB). Requiere conexión a internet.
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime

warnings.filterwarnings('ignore')

INTENCIONES     = ['INF', 'COT', 'TEC', 'CUR', 'VEN']
RANDOM_STATE    = 42
RUTA_CONSENSO   = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_SALIDA     = Path('06_resultados/beto')
RUTA_CHECKPOINT = Path('06_resultados/modelos/beto_finetuned_best')
MODELO_BETO     = 'dccuchile/bert-base-spanish-wwm-cased'

# Hiperparámetros — igual a los documentados en DISEÑO_MLOPS_FASE2.md (Etapa 3, Modelo 3)
EPOCAS        = 5
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
MAX_LENGTH    = 128

# Resultados de referencia ya reportados (para la comparación final)
F1_TFIDF_LR        = 0.7516  # 06_resultados/reporte_holdout.txt / CHANGELOG.md Sprint 7
F1_BETO_EMBEDDINGS = 0.6370  # 06_resultados/beto/comparacion_tfidf_vs_beto.txt (script 11)


def cargar_datos():
    print("[1/5] Cargando dataset...")
    df = pd.read_csv(RUTA_CONSENSO)
    df = df[df['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    print(f"  ✓ {len(df)} registros, {len(INTENCIONES)} clases")
    for cls, n in df['intencion_consenso'].value_counts().items():
        print(f"    {cls}: {n}")
    return df


class DatasetIntenciones:
    """Wrapper mínimo tipo torch.utils.data.Dataset sobre encodings ya tokenizados."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def entrenar_beto_finetuned(df):
    """Fine-tuning real de BETO (clasificación de secuencias) sobre las 5 clases."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )

    usa_gpu = torch.cuda.is_available()
    print(f"[2/5] GPU disponible: {usa_gpu}"
          + (f" ({torch.cuda.get_device_name(0)})" if usa_gpu else " — será muy lento en CPU"))

    label2id = {c: i for i, c in enumerate(INTENCIONES)}
    id2label = {i: c for c, i in label2id.items()}

    textos = df['texto_conversacion'].astype(str).tolist()
    labels = df['intencion_consenso'].map(label2id).tolist()

    # Mismo split de test (80/20, seed=42, estratificado) que usan 11_beto_clasificador.py
    # y la comparación TF-IDF+LR, para que el F1-macro final sea comparable entre los tres.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        textos, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    # Dentro del 80%, separa validación (para elegir la mejor época) del entrenamiento.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=RANDOM_STATE, stratify=y_trainval
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("[3/5] Cargando BETO + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BETO)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BETO, num_labels=len(INTENCIONES), id2label=id2label, label2id=label2id
    )

    def tokenizar(textos_batch):
        return tokenizer(textos_batch, padding=True, truncation=True, max_length=MAX_LENGTH)

    train_ds = DatasetIntenciones(tokenizar(X_train), y_train)
    val_ds = DatasetIntenciones(tokenizar(X_val), y_val)
    test_ds = DatasetIntenciones(tokenizar(X_test), y_test)

    def compute_metrics(eval_pred):
        logits, etiquetas = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {'f1_macro': f1_score(etiquetas, preds, average='macro', zero_division=0)}

    args = TrainingArguments(
        output_dir=str(RUTA_CHECKPOINT.parent / 'beto_finetuned_checkpoints'),
        num_train_epochs=EPOCAS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        fp16=usa_gpu,
        report_to='none',
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("[4/5] Fine-tuning (5 épocas)...")
    trainer.train()

    print("[5/5] Evaluando en test set (held-out, no visto en entrenamiento/validación)...")
    pred_output = trainer.predict(test_ds)
    y_pred = np.argmax(pred_output.predictions, axis=-1)
    y_test_lbl = [id2label[i] for i in y_test]
    y_pred_lbl = [id2label[i] for i in y_pred]

    f1 = f1_score(y_test_lbl, y_pred_lbl, labels=INTENCIONES, average='macro', zero_division=0)
    reporte = classification_report(y_test_lbl, y_pred_lbl, labels=INTENCIONES, zero_division=0)
    print(f"\n  [BETO fine-tuned]")
    print(f"  F1-macro: {f1:.4f}")
    print(reporte)

    RUTA_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(RUTA_CHECKPOINT))
    tokenizer.save_pretrained(str(RUTA_CHECKPOINT))
    print(f"  ✓ Checkpoint guardado en {RUTA_CHECKPOINT}")

    return f1, reporte


def main():
    print("=" * 70)
    print("BETO FINE-TUNING — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    df = cargar_datos()
    f1_ft, rep_ft = entrenar_beto_finetuned(df)

    comparacion = f"""
================================================================================
COMPARACIÓN TF-IDF vs BETO (embeddings) vs BETO (fine-tuned) — ROCKTEC MIA 2026
================================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: consenso humano ({len(df)} registros, 5 clases)
Split test: 80/20 estratificado, random_state=42 (mismo split que script 11)

RESULTADOS:
  TF-IDF + LR (full dataset):              F1-macro = {F1_TFIDF_LR:.4f}  ✅ META ≥ 0.75
  BETO embeddings + LR (sin fine-tuning):  F1-macro = {F1_BETO_EMBEDDINGS:.4f}
  BETO fine-tuned ({EPOCAS} épocas):              F1-macro = {f1_ft:.4f}{"  ✅ META ≥ 0.75" if f1_ft >= 0.75 else ""}

INTERPRETACIÓN:
  - BETO fine-tuned {"supera" if f1_ft > F1_TFIDF_LR else "no supera"} a TF-IDF + LR
  - BETO fine-tuned {"supera" if f1_ft > F1_BETO_EMBEDDINGS else "no supera"} a BETO sin fine-tuning (como era de esperar,
    el ajuste de pesos sobre el dominio de construcción/concreto decorativo
    ecuatoriano debería mejorar sobre los embeddings genéricos)

CONCLUSIÓN:
  {"El fine-tuning de BETO iguala o mejora el F1-macro sobre TF-IDF+LR; evaluar" if f1_ft >= F1_TFIDF_LR else "TF-IDF + LR sigue siendo el modelo recomendado para producción:"}
  {"si el costo de infraestructura GPU se justifica para producción." if f1_ft >= F1_TFIDF_LR else "- Ya alcanza la meta F1-macro ≥ 0.75, es interpretable y no requiere GPU."}

================================================================================
"""
    print(comparacion)

    (RUTA_SALIDA / 'reporte_beto_finetuned.txt').write_text(rep_ft, encoding='utf-8')
    (RUTA_SALIDA / 'comparacion_tfidf_vs_beto_finetuned.txt').write_text(
        comparacion, encoding='utf-8'
    )
    print(f"✓ Guardado: {RUTA_SALIDA}/reporte_beto_finetuned.txt")
    print(f"✓ Guardado: {RUTA_SALIDA}/comparacion_tfidf_vs_beto_finetuned.txt")
    print("\n✅ BETO FINE-TUNING COMPLETADO")


if __name__ == '__main__':
    main()
