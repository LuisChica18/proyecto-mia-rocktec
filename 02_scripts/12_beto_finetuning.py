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

IMPORTANTE — evita fuga de datos hacia el holdout: este script entrena
EXCLUSIVAMENTE con `04_anotaciones/train_val.csv` (las 1,115 filas que
09_crear_holdout_set.py separó para entrenamiento/validación). NUNCA lee
`holdout_test.csv` ni `dataset_consenso_final.csv` completo — esas 197 filas
de holdout deben quedar totalmente fuera del entrenamiento para que
13_evaluacion_holdout.py pueda reportar una métrica final honesta. (Versiones
anteriores de este script entrenaban sobre el dataset completo, incluyendo por
error las filas de holdout — ver CHANGELOG.md Sprint 7, 25 Jul 2026.)

Pensado para correr en un entorno con GPU (Google Colab / Kaggle Notebooks con
runtime T4 gratuito, o una máquina con CUDA). En CPU también corre, pero muy
lento (no recomendado).

Salida:
    06_resultados/beto/reporte_beto_finetuned_val.txt       (métricas de validación interna,
                                                               NO son la métrica final)
    06_resultados/modelos/beto_finetuned_best/               (checkpoint HuggingFace)
    06_resultados/modelos/beto_finetuned_best/FUENTE_ENTRENAMIENTO.txt
        (marca de qué archivo se usó para entrenar — 13_evaluacion_holdout.py la valida
        antes de evaluar sobre el holdout, para no reportar por error un resultado con fuga)

La métrica final de este modelo (F1-macro sobre holdout_test.csv, una sola vez) la calcula
13_evaluacion_holdout.py — no este script.

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
RUTA_TRAIN_VAL  = Path('04_anotaciones/train_val.csv')
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


def cargar_datos():
    print("[1/5] Cargando train_val.csv (holdout_test.csv NO se toca en este script)...")
    df = pd.read_csv(RUTA_TRAIN_VAL)
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

    # Todo train_val.csv se reparte entre entrenamiento y validación (para elegir la mejor
    # época). No se aparta un "test" interno aquí: el test final es holdout_test.csv, evaluado
    # una sola vez por 13_evaluacion_holdout.py — nunca visto por este script.
    X_train, X_val, y_train, y_val = train_test_split(
        textos, labels, test_size=0.15, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    print("[3/5] Cargando BETO + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BETO)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BETO, num_labels=len(INTENCIONES), id2label=id2label, label2id=label2id
    )

    def tokenizar(textos_batch):
        return tokenizer(textos_batch, padding=True, truncation=True, max_length=MAX_LENGTH)

    train_ds = DatasetIntenciones(tokenizar(X_train), y_train)
    val_ds = DatasetIntenciones(tokenizar(X_val), y_val)

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

    print("[5/5] Evaluando en validación interna (NO es la métrica final — "
          "esa la calcula 13_evaluacion_holdout.py sobre holdout_test.csv)...")
    pred_output = trainer.predict(val_ds)
    y_pred = np.argmax(pred_output.predictions, axis=-1)
    y_val_lbl = [id2label[i] for i in y_val]
    y_pred_lbl = [id2label[i] for i in y_pred]

    f1 = f1_score(y_val_lbl, y_pred_lbl, labels=INTENCIONES, average='macro', zero_division=0)
    reporte = classification_report(y_val_lbl, y_pred_lbl, labels=INTENCIONES, zero_division=0)
    print(f"\n  [BETO fine-tuned — validación interna]")
    print(f"  F1-macro (validación, NO final): {f1:.4f}")
    print(reporte)

    RUTA_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(RUTA_CHECKPOINT))
    tokenizer.save_pretrained(str(RUTA_CHECKPOINT))
    (RUTA_CHECKPOINT / 'FUENTE_ENTRENAMIENTO.txt').write_text(
        f"fuente={RUTA_TRAIN_VAL}\n"
        f"filas={len(df)}\n"
        f"fecha={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"f1_macro_validacion_interna={f1:.4f}\n"
        f"nota=Checkpoint entrenado SOLO con train_val.csv — holdout_test.csv no fue visto.\n"
        f"     13_evaluacion_holdout.py verifica esta marca antes de evaluar en el holdout.\n",
        encoding='utf-8'
    )
    print(f"  ✓ Checkpoint guardado en {RUTA_CHECKPOINT}")

    return f1, reporte


def main():
    print("=" * 70)
    print("BETO FINE-TUNING — ROCKTEC MIA 2026 (solo train_val.csv)")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    df = cargar_datos()
    f1_val, rep_val = entrenar_beto_finetuned(df)

    resumen = f"""
================================================================================
BETO FINE-TUNED — VALIDACIÓN INTERNA (NO ES LA MÉTRICA FINAL) — ROCKTEC MIA 2026
================================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset de entrenamiento: {RUTA_TRAIN_VAL} ({len(df)} registros, 5 clases)
Split interno: 85% train / 15% validación, estratificado, random_state=42

F1-macro (validación interna): {f1_val:.4f}

Este número NO es la métrica final del modelo — es solo la señal usada para elegir la
mejor época durante el entrenamiento. La métrica final (F1-macro sobre holdout_test.csv,
evaluada una única vez, comparada con TF-IDF+LR) la calcula:

    python 02_scripts/13_evaluacion_holdout.py

================================================================================
"""
    print(resumen)

    (RUTA_SALIDA / 'reporte_beto_finetuned_val.txt').write_text(rep_val, encoding='utf-8')
    (RUTA_SALIDA / 'resumen_beto_finetuned_val.txt').write_text(resumen, encoding='utf-8')
    print(f"✓ Guardado: {RUTA_SALIDA}/reporte_beto_finetuned_val.txt")
    print(f"✓ Guardado: {RUTA_SALIDA}/resumen_beto_finetuned_val.txt")
    print("\n✅ BETO FINE-TUNING (VALIDACIÓN) COMPLETADO — corre 13_evaluacion_holdout.py para la métrica final")


if __name__ == '__main__':
    main()
