"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 14_explicabilidad_beto.py
Ajuste S7 #6b — Explicabilidad de BETO fine-tuned (SHAP + LIME)
================================================================================

Equivalente a 10_shap_lime_explicabilidad.py pero para el modelo BETO
fine-tuned (06_resultados/modelos/beto_finetuned_best/) en vez de LR.

Diferencia clave con el script 10: para LR, SHAP usa LinearExplainer sobre
300 muestras para dar una importancia GLOBAL de features TF-IDF por clase.
BETO no tiene "features" interpretables de esa forma (son embeddings), así
que aquí SHAP y LIME son ambos LOCALES: explican token por token por qué UN
mensaje específico se clasificó en una clase — no hay equivalente directo al
ranking global de palabras que sí existe para LR.

Usa los mismos ejemplos (mismo dataset, mismo random_state=42) que el script
10 para que las explicaciones de LR y BETO sean comparables lado a lado sobre
el mismo texto.

Salida:
    06_resultados/explicabilidad/shap_beto_ejemplos.png
    06_resultados/explicabilidad/shap_beto_reporte.txt
    06_resultados/explicabilidad/lime_beto_ejemplos.txt
    06_resultados/explicabilidad/reporte_explicabilidad_beto.txt

Uso (desde la raíz del repo, no requiere GPU — es solo inferencia):
    python 02_scripts/14_explicabilidad_beto.py
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

INTENCIONES     = ['INF', 'COT', 'TEC', 'CUR', 'VEN']
RUTA_CONSENSO   = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_CHECKPOINT = Path('06_resultados/modelos/beto_finetuned_best')
RUTA_SALIDA     = Path('06_resultados/explicabilidad')


def cargar_modelo_y_datos():
    print("[1/4] Cargando BETO fine-tuned y datos...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not RUTA_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_CHECKPOINT} — corre 02_scripts/12_beto_finetuning.py "
            "(en Colab/GPU) primero."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(RUTA_CHECKPOINT))
    modelo = AutoModelForSequenceClassification.from_pretrained(str(RUTA_CHECKPOINT))
    modelo.eval()
    print(f"  ✓ Modelo BETO fine-tuned cargado ({RUTA_CHECKPOINT})")

    df = pd.read_csv(RUTA_CONSENSO)
    df = df[df['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    print(f"  ✓ Dataset: {len(df)} registros, {len(INTENCIONES)} clases")

    def predict_proba(textos):
        import torch as _torch
        textos = list(textos)
        inputs = tokenizer(textos, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with _torch.no_grad():
            logits = modelo(**inputs).logits
        return _torch.softmax(logits, dim=-1).numpy()

    return modelo, tokenizer, df, predict_proba


def seleccionar_ejemplos(df):
    """Mismo criterio que 10_shap_lime_explicabilidad.py: 1 ejemplo por clase,
    random_state=42 — para que sean los mismos textos usados al explicar LR."""
    ejemplos = []
    for clase in INTENCIONES:
        subset = df[df['intencion_consenso'] == clase]
        if len(subset) > 0:
            ej = subset.sample(1, random_state=42).iloc[0]
            ejemplos.append((clase, ej['texto_conversacion']))
    return ejemplos


def lime_explicabilidad_beto(predict_proba, ejemplos):
    from lime.lime_text import LimeTextExplainer

    print("\n[2/4] Calculando LIME (BETO)...")
    explainer = LimeTextExplainer(class_names=INTENCIONES)
    reporte = "EXPLICACIONES LIME POR EJEMPLO — BETO FINE-TUNED\n" + "=" * 60 + "\n"

    for clase, texto in ejemplos:
        if len(str(texto).split()) < 3:
            continue
        try:
            exp = explainer.explain_instance(
                texto, predict_proba, num_features=8, num_samples=100,
                labels=[INTENCIONES.index(clase)]
            )
            features_exp = exp.as_list(label=INTENCIONES.index(clase))
            pred = INTENCIONES[np.argmax(predict_proba([texto])[0])]

            reporte += f"\n{'─'*50}\n"
            reporte += f"Clase real:      {clase}\n"
            reporte += f"Predicción:      {pred}\n"
            reporte += f"Texto (preview): {str(texto)[:100]}...\n"
            reporte += "Features que influyeron:\n"
            for feat, peso in features_exp:
                signo = "→ FAVOR" if peso > 0 else "→ CONTRA"
                reporte += f"  '{feat}': {peso:+.4f}  {signo}\n"

            print(f"  ✓ LIME {clase}: pred={pred}, top_feat='{features_exp[0][0] if features_exp else 'N/A'}'")
        except Exception as e:
            print(f"  ⚠ LIME {clase}: {e}")
            continue

    return reporte


def shap_explicabilidad_beto(predict_proba, tokenizer, ejemplos):
    import shap

    print("\n[3/4] Calculando SHAP (BETO, local por ejemplo — puede tardar ~10s/ejemplo)...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, output_names=INTENCIONES)

    reporte = "EXPLICACIONES SHAP POR EJEMPLO (LOCAL) — BETO FINE-TUNED\n" + "=" * 60 + "\n"
    resultados_plot = []

    for clase, texto in ejemplos:
        if len(str(texto).split()) < 3:
            continue
        try:
            sv = explainer([str(texto)])
            idx_clase = INTENCIONES.index(clase)
            tokens = sv.data[0]
            valores = sv.values[0][:, idx_clase]

            orden = np.argsort(np.abs(valores))[::-1][:8]
            pred = INTENCIONES[np.argmax(predict_proba([texto])[0])]

            reporte += f"\n{'─'*50}\n"
            reporte += f"Clase real:      {clase}\n"
            reporte += f"Predicción:      {pred}\n"
            reporte += f"Texto (preview): {str(texto)[:100]}...\n"
            reporte += "Tokens que más influyeron (SHAP local, respecto a la clase real):\n"
            for j in orden:
                tok = tokens[j].strip() or '(espacio)'
                signo = "→ FAVOR" if valores[j] > 0 else "→ CONTRA"
                reporte += f"  '{tok}': {valores[j]:+.4f}  {signo}\n"

            resultados_plot.append((clase, [tokens[j].strip() or '·' for j in orden], [valores[j] for j in orden]))
            print(f"  ✓ SHAP {clase}: pred={pred}, top_token='{tokens[orden[0]].strip()}'")
        except Exception as e:
            print(f"  ⚠ SHAP {clase}: {e}")
            continue

    # Gráfico: un subplot por ejemplo, barras firmadas (positivo=favor, negativo=contra)
    if resultados_plot:
        fig, axes = plt.subplots(1, len(resultados_plot), figsize=(5 * len(resultados_plot), 6))
        if len(resultados_plot) == 1:
            axes = [axes]
        for ax, (clase, toks, vals) in zip(axes, resultados_plot):
            colores = ['seagreen' if v > 0 else 'indianred' for v in vals]
            ax.barh(toks[::-1], vals[::-1], color=colores[::-1])
            ax.set_title(clase, fontsize=10, fontweight='bold')
            ax.set_xlabel('SHAP (local)')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.tick_params(labelsize=7)
        plt.suptitle('SHAP local — 1 ejemplo por clase\nBETO fine-tuned, Rocktec MIA 2026',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        ruta_plot = RUTA_SALIDA / 'shap_beto_ejemplos.png'
        plt.savefig(ruta_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Gráfico guardado: {ruta_plot}")

    return reporte


def main():
    print("=" * 70)
    print("EXPLICABILIDAD SHAP + LIME — BETO FINE-TUNED — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    modelo, tokenizer, df, predict_proba = cargar_modelo_y_datos()
    ejemplos = seleccionar_ejemplos(df)

    reporte_lime = lime_explicabilidad_beto(predict_proba, ejemplos)
    reporte_shap = shap_explicabilidad_beto(predict_proba, tokenizer, ejemplos)

    print("\n[4/4] Guardando reportes...")
    (RUTA_SALIDA / 'lime_beto_ejemplos.txt').write_text(reporte_lime, encoding='utf-8')
    (RUTA_SALIDA / 'shap_beto_reporte.txt').write_text(reporte_shap, encoding='utf-8')

    reporte_final = f"""
================================================================================
REPORTE EXPLICABILIDAD — BETO FINE-TUNED — ROCKTEC MIA 2026
================================================================================

A diferencia de 10_shap_lime_explicabilidad.py (LR), aquí SHAP también es LOCAL
(un ejemplo por clase), no una importancia global de features — BETO no tiene
un vocabulario de features TF-IDF sobre el cual calcular esa importancia global
de forma barata; el equivalente honesto para un transformer sería agregar
atribuciones locales sobre muchos ejemplos, que queda fuera de alcance aquí.

LIME (Local Interpretable Model-agnostic Explanations):
  - Método: LimeTextExplainer (igual que en LR — model-agnostic, no le importa
    que el modelo por dentro sea TF-IDF+LR o un transformer)
  - Output: explicación local por ejemplo representativo de cada clase
  - Archivo: lime_beto_ejemplos.txt

SHAP (SHapley Additive exPlanations) — variante de texto:
  - Método: shap.Explainer con shap.maskers.Text (enmascara tokens y mide el
    cambio en la probabilidad predicha — Partition explainer internamente)
  - Output: atribución por token, local a cada ejemplo (NO comparable
    directamente con el ranking global de shap_top_features.png de LR)
  - Archivo: shap_beto_ejemplos.png, shap_beto_reporte.txt

Mismos ejemplos que 10_shap_lime_explicabilidad.py (mismo dataset, mismo
random_state=42) — compara lime_ejemplos.txt / shap_top_features_reporte.txt
(LR) contra lime_beto_ejemplos.txt / shap_beto_reporte.txt (BETO) sobre el
mismo texto para ver si ambos modelos se fijan en las mismas palabras.

================================================================================
"""
    (RUTA_SALIDA / 'reporte_explicabilidad_beto.txt').write_text(reporte_final, encoding='utf-8')

    print(f"  ✓ lime_beto_ejemplos.txt")
    print(f"  ✓ shap_beto_reporte.txt")
    print(f"  ✓ shap_beto_ejemplos.png")
    print(f"  ✓ reporte_explicabilidad_beto.txt")

    print("\n" + "=" * 70)
    print("✅ EXPLICABILIDAD BETO COMPLETADA")
    print("=" * 70)


if __name__ == '__main__':
    main()
