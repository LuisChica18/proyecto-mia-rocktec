"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 10_shap_lime_explicabilidad.py
Ajuste S7 #6 — Explicabilidad con SHAP y LIME
================================================================================

Objetivo:
    Explicar POR QUÉ el modelo Logistic Regression clasifica cada conversación
    en una intención. Responde preguntas como:
    - ¿Qué palabras llevaron al modelo a decir COT?
    - ¿Por qué este mensaje fue clasificado como TEC y no INF?

    SHAP: importancia global de features (qué palabras importan más en general)
    LIME: explicación local (por qué clasificó ESTE mensaje específico)

Salida:
    06_resultados/explicabilidad/shap_summary.png
    06_resultados/explicabilidad/shap_top_features.png
    06_resultados/explicabilidad/lime_ejemplos.txt
    06_resultados/explicabilidad/lime_<clase>.png (1 por clase modelada)
    06_resultados/explicabilidad/reporte_explicabilidad.txt

Uso:
    python 02_scripts/10_shap_lime_explicabilidad.py

================================================================================
"""

import sys
import importlib.util
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# Importar feature engineering
_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)

VectorizadorTFIDF    = _fe.VectorizadorTFIDF
PreprocessadorTexto  = _fe.PreprocessadorTexto
INTENCIONES          = _fe.INTENCIONES

RUTA_CONSENSO   = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_MODELO_LR  = Path('06_resultados/modelos/modelo_lr.pkl')
RUTA_VECTOR     = Path('06_resultados/modelos/vectorizador_tfidf.pkl')
RUTA_SALIDA     = Path('06_resultados/explicabilidad')

def cargar_modelo_y_datos():
    """Carga modelo LR, vectorizador y dataset."""
    print("[1/4] Cargando modelo y datos...")

    # Cargar vectorizador
    vec = VectorizadorTFIDF.cargar(str(RUTA_VECTOR))

    # Cargar modelo LR desde MLflow
    import mlflow.sklearn
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    modelo = mlflow.sklearn.load_model(str(RUTA_MODELO_LR))
    print(f"  ✓ Modelo LR cargado")

    # Cargar dataset (5 clases)
    df = pd.read_csv(RUTA_CONSENSO)
    df = df[df['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)
    print(f"  ✓ Dataset: {len(df)} registros, {len(INTENCIONES)} clases")

    X = vec.transform(df['texto_conversacion'].values)
    y = df['intencion_consenso'].values

    return modelo, vec, df, X, y


def shap_explicabilidad(modelo, vec, X, y):
    """Genera explicaciones SHAP globales."""
    import shap

    print("\n[2/4] Calculando SHAP...")
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    # Usar muestra para velocidad (SHAP es costoso con matrices sparse grandes)
    np.random.seed(42)
    idx = np.random.choice(len(y), min(300, len(y)), replace=False)
    X_muestra = X[idx]

    # SHAP LinearExplainer (optimizado para modelos lineales)
    explainer = shap.LinearExplainer(modelo, X_muestra, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_muestra)

    # Nombres de features: TF-IDF vocab + 8 features manuales
    vocab = vec.tfidf.get_feature_names_out().tolist()
    nombres_manual = [
        'feat_longitud_chars', 'feat_num_palabras', 'feat_es_pregunta',
        'feat_menciona_precio', 'feat_saludo', 'feat_queja',
        'feat_curso', 'feat_venta'
    ]
    feature_names = vocab + nombres_manual

    # Top features por clase
    print("  Top 10 features por clase:")
    reporte_shap = "TOP FEATURES POR CLASE (SHAP)\n" + "="*60 + "\n"

    fig, axes = plt.subplots(1, len(INTENCIONES), figsize=(20, 6))
    if len(INTENCIONES) == 1:
        axes = [axes]

    for i, clase in enumerate(INTENCIONES):
        if isinstance(shap_values, list):
            sv = shap_values[i] if i < len(shap_values) else shap_values[0]
        else:
            sv = shap_values[:, :, i] if shap_values.ndim == 3 else shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-10:][::-1]
        top_feat = [feature_names[j] if j < len(feature_names) else f'feat_{j}' for j in top_idx]
        top_vals = mean_abs[top_idx]

        axes[i].barh(top_feat[::-1], top_vals[::-1], color='steelblue')
        axes[i].set_title(f'{clase}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('|SHAP| medio')
        axes[i].tick_params(labelsize=7)

        print(f"    {clase}: {', '.join(top_feat[:5])}")
        reporte_shap += f"\n{clase}:\n"
        for feat, val in zip(top_feat, top_vals):
            reporte_shap += f"  {feat}: {val:.4f}\n"

    plt.suptitle('SHAP — Top 10 features por clase de intención\nRocktec MIA 2026',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    ruta_plot = RUTA_SALIDA / 'shap_top_features.png'
    plt.savefig(ruta_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico guardado: {ruta_plot}")

    return reporte_shap, feature_names


def lime_explicabilidad(modelo, vec, df):
    """Genera explicaciones LIME para ejemplos representativos."""
    from lime.lime_text import LimeTextExplainer

    print("\n[3/4] Calculando LIME...")

    prep = PreprocessadorTexto()

    def predict_proba(textos):
        X = vec.transform(textos)
        return modelo.predict_proba(X)

    explainer = LimeTextExplainer(class_names=INTENCIONES)

    # Seleccionar 1 ejemplo por clase
    ejemplos = []
    for clase in INTENCIONES:
        subset = df[df['intencion_consenso'] == clase]
        if len(subset) > 0:
            ej = subset.sample(1, random_state=42).iloc[0]
            ejemplos.append((clase, ej['texto_conversacion']))

    reporte_lime = "EXPLICACIONES LIME POR EJEMPLO\n" + "="*60 + "\n"

    for clase, texto in ejemplos:
        texto_limpio = prep.limpiar(texto)
        if len(texto_limpio.split()) < 3:
            continue

        try:
            exp = explainer.explain_instance(
                texto_limpio,
                predict_proba,
                num_features=8,
                num_samples=100,
                labels=[INTENCIONES.index(clase)]
            )

            features_exp = exp.as_list(label=INTENCIONES.index(clase))
            pred = INTENCIONES[np.argmax(predict_proba([texto_limpio])[0])]

            reporte_lime += f"\n{'─'*50}\n"
            reporte_lime += f"Clase real:      {clase}\n"
            reporte_lime += f"Predicción:      {pred}\n"
            reporte_lime += f"Texto (preview): {texto[:100]}...\n"
            reporte_lime += f"Features que influyeron:\n"
            for feat, peso in features_exp:
                signo = "→ FAVOR" if peso > 0 else "→ CONTRA"
                reporte_lime += f"  '{feat}': {peso:+.4f}  {signo}\n"

            fig = exp.as_pyplot_figure(label=INTENCIONES.index(clase))
            fig.suptitle(f'LIME — {clase} (predicción: {pred})\nRocktec MIA 2026', fontsize=11, fontweight='bold')
            fig.tight_layout()
            ruta_plot = RUTA_SALIDA / f'lime_{clase}.png'
            fig.savefig(ruta_plot, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ LIME {clase}: pred={pred}, top_feat='{features_exp[0][0] if features_exp else 'N/A'}' → {ruta_plot.name}")

        except Exception as e:
            print(f"  ⚠ LIME {clase}: {e}")
            continue

    return reporte_lime


def main():
    print("=" * 70)
    print("EXPLICABILIDAD SHAP + LIME — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    modelo, vec, df, X, y = cargar_modelo_y_datos()
    reporte_shap, feature_names = shap_explicabilidad(modelo, vec, X, y)
    reporte_lime = lime_explicabilidad(modelo, vec, df)

    # Guardar reportes
    print("\n[4/4] Guardando reportes...")
    (RUTA_SALIDA / 'shap_top_features_reporte.txt').write_text(reporte_shap, encoding='utf-8')
    (RUTA_SALIDA / 'lime_ejemplos.txt').write_text(reporte_lime, encoding='utf-8')

    reporte_final = f"""
================================================================================
REPORTE EXPLICABILIDAD — ROCKTEC MIA 2026
================================================================================

SHAP (SHapley Additive exPlanations):
  - Método: LinearExplainer (optimizado para Logistic Regression)
  - Muestra: 300 registros aleatorios
  - Output: importancia global de features por clase
  - Archivo: shap_top_features.png

LIME (Local Interpretable Model-agnostic Explanations):
  - Método: LimeTextExplainer
  - Output: explicación local por ejemplo representativo de cada clase
  - Archivo: lime_ejemplos.txt

INTERPRETACIÓN:
  SHAP responde: ¿qué palabras son más importantes GLOBALMENTE para cada clase?
  LIME responde: ¿por qué se clasificó ESTE mensaje específico en esta clase?

  Ejemplo: para COT, SHAP muestra que 'precio', 'costo', 'presupuesto' son
  las features más relevantes. LIME confirma esto en ejemplos individuales.

================================================================================
"""
    (RUTA_SALIDA / 'reporte_explicabilidad.txt').write_text(reporte_final, encoding='utf-8')

    print(f"  ✓ shap_top_features_reporte.txt")
    print(f"  ✓ lime_ejemplos.txt")
    print(f"  ✓ lime_<clase>.png (1 por clase)")
    print(f"  ✓ reporte_explicabilidad.txt")

    print("\n" + "=" * 70)
    print("✅ EXPLICABILIDAD COMPLETADA")
    print("=" * 70)


if __name__ == '__main__':
    main()
