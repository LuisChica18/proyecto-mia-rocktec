"""
07_validacion_estadistica.py
Validación estadística rigurosa — Rocktec MIA 2026

Compara, con un diseño de validación cruzada PAREADO, el modelo baseline
(etiquetas heurísticas por palabras clave) contra el modelo ajustado
(etiquetas de consenso humano, voto mayoritario 2/3 entre los 3 anotadores),
evaluando AMBAS etiquetas sobre EXACTAMENTE las mismas 1,297 filas y los
mismos folds de validación cruzada. Esto aísla el efecto de la calidad del
etiquetado del efecto del tamaño de la muestra (que sí varía entre el
dataset heurístico completo, ~9,300 filas, y el dataset de consenso, 1,297).

Metodología:
  - RepeatedKFold(n_splits=5, n_repeats=5) sobre los ÍNDICES de fila
    (no estratificado), de modo que el mismo split train/test se reutiliza
    para ambas fuentes de etiqueta -> comparación pareada válida.
  - Por fold: TF-IDF + features manuales re-ajustado solo con el train del
    fold (evita fuga de información del vocabulario hacia el test).
  - Métrica: F1-macro sobre las 7 clases fijas del catálogo (zero_division=0),
    para que el denominador de "macro" no cambie según qué clases aparezcan
    en cada fold.
  - Incertidumbre: media, desviación estándar e IC 95% (t de Student, n=25).
  - Prueba de hipótesis: Wilcoxon signed-rank (no paramétrica) + t-test
    pareado (verificación cruzada) sobre las diferencias fold-a-fold.

Uso: python 07_validacion_estadistica.py   (ejecutar desde la raíz del repo)
"""

import json
import sys
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

warnings.filterwarnings('ignore')

_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)

VectorizadorTFIDF = _fe.VectorizadorTFIDF
INTENCIONES       = _fe.INTENCIONES

SEED       = 42
N_SPLITS   = 5
N_REPEATS  = 5
LR_C       = 0.1   # mejor hiperparámetro encontrado por GridSearchCV en 05_entrenar_modelos.py
SVM_C      = 1.0
RUTA_CONSENSO = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_SALIDA   = Path('06_resultados/validacion_estadistica.json')
RUTA_PLOT     = Path('06_resultados/modelos/validacion_cv_boxplot.png')


def cargar_datos_pareados():
    df = pd.read_csv(RUTA_CONSENSO)
    df = df[df['intencion_consenso'] != 'SIN_CONSENSO'].reset_index(drop=True)
    df = df[df['intencion_consenso'].isin(INTENCIONES) &
             df['intencion_heuristica_baseline'].isin(INTENCIONES)].reset_index(drop=True)
    return (df['texto_conversacion'].values,
            df['intencion_consenso'].values,
            df['intencion_heuristica_baseline'].values)


def features_fold(X_texto, train_idx, test_idx):
    """TF-IDF depende solo del texto (compartido por ambos brazos y modelos);
    se calcula una única vez por fold y se reutiliza 4x (2 modelos x 2 brazos)."""
    vec = VectorizadorTFIDF()
    return vec.fit_transform(X_texto[train_idx]), vec.transform(X_texto[test_idx])


def evaluar_fold(modelo_factory, X_train, X_test, y, train_idx, test_idx):
    y_train, y_test = y[train_idx], y[test_idx]

    modelo = modelo_factory()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    f1  = f1_score(y_test, y_pred, labels=INTENCIONES, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    return f1, acc


def resumen_estadistico(scores):
    scores = np.asarray(scores, dtype=float)
    n      = len(scores)
    media  = scores.mean()
    std    = scores.std(ddof=1)
    err    = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return {
        'media': float(media), 'std': float(std), 'n': int(n),
        'ic95_low': float(media - t_crit * err),
        'ic95_high': float(media + t_crit * err),
        'scores': scores.tolist(),
    }


def comparar_pareado(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    t_stat, t_p = stats.ttest_rel(a, b)
    return {
        'diferencia_media': float(diffs.mean()),
        'wilcoxon_statistic': float(w_stat), 'wilcoxon_pvalue': float(w_p),
        'ttest_pareado_statistic': float(t_stat), 'ttest_pareado_pvalue': float(t_p),
    }


def main():
    print("=" * 70)
    print("VALIDACIÓN ESTADÍSTICA PAREADA — BASELINE vs. AJUSTADO")
    print("=" * 70)

    X_texto, y_consenso, y_heuristico = cargar_datos_pareados()
    n = len(X_texto)
    print(f"✓ Filas con consenso válido: {n}")

    modelos = {
        'lr':  lambda: LogisticRegression(C=LR_C, class_weight='balanced',
                                          solver='lbfgs', max_iter=1000, random_state=SEED),
        'svm': lambda: LinearSVC(C=SVM_C, class_weight='balanced',
                                 max_iter=2000, random_state=SEED),
    }
    arms = {'consenso': y_consenso, 'heuristico': y_heuristico}

    rkf    = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rkf.split(np.arange(n)))
    print(f"✓ Folds pareados generados: {len(splits)} (n_splits={N_SPLITS}, n_repeats={N_REPEATS})")

    # fold-mayor, (modelo,brazo)-menor: el TF-IDF (que solo depende del texto)
    # se calcula una vez por fold y se reutiliza en las 4 combinaciones de
    # modelo x brazo, en vez de recalcularlo 4 veces por fold.
    puntajes = {f'{m}_{a}': {'f1_macro': [], 'accuracy': []}
                for m in modelos for a in arms}

    for i, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = features_fold(X_texto, train_idx, test_idx)
        for nombre_modelo, factory in modelos.items():
            for nombre_arm, y in arms.items():
                f1, acc = evaluar_fold(factory, X_train, X_test, y, train_idx, test_idx)
                clave = f'{nombre_modelo}_{nombre_arm}'
                puntajes[clave]['f1_macro'].append(f1)
                puntajes[clave]['accuracy'].append(acc)
        print(f"  Fold {i}/{len(splits)} completado", end='\r')

    print()
    resultados = {}
    for clave, vals in puntajes.items():
        resultados[clave] = {
            'f1_macro': resumen_estadistico(vals['f1_macro']),
            'accuracy': resumen_estadistico(vals['accuracy']),
        }
        r = resultados[clave]['f1_macro']
        print(f"[{clave}] F1-macro: {r['media']:.4f} ± {r['std']:.4f}  "
              f"(IC95%: [{r['ic95_low']:.4f}, {r['ic95_high']:.4f}])")

    comparaciones = {}
    for nombre_modelo in modelos:
        comparaciones[f'{nombre_modelo}_consenso_vs_heuristico'] = comparar_pareado(
            resultados[f'{nombre_modelo}_consenso']['f1_macro']['scores'],
            resultados[f'{nombre_modelo}_heuristico']['f1_macro']['scores'],
        )
    comparaciones['svm_vs_lr_consenso'] = comparar_pareado(
        resultados['svm_consenso']['f1_macro']['scores'],
        resultados['lr_consenso']['f1_macro']['scores'],
    )

    print("\n" + "=" * 70)
    print("COMPARACIONES PAREADAS (Wilcoxon signed-rank / t-test pareado)")
    print("=" * 70)
    for nombre, c in comparaciones.items():
        print(f"\n{nombre}:")
        print(f"  Δ media F1-macro:  {c['diferencia_media']:+.4f}")
        print(f"  Wilcoxon:  stat={c['wilcoxon_statistic']:.3f}  p={c['wilcoxon_pvalue']:.4f}")
        print(f"  t-test pareado: stat={c['ttest_pareado_statistic']:.3f}  p={c['ttest_pareado_pvalue']:.4f}")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA.write_text(json.dumps({
        'metodologia': {
            'n_filas': n, 'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
            'lr_C': LR_C, 'svm_C': SVM_C,
        },
        'resultados': resultados,
        'comparaciones': comparaciones,
    }, indent=2), encoding='utf-8')
    print(f"\n✓ Resultados guardados: {RUTA_SALIDA}")

    # ── Boxplot ──
    etiquetas = ['LR\nheurístico', 'LR\nconsenso', 'SVM\nheurístico', 'SVM\nconsenso']
    datos = [
        resultados['lr_heuristico']['f1_macro']['scores'],
        resultados['lr_consenso']['f1_macro']['scores'],
        resultados['svm_heuristico']['f1_macro']['scores'],
        resultados['svm_consenso']['f1_macro']['scores'],
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(datos, tick_labels=etiquetas)
    ax.set_ylabel('F1-macro (25 folds pareados)')
    ax.set_title('Distribución de F1-macro por fold — baseline (heurístico) vs. ajustado (consenso)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    RUTA_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RUTA_PLOT, dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {RUTA_PLOT}")


if __name__ == '__main__':
    main()
