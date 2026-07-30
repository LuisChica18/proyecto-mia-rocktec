"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 18_active_learning_tec.py
S8 Ajuste #2 — Active Learning para ampliar dataset TEC
================================================================================

Objetivo:
    TEC es la clase con menor F1 (0.40-0.47) por falta de datos (51 casos).
    En lugar de recolectar datos al azar, el Active Learning identifica los
    20-30 mensajes donde el modelo tiene MENOR CONFIANZA en TEC — esos son
    los más informativos para anotar primero.

    Estrategia: uncertainty sampling
    - Entrenar LR sobre el dataset actual
    - Aplicar el modelo sobre mensajes NO anotados (pool crudo)
    - Seleccionar los que tienen mayor incertidumbre para TEC
    - Esos son los candidatos prioritarios para que los asesores de Rocktec anoten

Salida:
    04_anotaciones/active_learning_tec_candidatos.xlsx
    06_resultados/active_learning/reporte_active_learning.txt

Uso:
    python 02_scripts/18_active_learning_tec.py
================================================================================
"""

import sys
import warnings
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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

RUTA_TRAIN      = Path('04_anotaciones/train_val.csv')
RUTA_CONSENSO   = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_SALIDA_AL  = Path('04_anotaciones/active_learning_tec_candidatos.xlsx')
RUTA_REPORTE    = Path('06_resultados/active_learning/reporte_active_learning.txt')
N_CANDIDATOS    = 30
RANDOM_STATE    = 42


def cargar_datos():
    print("[1/4] Cargando datos...")
    train = pd.read_csv(RUTA_TRAIN)
    train = train[train['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)

    print(f"  Train: {len(train)} registros")
    print(f"  TEC en train: {(train['intencion_consenso']=='TEC').sum()} casos")
    return train


def entrenar_modelo(train):
    print("[2/4] Entrenando modelo LR...")
    vec = VectorizadorTFIDF()
    X   = vec.fit_transform(train['texto_conversacion'].values)
    y   = train['intencion_consenso'].values

    modelo = LogisticRegression(
        C=10, class_weight='balanced',
        max_iter=1000, random_state=RANDOM_STATE
    )
    modelo.fit(X, y)
    print(f"  ✓ Modelo entrenado sobre {len(train)} registros")
    return modelo, vec


def generar_pool_no_anotado(train):
    """
    Genera un pool de mensajes no anotados combinando:
    1. Mensajes del dataset consenso que el modelo predice como TEC
    2. Variaciones de mensajes existentes de TEC para ampliar cobertura
    """
    print("[3/4] Generando pool de candidatos...")

    # Pool: mensajes del consenso completo que no están en train
    consenso = pd.read_csv(RUTA_CONSENSO)
    consenso = consenso[consenso['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)

    # Textos ya en train
    textos_train = set(train['texto_conversacion'].str.strip().str.lower())

    # Candidatos: textos del consenso no usados en train
    pool = consenso[~consenso['texto_conversacion'].str.strip().str.lower().isin(textos_train)].copy()

    # Agregar ejemplos sintéticos de TEC para ampliar el pool
    ejemplos_tec_adicionales = [
        "¿Cuántas capas de sellador necesito aplicar sobre el concreto estampado?",
        "El microcemento se puede aplicar sobre cerámica existente sin removerla?",
        "¿Cuál es el tiempo de curado del oxidante antes de sellar?",
        "¿Qué herramientas necesito para aplicar el concreto decorativo?",
        "¿Cómo preparo la superficie para aplicar el microcemento en baño?",
        "¿El sellador es resistente al agua en pisos exteriores?",
        "¿Puedo aplicar el producto sobre piso de madera?",
        "¿Cuántos metros cuadrados rinde un galón de sellador?",
        "¿Qué pasa si aplico una segunda capa antes de que seque la primera?",
        "¿El concreto decorativo se puede usar en paredes o solo en pisos?",
        "¿Necesito lijar entre capas de microcemento?",
        "¿Cómo limpio las herramientas después de aplicar el oxidante?",
        "¿Cuál es la diferencia entre sellador mate y brillante?",
        "¿El producto es apto para zonas de alto tráfico?",
        "¿Cómo reparo una zona que quedó con burbujas después de secar?",
    ]

    pool_extra = pd.DataFrame({
        'texto_conversacion': ejemplos_tec_adicionales,
        'intencion_consenso': 'POOL_SINTETICO'
    })

    pool_completo = pd.concat([
        pool[['texto_conversacion', 'intencion_consenso']],
        pool_extra
    ], ignore_index=True)

    print(f"  Pool generado: {len(pool_completo)} mensajes")
    return pool_completo


def seleccionar_candidatos_incertidumbre(modelo, vec, pool):
    """
    Uncertainty sampling: selecciona mensajes donde el modelo
    tiene MENOR confianza en su clasificación (mayor incertidumbre).
    Estos son los más informativos para anotar.
    """
    X_pool = vec.transform(pool['texto_conversacion'].values)
    proba  = modelo.predict_proba(X_pool)
    clases = modelo.classes_

    idx_tec = list(clases).index('TEC') if 'TEC' in clases else 0

    # Probabilidad de TEC para cada mensaje
    proba_tec = proba[:, idx_tec]

    # Confianza máxima del modelo (incertidumbre = 1 - confianza_max)
    confianza_max   = proba.max(axis=1)
    incertidumbre   = 1 - confianza_max
    pred_clase      = clases[proba.argmax(axis=1)]

    pool = pool.copy()
    pool['pred_clase']       = pred_clase
    pool['proba_TEC']        = proba_tec
    pool['confianza_modelo'] = confianza_max
    pool['incertidumbre']    = incertidumbre

    # Estrategia: mensajes donde TEC tiene probabilidad media (0.15-0.60)
    # — el modelo duda entre TEC y otra clase
    candidatos_tec = pool[
        (pool['proba_TEC'] >= 0.10) &
        (pool['proba_TEC'] <= 0.70)
    ].copy()

    # Ordenar por mayor incertidumbre
    candidatos_tec = candidatos_tec.sort_values(
        'incertidumbre', ascending=False
    ).head(N_CANDIDATOS)

    return candidatos_tec, pool


def main():
    print("=" * 70)
    print("ACTIVE LEARNING PARA TEC — ROCKTEC MIA 2026")
    print("=" * 70)

    Path('06_resultados/active_learning').mkdir(parents=True, exist_ok=True)

    train          = cargar_datos()
    modelo, vec    = entrenar_modelo(train)
    pool           = generar_pool_no_anotado(train)
    candidatos, pool_full = seleccionar_candidatos_incertidumbre(modelo, vec, pool)

    print(f"\n[4/4] Guardando {len(candidatos)} candidatos TEC...")

    # Preparar Excel para anotación
    salida = candidatos[['texto_conversacion', 'pred_clase',
                          'proba_TEC', 'confianza_modelo', 'incertidumbre']].copy()
    salida.columns = ['texto_conversacion', 'prediccion_modelo',
                      'prob_TEC', 'confianza_modelo', 'incertidumbre']
    salida['intencion_correcta'] = ''    # columna para que los asesores anoten
    salida['es_TEC_real'] = ''           # SÍ / NO
    salida['notas'] = ''
    salida = salida.reset_index(drop=True)

    salida.to_excel(RUTA_SALIDA_AL, index=False)
    print(f"  ✓ Candidatos guardados: {RUTA_SALIDA_AL}")

    # Reporte
    reporte = f"""
================================================================================
REPORTE ACTIVE LEARNING TEC — ROCKTEC MIA 2026
================================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Estrategia: Uncertainty Sampling sobre pool de mensajes

CONTEXTO:
  TEC es la clase con menor F1 (LR: 0.47, BETO: 0.40) por falta de datos.
  Solo tiene 51 casos en el dataset de 1,312. El active learning identifica
  los mensajes más informativos para anotar PRIMERO, en lugar de recolectar
  datos al azar.

METODOLOGÍA:
  1. Entrenar LR sobre train_val.csv ({len(train)} registros)
  2. Aplicar sobre pool de mensajes no anotados ({len(pool)} mensajes)
  3. Seleccionar los {N_CANDIDATOS} mensajes con mayor incertidumbre en TEC
     (prob_TEC entre 0.10 y 0.70 — el modelo duda entre TEC y otra clase)

CANDIDATOS SELECCIONADOS: {len(candidatos)}
{'─'*60}
"""
    for i, row in salida.iterrows():
        reporte += f"\n[{i+1:2d}] Pred: {row['prediccion_modelo']:5s} | "
        reporte += f"P(TEC)={row['prob_TEC']:.3f} | "
        reporte += f"Incert={row['incertidumbre']:.3f}\n"
        reporte += f"     Texto: {str(row['texto_conversacion'])[:100]}\n"

    reporte += f"""
{'─'*60}
INSTRUCCIONES PARA ASESORES DE ROCKTEC:
  1. Abrir active_learning_tec_candidatos.xlsx
  2. Para cada mensaje, completar:
     - 'intencion_correcta': la categoría real (INF/COT/TEC/CUR/VEN)
     - 'es_TEC_real': SÍ si es consulta técnica, NO si no lo es
     - 'notas': observaciones sobre el caso
  3. Meta: confirmar al menos 30-40 casos reales de TEC
  4. Estos casos se agregarán al dataset y se reentrenará el modelo

IMPACTO ESPERADO:
  Con 30-40 casos TEC adicionales bien anotados:
  - F1 TEC estimado: de 0.47 → ~0.65+
  - F1-macro general: de 0.7938 → ~0.82+

================================================================================
"""
    RUTA_REPORTE.write_text(reporte, encoding='utf-8')
    print(f"  ✓ Reporte guardado: {RUTA_REPORTE}")

    print(f"\n{'='*70}")
    print(f"✅ ACTIVE LEARNING COMPLETADO")
    print(f"   {len(candidatos)} candidatos TEC identificados")
    print(f"   Archivo: {RUTA_SALIDA_AL}")
    print(f"   → Entregar a asesores de Rocktec para anotación")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
