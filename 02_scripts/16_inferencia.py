"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: 16_inferencia.py
Fase 3 — Componente de inferencia (batch)
================================================================================

Cierra el gap identificado en DIAGNOSTICO_FASES_3_4_5.md (Fase 3): hasta ahora
todo el proyecto era entrenamiento/evaluación offline sobre CSVs históricos —
no existía ninguna forma de tomar un mensaje NUEVO y devolver una predicción.

Diseño: inferencia POR LOTES, no una API en vivo. Los mensajes de WhatsApp de
Rocktec se descargan manualmente (no hay integración en vivo con la WhatsApp
Business API — ver PROPUESTA_REVISADA_FASE2.md §2, ajuste #8). El flujo real es:
alguien exporta un Excel/CSV nuevo de conversaciones → corre este script → recibe
el mismo archivo con columnas de predicción añadidas, listo para priorizar al
equipo de ventas.

Usa el artefacto de 06_resultados/modelos/produccion/ (generado por
15_entrenar_produccion.py) — NO el de 06_resultados/modelos/ (ese es el
experimento de 05_entrenar_modelos.py, entrenado antes de que existiera el split
train_val/holdout).

Mensajes con confianza por debajo de UMBRAL_REVISION_MANUAL se marcan
revisar_manual=True en vez de enrutarse automáticamente — mitigación ya prevista
en ANALISIS_RIESGOS_FASE2.md (R1: TEC débil, R2: SEG/QUE fuera del alcance del
modelo) para no forzar en silencio esos mensajes a una de las 5 clases modeladas.

Cada corrida además agrega un log a 06_resultados/predicciones/log_predicciones.csv
— sustituto de la base de datos pospuesta (PostgreSQL, ver PROPUESTA_REVISADA_FASE2.md
§2 ajuste #8) y el insumo que alimentará el cálculo de PSI para el monitoreo de
drift (Etapa 5, todavía sin implementar).

Uso como CLI:
    python 02_scripts/16_inferencia.py --input nuevas_conversaciones.xlsx --output predicciones.xlsx
    python 02_scripts/16_inferencia.py --input nuevas_conversaciones.csv  --output predicciones.csv --columna-texto detalle

Uso como librería:
    from importlib import util as _u
    spec = _u.spec_from_file_location("inferencia", "02_scripts/16_inferencia.py")
    inferencia = _u.module_from_spec(spec); spec.loader.exec_module(inferencia)
    inferencia.predecir(["¿Cuánto cuesta el microcemento para 30 m²?"])
================================================================================
"""

import sys
import argparse
import importlib.util
import warnings
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

warnings.filterwarnings('ignore')

UMBRAL_REVISION_MANUAL = 0.50

RUTA_PRODUCCION = Path('06_resultados/modelos/produccion')
RUTA_VECTOR     = RUTA_PRODUCCION / 'vectorizador_tfidf.pkl'
RUTA_MODELO     = RUTA_PRODUCCION / 'modelo_lr.pkl'
RUTA_LOG        = Path('06_resultados/predicciones/log_predicciones.csv')

# ── Reutiliza el vectorizador TF-IDF + features manuales del pipeline oficial ──
_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec    = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe      = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)
VectorizadorTFIDF = _fe.VectorizadorTFIDF

_vec = None
_modelo = None


def _cargar_artefactos():
    global _vec, _modelo
    if _vec is None or _modelo is None:
        if not RUTA_VECTOR.exists() or not RUTA_MODELO.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto de producción en {RUTA_PRODUCCION}/. "
                "Corre primero: python 02_scripts/15_entrenar_produccion.py"
            )
        _vec = VectorizadorTFIDF.cargar(str(RUTA_VECTOR))
        with open(RUTA_MODELO, 'rb') as f:
            _modelo = pickle.load(f)
    return _vec, _modelo


def predecir(textos):
    """
    Predice la intención de una lista de mensajes.

    Devuelve un DataFrame con columnas: texto, intencion_predicha, confianza, revisar_manual.
    """
    vec, modelo = _cargar_artefactos()
    textos = list(textos)
    X = vec.transform(textos)
    proba = modelo.predict_proba(X)
    clases = modelo.classes_
    idx_max = proba.argmax(axis=1)
    confianza = proba[range(len(textos)), idx_max]

    return pd.DataFrame({
        'texto': textos,
        'intencion_predicha': [clases[i] for i in idx_max],
        'confianza': confianza.round(4),
        'revisar_manual': confianza < UMBRAL_REVISION_MANUAL,
    })


def _leer_entrada(ruta, columna_texto):
    if ruta.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(ruta)
    else:
        df = pd.read_csv(ruta)
    if columna_texto not in df.columns:
        raise ValueError(
            f"La columna de texto '{columna_texto}' no existe en {ruta}. "
            f"Columnas disponibles: {list(df.columns)}. Usa --columna-texto para indicar la correcta."
        )
    return df


def _guardar_log(df_resultado):
    RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    df_log = df_resultado.copy()
    df_log.insert(0, 'timestamp', datetime.now().isoformat(timespec='seconds'))
    encabezado = not RUTA_LOG.exists()
    df_log.to_csv(RUTA_LOG, mode='a', header=encabezado, index=False)


def main():
    parser = argparse.ArgumentParser(description="Inferencia por lotes — Rocktec MIA 2026")
    parser.add_argument('--input', required=True, help="Excel o CSV con mensajes nuevos")
    parser.add_argument('--output', required=True, help="Ruta de salida (mismo formato recomendado que --input)")
    parser.add_argument('--columna-texto', default='texto_conversacion',
                         help="Columna con el texto del mensaje (default: texto_conversacion)")
    args = parser.parse_args()

    ruta_in = Path(args.input)
    ruta_out = Path(args.output)

    print("=" * 70)
    print("INFERENCIA POR LOTES — ROCKTEC MIA 2026")
    print("=" * 70)

    df_in = _leer_entrada(ruta_in, args.columna_texto)
    print(f"  {len(df_in)} mensajes cargados de {ruta_in}")

    resultado = predecir(df_in[args.columna_texto].astype(str).tolist())
    df_out = df_in.copy()
    df_out['intencion_predicha'] = resultado['intencion_predicha'].values
    df_out['confianza'] = resultado['confianza'].values
    df_out['revisar_manual'] = resultado['revisar_manual'].values

    n_revisar = int(df_out['revisar_manual'].sum())
    print(f"  Predicciones completadas. {n_revisar}/{len(df_out)} mensajes bajo el umbral "
          f"de confianza ({UMBRAL_REVISION_MANUAL}) — marcados para revisión manual.")

    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    if ruta_out.suffix.lower() in ('.xlsx', '.xls'):
        df_out.to_excel(ruta_out, index=False)
    else:
        df_out.to_csv(ruta_out, index=False)
    print(f"  ✓ Guardado: {ruta_out}")

    _guardar_log(
        df_out[[args.columna_texto, 'intencion_predicha', 'confianza', 'revisar_manual']]
        .rename(columns={args.columna_texto: 'texto'})
    )
    print(f"  ✓ Log actualizado: {RUTA_LOG}")
    print("\n✅ INFERENCIA COMPLETADA")


if __name__ == '__main__':
    main()
