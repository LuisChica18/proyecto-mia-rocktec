"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script 18: Monitoreo de Equidad por Perfil de Cliente
================================================================================
Autor: Equipo Rocktec MIA 2026 | Versión: 1.1 | Julio 2026

LIMITACIÓN DOCUMENTADA:
    Rocktec no captura el perfil de cliente en la misma fuente que contiene
    el texto de las conversaciones. JEVA tiene TIPO DE CLIENTE pero sin texto.
    El dataset de consenso tiene texto pero etiqueta_crm es una etiqueta de
    campaña CRM, no un perfil limpio. El perfil usado es un PROXY inferido
    desde etiqueta_crm. Filas sin etiqueta_crm (WhatsApp) quedan excluidas.

RECOMENDACIÓN:
    Incorporar menú de selección de perfil en el primer mensaje de WhatsApp
    Business para capturar este dato en la misma conversación.

Entrada:
    - 04_anotaciones/dataset_consenso_final.csv
    - 06_resultados/modelos/produccion/modelo_lr.pkl
    - 06_resultados/modelos/produccion/vectorizador_tfidf.pkl

Salida:
    - 06_resultados/equidad/reporte_equidad_por_perfil.html
    - 06_resultados/equidad/tabla_equidad_por_perfil.csv
    - 06_resultados/equidad/reporte_equidad.txt
================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '02_scripts')

import feature_engineering  # necesario para deserializar vectorizador_tfidf.pkl

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Rutas ────────────────────────────────────────────────────────────────────
RUTA_DATASET      = Path('04_anotaciones/dataset_consenso_final.csv')
RUTA_MODELO       = Path('06_resultados/modelos/produccion/modelo_lr.pkl')
RUTA_VECTORIZADOR = Path('06_resultados/modelos/produccion/vectorizador_tfidf.pkl')
RUTA_SALIDA       = Path('06_resultados/equidad')
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ── Mapeo etiqueta_crm → perfil proxy ────────────────────────────────────────
MAPEO_PERFIL = {
    'informacion general':                        'Prospecto_General',
    'información general':                        'Prospecto_General',
    'cotizacion materiales':                      'Comprador_Activo',
    'cotización materiales':                      'Comprador_Activo',
    'cotizacion materiales, informacion general': 'Comprador_Activo',
    'venta ganada':                               'Comprador_Activo',
    'venta perdida':                              'Comprador_Activo',
    'curso':                                      'Tecnico_Profesional',
    'capacitacion':                               'Tecnico_Profesional',
    'capacitación':                               'Tecnico_Profesional',
    'visita tecnica':                             'Tecnico_Profesional',
    'visita técnica':                             'Tecnico_Profesional',
}

def inferir_perfil(etiqueta):
    if pd.isna(etiqueta):
        return None
    etiqueta_lower = str(etiqueta).lower().strip()
    if etiqueta_lower in MAPEO_PERFIL:
        return MAPEO_PERFIL[etiqueta_lower]
    for clave, perfil in MAPEO_PERFIL.items():
        if clave in etiqueta_lower:
            return perfil
    return None

def calcular_metricas_perfil(df_perfil, nombre_perfil):
    y_true = df_perfil['intencion_consenso'].astype(str)
    y_pred = df_perfil['prediccion'].astype(str)
    # Filtrar solo filas donde ambos son strings válidos (no 'nan')
    mask = (y_true != 'nan') & (y_pred != 'nan')
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return None
    return {
        'perfil':           nombre_perfil,
        'n_registros':      len(y_true),
        'accuracy':         round((y_true == y_pred).mean(), 4),
        'precision_macro':  round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro':     round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_macro':         round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_weighted':      round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }

def generar_reporte_evidently(df_con_perfil):
    try:
        from evidently import Report, Dataset, DataDefinition, MulticlassClassification
        from evidently.presets import ClassificationPreset

        df_ev = df_con_perfil[['texto_conversacion', 'intencion_consenso', 'prediccion', 'perfil_cliente']].copy()
        df_ev['intencion_consenso'] = df_ev['intencion_consenso'].astype(str)
        df_ev['prediccion']         = df_ev['prediccion'].astype(str)

        data_def = DataDefinition(
            classification=[
                MulticlassClassification(
                    target='intencion_consenso',
                    prediction_labels='prediccion'
                )
            ],
            categorical_columns=['intencion_consenso', 'prediccion', 'perfil_cliente']
        )

        ev_dataset = Dataset.from_pandas(df_ev, data_definition=data_def)
        report = Report([ClassificationPreset()])
        resultado = report.run(ev_dataset, None)

        ruta_html = RUTA_SALIDA / 'reporte_equidad_por_perfil.html'
        resultado.save_html(str(ruta_html))
        print(f"  ✓ Reporte Evidently guardado: {ruta_html}")
        return True
    except Exception as e:
        print(f"  ⚠ Evidently HTML no generado: {e}")
        return False


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SCRIPT 18 — MONITOREO DE EQUIDAD POR PERFIL DE CLIENTE")
    print("Proyecto MIA 2026 - Rocktec")
    print("=" * 70)

    # 1. Cargar dataset
    print("\n[1] Cargando dataset de consenso...")
    df = pd.read_csv(RUTA_DATASET)
    print(f"    Total registros: {len(df)}")

    # Limpiar intencion_consenso — eliminar NaN
    df = df.dropna(subset=['intencion_consenso']).copy()
    df['intencion_consenso'] = df['intencion_consenso'].astype(str).str.strip()
    df = df[df['intencion_consenso'] != ''].copy()
    print(f"    Registros con intención válida: {len(df)}")

    # 2. Cargar modelo
    print("\n[2] Cargando modelo LR de producción...")
    modelo       = joblib.load(RUTA_MODELO)
    vectorizador = joblib.load(RUTA_VECTORIZADOR)
    print(f"    Clases del modelo: {list(modelo.classes_)}")

    # 3. Predicciones
    print("\n[3] Generando predicciones...")
    X = vectorizador.transform(df['texto_conversacion'].fillna('').astype(str))
    df['prediccion'] = modelo.predict(X).astype(str)
    print(f"    Predicciones generadas: {len(df)}")

    # 4. Inferir perfil proxy
    print("\n[4] Infiriendo perfil de cliente desde etiqueta_crm (proxy)...")
    df['perfil_cliente'] = df['etiqueta_crm'].apply(inferir_perfil)
    con_perfil    = df['perfil_cliente'].notna().sum()
    sin_perfil    = df['perfil_cliente'].isna().sum()
    print(f"    Con perfil:    {con_perfil} ({con_perfil/len(df)*100:.1f}%)")
    print(f"    Sin perfil:    {sin_perfil} ({sin_perfil/len(df)*100:.1f}%) — excluidos")
    print(f"\n    Distribución por perfil:")
    for perfil, cnt in df['perfil_cliente'].value_counts(dropna=True).items():
        print(f"      {perfil}: {cnt}")

    # 5. Métricas por perfil
    print("\n[5] Calculando métricas por perfil...")
    df_filtrado = df[df['perfil_cliente'].notna()].copy()
    filas = []
    for perfil in sorted(df_filtrado['perfil_cliente'].unique()):
        subdf = df_filtrado[df_filtrado['perfil_cliente'] == perfil]
        if len(subdf) < 10:
            print(f"    ⚠ {perfil}: {len(subdf)} registros — omitido (mínimo 10)")
            continue
        m = calcular_metricas_perfil(subdf, perfil)
        if m:
            filas.append(m)
            print(f"    {perfil}: n={m['n_registros']} | F1-macro={m['f1_macro']} | Accuracy={m['accuracy']}")

    tabla = pd.DataFrame(filas)

    # 6. Brecha de equidad
    print("\n[6] Brecha de equidad...")
    if len(tabla) >= 2:
        f1_max       = tabla['f1_macro'].max()
        f1_min       = tabla['f1_macro'].min()
        brecha       = round(f1_max - f1_min, 4)
        perfil_mejor = tabla.loc[tabla['f1_macro'].idxmax(), 'perfil']
        perfil_peor  = tabla.loc[tabla['f1_macro'].idxmin(), 'perfil']
        evaluacion   = '✅ aceptable (<0.10)' if brecha < 0.10 else '⚠ significativa (≥0.10)'
        print(f"    Mejor: {perfil_mejor} (F1={f1_max})")
        print(f"    Peor:  {perfil_peor} (F1={f1_min})")
        print(f"    Brecha: {brecha} — {evaluacion}")
    else:
        brecha = f1_max = f1_min = None
        perfil_mejor = perfil_peor = evaluacion = 'N/A'
        print("    ⚠ Menos de 2 perfiles con datos suficientes.")

    # 7. Guardar CSV
    ruta_csv = RUTA_SALIDA / 'tabla_equidad_por_perfil.csv'
    tabla.to_csv(ruta_csv, index=False, encoding='utf-8')
    print(f"\n[7] CSV guardado: {ruta_csv}")

    # 8. Evidently
    print("\n[8] Reporte Evidently...")
    evidently_ok = generar_reporte_evidently(df_filtrado)

    # 9. Reporte txt
    reporte = f"""
{'='*70}
REPORTE DE EQUIDAD — PROYECTO MIA 2026 ROCKTEC
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

LIMITACIÓN DE DATOS (documentada):
{'-'*70}
Rocktec no captura el perfil del cliente en la misma fuente que contiene
el texto de las conversaciones.
  - JEVA: tiene TIPO DE CLIENTE pero sin texto de conversación (OBSERVACIONES
    vacías en los 1,155 registros).
  - Dataset de consenso: tiene texto pero etiqueta_crm es una etiqueta de
    campaña CRM con 61 valores distintos, no un perfil limpio.
El perfil usado es un PROXY inferido desde etiqueta_crm.
{sin_perfil} registros ({sin_perfil/len(df)*100:.1f}%) sin etiqueta_crm fueron excluidos.

COBERTURA:
{'-'*70}
  Total dataset (con intención válida): {len(df)}
  Con perfil asignado:                  {con_perfil} ({con_perfil/len(df)*100:.1f}%)
  Excluidos (sin perfil):               {sin_perfil} ({sin_perfil/len(df)*100:.1f}%)

MÉTRICAS POR PERFIL:
{'-'*70}
""" + "\n".join([
        f"  {r['perfil']}\n"
        f"    n={r['n_registros']} | Accuracy={r['accuracy']} | "
        f"Precision={r['precision_macro']} | Recall={r['recall_macro']} | F1-macro={r['f1_macro']}"
        for _, r in tabla.iterrows()
    ]) + f"""

BRECHA DE EQUIDAD (F1-macro):
{'-'*70}
  Mejor perfil: {perfil_mejor} (F1={f1_max})
  Peor perfil:  {perfil_peor} (F1={f1_min})
  Brecha:       {brecha} — {evaluacion}

RECOMENDACIÓN PARA ROCKTEC:
{'-'*70}
Incorporar en WhatsApp Business un mensaje de bienvenida con menú de
selección de perfil al inicio de cada conversación:
  "¿Con qué perfil nos contactas?"
  1️⃣  Arquitecto / Diseñador
  2️⃣  Constructor / Ingeniero
  3️⃣  Aplicador / Técnico
  4️⃣  Cliente final / Particular
Esto permitirá en futuras versiones calcular métricas de equidad reales
por segmento y personalizar las respuestas automáticas.

Reporte Evidently HTML: {'generado ✅' if evidently_ok else 'no generado ⚠'}
Tabla CSV: {ruta_csv}
{'='*70}
"""
    print(reporte)
    ruta_txt = RUTA_SALIDA / 'reporte_equidad.txt'
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write(reporte)
    print(f"✅ Script 18 completado. Outputs en: {RUTA_SALIDA}")


if __name__ == '__main__':
    main()
