"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script 21: Monitoreo de Drift (PSI) — Etapa 5 del diseño MLOps
================================================================================
Cierra el gap identificado en DIAGNOSTICO_FASES_3_4_5.md (Fase 3/5): hasta ahora
el monitoreo de drift solo existía como diseño en DISEÑO_MLOPS_FASE2.md §3, sin
ningún script que calculara PSI sobre datos reales.

MÉTODO (igual al diseño original, DISEÑO_MLOPS_FASE2.md §3):
    PSI = Σ (P_actual - P_esperado) × ln(P_actual / P_esperado)

    PSI < 0.10        -> sin drift significativo
    0.10 <= PSI <= 0.25 -> drift moderado, revisar
    PSI > 0.25        -> drift severo, reentrenar

Se calculan DOS PSI independientes:
    1. Distribución de intención predicha (5 clases INF/COT/TEC/CUR/VEN).
    2. Distribución de confianza del modelo (bins de 0.1) — detecta que el
       modelo empiece a "dudar más" aunque la mezcla de intenciones no cambie.

Referencia (P_esperado): distribución real de `train_val.csv` (el conjunto de
entrenamiento final), tal como especifica el diseño original. Actual (P_actual):
distribución de `06_resultados/predicciones/log_predicciones.csv`, el log que
genera 16_inferencia.py en cada corrida.

LIMITACIÓN DOCUMENTADA:
    Al momento de escribir este script no existe todavía tráfico real de un
    piloto en producción (Fase 5 no ha empezado). El log usado para validar
    este script se sembró corriendo 16_inferencia.py sobre un lote de 345
    mensajes de WhatsApp históricos de rocktec_base_validada.csv que NUNCA
    fueron parte de la anotación (0% de solape con train_val/holdout_test,
    verificado por texto exacto) — es un sustituto razonable de "mensajes
    nuevos" para poder probar el script end-to-end, no un piloto real. El
    resultado de esta corrida debe leerse como validación del mecanismo, no
    como una medición de drift de producción.

    Drift de VOCABULARIO (palabras nuevas fuera del vocabulario TF-IDF) queda
    fuera de alcance de este script — el diseño original solo cubre drift de
    la distribución de predicciones, no de features. Documentado como
    extensión futura.

Salida:
    06_resultados/drift/reporte_drift.txt
    06_resultados/drift/tabla_drift_por_clase.csv
    06_resultados/drift/drift_resultado.json
    06_resultados/drift/grafico_drift.png

Uso:
    python 02_scripts/21_monitoreo_drift.py
    python 02_scripts/21_monitoreo_drift.py --log ruta/a/otro_log.csv
================================================================================
"""

import sys
import json
import argparse
import warnings
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

_FE_PATH = Path(__file__).parent / '04_feature_engineering.py'
_spec = importlib.util.spec_from_file_location('feature_engineering', _FE_PATH)
_fe = importlib.util.module_from_spec(_spec)
sys.modules['feature_engineering'] = _fe
_spec.loader.exec_module(_fe)
INTENCIONES = _fe.INTENCIONES

RUTA_TRAIN   = Path('04_anotaciones/train_val.csv')
RUTA_LOG     = Path('06_resultados/predicciones/log_predicciones.csv')
RUTA_SALIDA  = Path('06_resultados/drift')

BINS_CONFIANZA = np.arange(0.0, 1.01, 0.1)

UMBRAL_MODERADO = 0.10
UMBRAL_SEVERO   = 0.25


def calcular_psi(esperado, actual, epsilon=1e-4):
    """
    PSI = Sum (P_actual - P_esperado) * ln(P_actual / P_esperado), por bucket.
    epsilon evita división por cero / log(0) en buckets sin observaciones.
    """
    esperado = np.clip(esperado, epsilon, None)
    actual   = np.clip(actual, epsilon, None)
    return float(np.sum((actual - esperado) * np.log(actual / esperado)))


def interpretar_psi(psi):
    if psi < UMBRAL_MODERADO:
        return 'sin drift significativo'
    if psi <= UMBRAL_SEVERO:
        return 'drift moderado — revisar'
    return 'drift severo — reentrenar'


def distribucion_por_categoria(serie, categorias):
    conteo = serie.value_counts().reindex(categorias, fill_value=0)
    return (conteo / conteo.sum()).values, conteo.values


def cargar_datos(ruta_log):
    print("[1/4] Cargando datos de referencia y de log...")
    train = pd.read_csv(RUTA_TRAIN)
    train = train[train['intencion_consenso'].isin(INTENCIONES)].reset_index(drop=True)

    if not ruta_log.exists():
        raise FileNotFoundError(
            f"No se encontró el log de predicciones en {ruta_log}. "
            "Corre primero 02_scripts/16_inferencia.py sobre al menos un lote de mensajes."
        )
    log = pd.read_csv(ruta_log)

    print(f"  Referencia (train_val.csv): {len(train)} registros")
    print(f"  Log de predicciones:        {len(log)} registros")
    return train, log


def psi_distribucion_intenciones(train, log):
    print("[2/4] Calculando PSI — distribución de intenciones...")
    p_esperado, n_esperado = distribucion_por_categoria(train['intencion_consenso'], INTENCIONES)
    p_actual, n_actual     = distribucion_por_categoria(log['intencion_predicha'], INTENCIONES)

    psi_total = calcular_psi(p_esperado, p_actual)

    tabla = pd.DataFrame({
        'intencion': INTENCIONES,
        'n_esperado': n_esperado,
        'pct_esperado': (p_esperado * 100).round(2),
        'n_actual': n_actual,
        'pct_actual': (p_actual * 100).round(2),
        'contribucion_psi': [
            (a - e) * np.log(max(a, 1e-4) / max(e, 1e-4))
            for e, a in zip(p_esperado, p_actual)
        ],
    })
    tabla['contribucion_psi'] = tabla['contribucion_psi'].round(4)

    print(f"  ✓ PSI (intenciones) = {psi_total:.4f} — {interpretar_psi(psi_total)}")
    return psi_total, tabla


def psi_distribucion_confianza(train, log):
    print("[3/4] Calculando PSI — distribución de confianza...")
    # Referencia: confianza del modelo sobre train_val, estimada OUT-OF-FOLD
    # (cross_val_predict, 5 folds) — NO con predict_proba directo del modelo
    # ya ajustado sobre esos mismos datos. Un modelo evaluado in-sample sobre
    # su propio set de entrenamiento es sistemáticamente sobreconfiado, lo que
    # inflaría artificialmente el PSI frente a cualquier batch nuevo (el mismo
    # tipo de error que motivó el fix de fuga de datos de Sprint 7).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    vec = _fe.VectorizadorTFIDF()
    X = vec.fit_transform(train['texto_conversacion'].values)
    y = train['intencion_consenso'].values

    modelo_cv = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_oof = cross_val_predict(modelo_cv, X, y, cv=cv, method='predict_proba')
    confianza_train = proba_oof.max(axis=1)

    bins_esperado, _ = np.histogram(confianza_train, bins=BINS_CONFIANZA)
    bins_actual, _   = np.histogram(log['confianza'].values, bins=BINS_CONFIANZA)

    p_esperado = bins_esperado / bins_esperado.sum()
    p_actual   = bins_actual / bins_actual.sum()

    psi_confianza = calcular_psi(p_esperado, p_actual)
    print(f"  ✓ PSI (confianza)   = {psi_confianza:.4f} — {interpretar_psi(psi_confianza)}")

    tabla_confianza = pd.DataFrame({
        'bin_confianza': [f"{BINS_CONFIANZA[i]:.1f}-{BINS_CONFIANZA[i+1]:.1f}" for i in range(len(BINS_CONFIANZA)-1)],
        'pct_esperado': (p_esperado * 100).round(2),
        'pct_actual':   (p_actual * 100).round(2),
    })
    return psi_confianza, tabla_confianza


def generar_grafico(tabla_intenciones, ruta):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(tabla_intenciones))
    ancho = 0.35

    ax.bar(x - ancho/2, tabla_intenciones['pct_esperado'], ancho, label='Esperado (train_val)', color='#2196F3')
    ax.bar(x + ancho/2, tabla_intenciones['pct_actual'], ancho, label='Actual (log predicciones)', color='#8B0000')

    ax.set_xticks(x)
    ax.set_xticklabels(tabla_intenciones['intencion'])
    ax.set_ylabel('% de mensajes')
    ax.set_title('Monitoreo de Drift — Distribución de Intenciones\nRocktec MIA 2026', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico guardado: {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Monitoreo de drift (PSI) — Rocktec MIA 2026")
    parser.add_argument('--log', default=str(RUTA_LOG), help="Ruta al log de predicciones (default: log_predicciones.csv)")
    args = parser.parse_args()

    print("=" * 70)
    print("MONITOREO DE DRIFT (PSI) — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_log = Path(args.log)

    train, log = cargar_datos(ruta_log)
    psi_intenciones, tabla_intenciones = psi_distribucion_intenciones(train, log)
    psi_confianza, tabla_confianza = psi_distribucion_confianza(train, log)

    print("[4/4] Guardando reportes...")
    generar_grafico(tabla_intenciones, RUTA_SALIDA / 'grafico_drift.png')
    tabla_intenciones.to_csv(RUTA_SALIDA / 'tabla_drift_por_clase.csv', index=False)

    resultado_json = {
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'log_analizado': str(ruta_log),
        'n_mensajes_log': int(len(log)),
        'psi_distribucion_intenciones': round(psi_intenciones, 4),
        'interpretacion_intenciones': interpretar_psi(psi_intenciones),
        'psi_distribucion_confianza': round(psi_confianza, 4) if psi_confianza is not None else None,
        'interpretacion_confianza': interpretar_psi(psi_confianza) if psi_confianza is not None else 'no calculado (falta artefacto de producción)',
        'umbrales': {'moderado': UMBRAL_MODERADO, 'severo': UMBRAL_SEVERO},
        'limitacion': (
            "Log sembrado con un lote simulado de mensajes históricos nunca anotados "
            "(no es tráfico real de piloto en producción, que todavía no existe)."
        ),
    }
    ruta_json = RUTA_SALIDA / 'drift_resultado.json'
    ruta_json.write_text(json.dumps(resultado_json, indent=2, ensure_ascii=False), encoding='utf-8')

    reporte = f"""
================================================================================
REPORTE DE MONITOREO DE DRIFT — ROCKTEC MIA 2026
================================================================================
Fecha: {resultado_json['fecha']}
Log analizado: {ruta_log} ({len(log)} registros)

LIMITACIÓN DOCUMENTADA:
  Este log fue sembrado con un lote simulado de 345 mensajes históricos de
  WhatsApp nunca anotados (0% de solape con train_val/holdout_test), NO con
  tráfico real de un piloto en producción — Fase 5 no ha comenzado todavía.
  Este reporte valida que el mecanismo de PSI funciona correctamente; no debe
  citarse como una medición de drift real de producción. El PSI de confianza
  en particular sale alto en esta corrida porque el lote simulado incluye
  comentarios de Instagram/Facebook ajenos a Rocktec (temas sin relación,
  ej. crypto) que quedaron mezclados en rocktec_base_validada.csv por un
  problema de calidad de datos previo, no por drift real del negocio — un
  lote de mensajes genuinamente nuevos de clientes de Rocktec debería mostrar
  un PSI de confianza más bajo.

PSI — DISTRIBUCIÓN DE INTENCIONES: {psi_intenciones:.4f} ({interpretar_psi(psi_intenciones)})
{'─'*70}
{tabla_intenciones.to_string(index=False)}

PSI — DISTRIBUCIÓN DE CONFIANZA: {f"{psi_confianza:.4f} ({interpretar_psi(psi_confianza)})" if psi_confianza is not None else "no calculado"}
{'─'*70}
{tabla_confianza.to_string(index=False) if tabla_confianza is not None else ''}

UMBRALES DE REFERENCIA:
  PSI < {UMBRAL_MODERADO}          -> sin drift significativo
  {UMBRAL_MODERADO} <= PSI <= {UMBRAL_SEVERO} -> drift moderado, revisar
  PSI > {UMBRAL_SEVERO}          -> drift severo, reentrenar

FUERA DE ALCANCE:
  Drift de vocabulario (palabras nuevas fuera del TF-IDF) — no cubierto por
  este script. Ver docstring del script para detalle.
================================================================================
"""
    ruta_txt = RUTA_SALIDA / 'reporte_drift.txt'
    ruta_txt.write_text(reporte, encoding='utf-8')
    print(f"  ✓ Reporte guardado: {ruta_txt}")

    print(f"\n{'='*70}")
    print(f"✅ MONITOREO DE DRIFT COMPLETADO")
    print(f"   PSI intenciones: {psi_intenciones:.4f} ({interpretar_psi(psi_intenciones)})")
    if psi_confianza is not None:
        print(f"   PSI confianza:   {psi_confianza:.4f} ({interpretar_psi(psi_confianza)})")
    print(f"{'='*70}")

    if psi_intenciones > UMBRAL_SEVERO:
        sys.exit(1)


if __name__ == '__main__':
    main()
