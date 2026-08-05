"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script 22: Pipeline Orquestado End-to-End
================================================================================
Cierra el último gap técnico de Fase 3 identificado en DIAGNOSTICO_FASES_3_4_5.md:
"No hay un comando único que corra desde datos crudos hasta el modelo evaluado
— hoy se ejecuta script por script, manualmente."

`06_pipeline_completo.py` ya orquesta la Etapa 1 (ETL: limpieza → consolidación →
duplicados) pero nada más. Este script NO la reemplaza: la invoca como una etapa
opcional y encadena todo lo que falta (features → evaluación → producción →
monitoreo) usando los scripts numerados que ya existen — sin reimplementar su
lógica, cada etapa corre el script real como subproceso.

DECISIONES DE ALCANCE (deliberadas, no simplificaciones de conveniencia):
  - La Etapa 1 (ETL) es OPCIONAL (--incluir-etl) y apagada por defecto: opera
    sobre 01_datos_crudos/ (gitignored, no siempre presente) y regenera CSVs ya
    trackeados en 03_datos_procesados/ — no es algo que se deba rehacer en cada
    corrida de reentrenamiento rutinario.
  - NO se re-ejecuta 09_crear_holdout_set.py automáticamente. Re-partir el
    holdout en cada corrida podría filtrar filas ya "vistas" a un holdout nuevo
    si dataset_consenso_final.csv cambia (p. ej. tras confirmar anotaciones de
    active learning) — exactamente el tipo de error que motivó el fix de fuga
    de datos de Sprint 7. Repartir el holdout debe seguir siendo una decisión
    manual y deliberada, documentada en CHANGELOG.md, no un paso automático.
  - `05_entrenar_modelos.py` (grid search LR/SVM/BETO con MLflow) y
    `07_validacion_estadistica.py` (RepeatedKFold 5×5) son EXPERIMENTOS de
    comparación de arquitecturas, no producen el artefacto de producción —
    quedan detrás de flags opcionales por ser más lentos y no bloquear el
    pipeline de despliegue.
  - El monitoreo de drift (21_monitoreo_drift.py) se salta con una advertencia
    (no aborta el pipeline) si todavía no existe `log_predicciones.csv` — el
    drift depende de que 16_inferencia.py se haya corrido sobre mensajes
    nuevos, algo externo a este pipeline de entrenamiento.

ETAPAS POR DEFECTO (sin flags):
    1. Feature engineering (verificación)      -> 04_feature_engineering.py
    2. Evaluación honesta en holdout            -> 13_evaluacion_holdout.py
    3. Reentrenamiento de producción            -> 15_entrenar_produccion.py
    4. Monitoreo de equidad                     -> 20_monitoreo_equidad.py
    5. Monitoreo de drift (si hay log)          -> 21_monitoreo_drift.py

Uso:
    python 02_scripts/22_pipeline_orquestado.py
    python 02_scripts/22_pipeline_orquestado.py --incluir-etl
    python 02_scripts/22_pipeline_orquestado.py --incluir-entrenamiento-experimental --incluir-validacion-estadistica

Salida:
    06_resultados/pipeline/reporte_pipeline.txt
    06_resultados/pipeline/reporte_pipeline.json
================================================================================
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

RAIZ        = Path(__file__).resolve().parent.parent
DIR_SCRIPTS = RAIZ / '02_scripts'
RUTA_LOG    = RAIZ / '06_resultados' / 'predicciones' / 'log_predicciones.csv'
RUTA_SALIDA = RAIZ / '06_resultados' / 'pipeline'


class Etapa:
    def __init__(self, nombre, script, cwd=RAIZ, args=None, critica=True, condicion=None, motivo_salto=""):
        self.nombre = nombre
        self.script = script
        self.cwd = cwd
        self.args = args or []
        self.critica = critica          # si falla, aborta el resto del pipeline
        self.condicion = condicion      # callable() -> bool; si False, se salta
        self.motivo_salto = motivo_salto
        self.estado = None               # 'ok' | 'error' | 'saltada'
        self.duracion_seg = None
        self.detalle = ""


def ejecutar_etapa(etapa):
    if etapa.condicion is not None and not etapa.condicion():
        etapa.estado = 'saltada'
        etapa.detalle = etapa.motivo_salto
        print(f"  ⏭  Saltada: {etapa.motivo_salto}")
        return

    inicio = datetime.now()
    ruta_script = DIR_SCRIPTS / etapa.script
    comando = [sys.executable, str(ruta_script)] + etapa.args

    resultado = subprocess.run(comando, cwd=str(etapa.cwd), capture_output=True, text=True)
    etapa.duracion_seg = round((datetime.now() - inicio).total_seconds(), 1)

    if resultado.returncode == 0:
        etapa.estado = 'ok'
        print(f"  ✓ OK ({etapa.duracion_seg}s)")
    else:
        etapa.estado = 'error'
        etapa.detalle = resultado.stderr.strip()[-800:]
        print(f"  ✗ ERROR (código {resultado.returncode}, {etapa.duracion_seg}s)")
        print(f"    {etapa.detalle[:300]}")


def construir_etapas(args):
    etapas = []

    etapas.append(Etapa(
        nombre="Etapa 1 — ETL (limpieza, consolidación, duplicados)",
        script="06_pipeline_completo.py",
        cwd=DIR_SCRIPTS,
        critica=True,
        condicion=(lambda: args.incluir_etl),
        motivo_salto="--incluir-etl no especificado (por defecto se asume que 03_datos_procesados/ ya está vigente)",
    ))

    etapas.append(Etapa(
        nombre="Etapa 2 — Feature engineering (verificación)",
        script="04_feature_engineering.py",
        critica=True,
    ))

    if args.incluir_entrenamiento_experimental:
        etapas.append(Etapa(
            nombre="Experimento — Entrenamiento LR+SVM con MLflow",
            script="05_entrenar_modelos.py",
            args=["--skip-beto"],
            critica=False,
        ))

    if args.incluir_validacion_estadistica:
        etapas.append(Etapa(
            nombre="Experimento — Validación estadística pareada (RepeatedKFold 5x5)",
            script="07_validacion_estadistica.py",
            critica=False,
        ))

    etapas.append(Etapa(
        nombre="Etapa 4 — Evaluación honesta en holdout (F1-macro >= 0.75)",
        script="13_evaluacion_holdout.py",
        critica=True,
    ))

    etapas.append(Etapa(
        nombre="Producción — Reentrenar artefacto de despliegue (100% datos etiquetados)",
        script="15_entrenar_produccion.py",
        critica=True,
    ))

    etapas.append(Etapa(
        nombre="Etapa 5 — Monitoreo de equidad por perfil de cliente",
        script="20_monitoreo_equidad.py",
        critica=False,
    ))

    etapas.append(Etapa(
        nombre="Etapa 5 — Monitoreo de drift (PSI)",
        script="21_monitoreo_drift.py",
        critica=False,
        condicion=(lambda: RUTA_LOG.exists()),
        motivo_salto=(
            f"no existe {RUTA_LOG.relative_to(RAIZ)} todavía — corre 16_inferencia.py "
            "sobre un lote de mensajes antes de esperar una lectura de drift"
        ),
    ))

    return etapas


def main():
    parser = argparse.ArgumentParser(description="Pipeline orquestado end-to-end — Rocktec MIA 2026")
    parser.add_argument('--incluir-etl', action='store_true',
                         help="Incluye la Etapa 1 (ETL) sobre 01_datos_crudos/. Apagado por defecto.")
    parser.add_argument('--incluir-entrenamiento-experimental', action='store_true',
                         help="Corre 05_entrenar_modelos.py (grid search LR/SVM con MLflow). Lento, no crítico.")
    parser.add_argument('--incluir-validacion-estadistica', action='store_true',
                         help="Corre 07_validacion_estadistica.py (RepeatedKFold 5x5). Lento, no crítico.")
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE ORQUESTADO END-TO-END — ROCKTEC MIA 2026")
    print("=" * 70)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    etapas = construir_etapas(args)
    inicio_total = datetime.now()

    for etapa in etapas:
        print(f"\n[{etapa.nombre}]")
        ejecutar_etapa(etapa)
        if etapa.estado == 'error' and etapa.critica:
            print(f"\n❌ Etapa crítica falló — se detiene el pipeline.")
            break

    duracion_total = round((datetime.now() - inicio_total).total_seconds(), 1)

    # ── Reporte ──────────────────────────────────────────────────────────────
    resumen = [{
        'etapa': e.nombre, 'estado': e.estado, 'duracion_seg': e.duracion_seg,
        'critica': e.critica, 'detalle': e.detalle,
    } for e in etapas]

    hubo_error_critico = any(e.estado == 'error' and e.critica for e in etapas)

    reporte_json = {
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duracion_total_seg': duracion_total,
        'exito': not hubo_error_critico,
        'etapas': resumen,
    }
    (RUTA_SALIDA / 'reporte_pipeline.json').write_text(
        json.dumps(reporte_json, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    reporte_txt = [
        "=" * 70,
        "REPORTE PIPELINE ORQUESTADO — ROCKTEC MIA 2026",
        "=" * 70,
        f"Fecha: {reporte_json['fecha']}",
        f"Duración total: {duracion_total}s",
        "-" * 70,
    ]
    for e in etapas:
        icono = {'ok': '✓', 'error': '✗', 'saltada': '⏭', None: '·'}[e.estado]
        linea = f"{icono} {e.nombre}"
        if e.duracion_seg is not None:
            linea += f" ({e.duracion_seg}s)"
        reporte_txt.append(linea)
        if e.detalle:
            reporte_txt.append(f"    {e.detalle[:200]}")
    reporte_txt.append("=" * 70)
    (RUTA_SALIDA / 'reporte_pipeline.txt').write_text("\n".join(reporte_txt), encoding='utf-8')

    print(f"\n{'='*70}")
    if hubo_error_critico:
        print("❌ PIPELINE INCOMPLETO — revisa 06_resultados/pipeline/reporte_pipeline.txt")
    else:
        print("✅ PIPELINE COMPLETADO")
    print(f"   Duración total: {duracion_total}s")
    print(f"   Reporte: 06_resultados/pipeline/reporte_pipeline.txt")
    print(f"{'='*70}")

    sys.exit(1 if hubo_error_critico else 0)


if __name__ == '__main__':
    main()
