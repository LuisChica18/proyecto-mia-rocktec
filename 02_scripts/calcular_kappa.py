"""
═══════════════════════════════════════════════════════════════════════════════════════
ROCKTEC MIA 2026 - CÁLCULO DE COHEN'S KAPPA
═══════════════════════════════════════════════════════════════════════════════════════

Script para calcular el coeficiente de acuerdo inter-anotador (Cohen's Kappa)
entre los 3 anotadores (PATTY, LUIS CRUEL, LUIS CHICA)

Uso:
    python calcular_kappa.py <archivo_anotado.xlsx>

Ejemplo:
    python calcular_kappa.py ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx

═══════════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import sys
import importlib.util
from pathlib import Path

_PIPELINE_PATH = Path(__file__).parent / '06_pipeline_completo.py'
_spec = importlib.util.spec_from_file_location('pipeline_completo', _PIPELINE_PATH)
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
MapeoIntenciones = _pipeline.MapeoIntenciones

CODIGOS_VALIDOS = {'INF', 'COT', 'TEC', 'CUR', 'VEN', 'SEG', 'QUE'}
RUTA_CONSENSO = Path(__file__).parent.parent / '04_anotaciones' / 'dataset_consenso_final.csv'


def generar_dataset_consenso(df, anotadores, ruta_salida=RUTA_CONSENSO):
    """
    Genera el dataset de consenso (voto mayoritario 2/3) por fila, junto con una
    etiqueta heurística baseline calculada sobre el mismo texto (MapeoIntenciones),
    para permitir una comparación pareada baseline-vs-consenso en las mismas filas.
    """
    nombres = list(anotadores.keys())
    filas = []

    for idx in range(len(df)):
        anots = [df[anotadores[n]].fillna('').astype(str).str.strip().str.upper().iloc[idx]
                 for n in nombres]

        if all(a in CODIGOS_VALIDOS for a in anots):
            consenso = None
            for a in anots:
                if anots.count(a) >= 2:
                    consenso = a
                    break
            if consenso is None:
                consenso = 'SIN_CONSENSO'
        else:
            consenso = 'SIN_CONSENSO'

        texto = str(df.iloc[idx]['TEXTO'])
        filas.append({
            'id': df.iloc[idx]['ID'],
            'texto_conversacion': texto,
            'fuente': df.iloc[idx].get('FUENTE', ''),
            'intencion_consenso': consenso,
            'intencion_heuristica_baseline': MapeoIntenciones.mapear_desde_texto(texto),
            'n_acuerdo': max((anots.count(a) for a in set(anots)), default=0),
        })

    out = pd.DataFrame(filas)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ruta_salida, index=False, encoding='utf-8')

    n_consenso = (out['intencion_consenso'] != 'SIN_CONSENSO').sum()
    print(f"\n✓ Dataset de consenso guardado: {ruta_salida}")
    print(f"  Filas totales: {len(out)}  |  Con consenso válido: {n_consenso}  "
          f"|  Sin consenso: {len(out) - n_consenso}")


def calcular_kappa(archivo):
    """Calcular Cohen's Kappa para anotaciones"""
    
    print("="*150)
    print("CÁLCULO DE COHEN'S KAPPA - ROCKTEC MIA 2026")
    print("="*150)
    
    # Cargar archivo
    try:
        df = pd.read_excel(archivo, sheet_name='DATOS_ANOTACIÓN')
        print(f"\n✓ Archivo cargado: {archivo}")
    except Exception as e:
        print(f"\n✗ Error al cargar archivo: {e}")
        return
    
    # Columnas de anotadores
    anotadores = {
        'PATRICIA': 'PATRICIA',
        'LUIS_CRUEL': 'LUIS_CRUEL',
        'LUIS_CHICA': 'LUIS_CHICA'
    }
    
    # Obtener anotaciones
    anotaciones = {}
    for nombre, col in anotadores.items():
        if col not in df.columns:
            print(f"✗ Error: Columna '{col}' no encontrada")
            return
        anotaciones[nombre] = df[col].fillna('').astype(str).str.strip().str.upper()
    
    # Filtrar registros anotados (válidos)
    codigos_validos = {'INF', 'COT', 'TEC', 'CUR', 'VEN', 'SEG', 'QUE'}
    
    # Encontrar registros donde todos los anotadores tienen código válido
    registros_validos = []
    for idx in range(len(df)):
        anots = [anotaciones[nombre].iloc[idx] for nombre in anotadores.keys()]
        if all(a in codigos_validos for a in anots):
            registros_validos.append((idx, anots))
    
    print(f"\n✓ Total registros anotados: {len(df)}")
    print(f"✓ Registros con 3 anotaciones VÁLIDAS: {len(registros_validos)}")
    print(f"✓ Registros vacíos/inválidos: {len(df) - len(registros_validos)}")
    
    if len(registros_validos) == 0:
        print("\n✗ No hay registros con 3 anotaciones válidas. Revisa el archivo.")
        return
    
    # Calcular Cohen's Kappa por pares
    print(f"\n{'='*150}")
    print("COHEN'S KAPPA POR PARES")
    print(f"{'='*150}\n")
    
    pares = [
        ('PATRICIA', 'LUIS_CRUEL'),
        ('PATRICIA', 'LUIS_CHICA'),
        ('LUIS_CRUEL', 'LUIS_CHICA')
    ]
    
    kappas = []
    
    for nombre1, nombre2 in pares:
        anots1 = []
        anots2 = []
        
        for idx, anots in registros_validos:
            idx_n1 = list(anotadores.keys()).index(nombre1)
            idx_n2 = list(anotadores.keys()).index(nombre2)
            anots1.append(anots[idx_n1])
            anots2.append(anots[idx_n2])
        
        kappa = cohen_kappa_score(anots1, anots2)
        kappas.append(kappa)
        
        acuerdo = sum(1 for a, b in zip(anots1, anots2) if a == b)
        acuerdo_pct = (acuerdo / len(anots1)) * 100
        
        print(f"{nombre1} vs {nombre2}:")
        print(f"  Cohen's Kappa:    {kappa:.4f}")
        print(f"  Acuerdo exacto:   {acuerdo}/{len(anots1)} ({acuerdo_pct:.1f}%)")
        print()
    
    # Kappa promedio
    kappa_promedio = np.mean(kappas)
    
    print(f"{'='*150}")
    print("RESULTADO FINAL")
    print(f"{'='*150}\n")
    
    print(f"Cohen's Kappa PROMEDIO: {kappa_promedio:.4f}")
    
    if kappa_promedio >= 0.70:
        print(f"✓ META ALCANZADA ✓ (≥ 0.70)")
        print(f"\n→ PROCEDER A FASE 2: Diseño del Pipeline MLOps")
    else:
        print(f"✗ META NO ALCANZADA (< 0.70)")
        print(f"\n→ ACCIÓN: Sesión de alineación + revisión de desacuerdos")
        diferencia = 0.70 - kappa_promedio
        print(f"→ Diferencia: {diferencia:.4f}")
    
    # Distribución de anotaciones
    print(f"\n{'='*150}")
    print("DISTRIBUCIÓN DE INTENCIONES (PROMEDIO DE 3 ANOTADORES)")
    print(f"{'='*150}\n")
    
    dist = {}
    for codigo in codigos_validos:
        for nombre in anotadores.keys():
            anots = anotaciones[nombre]
            count = (anots == codigo).sum()
            if codigo not in dist:
                dist[codigo] = []
            dist[codigo].append(count)
    
    for codigo in sorted(codigos_validos):
        counts = dist.get(codigo, [0, 0, 0])
        promedio = np.mean(counts)
        pct = (promedio / len(registros_validos)) * 100
        print(f"{codigo}: {promedio:.0f} registros ({pct:.1f}%)")
    
    # Desacuerdos principales
    print(f"\n{'='*150}")
    print("ANÁLISIS DE DESACUERDOS (MUESTRA)")
    print(f"{'='*150}\n")
    
    desacuerdos = []
    for idx, anots in registros_validos:
        if len(set(anots)) > 1:  # Hay desacuerdo
            desacuerdos.append((idx, anots))
    
    print(f"Total desacuerdos: {len(desacuerdos)}/{len(registros_validos)} ({(len(desacuerdos)/len(registros_validos)*100):.1f}%)")
    
    if desacuerdos:
        print(f"\nPrimeros 5 casos de desacuerdo:")
        for i, (idx, anots) in enumerate(desacuerdos[:5], 1):
            nombres = list(anotadores.keys())
            texto = str(df.iloc[idx]['TEXTO'])[:80]
            print(f"\n{i}. Fila {idx+2}:")
            print(f"   Texto: {texto}...")
            for nombre, anot in zip(nombres, anots):
                print(f"   {nombre}: {anot}")
    
    # Estadísticas finales
    print(f"\n{'='*150}")
    print("PRÓXIMOS PASOS")
    print(f"{'='*150}\n")
    
    if kappa_promedio >= 0.70:
        print(f"✓ Dataset validado con Kappa = {kappa_promedio:.4f}")
        print(f"✓ Proceder a Fase 2: MLOps Pipeline")
        print(f"✓ Guardando dataset etiquetado para entrenamiento...")
        generar_dataset_consenso(df, anotadores)
    else:
        print(f"✗ Kappa = {kappa_promedio:.4f} (meta: ≥ 0.70)")
        print(f"✗ Acciones recomendadas:")
        print(f"  1. Analizar {len(desacuerdos)} casos de desacuerdo")
        print(f"  2. Sesión de alineación del equipo")
        print(f"  3. Revisar GUÍA MEJORADA v2.0")
        print(f"  4. Re-anotar registros problemáticos")
        print(f"  5. Recalcular Kappa")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python calcular_kappa.py <archivo_anotado.xlsx>")
        print("\nEjemplo:")
        print("  python calcular_kappa.py ROCKTEC_BASE_FINAL_ANOTACION_1500.xlsx")
        sys.exit(1)
    
    archivo = sys.argv[1]
    calcular_kappa(archivo)
