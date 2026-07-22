"""
08_buscar_candidatos_que_seg.py
Rocktec MIA 2026 — Búsqueda dirigida de candidatos QUE y SEG

Problema: en las 1,500 anotaciones actuales solo hay 4 casos de QUE y 5 de SEG.
Este script NO re-muestrea al azar — busca específicamente en las 4 fuentes
crudas (CRM1, CRM2, JEVA, WhatsApp) conversaciones con alta probabilidad de
ser QUE (queja/reclamo) o SEG (seguimiento), para que el equipo las anote
manualmente y así llegar a la meta de 40-50 ejemplos reales por clase.

Salida:
    04_anotaciones/candidatos_QUE.xlsx
    04_anotaciones/candidatos_SEG.xlsx

Uso:
    python 02_scripts/08_buscar_candidatos_que_seg.py
"""

import re
import pandas as pd
from pathlib import Path

RUTA_CRUDOS = Path('01_datos_crudos')
RUTA_SALIDA = Path('04_anotaciones')
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Palabras clave (alineadas al Catálogo de Intenciones v2.0)
# ─────────────────────────────────────────────────────────────────────────

KW_QUE = [
    'reclam', 'queja', 'dañ', 'defectuos', 'mal aplicado', 'mal instalado',
    'no funciona', 'no sirve', 'insatisfech', 'inconform', 'devoluci',
    'garantia', 'garantía', 'se desprend', 'se descascar', 'se agriet',
    'cobraron mal', 'cobro mal', 'error en la factura', 'mal servicio',
    'no cumpli', 'demora', 'retraso', 'lento', 'pesim', 'terrible',
    'nunca llego', 'nunca llegó', 'no llego', 'no llegó', 'perdid',
    # ampliado: atención al cliente y no-respuesta
    'mala atenc', 'pesima atenc', 'pésima atenc', 'no responden',
    'no responde', 'no contestan', 'no contesta', 'no atienden',
    'no me atendieron', 'no me atienden', 'nadie contesta',
    'nadie responde', 'sin respuesta', 'no dan respuesta',
    'no me han contestado', 'no me han respondido', 'no regresaron a llamar',
    'no volvieron a llamar', 'descontent', 'grosero', 'grosera',
    'mal trato', 'maltrato', 'falta de respeto',
    # ampliado: precio (queja de precio, no simple cotización)
    'muy caro', 'demasiado caro', 'carisimo', 'carísimo', 'muy costoso',
    'precio abusivo', 'me parece caro', 'esta muy caro', 'está muy caro',
]

KW_SEG = [
    'estado de mi', 'estado del pedido', 'estado de la cotiz', 'ya llego',
    'ya llegó', 'confirmaron', 'cuando despach', 'cuándo despach',
    'seguimiento', 'que paso con', 'qué pasó con', 'sigo esperando',
    'me pueden confirmar', 'aun no', 'aún no', 'todavia no', 'todavía no',
    'el mes pasado', 'la semana pasada', 'hace dias', 'hace días',
    'anteriormente', 'ya me habian', 'ya me habían',
]

def contiene_alguna(texto, palabras):
    if not isinstance(texto, str):
        return False
    t = texto.lower()
    return any(p in t for p in palabras)


def buscar_en_crm():
    print("Cargando CRM (2 archivos)...")
    crm1 = pd.read_excel(RUTA_CRUDOS / 'Copia de clienty-prospectos 1.xlsx', sheet_name='Worksheet')
    crm2 = pd.read_excel(RUTA_CRUDOS / 'Copia de clienty-prospectos 2.xlsx', sheet_name='Worksheet')
    crm = pd.concat([crm1, crm2], ignore_index=True)

    texto_combinado = (
        crm['Consulta'].fillna('').astype(str) + ' | ' +
        crm['Notas'].fillna('').astype(str) + ' | ' +
        crm['Etiquetas'].fillna('').astype(str)
    )

    # Señal fuerte adicional por Estado (no reemplaza la búsqueda de texto)
    estado_seg = crm['Estado'].astype(str).str.lower().str.contains('seguimiento', na=False)
    estado_perdida = crm['Estado'].astype(str).str.lower().str.contains('perdida', na=False)

    mask_que = texto_combinado.apply(lambda t: contiene_alguna(t, KW_QUE)) | estado_perdida
    mask_seg = texto_combinado.apply(lambda t: contiene_alguna(t, KW_SEG)) | estado_seg

    crm['texto_conversacion'] = texto_combinado
    crm['fuente'] = 'CRM'

    return crm[mask_que].copy(), crm[mask_seg].copy()


def buscar_en_jeva():
    print("Cargando JEVA...")
    jeva = pd.read_excel(RUTA_CRUDOS / 'ROCKTEC - JEVA base datos.xlsx', sheet_name='DATOS JEVA')
    texto = jeva['OBSERVACIONES'].fillna('').astype(str)
    jeva['texto_conversacion'] = texto
    jeva['fuente'] = 'JEVA'

    mask_que = texto.apply(lambda t: contiene_alguna(t, KW_QUE))
    mask_seg = texto.apply(lambda t: contiene_alguna(t, KW_SEG))

    return jeva[mask_que].copy(), jeva[mask_seg].copy()


def buscar_en_whatsapp():
    print("Cargando WhatsApp...")
    wa = pd.read_excel(RUTA_CRUDOS / 'base_maestra_raw_total_rocktec.xlsx', sheet_name='BASE_TOTAL_RAW')
    texto = wa['Detalle'].fillna('').astype(str)
    wa['texto_conversacion'] = texto
    wa['fuente'] = 'WhatsApp'

    mask_que = texto.apply(lambda t: contiene_alguna(t, KW_QUE))
    mask_seg = texto.apply(lambda t: contiene_alguna(t, KW_SEG))

    return wa[mask_que].copy(), wa[mask_seg].copy()


def main():
    print("=" * 70)
    print("BÚSQUEDA DIRIGIDA DE CANDIDATOS — QUE y SEG")
    print("=" * 70)

    crm_que, crm_seg = buscar_en_crm()
    jeva_que, jeva_seg = buscar_en_jeva()
    wa_que, wa_seg = buscar_en_whatsapp()

    cols_comunes = ['fuente', 'texto_conversacion']

    candidatos_que = pd.concat([
        crm_que[cols_comunes], jeva_que[cols_comunes], wa_que[cols_comunes]
    ], ignore_index=True).drop_duplicates(subset='texto_conversacion')

    candidatos_seg = pd.concat([
        crm_seg[cols_comunes], jeva_seg[cols_comunes], wa_seg[cols_comunes]
    ], ignore_index=True).drop_duplicates(subset='texto_conversacion')

    # Columnas para anotación manual (igual formato que el dataset ya anotado)
    for df in (candidatos_que, candidatos_seg):
        df['intencion_patricia'] = ''
        df['intencion_luis_cruel'] = ''
        df['intencion_luis_chica'] = ''
        df['notas_anotacion'] = ''

    candidatos_que.to_excel(RUTA_SALIDA / 'candidatos_QUE.xlsx', index=False)
    candidatos_seg.to_excel(RUTA_SALIDA / 'candidatos_SEG.xlsx', index=False)

    print()
    print("-" * 70)
    print("RESULTADOS")
    print("-" * 70)
    print(f"Candidatos QUE encontrados: {len(candidatos_que)}  (por fuente: CRM={len(crm_que)}, JEVA={len(jeva_que)}, WhatsApp={len(wa_que)})")
    print(f"Candidatos SEG encontrados: {len(candidatos_seg)}  (por fuente: CRM={len(crm_seg)}, JEVA={len(jeva_seg)}, WhatsApp={len(wa_seg)})")
    print()
    print(f"✓ Guardado: {RUTA_SALIDA / 'candidatos_QUE.xlsx'}")
    print(f"✓ Guardado: {RUTA_SALIDA / 'candidatos_SEG.xlsx'}")
    print()
    print("SIGUIENTE PASO: revisar manualmente estos candidatos (no todos serán")
    print("QUE/SEG reales — son candidatos por palabra clave). Meta: confirmar")
    print("40-50 casos reales de cada clase y agregarlos al dataset de anotación.")
    print("=" * 70)


if __name__ == '__main__':
    main()
