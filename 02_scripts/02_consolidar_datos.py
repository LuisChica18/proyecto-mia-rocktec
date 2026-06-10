"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: Consolidación y Mapeo de Intenciones
================================================================================

Descripción:
    Consolida datos de múltiples fuentes (CRM, WhatsApp) en una base única.
    Mapea estados/etiquetas CRM a categorías de intención según
    Catálogo de Intenciones v2.0.

Autor: Equipo Rocktec MIA 2026
Versión: 1.0

Intenciones definidas:
    INF - Información General
    COT - Cotización
    TEC - Consulta Técnica
    CUR - Consulta de Cursos
    VEN - Venta/Confirmación
    SEG - Seguimiento
    QUE - Queja/Reclamo

Entrada:
    - ../03_datos_procesados/crm_limpio.csv
    - ../03_datos_procesados/whatsapp_limpio.csv

Salida:
    - ../03_datos_procesados/rocktec_base_consolidada.csv
    - reporte_consolidacion.txt

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapeoIntenciones:
    """Mapear datos a categorías de intención según Catálogo v2.0"""
    
    INTENCIONES = {
        'INF': 'Información General',
        'COT': 'Cotización',
        'TEC': 'Consulta Técnica',
        'CUR': 'Consulta de Cursos',
        'VEN': 'Venta/Confirmación',
        'SEG': 'Seguimiento',
        'QUE': 'Queja/Reclamo',
    }
    
    @staticmethod
    def mapear_desde_texto(texto):
        """Mapear intención desde texto de conversación"""
        if not isinstance(texto, str) or len(texto) < 5:
            return 'INF'
        
        texto_lower = texto.lower()
        
        palabras_clave = {
            'COT': ['cotizacion', 'presupuesto', 'precio', 'proforma', 'costo', 'valor'],
            'TEC': ['aplicacion', 'instalacion', 'como', 'tecnica', 'herramientas', 'durabilidad'],
            'CUR': ['curso', 'capacitacion', 'taller', 'certificacion', 'formacion'],
            'VEN': ['confirmo', 'adelante', 'compra', 'pago', 'factura'],
            'QUE': ['problema', 'queja', 'reclamo', 'dañado', 'insatisfecho', 'falla'],
            'SEG': ['seguimiento', 'estado', 'cuando', 'confirmacion', 'status'],
        }
        
        for intencion, palabras in palabras_clave.items():
            if any(palabra in texto_lower for palabra in palabras):
                return intencion
        
        return 'INF'
    
    @staticmethod
    def mapear_desde_crm(row):
        """Mapear intención desde campos CRM"""
        estado = str(row.get('estado', '')).lower()
        etiquetas = str(row.get('etiquetas', '')).lower()
        consulta = str(row.get('consulta', '')).lower()
        
        if 'venta ganada' in estado:
            return 'VEN'
        elif 'cotizacion' in etiquetas:
            return 'COT'
        elif 'visita tecnica' in etiquetas or 'visita' in etiquetas:
            return 'TEC'
        elif any(w in etiquetas for w in ['curso', 'capacitacion']):
            return 'CUR'
        elif 'venta perdida' in estado:
            return 'QUE'
        elif any(w in estado for w in ['seguimiento', 'en seguimiento']):
            return 'SEG'
        elif len(consulta) > 10:
            return MapeoIntenciones.mapear_desde_texto(consulta)
        else:
            return 'INF'


class ConsolidadorDatos:
    """Consolidar datos de múltiples fuentes"""
    
    def __init__(self, ruta_datos_limpios, ruta_salida):
        self.ruta_limpios = Path(ruta_datos_limpios)
        self.ruta_salida = Path(ruta_salida)
        self.estadisticas = {}
        self.ruta_salida.mkdir(parents=True, exist_ok=True)
        logger.info("Consolidador iniciado")
    
    def consolidar_crm(self, crm):
        """Convertir registros CRM a formato de conversaciones"""
        logger.info("Consolidando CRM...")
        conversaciones = []
        
        for idx, row in crm.iterrows():
            partes = []
            
            if pd.notna(row.get('estado')):
                partes.append(f"Estado: {row['estado']}")
            if pd.notna(row.get('etiquetas')):
                partes.append(f"Etiquetas: {row['etiquetas']}")
            if pd.notna(row.get('consulta')) and len(str(row['consulta']).strip()) > 5:
                partes.append(f"Consulta: {str(row['consulta'])[:200]}")
            if pd.notna(row.get('notas')) and len(str(row['notas']).strip()) > 20:
                partes.append(f"Notas: {str(row['notas'])[:300]}")
            
            texto = ' | '.join(partes)
            
            if len(texto) > 20:
                conversaciones.append({
                    'id_cliente': f"CRM_{idx}",
                    'nombre_cliente': str(row.get('nombre', 'Cliente')),
                    'empresa': str(row.get('empresa', '')),
                    'canal': 'CRM',
                    'fecha_contacto': str(row.get('fecha_de_ingreso', '')),
                    'intencion_catalogo': MapeoIntenciones.mapear_desde_crm(row),
                    'texto_conversacion': texto,
                    'estado_crm': str(row.get('estado', '')),
                    'etiquetas_crm': str(row.get('etiquetas', '')),
                    'fuente': 'CRM Real'
                })
        
        self.estadisticas['crm_consolidado'] = len(conversaciones)
        logger.info(f"  {len(conversaciones)} registros CRM consolidados")
        return conversaciones
    
    def consolidar_whatsapp(self, whatsapp):
        """Consolidar conversaciones WhatsApp"""
        logger.info("Consolidando WhatsApp...")
        consolidadas = []
        
        for idx, row in whatsapp.iterrows():
            detalle = str(row.get('detalle', '')).strip()
            
            if len(detalle) > 20:
                consolidadas.append({
                    'id_cliente': f"WA_{idx}",
                    'nombre_cliente': str(row.get('remitente', 'Cliente')),
                    'empresa': str(row.get('ciudad', '')),
                    'canal': 'WhatsApp',
                    'fecha_contacto': str(row.get('fecha', '')),
                    'intencion_catalogo': MapeoIntenciones.mapear_desde_texto(detalle),
                    'texto_conversacion': detalle[:800],
                    'estado_crm': '',
                    'etiquetas_crm': '',
                    'fuente': 'WhatsApp Real'
                })
        
        self.estadisticas['whatsapp_consolidado'] = len(consolidadas)
        logger.info(f"  {len(consolidadas)} conversaciones WhatsApp consolidadas")
        return consolidadas
    
    def crear_base_final(self, crm_consolidado, wa_consolidado):
        """Crear base final consolidada"""
        logger.info("Creando base final consolidada...")
        
        df_crm = pd.DataFrame(crm_consolidado)
        df_wa = pd.DataFrame(wa_consolidado)
        
        base_final = pd.concat([df_crm, df_wa], ignore_index=True)
        base_final = base_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        base_final['intencion_patricia'] = ''
        base_final['intencion_luis_cruel'] = ''
        base_final['intencion_luis_chica'] = ''
        base_final['notas_anotacion'] = ''
        
        self.estadisticas['total_final'] = len(base_final)
        logger.info(f"  Base final: {len(base_final)} registros")
        
        logger.info(f"\nDistribución por intención:")
        for intencion, count in base_final['intencion_catalogo'].value_counts().items():
            pct = (count / len(base_final)) * 100
            logger.info(f"  {intencion}: {count} ({pct:.1f}%)")
        
        return base_final
    
    def guardar_base(self, df):
        """Guardar base procesada"""
        logger.info(f"Guardando base consolidada...")
        ruta_csv = self.ruta_salida / 'rocktec_base_consolidada.csv'
        df.to_csv(ruta_csv, index=False, encoding='utf-8')
        logger.info(f"  ✓ Guardado: {ruta_csv}")
    
    def generar_reporte(self, df):
        """Generar reporte de consolidación"""
        logger.info("Generando reporte...")
        
        reporte = f"""
================================================================================
REPORTE DE CONSOLIDACION - ROCKTEC MIA 2026
================================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTADISTICAS:
{'-'*80}

CRM consolidado: {self.estadisticas.get('crm_consolidado', 0)} registros
WhatsApp consolidado: {self.estadisticas.get('whatsapp_consolidado', 0)} registros
Total base final: {self.estadisticas.get('total_final', 0)} registros

DISTRIBUCION POR INTENCION:
{'-'*80}
"""
        
        for intencion, count in df['intencion_catalogo'].value_counts().items():
            pct = (count / len(df)) * 100
            reporte += f"\n{intencion}: {count:5} ({pct:5.1f}%)"
        
        reporte += f"""

DISTRIBUCION POR FUENTE:
{'-'*80}
"""
        
        for fuente, count in df['fuente'].value_counts().items():
            pct = (count / len(df)) * 100
            reporte += f"\n{fuente}: {count:5} ({pct:5.1f}%)"
        
        reporte += f"""

ARCHIVO GENERADO:
  - ../03_datos_procesados/rocktec_base_consolidada.csv

================================================================================
"""
        
        ruta_reporte = self.ruta_salida / 'reporte_consolidacion.txt'
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        logger.info(f"Reporte guardado: {ruta_reporte}")
        print(reporte)


def main():
    logger.info("="*80)
    logger.info("CONSOLIDACION DE DATOS - ROCKTEC MIA 2026")
    logger.info("="*80)
    
    RUTA_LIMPIOS = Path('../03_datos_procesados')
    RUTA_SALIDA = RUTA_LIMPIOS
    
    logger.info("\nCargando datos limpios...")
    crm = pd.read_csv(RUTA_LIMPIOS / 'crm_limpio.csv')
    whatsapp = pd.read_csv(RUTA_LIMPIOS / 'whatsapp_limpio.csv')
    
    logger.info(f"  CRM cargado: {len(crm)} registros")
    logger.info(f"  WhatsApp cargado: {len(whatsapp)} registros")
    
    consolidador = ConsolidadorDatos(RUTA_LIMPIOS, RUTA_SALIDA)
    crm_consolidado = consolidador.consolidar_crm(crm)
    wa_consolidado = consolidador.consolidar_whatsapp(whatsapp)
    base_final = consolidador.crear_base_final(crm_consolidado, wa_consolidado)
    
    consolidador.guardar_base(base_final)
    consolidador.generar_reporte(base_final)
    
    logger.info("\n" + "="*80)
    logger.info("CONSOLIDACION COMPLETADA")
    logger.info("="*80)


if __name__ == '__main__':
    main()
