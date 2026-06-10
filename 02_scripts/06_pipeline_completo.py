"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: Pipeline Completo - Limpieza a Validación
================================================================================

Descripción:
    Ejecuta la secuencia completa de procesamiento de datos:
    1. Limpieza de datos crudos
    2. Consolidación y mapeo de intenciones
    3. Validación y detección de duplicados
    
    Genera base final lista para anotación inter-anotador (Fase 1)

Autor: Equipo Rocktec MIA 2026
Versión: 1.0

Ejecución:
    python 06_pipeline_completo.py

Salida:
    - ../03_datos_procesados/rocktec_base_validada.csv
    - Reportes de cada fase
    - pipeline_completo.log

================================================================================
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_completo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FASE 1: LIMPIEZA
# ============================================================================

class LimpiadoDatos:
    """Limpiador de datos crudos"""
    
    def __init__(self, ruta_crudos, ruta_salida):
        self.ruta_crudos = Path(ruta_crudos)
        self.ruta_salida = Path(ruta_salida)
        self.estadisticas = {}
        self.ruta_salida.mkdir(parents=True, exist_ok=True)
    
    def cargar_crm(self, archivo_1, archivo_2):
        logger.info("  Cargando CRM...")
        crm_1 = pd.read_excel(self.ruta_crudos / archivo_1, sheet_name='Worksheet')
        crm_2 = pd.read_excel(self.ruta_crudos / archivo_2, sheet_name='Worksheet')
        crm = pd.concat([crm_1, crm_2], ignore_index=True)
        self.estadisticas['crm_cargado'] = len(crm)
        return crm
    
    def cargar_whatsapp(self, archivo):
        logger.info("  Cargando WhatsApp...")
        df = pd.read_excel(self.ruta_crudos / archivo, sheet_name='BASE_TOTAL_RAW')
        self.estadisticas['whatsapp_cargado'] = len(df)
        return df
    
    def normalizar_columnas(self, df):
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df
    
    def limpiar_crm(self, crm):
        logger.info("  Limpiando CRM...")
        crm = self.normalizar_columnas(crm)
        crm = crm.dropna(how='all')
        self.estadisticas['crm_final'] = len(crm)
        return crm
    
    def limpiar_whatsapp(self, whatsapp):
        logger.info("  Limpiando WhatsApp...")
        whatsapp = self.normalizar_columnas(whatsapp)
        whatsapp = whatsapp.dropna(how='all')
        self.estadisticas['whatsapp_final'] = len(whatsapp)
        return whatsapp
    
    def guardar_datos(self, crm, whatsapp):
        crm.to_csv(self.ruta_salida / 'crm_limpio.csv', index=False, encoding='utf-8')
        whatsapp.to_csv(self.ruta_salida / 'whatsapp_limpio.csv', index=False, encoding='utf-8')


# ============================================================================
# FASE 2: CONSOLIDACION
# ============================================================================

class MapeoIntenciones:
    """Mapear intenciones"""
    
    @staticmethod
    def mapear_desde_texto(texto):
        if not isinstance(texto, str) or len(texto) < 5:
            return 'INF'
        
        texto_lower = texto.lower()
        palabras_clave = {
            'COT': ['cotizacion', 'presupuesto', 'precio', 'proforma', 'costo'],
            'TEC': ['aplicacion', 'instalacion', 'como', 'tecnica', 'herramientas'],
            'CUR': ['curso', 'capacitacion', 'taller', 'certificacion'],
            'VEN': ['confirmo', 'adelante', 'compra', 'pago', 'factura'],
            'QUE': ['problema', 'queja', 'reclamo', 'dañado', 'insatisfecho'],
            'SEG': ['seguimiento', 'estado', 'cuando', 'confirmacion'],
        }
        
        for intencion, palabras in palabras_clave.items():
            if any(palabra in texto_lower for palabra in palabras):
                return intencion
        return 'INF'
    
    @staticmethod
    def mapear_desde_crm(row):
        estado = str(row.get('estado', '')).lower()
        etiquetas = str(row.get('etiquetas', '')).lower()
        consulta = str(row.get('consulta', '')).lower()
        
        if 'venta ganada' in estado:
            return 'VEN'
        elif 'cotizacion' in etiquetas:
            return 'COT'
        elif 'visita tecnica' in etiquetas or 'visita' in etiquetas:
            return 'TEC'
        elif 'curso' in etiquetas or 'capacitacion' in etiquetas:
            return 'CUR'
        elif 'venta perdida' in estado:
            return 'QUE'
        elif 'seguimiento' in estado:
            return 'SEG'
        elif len(consulta) > 10:
            return MapeoIntenciones.mapear_desde_texto(consulta)
        return 'INF'


class ConsolidadorDatos:
    """Consolidar datos"""
    
    def __init__(self, ruta_datos_limpios, ruta_salida):
        self.ruta_limpios = Path(ruta_datos_limpios)
        self.ruta_salida = Path(ruta_salida)
        self.estadisticas = {}
    
    def consolidar_crm(self, crm):
        logger.info("  Consolidando CRM...")
        conversaciones = []
        
        for idx, row in crm.iterrows():
            partes = []
            if pd.notna(row.get('estado')):
                partes.append(f"Estado: {row['estado']}")
            if pd.notna(row.get('etiquetas')):
                partes.append(f"Etiquetas: {row['etiquetas']}")
            if pd.notna(row.get('consulta')) and len(str(row['consulta']).strip()) > 5:
                partes.append(f"Consulta: {str(row['consulta'])[:200]}")
            
            texto = ' | '.join(partes)
            if len(texto) > 20:
                conversaciones.append({
                    'id_cliente': f"CRM_{idx}",
                    'nombre_cliente': str(row.get('nombre', 'Cliente')),
                    'canal': 'CRM',
                    'intencion_catalogo': MapeoIntenciones.mapear_desde_crm(row),
                    'texto_conversacion': texto,
                    'fuente': 'CRM Real'
                })
        
        self.estadisticas['crm_consolidado'] = len(conversaciones)
        return conversaciones
    
    def consolidar_whatsapp(self, whatsapp):
        logger.info("  Consolidando WhatsApp...")
        consolidadas = []
        
        for idx, row in whatsapp.iterrows():
            detalle = str(row.get('detalle', '')).strip()
            if len(detalle) > 20:
                consolidadas.append({
                    'id_cliente': f"WA_{idx}",
                    'nombre_cliente': str(row.get('remitente', 'Cliente')),
                    'canal': 'WhatsApp',
                    'intencion_catalogo': MapeoIntenciones.mapear_desde_texto(detalle),
                    'texto_conversacion': detalle[:800],
                    'fuente': 'WhatsApp Real'
                })
        
        self.estadisticas['whatsapp_consolidado'] = len(consolidadas)
        return consolidadas
    
    def crear_base_final(self, crm_consolidado, wa_consolidado):
        logger.info("  Creando base final...")
        df_crm = pd.DataFrame(crm_consolidado)
        df_wa = pd.DataFrame(wa_consolidado)
        
        base_final = pd.concat([df_crm, df_wa], ignore_index=True)
        base_final = base_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        base_final['intencion_patricia'] = ''
        base_final['intencion_luis_cruel'] = ''
        base_final['intencion_luis_chica'] = ''
        base_final['notas_anotacion'] = ''
        
        self.estadisticas['total_final'] = len(base_final)
        return base_final
    
    def guardar_base(self, df):
        df.to_csv(self.ruta_salida / 'rocktec_base_consolidada.csv', index=False, encoding='utf-8')


# ============================================================================
# FASE 3: VALIDACION
# ============================================================================

class DetectorDuplicados:
    """Detector de duplicados"""
    
    def __init__(self, ruta_entrada, ruta_salida):
        self.ruta_entrada = Path(ruta_entrada)
        self.ruta_salida = Path(ruta_salida)
        self.duplicados_similares = []
    
    def detectar_duplicados_exactos(self, df):
        logger.info("  Detectando duplicados exactos...")
        columnas_comparar = ['texto_conversacion', 'nombre_cliente', 'canal']
        duplicados = df[df.duplicated(subset=columnas_comparar, keep=False)]
        logger.info(f"    Encontrados: {len(duplicados)}")
        return duplicados
    
    def eliminar_duplicados(self, df):
        logger.info("  Eliminando duplicados exactos...")
        columnas_comparar = ['texto_conversacion', 'nombre_cliente', 'canal']
        registros_antes = len(df)
        df_limpio = df.drop_duplicates(subset=columnas_comparar, keep='first')
        eliminados = registros_antes - len(df_limpio)
        logger.info(f"    Eliminados: {eliminados}")
        return df_limpio
    
    def guardar_datos(self, df_limpio):
        df_limpio.to_csv(self.ruta_salida / 'rocktec_base_validada.csv', index=False, encoding='utf-8')


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Ejecutar pipeline completo"""
    
    inicio = datetime.now()
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETO - ROCKTEC MIA 2026")
    logger.info("="*80)
    logger.info(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    RUTA_CRUDOS = Path('../01_datos_crudos')
    RUTA_PROCESADOS = Path('../03_datos_procesados')
    
    try:
        # FASE 1: LIMPIEZA
        logger.info("\n[FASE 1] LIMPIEZA DE DATOS")
        logger.info("-"*80)
        limpiador = LimpiadoDatos(RUTA_CRUDOS, RUTA_PROCESADOS)
        crm = limpiador.cargar_crm(
            'Copia_de_clienty-prospectos_1.xlsx',
            'Copia_de_clienty-prospectos_2.xlsx'
        )
        whatsapp = limpiador.cargar_whatsapp('base_maestra_raw_total_rocktec.xlsx')
        crm_limpio = limpiador.limpiar_crm(crm)
        whatsapp_limpio = limpiador.limpiar_whatsapp(whatsapp)
        limpiador.guardar_datos(crm_limpio, whatsapp_limpio)
        logger.info(f"✓ Limpieza completada")
        
        # FASE 2: CONSOLIDACION
        logger.info("\n[FASE 2] CONSOLIDACION Y MAPEO")
        logger.info("-"*80)
        consolidador = ConsolidadorDatos(RUTA_PROCESADOS, RUTA_PROCESADOS)
        crm_consolidado = consolidador.consolidar_crm(crm_limpio)
        wa_consolidado = consolidador.consolidar_whatsapp(whatsapp_limpio)
        base_final = consolidador.crear_base_final(crm_consolidado, wa_consolidado)
        consolidador.guardar_base(base_final)
        logger.info(f"✓ Consolidación completada: {len(base_final)} registros")
        
        # FASE 3: VALIDACION
        logger.info("\n[FASE 3] VALIDACION Y DUPLICADOS")
        logger.info("-"*80)
        detector = DetectorDuplicados(RUTA_PROCESADOS, RUTA_PROCESADOS)
        detector.detectar_duplicados_exactos(base_final)
        base_validada = detector.eliminar_duplicados(base_final)
        detector.guardar_datos(base_validada)
        logger.info(f"✓ Validación completada: {len(base_validada)} registros finales")
        
        # RESUMEN
        fin = datetime.now()
        duracion = (fin - inicio).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("RESUMEN FINAL")
        logger.info("="*80)
        logger.info(f"\nARCHIVOS GENERADOS:")
        logger.info(f"  ✓ rocktec_base_validada.csv ({len(base_validada)} registros)")
        logger.info(f"\nESTADISTICAS:")
        logger.info(f"  CRM: {len(crm_limpio)} registros")
        logger.info(f"  WhatsApp: {len(whatsapp_limpio)} registros")
        logger.info(f"  Total consolidado: {len(base_final)} registros")
        logger.info(f"  Total validado: {len(base_validada)} registros")
        logger.info(f"\nTIEMPO:")
        logger.info(f"  Duración: {duracion:.2f} segundos ({duracion/60:.2f} minutos)")
        logger.info(f"  Fin: {fin.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n✗ ERROR EN PIPELINE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
