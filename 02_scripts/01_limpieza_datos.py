"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: Limpieza y Preprocesamiento de Datos
================================================================================

Descripción:
    Script para limpiar y preprocesar datos crudos de múltiples fuentes
    (CRM, WhatsApp, JEVA). Elimina registros inválidos, normaliza estructura
    y genera reportes de calidad.

Autor: Equipo Rocktec MIA 2026 (Patricia MC, Luis Cruel, Luis Chica)
Fecha: Junio 2026
Versión: 1.0

Entrada:
    - Copia_de_clienty-prospectos_1.xlsx
    - Copia_de_clienty-prospectos_2.xlsx
    - base_maestra_raw_total_rocktec.xlsx
    - ROCKTEC_-_JEVA_base_datos.xlsx

Salida:
    - ../03_datos_procesados/crm_limpio.csv
    - ../03_datos_procesados/whatsapp_limpio.csv
    - reporte_limpieza.txt

================================================================================
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('limpieza_datos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LimpiadoDatos:
    """Clase para limpiar y validar datos crudos de Rocktec"""
    
    def __init__(self, ruta_crudos, ruta_salida):
        """
        Inicializar limpiador de datos
        
        Args:
            ruta_crudos (str): Ruta a carpeta con datos crudos
            ruta_salida (str): Ruta para guardar datos procesados
        """
        self.ruta_crudos = Path(ruta_crudos)
        self.ruta_salida = Path(ruta_salida)
        self.estadisticas = {}
        
        # Crear carpeta salida si no existe
        self.ruta_salida.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Limpiador iniciado")
        logger.info(f"  Ruta datos crudos: {self.ruta_crudos}")
        logger.info(f"  Ruta salida: {self.ruta_salida}")
    
    def cargar_crm(self, archivo_1, archivo_2):
        """
        Cargar y combinar datos CRM de dos fuentes
        
        Args:
            archivo_1, archivo_2: Nombres de archivos CRM
            
        Returns:
            pd.DataFrame: CRM combinado
        """
        logger.info("Cargando datos CRM...")
        
        try:
            crm_1 = pd.read_excel(self.ruta_crudos / archivo_1, sheet_name='Worksheet')
            crm_2 = pd.read_excel(self.ruta_crudos / archivo_2, sheet_name='Worksheet')
            
            logger.info(f"  CRM 1: {len(crm_1)} registros")
            logger.info(f"  CRM 2: {len(crm_2)} registros")
            
            crm = pd.concat([crm_1, crm_2], ignore_index=True)
            
            self.estadisticas['crm_cargado'] = len(crm)
            logger.info(f"  Total CRM: {len(crm)} registros")
            
            return crm
            
        except Exception as e:
            logger.error(f"Error cargando CRM: {e}")
            raise
    
    def cargar_whatsapp(self, archivo):
        """
        Cargar datos WhatsApp
        
        Args:
            archivo: Nombre de archivo WhatsApp
            
        Returns:
            pd.DataFrame: Datos WhatsApp
        """
        logger.info("Cargando datos WhatsApp...")
        
        try:
            df = pd.read_excel(self.ruta_crudos / archivo, sheet_name='BASE_TOTAL_RAW')
            
            self.estadisticas['whatsapp_cargado'] = len(df)
            logger.info(f"  WhatsApp: {len(df)} registros")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando WhatsApp: {e}")
            raise
    
    def limpiar_texto(self, texto):
        """
        Limpiar texto: espacios múltiples, caracteres especiales
        
        Args:
            texto (str): String a limpiar
            
        Returns:
            str: Texto limpio
        """
        if pd.isna(texto) or not isinstance(texto, str):
            return ""
        
        # Eliminar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        # Eliminar caracteres de control
        texto = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', texto)
        
        return texto
    
    def normalizar_columnas(self, df):
        """
        Normalizar nombres de columnas
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: DataFrame con columnas normalizadas
        """
        logger.info("Normalizando columnas...")
        
        # Convertir a minúsculas y reemplazar espacios
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        logger.info(f"  Columnas normalizadas: {len(df.columns)}")
        
        return df
    
    def eliminar_registros_vacios(self, df):
        """
        Eliminar registros completamente vacíos
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame, dict: DataFrame limpio y estadísticas
        """
        logger.info("Eliminando registros vacíos...")
        
        registros_antes = len(df)
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        registros_eliminados = registros_antes - len(df)
        
        logger.info(f"  Registros eliminados: {registros_eliminados}")
        
        return df, {
            'registros_antes': registros_antes,
            'registros_eliminados': registros_eliminados
        }
    
    def limpiar_crm(self, crm):
        """
        Pipeline de limpieza para CRM
        
        Args:
            crm: DataFrame de CRM
            
        Returns:
            pd.DataFrame: CRM limpio
        """
        logger.info("\nLimpiando CRM...")
        
        # Normalizar columnas
        crm = self.normalizar_columnas(crm)
        
        # Eliminar vacíos
        crm, stats = self.eliminar_registros_vacios(crm)
        self.estadisticas['crm_eliminados'] = stats['registros_eliminados']
        
        # Limpiar texto en columnas clave
        if 'consulta' in crm.columns:
            crm['consulta'] = crm['consulta'].apply(self.limpiar_texto)
        
        if 'notas' in crm.columns:
            crm['notas'] = crm['notas'].apply(self.limpiar_texto)
        
        self.estadisticas['crm_final'] = len(crm)
        logger.info(f"  CRM final: {len(crm)} registros")
        
        return crm
    
    def limpiar_whatsapp(self, whatsapp):
        """
        Pipeline de limpieza para WhatsApp
        
        Args:
            whatsapp: DataFrame de WhatsApp
            
        Returns:
            pd.DataFrame: WhatsApp limpio
        """
        logger.info("\nLimpiando WhatsApp...")
        
        # Normalizar columnas
        whatsapp = self.normalizar_columnas(whatsapp)
        
        # Eliminar vacíos
        whatsapp, stats = self.eliminar_registros_vacios(whatsapp)
        self.estadisticas['whatsapp_eliminados'] = stats['registros_eliminados']
        
        # Limpiar texto
        if 'detalle' in whatsapp.columns:
            whatsapp['detalle'] = whatsapp['detalle'].apply(self.limpiar_texto)
        
        self.estadisticas['whatsapp_final'] = len(whatsapp)
        logger.info(f"  WhatsApp final: {len(whatsapp)} registros")
        
        return whatsapp
    
    def guardar_datos(self, crm, whatsapp):
        """
        Guardar datos limpios en CSV
        
        Args:
            crm: DataFrame de CRM
            whatsapp: DataFrame de WhatsApp
        """
        logger.info("\nGuardando datos procesados...")
        
        # Guardar CRM
        ruta_crm = self.ruta_salida / 'crm_limpio.csv'
        crm.to_csv(ruta_crm, index=False, encoding='utf-8')
        logger.info(f"  ✓ CRM guardado: {ruta_crm}")
        
        # Guardar WhatsApp
        ruta_wa = self.ruta_salida / 'whatsapp_limpio.csv'
        whatsapp.to_csv(ruta_wa, index=False, encoding='utf-8')
        logger.info(f"  ✓ WhatsApp guardado: {ruta_wa}")
    
    def generar_reporte(self):
        """Generar reporte de limpieza"""
        logger.info("\nGenerando reporte...")
        
        reporte = f"""
================================================================================
REPORTE DE LIMPIEZA DE DATOS - ROCKTEC MIA 2026
================================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTADISTICAS DE LIMPIEZA:
{'-'*80}

CRM:
  - Registros cargados: {self.estadisticas.get('crm_cargado', 0)}
  - Registros eliminados: {self.estadisticas.get('crm_eliminados', 0)}
  - Registros finales: {self.estadisticas.get('crm_final', 0)}

WhatsApp:
  - Registros cargados: {self.estadisticas.get('whatsapp_cargado', 0)}
  - Registros eliminados: {self.estadisticas.get('whatsapp_eliminados', 0)}
  - Registros finales: {self.estadisticas.get('whatsapp_final', 0)}

TOTAL FINAL:
  - Registros disponibles para consolidación: {self.estadisticas.get('crm_final', 0) + self.estadisticas.get('whatsapp_final', 0)}

ARCHIVOS GENERADOS:
  - ../03_datos_procesados/crm_limpio.csv
  - ../03_datos_procesados/whatsapp_limpio.csv

================================================================================
"""
        
        # Guardar reporte
        ruta_reporte = self.ruta_salida / 'reporte_limpieza.txt'
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        logger.info(f"Reporte guardado: {ruta_reporte}")
        print(reporte)


def main():
    """Función principal"""
    
    logger.info("="*80)
    logger.info("INICIO: Limpieza de Datos Rocktec MIA 2026")
    logger.info("="*80)
    
    # Rutas
    RUTA_CRUDOS = Path('../01_datos_crudos')
    RUTA_SALIDA = Path('../03_datos_procesados')
    
    # Crear limpiador
    limpiador = LimpiadoDatos(RUTA_CRUDOS, RUTA_SALIDA)
    
    # Cargar datos
    crm = limpiador.cargar_crm(
        'Copia_de_clienty-prospectos_1.xlsx',
        'Copia_de_clienty-prospectos_2.xlsx'
    )
    
    whatsapp = limpiador.cargar_whatsapp('base_maestra_raw_total_rocktec.xlsx')
    
    # Limpiar
    crm_limpio = limpiador.limpiar_crm(crm)
    whatsapp_limpio = limpiador.limpiar_whatsapp(whatsapp)
    
    # Guardar
    limpiador.guardar_datos(crm_limpio, whatsapp_limpio)
    
    # Reporte
    limpiador.generar_reporte()
    
    logger.info("\n" + "="*80)
    logger.info("FIN: Limpieza completada exitosamente")
    logger.info("="*80)


if __name__ == '__main__':
    main()
