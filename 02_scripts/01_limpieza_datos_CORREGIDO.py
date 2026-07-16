import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("LIMPIEZA DE DATOS ROCKTEC MIA 2026")
logger.info("="*80)

# ARCHIVOS CORRECTOS
try:
    logger.info("Leyendo Copia de clienty-prospectos 1.xlsx...")
    crm1 = pd.read_excel("01_datos_crudos/Copia de clienty-prospectos 1.xlsx")
    logger.info(f"  OK: {len(crm1)} registros")
    
    logger.info("Leyendo Copia de clienty-prospectos 2.xlsx...")
    crm2 = pd.read_excel("01_datos_crudos/Copia de clienty-prospectos 2.xlsx")
    logger.info(f"  OK: {len(crm2)} registros")
    
    logger.info("Leyendo ROCKTEC - JEVA base datos.xlsx...")
    jeva = pd.read_excel("01_datos_crudos/ROCKTEC - JEVA base datos.xlsx")
    logger.info(f"  OK: {len(jeva)} registros")
    
    logger.info("Leyendo base_maestra_raw_total_rocktec.xlsx...")
    whatsapp = pd.read_excel("01_datos_crudos/base_maestra_raw_total_rocktec.xlsx")
    logger.info(f"  OK: {len(whatsapp)} registros")
    
    logger.info(f"\nTOTAL BRUTO: {len(crm1) + len(crm2) + len(jeva) + len(whatsapp)}")
    
    # Llenar NaN
    crm1 = crm1.fillna('')
    crm2 = crm2.fillna('')
    jeva = jeva.fillna('')
    whatsapp = whatsapp.fillna('')
    
    # Guardar
    logger.info("\nGuardando datos limpios...")
    crm1.to_csv("03_datos_procesados/crm_limpio_1.csv", index=False)
    crm2.to_csv("03_datos_procesados/crm_limpio_2.csv", index=False)
    jeva.to_csv("03_datos_procesados/jeva_limpio.csv", index=False)
    whatsapp.to_csv("03_datos_procesados/whatsapp_limpio.csv", index=False)
    
    logger.info("\nOK: Datos guardados en 03_datos_procesados/")
    logger.info("COMPLETA")
    
except Exception as e:
    logger.error(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
