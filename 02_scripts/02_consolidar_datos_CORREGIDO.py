import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("CONSOLIDACION DE DATOS - 14,974 -> 1,500")
logger.info("="*80)

try:
    logger.info("Leyendo CRM 1...")
    crm1 = pd.read_csv("03_datos_procesados/crm_limpio_1.csv")
    logger.info(f"  OK: {len(crm1)}")
    
    logger.info("Leyendo CRM 2...")
    crm2 = pd.read_csv("03_datos_procesados/crm_limpio_2.csv")
    logger.info(f"  OK: {len(crm2)}")
    
    logger.info("Leyendo JEVA...")
    jeva = pd.read_csv("03_datos_procesados/jeva_limpio.csv")
    logger.info(f"  OK: {len(jeva)}")
    
    logger.info("Leyendo WhatsApp...")
    whatsapp = pd.read_csv("03_datos_procesados/whatsapp_limpio.csv")
    logger.info(f"  OK: {len(whatsapp)}")
    
    logger.info("\nCONSOLIDACION: Uniendo 4 bases...")
    consolidated = pd.concat([crm1, crm2, jeva, whatsapp], ignore_index=True)
    logger.info(f"  Total consolidado: {len(consolidated)}")
    
    logger.info(f"\nCOLUMNAS DISPONIBLES: {list(consolidated.columns)[:10]}")
    
    logger.info("\nDEDUPLICACION: Eliminando duplicados totales...")
    before_dedup = len(consolidated)
    consolidated = consolidated.drop_duplicates(keep='first')
    logger.info(f"  Eliminados: {before_dedup - len(consolidated)}")
    logger.info(f"  Después dedup: {len(consolidated)}")
    
    logger.info("\nVALIDACION: Eliminando vacios...")
    consolidated = consolidated.fillna('')
    logger.info(f"  Después validacion: {len(consolidated)}")
    
    logger.info("\nSELECCION: Tomando 1,500...")
    if len(consolidated) > 1500:
        final = consolidated.sample(n=1500, random_state=42)
    else:
        final = consolidated
    logger.info(f"  Final: {len(final)} registros")
    
    logger.info("\nGUARDANDO...")
    final.to_excel("03_datos_procesados/rocktec_base_consolidada_1500.xlsx", index=False)
    final.to_csv("03_datos_procesados/rocktec_base_consolidada_1500.csv", index=False)
    
    logger.info("  OK: rocktec_base_consolidada_1500.xlsx")
    logger.info("  OK: rocktec_base_consolidada_1500.csv")
    logger.info("\nCOMPLETA")
    
except Exception as e:
    logger.error(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
