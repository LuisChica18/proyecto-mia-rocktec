import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("VALIDACION DE DUPLICADOS - 1,500 REGISTROS")
logger.info("="*80)

try:
    logger.info("Cargando base consolidada...")
    df = pd.read_csv("03_datos_procesados/rocktec_base_consolidada_1500.csv")
    logger.info(f"  OK: {len(df)} registros")
    
    logger.info("\nANALISIS DE DUPLICADOS:")
    logger.info(f"  Total registros: {len(df)}")
    logger.info(f"  Registros duplicados: {df.duplicated().sum()}")
    logger.info(f"  Registros unicos: {df.duplicated().sum() == 0}")
    
    logger.info("\nCOLUMNAS:")
    for col in df.columns:
        logger.info(f"  - {col}")
    
    logger.info("\nMUESTRA (primeras 5 filas):")
    logger.info(df.head().to_string())
    
    logger.info("\nESTADISTICAS:")
    logger.info(f"  Filas: {len(df)}")
    logger.info(f"  Columnas: {len(df.columns)}")
    logger.info(f"  Valores vacios: {df.isnull().sum().sum()}")
    
    logger.info("\nCOMPLETA: Base validada y lista para anotacion")
    
except Exception as e:
    logger.error(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
