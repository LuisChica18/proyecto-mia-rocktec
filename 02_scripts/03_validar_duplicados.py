"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script: Validación y Detección de Duplicados
================================================================================

Descripción:
    Detecta y elimina registros duplicados usando:
    - Duplicados exactos (contenido idéntico)
    - Duplicados similares (similitud coseno TF-IDF > 0.95)
    
    Genera reportes detallados para revisión.

Autor: Equipo Rocktec MIA 2026
Versión: 1.0

Entrada:
    - ../03_datos_procesados/rocktec_base_consolidada.csv

Salida:
    - ../03_datos_procesados/rocktec_base_validada.csv
    - reporte_validacion_duplicados.txt
    - duplicados_encontrados.csv (para revisión manual)

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectorDuplicados:
    """Detectar y validar duplicados en base de datos"""
    
    def __init__(self, ruta_entrada, ruta_salida):
        self.ruta_entrada = Path(ruta_entrada)
        self.ruta_salida = Path(ruta_salida)
        self.duplicados_exactos = []
        self.duplicados_similares = []
        self.ruta_salida.mkdir(parents=True, exist_ok=True)
        logger.info("Detector de duplicados iniciado")
    
    def detectar_duplicados_exactos(self, df):
        """Detectar registros completamente idénticos"""
        logger.info("Detectando duplicados exactos...")
        columnas_comparar = ['texto_conversacion', 'nombre_cliente', 'canal']
        duplicados = df[df.duplicated(subset=columnas_comparar, keep=False)]
        
        logger.info(f"  Duplicados exactos encontrados: {len(duplicados)}")
        if len(duplicados) > 0:
            self.duplicados_exactos = duplicados.index.tolist()
        return duplicados
    
    def detectar_duplicados_similares(self, df, threshold=0.95):
        """Detectar registros similares usando TF-IDF + cosine similarity"""
        logger.info(f"Detectando duplicados similares (threshold={threshold})...")
        
        if len(df) < 2:
            logger.info("  No hay suficientes registros para comparar")
            return []
        
        textos = df['texto_conversacion'].fillna('')
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='spanish',
            max_features=500,
            analyzer='char',
            ngram_range=(2, 3)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(textos)
            similitud = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.warning(f"  Error calculando similitud: {e}")
            return []
        
        pares_similares = []
        for i in range(len(similitud)):
            for j in range(i + 1, len(similitud)):
                if similitud[i, j] >= threshold:
                    pares_similares.append({
                        'idx_1': i,
                        'idx_2': j,
                        'similitud': similitud[i, j],
                        'texto_1': textos.iloc[i][:100],
                        'texto_2': textos.iloc[j][:100]
                    })
        
        logger.info(f"  Pares similares encontrados: {len(pares_similares)}")
        self.duplicados_similares = pares_similares
        return pares_similares
    
    def eliminar_duplicados(self, df):
        """Eliminar duplicados exactos"""
        logger.info("Eliminando duplicados exactos...")
        columnas_comparar = ['texto_conversacion', 'nombre_cliente', 'canal']
        registros_antes = len(df)
        df_limpio = df.drop_duplicates(subset=columnas_comparar, keep='first')
        registros_eliminados = registros_antes - len(df_limpio)
        
        logger.info(f"  Registros eliminados: {registros_eliminados}")
        logger.info(f"  Registros finales: {len(df_limpio)}")
        return df_limpio
    
    def guardar_reportes(self, df, df_limpio):
        """Guardar archivo con duplicados similares encontrados"""
        logger.info("Guardando reportes...")
        
        if len(self.duplicados_similares) > 0:
            duplicados_df = pd.DataFrame(self.duplicados_similares)
            ruta_duplicados = self.ruta_salida / 'duplicados_similares_encontrados.csv'
            duplicados_df.to_csv(ruta_duplicados, index=False, encoding='utf-8')
            logger.info(f"  ✓ Duplicados similares: {ruta_duplicados}")
        
        ruta_validada = self.ruta_salida / 'rocktec_base_validada.csv'
        df_limpio.to_csv(ruta_validada, index=False, encoding='utf-8')
        logger.info(f"  ✓ Base validada: {ruta_validada}")
    
    def generar_reporte(self, registros_antes, registros_despues):
        """Generar reporte de validación"""
        logger.info("Generando reporte de validación...")
        
        reporte = f"""
================================================================================
REPORTE DE VALIDACION Y DETECCION DE DUPLICADOS
ROCKTEC MIA 2026
================================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTADISTICAS GENERALES:
{'-'*80}

Registros cargados: {registros_antes}
Registros después de eliminar duplicados exactos: {registros_despues}
Registros eliminados: {registros_antes - registros_despues}
Tasa de duplicados: {((registros_antes - registros_despues) / registros_antes * 100):.2f}%

DUPLICADOS DETECTADOS:
{'-'*80}

Duplicados exactos encontrados: {len(self.duplicados_exactos)}
Pares similares encontrados (threshold 0.95): {len(self.duplicados_similares)}

ARCHIVOS GENERADOS:
  ✓ rocktec_base_validada.csv ({registros_despues} registros)
  ✓ duplicados_similares_encontrados.csv (para revisión manual)

================================================================================
"""
        
        ruta_reporte = self.ruta_salida / 'reporte_validacion_duplicados.txt'
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        logger.info(f"Reporte guardado: {ruta_reporte}")
        print(reporte)


class ValidadorDatos:
    """Validar calidad general de datos"""
    
    @staticmethod
    def validar_columnas(df):
        columnas_requeridas = [
            'id_cliente', 'nombre_cliente', 'canal', 
            'intencion_catalogo', 'texto_conversacion'
        ]
        faltantes = [c for c in columnas_requeridas if c not in df.columns]
        if faltantes:
            logger.warning(f"Columnas faltantes: {faltantes}")
            return False
        return True
    
    @staticmethod
    def validar_intenciones(df):
        intenciones_validas = {'INF', 'COT', 'TEC', 'CUR', 'VEN', 'SEG', 'QUE'}
        invalidas = df[~df['intencion_catalogo'].isin(intenciones_validas)]
        if len(invalidas) > 0:
            logger.warning(f"Intenciones inválidas encontradas: {len(invalidas)}")
            return False
        return True


def main():
    logger.info("="*80)
    logger.info("VALIDACION Y DETECCION DE DUPLICADOS - ROCKTEC MIA 2026")
    logger.info("="*80)
    
    RUTA_ENTRADA = Path('../03_datos_procesados')
    RUTA_SALIDA = RUTA_ENTRADA
    
    logger.info("\nCargando base consolidada...")
    df = pd.read_csv(RUTA_ENTRADA / 'rocktec_base_consolidada.csv')
    registros_antes = len(df)
    logger.info(f"  Registros cargados: {registros_antes}")
    
    logger.info("\nValidando datos...")
    validador = ValidadorDatos()
    if not validador.validar_columnas(df):
        logger.error("Columnas inválidas. Abortando.")
        return
    
    detector = DetectorDuplicados(RUTA_ENTRADA, RUTA_SALIDA)
    duplicados_exactos = detector.detectar_duplicados_exactos(df)
    duplicados_similares = detector.detectar_duplicados_similares(df, threshold=0.95)
    df_limpio = detector.eliminar_duplicados(df)
    registros_despues = len(df_limpio)
    
    detector.guardar_reportes(df, df_limpio)
    detector.generar_reporte(registros_antes, registros_despues)
    
    logger.info("\n" + "="*80)
    logger.info("VALIDACION COMPLETADA")
    logger.info("="*80)


if __name__ == '__main__':
    main()
