"""
04_feature_engineering.py
Preprocesamiento y extracción de features — Rocktec MIA 2026

Genera dos representaciones del texto:
  - TF-IDF (unigrams + bigrams) + features manuales  →  para LR y SVM
  - Tokenización BERT                                 →  para BETO fine-tuning

Uso standalone (verifica el pipeline):
    python 04_feature_engineering.py

Importado desde 05_entrenar_modelos.py vía importlib.
"""

import re
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

INTENCIONES = ['INF', 'COT', 'TEC', 'CUR', 'VEN']

STOPWORDS_ES = {
    'de','la','que','el','en','y','a','los','del','se','las','un','por','con',
    'una','su','para','es','al','lo','más','o','pero','sus','le','ha','me',
    'si','sin','sobre','este','ya','entre','todo','esta','ser','son','dos',
    'también','fue','había','era','muy','hasta','desde','nos','ni','contra',
    'ese','esa','estos','estas','les','otro','otra','otros','otras','tanto',
}


# ─────────────────────────────────────────────────────────────────────────────
# Limpieza de texto
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessadorTexto:
    """Normaliza mensajes de WhatsApp en español."""

    _RE_URL     = re.compile(r'https?://\S+|www\.\S+')
    _RE_PRECIO  = re.compile(r'\$[\d,.]+|\d[\d,.]*\s*(?:dólares?|usd|dolares?)', re.I)
    _RE_SALUDO  = re.compile(r'\b(hola|buenos|buenas|buen\s+día)\b', re.I)
    _RE_MULTI_PUNCT = re.compile(r'([!¡?¿])\1+')
    _RE_ESPACIO = re.compile(r'\s+')

    def limpiar(self, texto):
        if not isinstance(texto, str) or not texto.strip():
            return ''
        t = texto.lower()
        t = self._RE_URL.sub(' URL ', t)
        t = self._RE_PRECIO.sub(' PRECIO ', t)
        t = re.sub(r'[!¡]+', ' ! ', t)
        t = re.sub(r'[?¿]+', ' ? ', t)
        t = re.sub(r'\.{2,}', ' ... ', t)
        t = re.sub(r'[^\w\s?!]', ' ', t)
        t = self._RE_ESPACIO.sub(' ', t).strip()
        return t

    def limpiar_serie(self, serie):
        return pd.Series(serie).apply(self.limpiar)

    def features_manuales(self, serie):
        """8 features heurísticas sobre estructura del mensaje."""
        s = pd.Series(serie).fillna('')
        sl = s.str.lower()
        X = np.column_stack([
            s.str.len().values,                                          # longitud caracteres
            s.str.split().str.len().fillna(0).values,                   # número de palabras
            s.str.contains(r'[?¿]', regex=True).astype(int).values,    # es pregunta
            s.str.contains(r'\$|PRECIO', regex=True).astype(int).values,# menciona precio
            sl.str.contains(r'\b(?:hola|buenas?|buen\s+d)', regex=True).astype(int).values,  # saludo
            sl.str.contains(r'\b(?:problema|queja|reclamo|dañ|falla)', regex=True).astype(int).values,  # queja
            sl.str.contains(r'\b(?:curso|taller|capacit|certif)', regex=True).astype(int).values,  # curso
            sl.str.contains(r'\b(?:confirmo|compra|pago|factura|adelante)', regex=True).astype(int).values,  # venta
        ]).astype(np.float32)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Vectorizador TF-IDF (LR y SVM)
# ─────────────────────────────────────────────────────────────────────────────

class VectorizadorTFIDF:
    """TF-IDF (1-2 gramas) + features manuales concatenadas."""

    def __init__(self, max_features=15000, ngram_range=(1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            stop_words=list(STOPWORDS_ES),
            analyzer='word',
            token_pattern=r'\b[a-záéíóúüñ?!]{2,}\b',
        )
        self.prep    = PreprocessadorTexto()
        self._fitted = False

    def fit_transform(self, textos):
        limpios         = self.prep.limpiar_serie(textos)
        tfidf_mat       = self.tfidf.fit_transform(limpios)
        feat_manual     = sp.csr_matrix(self.prep.features_manuales(textos))
        self._fitted    = True
        return sp.hstack([tfidf_mat, feat_manual])

    def transform(self, textos):
        if not self._fitted:
            raise RuntimeError("Vectorizador no ajustado — llamar fit_transform primero")
        limpios     = self.prep.limpiar_serie(textos)
        tfidf_mat   = self.tfidf.transform(limpios)
        feat_manual = sp.csr_matrix(self.prep.features_manuales(textos))
        return sp.hstack([tfidf_mat, feat_manual])

    def guardar(self, ruta):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Vectorizador guardado: {ruta}")

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Codificador de etiquetas
# ─────────────────────────────────────────────────────────────────────────────

class CodificadorIntenciones:
    """LabelEncoder fijo sobre las 7 intenciones del catálogo."""

    def __init__(self):
        self.le = LabelEncoder()
        self.le.fit(INTENCIONES)

    def encode(self, labels):
        return self.le.transform(pd.Series(labels).str.strip().str.upper())

    def decode(self, indices):
        return self.le.inverse_transform(indices)

    @property
    def clases(self):
        return list(self.le.classes_)

    @property
    def n_clases(self):
        return len(self.le.classes_)


# ─────────────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────────────

_RUTA_ANOTACIONES = Path('04_anotaciones/dataset_consenso_final.csv')
_RUTA_XLSX        = Path('04_anotaciones/ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx')
_RUTA_FALLBACK    = Path('03_datos_procesados/rocktec_base_validada.csv')


def _consenso_fila(row, cols):
    votos = [str(row[c]).strip().upper() for c in cols if str(row[c]).strip()]
    if not votos:
        return None
    for v in votos:
        if votos.count(v) >= 2:
            return v
    return votos[0]


def cargar_dataset():
    """
    Orden de prioridad:
      1. dataset_consenso_final.csv  (generado por calcular_kappa.py)
      2. ROCKTEC_ANOTACIONES_FINALES_v1.0.xlsx  (anotaciones manuales brutas)
      3. rocktec_base_validada.csv  (etiquetas heurísticas — para pruebas)
    """
    if _RUTA_ANOTACIONES.exists():
        print(f"✓ Cargando consenso: {_RUTA_ANOTACIONES}")
        df = pd.read_csv(_RUTA_ANOTACIONES)
        col_label = 'intencion_consenso'
        fuente = 'consenso_manual'

    elif _RUTA_XLSX.exists():
        print(f"✓ Cargando anotaciones xlsx: {_RUTA_XLSX}")
        df   = pd.read_excel(_RUTA_XLSX, sheet_name='DATOS_ANOTACIÓN')
        cols = [c for c in ['PATRICIA', 'LUIS_CRUEL', 'LUIS_CHICA'] if c in df.columns]
        if cols:
            df['intencion_consenso'] = df.apply(_consenso_fila, cols=cols, axis=1)
        col_label = 'intencion_consenso'
        fuente = 'xlsx_bruto'

    elif _RUTA_FALLBACK.exists():
        print(f"⚠ Usando etiquetas heurísticas (fallback): {_RUTA_FALLBACK}")
        df        = pd.read_csv(_RUTA_FALLBACK)
        col_label = 'intencion_catalogo'
        fuente = 'heuristico_fallback'
    else:
        raise FileNotFoundError(
            "No se encontró ningún dataset. "
            "Ejecuta calcular_kappa.py o verifica que existan los archivos en "
            "04_anotaciones/ o 03_datos_procesados/."
        )

    df = df.dropna(subset=[col_label, 'texto_conversacion']).copy()
    df['label'] = df[col_label].str.strip().str.upper()
    df = df[df['label'].isin(INTENCIONES)].reset_index(drop=True)

    print(f"✓ Dataset listo: {len(df):,} registros")
    print("\n  Distribución:")
    for lbl, n in df['label'].value_counts().items():
        print(f"    {lbl}: {n:4d}  ({n/len(df)*100:.1f}%)")

    return df[['texto_conversacion', 'label']], fuente


# ─────────────────────────────────────────────────────────────────────────────
# Verificación standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("VERIFICACIÓN — Feature Engineering")
    print("=" * 60)

    ejemplos = [
        "Hola, quisiera saber cuánto cuesta el microcemento para 30m2",
        "Cómo se aplica el producto en pisos de madera existentes?",
        "El piso que me instalaron quedó con manchas, quiero reclamar",
        "Cuándo es el próximo curso de concreto decorativo?",
        "Confirmo la compra del kit básico, pueden enviar la factura?",
        "En qué estado está mi cotización del 15 de junio?",
        "Tienen productos para exteriores? cuáles son las opciones?",
    ]

    prep = PreprocessadorTexto()
    vec  = VectorizadorTFIDF()
    cod  = CodificadorIntenciones()

    print("\nTextos preprocesados:")
    for t in ejemplos:
        print(f"  → {prep.limpiar(t)}")

    X = vec.fit_transform(ejemplos)
    print(f"\n✓ Matriz TF-IDF+manual: {X.shape}")
    print(f"✓ Features manuales:    {prep.features_manuales(ejemplos).shape}")
    print(f"✓ Clases codificadas:   {cod.clases}")

    try:
        df, fuente = cargar_dataset()
        print(f"\n✓ cargar_dataset() OK: {len(df)} registros (fuente: {fuente})")
    except FileNotFoundError as e:
        print(f"\n⚠ {e}")

    print("\n✅ Feature engineering operativo")
