"""
================================================================================
PROYECTO MIA 2026 - ROCKTEC
Script 19: Dashboard de Inteligencia Comercial - Streamlit (Dark Mode)
================================================================================
Uso: python -m streamlit run 02_scripts/19_dashboard_streamlit.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib, re
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import feature_engineering
except ImportError:
    st.error("No se encontró feature_engineering.py en 02_scripts/")
    st.stop()

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rocktec | Inteligencia Comercial",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DARK MODE CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; padding: 8px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #58a6ff !important; }

    /* Cards métricas */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem; font-weight: 700; }

    /* Inputs */
    .stTextArea textarea {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
    }
    .stTextArea textarea:focus { border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important; }

    /* Botón primario */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(31,111,235,0.4) !important; }

    /* Botones secundarios */
    .stButton > button {
        background: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { border-color: #58a6ff !important; color: #58a6ff !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }

    /* Tablas */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
    .dvn-scroller { background: #161b22 !important; }

    /* Alerts */
    .stAlert { border-radius: 8px; border: none; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Download button */
    .stDownloadButton > button {
        background: #21262d !important;
        color: #3fb950 !important;
        border: 1px solid #3fb950 !important;
        border-radius: 8px !important;
    }

    /* Header custom */
    .rt-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-top: 3px solid #1f6feb;
        border-radius: 12px;
        padding: 2rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .rt-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(31,111,235,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .rt-title { font-size: 2rem; font-weight: 700; color: #e6edf3; margin: 0; letter-spacing: -0.5px; }
    .rt-title span { color: #58a6ff; }
    .rt-subtitle { color: #8b949e; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .rt-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(31,111,235,0.1);
        border: 1px solid rgba(31,111,235,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #58a6ff;
        margin-top: 0.8rem;
    }

    /* Result card */
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .intent-title { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }
    .intent-code { font-family: monospace; background: #21262d; border-radius: 4px; padding: 2px 8px; font-size: 1rem; }
    .confidence-text { color: #8b949e; font-size: 0.9rem; margin: 0.5rem 0 0 0; }
    .method-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 0.8rem;
    }

    /* Intent bar */
    .intent-bar-row { margin-bottom: 0.6rem; }
    .intent-bar-label { color: #e6edf3; font-size: 0.85rem; margin-bottom: 3px; display: flex; justify-content: space-between; }
    .intent-bar-bg { background: #21262d; border-radius: 4px; height: 6px; }
    .intent-bar-fill { height: 6px; border-radius: 4px; transition: width 0.5s ease; }

    /* Recomendacion box */
    .rec-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #e6edf3;
    }

    /* Sidebar stat */
    .sidebar-stat {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sidebar-stat-label { color: #8b949e; font-size: 0.8rem; }
    .sidebar-stat-value { color: #58a6ff; font-weight: 700; font-size: 0.9rem; }

    /* Intent chip */
    .intent-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTES ───────────────────────────────────────────────────────────────
COLORES = {
    'COT': '#1f6feb',
    'TEC': '#388bfd',
    'INF': '#3fb950',
    'CUR': '#e3b341',
    'VEN': '#a371f7',
    'SEG': '#f0883e',
    'QUE': '#f85149',
}

ICONOS = {'COT': '💰', 'TEC': '🔧', 'INF': 'ℹ️', 'CUR': '📚', 'VEN': '✅', 'SEG': '🔄', 'QUE': '⚠️'}

DESCRIPCIONES = {
    'COT': 'Cotización',
    'TEC': 'Consulta Técnica',
    'INF': 'Información General',
    'CUR': 'Consulta de Cursos',
    'VEN': 'Venta / Confirmación',
    'SEG': 'Seguimiento',
    'QUE': 'Queja / Reclamo',
}

RECOMENDACIONES = {
    'COT': ('📌 Enviar cotización en menos de 2 horas. Alta probabilidad de conversión.', '#1f6feb'),
    'TEC': ('📌 Derivar a asesor técnico. El cliente evalúa viabilidad antes de comprar.', '#388bfd'),
    'INF': ('📌 Responder con información completa y catálogo. Nutrir al prospecto.', '#3fb950'),
    'CUR': ('📌 Compartir calendario de cursos y precios. Oportunidad de capacitación.', '#e3b341'),
    'VEN': ('🎉 ¡VENTA! Procesar pedido de inmediato. Enviar factura y confirmar despacho.', '#a371f7'),
    'SEG': ('📌 Cliente en seguimiento activo. Revisar cotización previa y hacer contacto.', '#f0883e'),
    'QUE': ('🚨 ATENCIÓN URGENTE. Escalar a supervisor. Responder en menos de 1 hora.', '#f85149'),
}

# ── Reglas léxicas expandidas con vocabulario real de Rocktec ────────────────
# QUE: queja, insatisfacción, problema con producto o servicio
PATRONES_QUE = [
    r'\b(queja|reclamo|reclam|molest|inconforme|inconformidad|insatisf|defect|defectu)\b',
    r'\b(daña|dañad|descascar|desprendiendo|burbujas|burbujeando|manchad|manchando|rajad|rajando)\b',
    r'\b(falla|fallo|no funciona|no sirve|no quedó|no quedó bien|quedó mal|quedó feo|quedó horrible)\b',
    r'\b(devoluci|garantí|garanti|no cumplieron|me engañaron|me cobraron mal|me cobró de más)\b',
    r'\b(pésimo|pésima|horrible|terrible|muy malo|muy mala|no me gustó|no me gusta|estoy molest)\b',
    r'\b(manchado|manchada|está manchado|está manchada|el piso manchado|quedó manchado)\b',
    r'\b(muy caro|carísimo|carísima|me parece caro|demasiado caro|cobran mucho|precio abusivo)\b',
]

# SEG: seguimiento de algo previo
PATRONES_SEG = [
    r'\b(seguimiento|en qué estado|cómo va|cómo está mi|ya fue despachado|cuándo llega|cuándo despachan)\b',
    r'\b(ya pagué|ya realicé el pago|ya deposité|ya transferí|ya hice el pago|ya cancelé)\b',
    r'\b(no contesta|no me han respondido|no me han llamado|sigo esperando|llevo esperando)\b',
    r'\b(mi pedido|mi cotización|mi solicitud|mi compra|mi orden|la cotización que pedí|el pedido que hice)\b',
    r'\b(me enviaron|mandaron|despacharon|ya llegó|cuándo me llega|cuándo me toca)\b',
    r'\b(no llega|no llego|no vien|no viene|no ha llegado|aun no llega|todavía no llega|no me ha llegado)\b',
    r'\b(pedi|pedí|hice un pedido|hice el pedido|ordené|ordene|compré|compre)\s+.{0,30}\b(no|noo|nun)\b',
    r'\b(esperando|espero|llevo esperando|horas esperando|dias esperando)\b',
    r'\b(gl|galon|galón|kilo|kg|saco|funda|producto)\s+.{0,20}\b(no viene|no llega|no ha llegado|no vienne|no llego)\b',
]

# VEN: confirmación de compra, pedido con cantidad y producto
PATRONES_VEN = [
    r'\b(factur|facturame|facturame|facturame|emite|emíteme)\b',
    r'\b(despach|despacha|despáchame|despáchame|envía|envíame|enviame|mándame|mandame)\b',
    r'\b(quiero comprar|voy a comprar|quiero llevar|me llevo|confirmado|confirmo|acepto|de acuerdo)\b',
    r'\b(procede|procedamos|adelante con|listo para pagar|voy a pagar|quiero pagar)\b',
    r'\b(orden de compra|quiero pedir|necesito pedir|me pueden vender|quiero adquirir)\b',
    r'(\d+\s*(kilos?|kg|quintales?|galones?|litros?|sacos?|fundas?|unidades?|metros?|m2|m²)\s+de\s+\w+)',
    r'\b(quiero|necesito|dame|deme)\s+\d+',
]

# COT: precio, presupuesto — incluyendo formas coloquiales ecuatorianas
PATRONES_COT = [
    r'\b(cotizaci|presupuest|proforma|precio|precios|costo|costos|valor|valores|tarifa)\b',
    r'\b(cuánto cuesta|cuanto cuesta|cuánto vale|cuanto vale|cuánto es|cuanto es|cuánto sale|cuanto sale)\b',
    r'\b(a cómo|a como|a cuánto|a cuanto|en cuánto|en cuanto|a qué precio|a que precio)\b',
    r'\b(cuánto me cobras|cuanto me cobras|cuánto me sale|cuanto me sale|cuánto me queda|cuanto me queda)\b',
    r'\b(hay descuento|tienen descuento|hacen descuento|precio especial|precio mayorista|precio por volumen)\b',
    r'\b(m2|m²|metro cuadrado|metros cuadrados)\s*(de\s+\w+)?\s*(cuánto|cuanto|precio|costo|vale|sale|está|esta)',
    r'(cuánto|cuanto|precio|costo|vale|sale|está|esta)\s*(el|la|los|las)?\s*(m2|m²|metro|kilo|kg|galon|litro)',
]

# TEC: consulta técnica, cómo aplicar, dosificación, proceso
PATRONES_TEC = [
    r'\b(cómo|como)\s+(se\s+)?(aplica|usa|prepara|mezcla|instala|coloca|sella|pule|limpia|mantiene|trabaja|pega)\b',
    r'\b(visita\s+técnica|visita\s+tecnica|asesoría\s+técnica|asesoria\s+tecnica|asistencia\s+técnica)\b',
    r'\b(dosis|dosificación|proporci|rendimiento|cobertura|cuánto\s+rinde|cuanto\s+rinde|cuánto\s+necesito|cuanto\s+necesito)\b',
    r'\b(tiempo\s+de\s+secado|secado|curado|fraguado|temperatura|humedad|superficie|preparación\s+de\s+superficie)\b',
    r'\b(sellador|sellante|imprimante|promotor|catalizador|endurecedor|pigmento|oxidante)\s+(cómo|como|cuánto|cuanto|para|se\s+aplica)\b',
    r'\b(cuántas\s+capas|cuantas\s+capas|cuánto\s+tiempo|cuanto\s+tiempo|puede\s+ir\s+sobre|compatible\s+con)\b',
    r'\b(herramientas|moldes|estampar|estampado|textura|patrón|patron)\s+(necesito|requiero|se\s+usa|se\s+usan)\b',
    r'\b(medida|medidas|cantidad|cantidades)\s+(de|del|para)\s+(sellante|sellador|microcemento|oxidante|material)\b',
]

def detectar_reglas(texto):
    t = texto.lower()
    # Orden de prioridad: QUE > SEG > VEN > COT > TEC (igual que catálogo v2.0)
    for p in PATRONES_QUE:
        if re.search(p, t): return 'QUE', 0.95
    for p in PATRONES_SEG:
        if re.search(p, t): return 'SEG', 0.95
    for p in PATRONES_VEN:
        if re.search(p, t): return 'VEN', 0.92
    for p in PATRONES_COT:
        if re.search(p, t): return 'COT', 0.90
    for p in PATRONES_TEC:
        if re.search(p, t): return 'TEC', 0.88
    return None, None

@st.cache_resource
def cargar_modelo():
    try:
        m = joblib.load('06_resultados/modelos/produccion/modelo_lr.pkl')
        v = joblib.load('06_resultados/modelos/produccion/vectorizador_tfidf.pkl')
        return m, v
    except Exception as e:
        return None, None

@st.cache_data
def cargar_dataset():
    try:
        return pd.read_csv('04_anotaciones/dataset_consenso_final.csv')
    except:
        return None

def clasificar(texto, modelo, vectorizador, umbral=0.65):
    if not texto or len(texto.strip()) < 5:
        return None, 0.0, {}, False
    intencion_r, conf_r = detectar_reglas(texto)
    if intencion_r:
        return intencion_r, conf_r, {intencion_r: conf_r}, True
    try:
        X = vectorizador.transform([texto])
        proba = modelo.predict_proba(X)[0]
        clases = modelo.classes_
        probas_dict = dict(zip(clases, proba))
        max_prob = float(np.max(proba))
        pred = clases[np.argmax(proba)]
        return (pred, max_prob, probas_dict, False) if max_prob >= umbral else (None, max_prob, probas_dict, False)
    except:
        return None, 0.0, {}, False

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0; border-bottom: 1px solid #30363d; margin-bottom: 1rem;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #e6edf3;">🏗️ Rocktec</div>
        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 2px;">Inteligencia Comercial · MIA 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Umbral de confianza**")
    umbral = st.slider("Umbral", 0.30, 0.95, 0.65, 0.05, label_visibility="collapsed")
    st.markdown(f"<div style='color:#8b949e; font-size:0.8rem; margin-top:-0.5rem'>Mensajes con confianza < {umbral:.0%} van a revisión humana</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #30363d;'></div>", unsafe_allow_html=True)
    st.markdown("**📊 Métricas del modelo**")
    for label, val in [("F1-macro CV", "0.75 ✅"), ("F1 Holdout", "0.72"), ("Kappa", "0.8854"), ("Dataset", "1,312")]:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">{label}</span>
            <span class="sidebar-stat-value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #30363d;'></div>", unsafe_allow_html=True)
    st.markdown("**🏷️ Intenciones**")
    for cod, desc in DESCRIPCIONES.items():
        color = COLORES[cod]
        icono = ICONOS[cod]
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <div style="width:8px; height:8px; border-radius:50%; background:{color}; flex-shrink:0;"></div>
            <span style="color:#8b949e; font-size:0.8rem;"><strong style="color:#e6edf3;">{cod}</strong> — {icono} {desc}</span>
        </div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rt-header">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
        <div>
            <h1 class="rt-title">🏗️ Rocktec <span>Intelligence</span></h1>
            <p class="rt-subtitle">Clasificación automática de intenciones comerciales · PLN + MLOps</p>
            <div class="rt-badge">
                <div style="width:6px; height:6px; border-radius:50%; background:#3fb950; animation: pulse 2s infinite;"></div>
                Modelo activo · LR · F1-macro 0.75
            </div>
        </div>
        <div style="text-align:right; color:#8b949e; font-size:0.8rem; line-height:1.8;">
            <div>Universidad de Las Américas</div>
            <div>Maestría en IA Aplicada · 2026</div>
            <div style="color:#30363d;">────────────────</div>
            <div>Mosquera · Cruel · Chica</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬  Clasificador", "📊  Dashboard", "📁  Lote de mensajes"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLASIFICADOR
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    modelo, vectorizador = cargar_modelo()
    if not modelo:
        st.error("❌ Modelo no encontrado. Verifica la ruta: 06_resultados/modelos/produccion/")
        st.stop()

    col_izq, col_der = st.columns([1.4, 1], gap="large")

    with col_izq:
        st.markdown("#### Texto de la conversación")
        if 'ejemplo_sel' not in st.session_state:
            st.session_state['ejemplo_sel'] = ""
        texto_input = st.text_area("Texto", value=st.session_state.get('ejemplo_sel', ''), placeholder="Escribe o pega aquí el mensaje del cliente de WhatsApp...", height=180, label_visibility="collapsed")

        st.markdown("**Ejemplos rápidos:**")
        ejemplos = {
            "💰 Cotización": "necesito presupuesto para 200m2 de microcemento gris",
            "🔧 Técnica": "puedo aplicar el concreto estampado sobre cerámica existente?",
            "📚 Curso": "cuándo es el próximo curso de aplicadores y cuánto cuesta",
            "✅ Venta": "listo acepto la propuesta envíame la factura para pagar",
            "⚠️ Queja": "el producto llegó dañado no estoy conforme con la calidad",
            "🔄 Seguimiento": "quería saber en qué estado está mi cotización ya no me han respondido",
        }
        col_e1, col_e2, col_e3 = st.columns(3)
        for i, (nombre, txt_ej) in enumerate(ejemplos.items()):
            with [col_e1, col_e2, col_e3][i % 3]:
                if st.button(nombre, key=f"ej_{i}", use_container_width=True):
                    st.session_state['txt_main'] = txt_ej
                    st.rerun()

        btn = st.button("🔍  Clasificar intención", type="primary", use_container_width=True)

    with col_der:
        st.markdown("#### Resultado")

        if btn and texto_input:
            intencion, confianza, probas, es_regla = clasificar(texto_input, modelo, vectorizador, umbral)

            if intencion is None and confianza > 0:
                st.markdown(f"""
                <div class="result-card" style="border-top-color: #f0883e;">
                    <div class="method-badge">⚠️ Baja confianza</div>
                    <div class="intent-title" style="color:#f0883e;">Revisión humana</div>
                    <p class="confidence-text">Confianza: {confianza:.0%} — por debajo del umbral {umbral:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""<div class="rec-box" style="border-left-color:#f0883e;">
                Un asesor debe revisar esta conversación manualmente.</div>""", unsafe_allow_html=True)

            elif intencion:
                color = COLORES.get(intencion, '#58a6ff')
                desc = DESCRIPCIONES.get(intencion, intencion)
                icono = ICONOS.get(intencion, '•')
                metodo = "📏 Reglas léxicas" if es_regla else "🤖 Modelo ML"
                rec_txt, rec_color = RECOMENDACIONES.get(intencion, ('', color))

                st.markdown(f"""
                <div class="result-card" style="border-top-color:{color};">
                    <div class="method-badge">{metodo}</div>
                    <div class="intent-title" style="color:{color};">{icono} {desc}</div>
                    <div><span class="intent-code">{intencion}</span></div>
                    <p class="confidence-text">Confianza: <strong style="color:{color};">{confianza:.0%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""<div class="rec-box" style="border-left-color:{rec_color};">{rec_txt}</div>""", unsafe_allow_html=True)

                if not es_regla and probas:
                    st.markdown("<br>**Probabilidades:**", unsafe_allow_html=True)
                    for k, v in sorted(probas.items(), key=lambda x: -x[1]):
                        c = COLORES.get(k, '#58a6ff')
                        ic = ICONOS.get(k, '•')
                        desc_k = DESCRIPCIONES.get(k, k)
                        pct = int(v * 100)
                        st.markdown(f"""
                        <div class="intent-bar-row">
                            <div class="intent-bar-label">
                                <span>{ic} {desc_k}</span>
                                <span style="color:{c}; font-weight:600;">{pct}%</span>
                            </div>
                            <div class="intent-bar-bg">
                                <div class="intent-bar-fill" style="width:{pct}%; background:{c};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
        elif btn and not texto_input:
            st.markdown("""<div class="result-card" style="border-top-color:#30363d; text-align:center; color:#8b949e; padding:2rem;">
            Escribe un mensaje para clasificar</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-card" style="border-top-color:#30363d; text-align:center; color:#8b949e; padding:3rem 2rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">💬</div>
            <div>Ingresa un texto y presiona<br><strong style="color:#58a6ff;">Clasificar intención</strong></div>
            </div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    df = cargar_dataset()

    if df is not None:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total registros", f"{len(df):,}")
        with col2: st.metric("Intenciones", df['intencion_consenso'].nunique())
        with col3: st.metric("Cohen's Kappa", "0.8854")
        with col4: st.metric("F1-macro CV", "0.75")

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid #30363d;'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown("#### 📊 Distribución de intenciones")
            dist = df['intencion_consenso'].value_counts()
            for cod, cnt in dist.items():
                color = COLORES.get(cod, '#58a6ff')
                icono = ICONOS.get(cod, '•')
                desc = DESCRIPCIONES.get(cod, cod)
                pct = cnt / len(df) * 100
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#e6edf3; font-size:0.85rem;">{icono} <strong>{cod}</strong> — {desc}</span>
                        <span style="color:{color}; font-weight:700; font-size:0.85rem;">{cnt} ({pct:.1f}%)</span>
                    </div>
                    <div style="background:#21262d; border-radius:4px; height:8px;">
                        <div style="width:{pct}%; height:8px; border-radius:4px; background:{color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("#### 🤖 Comparativa de modelos")
            metricas = pd.DataFrame({
                'Modelo': ['Logistic Regression', 'SVM (LinearSVC)', 'BETO sin GPU', 'BETO con GPU'],
                'F1-macro CV': ['0.75 ✅', '0.72', '0.637', '0.855'],
                'F1 Holdout': ['0.72', '0.69', '—', '0.797'],
                'Producción': ['✅', '—', '—', '⚠️ requiere GPU'],
            })
            st.dataframe(metricas, hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Historial Cohen's Kappa")
            kappa_hist = pd.DataFrame({
                'Ronda': ['Inicial', 'Post-alineación', 'Fase 1B', 'Fase 1C'],
                'Kappa': [0.6767, 0.8794, 0.8851, 0.8854],
                'Estado': ['❌', '✅', '✅', '✅'],
            })
            st.dataframe(kappa_hist, hide_index=True, use_container_width=True)

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid #30363d;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🗂️ Muestra del dataset")
        cols_m = [c for c in ['texto_conversacion', 'intencion_consenso', 'fuente'] if c in df.columns]
        st.dataframe(df[cols_m].sample(min(8, len(df)), random_state=42), hide_index=True, use_container_width=True)
    else:
        st.warning("No se pudo cargar el dataset.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — LOTE
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    modelo, vectorizador = cargar_modelo()

    st.markdown("#### Clasificar múltiples mensajes")
    st.markdown("<span style='color:#8b949e; font-size:0.9rem;'>Un mensaje por línea. Útil para procesar exportaciones de WhatsApp.</span>", unsafe_allow_html=True)

    texto_lote = st.text_area("Texto", placeholder="necesito cotización para 100m²\ncómo se aplica el microcemento sobre cerámica\ncuándo es el próximo curso de aplicadores\nya pagué cuándo me despachan", height=180, label_visibility="collapsed")

    if st.button("🔍  Clasificar todos", type="primary") and texto_lote and modelo:
        mensajes = [m.strip() for m in texto_lote.strip().split('\n') if m.strip()]
        if mensajes:
            resultados = []
            for msg in mensajes:
                intencion, confianza, _, es_regla = clasificar(msg, modelo, vectorizador, umbral)
                label = intencion if intencion else "⚠️ REVISAR"
                resultados.append({
                    'Mensaje': msg[:80] + ('...' if len(msg) > 80 else ''),
                    'Intención': label,
                    'Descripción': DESCRIPCIONES.get(label, 'Revisión humana'),
                    'Confianza': f"{confianza:.0%}" if confianza > 0 else "—",
                    'Método': "Reglas" if es_regla else ("ML" if intencion else "⚠️ Baja confianza"),
                })

            df_res = pd.DataFrame(resultados)
            n_auto = sum(1 for r in resultados if r['Intención'] != '⚠️ REVISAR')
            n_rev = len(resultados) - n_auto

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total mensajes", len(mensajes))
            with col2: st.metric("Clasificados auto", n_auto, f"{n_auto/len(mensajes):.0%}")
            with col3: st.metric("Requieren revisión", n_rev, f"{n_rev/len(mensajes):.0%}")

            st.dataframe(df_res, hide_index=True, use_container_width=True)

            csv = df_res.to_csv(index=False, encoding='utf-8')
            st.download_button("⬇️  Descargar CSV", data=csv,
                file_name=f"clasificaciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding:1.5rem; border-top: 1px solid #30363d; text-align:center;">
    <div style="color:#30363d; font-size:0.75rem; line-height:2;">
        Rocktec Intelligence Platform · MIA 2026 · UDLA<br>
        Mosquera Castro A.P. · Cruel Chang L.C. · Chica Moncayo L.M.<br>
        LR · F1=0.75 · Kappa=0.8854 · Dataset: 1,312 registros
    </div>
</div>
""", unsafe_allow_html=True)
