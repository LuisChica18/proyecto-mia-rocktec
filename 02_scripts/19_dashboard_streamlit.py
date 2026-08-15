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

try:
    from tab5_panel_comercial import render_tab5
    TAB5_DISPONIBLE = True
except ImportError:
    TAB5_DISPONIBLE = False

# ── AUTENTICACIÓN POR CONTRASEÑA ─────────────────────────────────────────────
def check_password():
    """Pide contraseña antes de mostrar el dashboard."""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False

    if st.session_state.autenticado:
        return True

    # Pantalla de login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 0 1rem 0;'>
            <h1 style='color:#C0392B; font-size:2rem; margin-bottom:0.5rem;'>🏗️ Rocktec</h1>
            <p style='color:#777; font-size:1rem; margin-bottom:2rem;'>Panel de Inteligencia Comercial</p>
        </div>
        """, unsafe_allow_html=True)

        password = st.text_input("Contraseña de acceso", type="password", placeholder="Ingresa la contraseña")

        if st.button("Ingresar", type="primary", use_container_width=True):
            # Contraseña almacenada en Streamlit Secrets o valor por defecto
            clave_correcta = st.secrets.get("DASHBOARD_PASSWORD", "Rocktec2026#")
            if password == clave_correcta:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta. Intenta de nuevo.")
    return False

if not check_password():
    st.stop()

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rocktec | Inteligencia Comercial",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LIGHT MODE CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base blanco */
    .stApp { background-color: #FFFFFF; color: #1A1A1A; }
    [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E8E8E8; }
    [data-testid="stSidebar"] * { color: #1A1A1A !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #F5F5F5; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #777; border-radius: 6px; padding: 8px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #FFFFFF !important; color: #C0392B !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

    /* Cards métricas */
    [data-testid="metric-container"] {
        background: #FAFAFA;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label { color: #777 !important; font-size: 0.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #C0392B !important; font-size: 1.8rem; font-weight: 700; }

    /* Inputs */
    .stTextArea textarea {
        background: #FAFAFA !important;
        color: #1A1A1A !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        font-size: 14px;
    }
    .stTextArea textarea:focus { border-color: #C0392B !important; box-shadow: 0 0 0 3px rgba(192,57,43,0.08) !important; }

    /* Botón primario */
    .stButton > button[kind="primary"] {
        background: #C0392B !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover { background: #A93226 !important; }

    /* Botones secundarios */
    .stButton > button {
        background: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { border-color: #C0392B !important; color: #C0392B !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }

    /* Tablas */
    [data-testid="stDataFrame"] { border: 1px solid #E8E8E8; border-radius: 8px; overflow: hidden; }

    /* Alerts */
    .stAlert { border-radius: 8px; border: none; }

    /* Divider */
    hr { border-color: #E8E8E8; }

    /* Download button */
    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #1E8449 !important;
        border: 1px solid #1E8449 !important;
        border-radius: 8px !important;
    }

    /* Header custom */
    .rt-header {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-top: 3px solid #C0392B;
        border-radius: 12px;
        padding: 2rem 3rem;
        margin-bottom: 1.5rem;
    }
    .rt-title { font-size: 2rem; font-weight: 700; color: #1A1A1A; margin: 0; letter-spacing: -0.5px; }
    .rt-title span { color: #C0392B; }
    .rt-subtitle { color: #777; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .rt-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(192,57,43,0.06);
        border: 1px solid rgba(192,57,43,0.2);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #C0392B;
        margin-top: 0.8rem;
    }

    /* Result card */
    .result-card {
        background: #FAFAFA;
        border: 1px solid #E8E8E8;
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
    .intent-title { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; color: #1A1A1A; }
    .intent-code { font-family: monospace; background: #F0F0F0; border-radius: 4px; padding: 2px 8px; font-size: 1rem; color: #1A1A1A; }
    .confidence-text { color: #777; font-size: 0.9rem; margin: 0.5rem 0 0 0; }
    .method-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #F5F5F5;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #777;
        margin-bottom: 0.8rem;
    }

    /* Intent bar */
    .intent-bar-row { margin-bottom: 0.6rem; }
    .intent-bar-label { color: #1A1A1A; font-size: 0.85rem; margin-bottom: 3px; display: flex; justify-content: space-between; }
    .intent-bar-bg { background: #F0F0F0; border-radius: 4px; height: 6px; }
    .intent-bar-fill { height: 6px; border-radius: 4px; transition: width 0.5s ease; }

    /* Recomendacion box */
    .rec-box {
        background: #FAFAFA;
        border: 1px solid #E8E8E8;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #1A1A1A;
    }

    /* Sidebar stat */
    .sidebar-stat {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sidebar-stat-label { color: #777; font-size: 0.8rem; }
    .sidebar-stat-value { color: #C0392B; font-weight: 700; font-size: 0.9rem; }

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
    r'\b(cómo|como)\s+(se\s+)?(aplica|usa|prepara|mezcla|instala|coloca|sella|pule|limpia|mantiene|trabaja|pega|realiza|hace|utiliza|maneja|diluye|mezcla|aplican|instalan)\b',
    r'\b(visita\s+técnica|visita\s+tecnica|asesoría\s+técnica|asesoria\s+tecnica|asistencia\s+técnica)\b',
    r'\b(dosis|dosificación|proporci|rendimiento|cobertura|cuánto\s+rinde|cuanto\s+rinde|cuánto\s+necesito|cuanto\s+necesito)\b',
    r'\b(tiempo\s+de\s+secado|secado|curado|fraguado|temperatura|humedad|superficie|preparación\s+de\s+superficie)\b',
    r'\b(sellador|sellante|imprimante|promotor|catalizador|endurecedor|pigmento|oxidante)\s+(cómo|como|cuánto|cuanto|para|se\s+aplica)\b',
    r'\b(cuántas\s+capas|cuantas\s+capas|cuánto\s+tiempo|cuanto\s+tiempo|puede\s+ir\s+sobre|compatible\s+con)\b',
    r'\b(herramientas|moldes|estampar|estampado|textura|patrón|patron)\s+(necesito|requiero|se\s+usa|se\s+usan)\b',
    r'\b(medida|medidas|cantidad|cantidades)\s+(de|del|para)\s+(sellante|sellador|microcemento|oxidante|material)\b',
    r'\b(mezcla|mezclado|mezclando|mezclar|realizo\s+la\s+mezcla|hago\s+la\s+mezcla|hacer\s+la\s+mezcla|preparar\s+la\s+mezcla)\b',
    r'\b(como|cómo)\s+(realizo|realizamos|realizan|hago|hacemos|haces|realiza|preparo|preparamos|prepara|aplico|aplicamos|aplica|instalo|instalamos|instala|coloco|colocamos|coloca|mezclo|mezclamos|uso|usamos|usa|diluyo|diluimos|diluye)\b',
    r'\b(visita|asesor[ií]a|asistencia|apoyo|ayuda)\s+(t[eé]cnica|tecnico|del\s+t[eé]cnico)\b',
    r'\b(para|antes\s+de)\s+(aplicar|instalar|colocar|sellar|mezclar|preparar)\b',
    r'\b(el\s+piso|las\s+paredes|la\s+superficie|el\s+concreto|el\s+microcemento)\s+(se\s+puede|puede|se\s+aplica|se\s+coloca|necesita|requiere)\b',
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
    <div style="padding: 1rem 0 0.5rem 0; border-bottom: 1px solid #E0E0E0; margin-bottom: 1rem;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #1A1A1A;">🏗️ Rocktec</div>
        <div style="font-size: 0.75rem; color: #777777; margin-top: 2px;">Inteligencia Comercial · MIA 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Umbral de confianza**")
    umbral = st.slider("Umbral", 0.30, 0.95, 0.65, 0.05, label_visibility="collapsed")
    st.markdown(f"<div style='color:#777777; font-size:0.8rem; margin-top:-0.5rem'>Mensajes con confianza < {umbral:.0%} van a revisión humana</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #E0E0E0;'></div>", unsafe_allow_html=True)
    st.markdown("**📊 Métricas del modelo**")
    for label, val in [("F1-macro CV", "0.75 ✅"), ("F1 Holdout", "0.72"), ("Kappa", "0.8854"), ("Dataset", "1,312")]:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">{label}</span>
            <span class="sidebar-stat-value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #E0E0E0;'></div>", unsafe_allow_html=True)
    st.markdown("**🏷️ Intenciones**")
    for cod, desc in DESCRIPCIONES.items():
        color = COLORES[cod]
        icono = ICONOS[cod]
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <div style="width:8px; height:8px; border-radius:50%; background:{color}; flex-shrink:0;"></div>
            <span style="color:#777777; font-size:0.8rem;"><strong style="color:#1A1A1A;">{cod}</strong> — {icono} {desc}</span>
        </div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rt-header">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
        <div>
            <h1 class="rt-title">🏗️ Rocktec <span>Intelligence</span></h1>
            <p class="rt-subtitle">Clasificación automática de intenciones comerciales · PLN + MLOps</p>
            <div class="rt-badge">
                <div style="width:6px; height:6px; border-radius:50%; background:#1E8449; animation: pulse 2s infinite;"></div>
                Modelo activo · LR · F1-macro 0.75
            </div>
        </div>
        <div style="text-align:right; color:#777777; font-size:0.8rem; line-height:1.8;">
            <div>Universidad de Las Américas</div>
            <div>Maestría en IA Aplicada · 2026</div>
            <div style="color:#E0E0E0;">────────────────</div>
            <div>Mosquera · Cruel · Chica</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab4, tab5 = st.tabs(["💬  Clasificador", "📊  Dashboard", "📱  Chat WhatsApp", "📊  Panel Comercial"])

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
                color = COLORES.get(intencion, '#C0392B')
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
                        c = COLORES.get(k, '#C0392B')
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
            st.markdown("""<div class="result-card" style="border-top-color:#E0E0E0; text-align:center; color:#777777; padding:2rem;">
            Escribe un mensaje para clasificar</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-card" style="border-top-color:#E0E0E0; text-align:center; color:#777777; padding:3rem 2rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">💬</div>
            <div>Ingresa un texto y presiona<br><strong style="color:#C0392B;">Clasificar intención</strong></div>
            </div>""", unsafe_allow_html=True)

    # ── SECCIÓN LOTE ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📁 Clasificar múltiples mensajes")
    st.markdown("<span style='color:#777777; font-size:0.9rem;'>Un mensaje por línea. Útil para probar varios textos de una vez.</span>", unsafe_allow_html=True)

    texto_lote = st.text_area("Lote", placeholder="necesito cotización para 100m²\ncómo se aplica el microcemento sobre cerámica\ncuándo es el próximo curso de aplicadores\nya pagué cuándo me despachan", height=150, label_visibility="collapsed", key="lote_txt")

    if st.button("🔍  Clasificar todos", type="primary", key="btn_lote") and texto_lote and modelo:
        mensajes = [m.strip() for m in texto_lote.strip().split('\n') if m.strip()]
        if mensajes:
            resultados = []
            for msg in mensajes:
                intencion_l, confianza_l, _, es_regla_l = clasificar(msg, modelo, vectorizador, umbral)
                label_l = intencion_l if intencion_l else "⚠️ REVISAR"
                resultados.append({
                    'Mensaje': msg[:80] + ('...' if len(msg) > 80 else ''),
                    'Intención': label_l,
                    'Descripción': DESCRIPCIONES.get(label_l, 'Revisión humana'),
                    'Confianza': f"{confianza_l:.0%}" if confianza_l > 0 else "—",
                    'Método': "Reglas" if es_regla_l else ("ML" if intencion_l else "⚠️ Baja confianza"),
                })
            df_res = pd.DataFrame(resultados)
            n_auto = sum(1 for r in resultados if r['Intención'] != '⚠️ REVISAR')
            n_rev = len(resultados) - n_auto
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total mensajes", len(mensajes))
            with col2: st.metric("Clasificados auto", n_auto, f"{n_auto/len(mensajes):.0%}")
            with col3: st.metric("Requieren revisión", n_rev, f"{n_rev/len(mensajes):.0%}")
            st.dataframe(df_res, hide_index=True, use_container_width=True)
            csv_l = df_res.to_csv(index=False, encoding='utf-8')
            st.download_button("⬇️  Descargar CSV", data=csv_l,
                file_name=f"clasificaciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

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

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid #E0E0E0;'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown("#### 📊 Distribución de intenciones")
            dist = df['intencion_consenso'].value_counts()
            for cod, cnt in dist.items():
                color = COLORES.get(cod, '#C0392B')
                icono = ICONOS.get(cod, '•')
                desc = DESCRIPCIONES.get(cod, cod)
                pct = cnt / len(df) * 100
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#1A1A1A; font-size:0.85rem;">{icono} <strong>{cod}</strong> — {desc}</span>
                        <span style="color:{color}; font-weight:700; font-size:0.85rem;">{cnt} ({pct:.1f}%)</span>
                    </div>
                    <div style="background:#F0F0F0; border-radius:4px; height:8px;">
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

        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid #E0E0E0;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🗂️ Muestra del dataset")
        cols_m = [c for c in ['texto_conversacion', 'intencion_consenso', 'fuente'] if c in df.columns]
        st.dataframe(df[cols_m].sample(min(8, len(df)), random_state=42), hide_index=True, use_container_width=True)
    else:
        st.warning("No se pudo cargar el dataset.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — LOTE
# ────────────────────────────────────────────────────────────────────────────
# ── TAB 4: CHAT WHATSAPP ──────────────────────────────────────────────────────
with tab4:
    modelo_t4, vec_t4 = cargar_modelo()
    st.markdown("#### 📱 Subir chat exportado de WhatsApp")
    st.markdown(
        "<span style='color:#777777;font-size:0.9rem;'>"
        "Martha: exporta el chat desde WhatsApp → sube el .txt aquí → el sistema hace el resto."
        "</span>", unsafe_allow_html=True
    )
    st.markdown(
        "<div style='background:#F8F9FA;border:1px solid #E0E0E0;border-left:3px solid #C0392B;"
        "border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#777777;'>"
        "<strong style='color:#1A1A1A;'>¿Cómo exportar el chat?</strong><br>"
        "WhatsApp → Abrir chat → ⋮ Más → Exportar chat → Sin archivos → Guardar el .txt"
        "</div>", unsafe_allow_html=True
    )
    
    archivo_wa = st.file_uploader("Chat WhatsApp (.txt)", type=["txt"], key="uploader_wa")
    
    if archivo_wa and modelo_t4:
        import re as re_wa
        contenido_wa = archivo_wa.read().decode("utf-8", errors="ignore")
        ROCKTEC_IDS = ["rocktec", "concreto decorativo"]
        patron_wa = r"\d+/\d+/\d+, \d+:\d+ - ([^:]+): (.+?)(?=\n\d+/\d+/\d+, |\Z)"
        matches_wa = re_wa.findall(patron_wa, contenido_wa, re_wa.DOTALL)
        
        mensajes_wa = []
        nombre_wa = None
        EXCLUIR = ["cifrados de extremo", "es un contacto", "multimedia omitido", "archivo adjunto", "imagen omitida", "<multimedia"]
        
        for rem, txt in matches_wa:
            rem = rem.strip(); txt = txt.strip()
            if any(e in txt.lower() for e in EXCLUIR): continue
            if len(txt) < 5: continue
            if any(r in rem.lower() for r in ROCKTEC_IDS): continue
            if nombre_wa is None: nombre_wa = rem
            mensajes_wa.append(txt[:200])
        
        if mensajes_wa:
            st.success(f"✅ {len(mensajes_wa)} mensajes del cliente — {nombre_wa}")
            
            rows = []
            for msg in mensajes_wa:
                intent, conf, _, regla = clasificar(msg, modelo_t4, vec_t4, umbral)
                label = intent if intent else "REVISAR"
                rows.append({
                    "Mensaje": msg[:80] + ("..." if len(msg) > 80 else ""),
                    "Intención": label,
                    "Descripción": DESCRIPCIONES.get(label, "Revisión humana"),
                    "Confianza": f"{conf:.0%}" if conf > 0 else "—",
                    "Acción": RECOMENDACIONES[label][0] if label in RECOMENDACIONES else "Revisar manualmente",
                })
            
            df_wa = pd.DataFrame(rows)
            n_auto_wa = sum(1 for r in rows if r["Intención"] != "REVISAR")
            n_rev_wa = len(rows) - n_auto_wa
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Mensajes cliente", len(mensajes_wa))
            with c2: st.metric("Clasificados auto", n_auto_wa, f"{n_auto_wa/len(mensajes_wa):.0%}")
            with c3: st.metric("Para revisión", n_rev_wa)
            
            ints = df_wa[df_wa["Intención"] != "REVISAR"]["Intención"].value_counts()
            if not ints.empty:
                st.markdown("**Resumen del cliente:**")
                cols_i = st.columns(min(len(ints), 6))
                for i, (intent, cnt) in enumerate(ints.items()):
                    color = COLORES.get(intent, "#C0392B")
                    with cols_i[i % 6]:
                        st.markdown(
                            f"<div style='background:#F8F9FA;border:1px solid #E0E0E0;border-top:3px solid {color};"
                            f"border-radius:8px;padding:0.8rem;text-align:center;'>"
                            f"<div style='font-size:1.3rem;'>{ICONOS.get(intent,'•')}</div>"
                            f"<div style='color:{color};font-weight:700;font-size:1.2rem;'>{cnt}</div>"
                            f"<div style='color:#777777;font-size:0.75rem;'>{DESCRIPCIONES.get(intent,intent)}</div>"
                            f"</div>", unsafe_allow_html=True
                        )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_wa, hide_index=True, use_container_width=True)
            
            csv_wa = df_wa.to_csv(index=False, encoding="utf-8")
            fn = (nombre_wa or "cliente").replace(" ", "_")
            st.download_button(
                "⬇️ Descargar reporte CSV",
                data=csv_wa,
                file_name=f"reporte_{fn}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No se encontraron mensajes del cliente. Verifica que sea un chat exportado de WhatsApp.")
    else:
        st.markdown(
            "<div style='background:#F8F9FA;border:1px solid #E0E0E0;border-radius:10px;"
            "padding:3rem;text-align:center;color:#777777;'>"
            "<div style='font-size:3rem;margin-bottom:1rem;'>📱</div>"
            "<div style='font-size:1rem;color:#1A1A1A;margin-bottom:0.5rem;'>Sube el archivo .txt del chat de WhatsApp</div>"
            "<div>El sistema filtra automáticamente los mensajes del cliente y clasifica sus intenciones</div>"
            "</div>", unsafe_allow_html=True
        )


# ── TAB 5: PANEL COMERCIAL ───────────────────────────────────────────────────
with tab5:
    if TAB5_DISPONIBLE:
        try:
            render_tab5()
        except Exception as e:
            import traceback
            st.error(f"Error en Panel Comercial: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("No se encontró tab5_panel_comercial.py en 02_scripts/")
        st.info("Asegúrate de que tab5_panel_comercial.py esté en la misma carpeta que este script.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding:1.5rem; border-top: 1px solid #E0E0E0; text-align:center;">
    <div style="color:#AAAAAA; font-size:0.75rem; line-height:2;">
        Rocktec Intelligence Platform · MIA 2026 · UDLA<br>
        Mosquera Castro A.P. · Cruel Chang L.C. · Chica Moncayo L.M.<br>
        LR · F1=0.75 · Kappa=0.8854 · Dataset: 1,312 registros
    </div>
</div>
""", unsafe_allow_html=True)
