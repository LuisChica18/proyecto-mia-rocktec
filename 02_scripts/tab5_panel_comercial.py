"""
Tab 5: Panel de Inteligencia Comercial - Rocktec MIA 2026
"""
import re
import json
import zipfile
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st

ASESORES = [
    {"nombre": "Admin Rocktec",    "numero": "+593 99 060 9023", "clave": "admin"},
    {"nombre": "Ventas Rocktec",   "numero": "+593 99 561 8025", "clave": "ventas"},
    {"nombre": "Gerencia Rocktec", "numero": "+593 99 380 2851", "clave": "gerencia"},
]

SHEET_ID = "1sgPf6RUl_T9eDv2B6R1Vpi5nNEsSYZY-i5r7QKT6r1g"
ESTADO_PATH = Path("estado_panel.json")  # fallback local


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS — conexión y operaciones
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gsheet_client():
    """Conecta a Google Sheets usando las credenciales de Streamlit Secrets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        return gspread.authorize(creds)
    except Exception:
        return None


# ── SENTIMIENTO INFERIDO DESDE INTENCIÓN ─────────────────────────────────────
SENTIMIENTO_MAP = {
    "QUE": ("Negativo", "😠", "#C0392B", "#FDEDEC"),
    "SEG": ("Negativo", "😟", "#C0392B", "#FDEDEC"),
    "VEN": ("Positivo", "😊", "#1E8449", "#EAFAF1"),
    "COT": ("Neutro",   "😐", "#B7770D", "#FEF9E7"),
    "TEC": ("Neutro",   "😐", "#B7770D", "#FEF9E7"),
    "CUR": ("Neutro",   "😐", "#B7770D", "#FEF9E7"),
    "INF": ("Neutro",   "😐", "#777777", "#F8F9FA"),
}

def inferir_sentimiento(intencion):
    return SENTIMIENTO_MAP.get(intencion, ("Neutro", "😐", "#777777", "#F8F9FA"))


def get_worksheets():
    """Retorna (ws_mensajes, ws_estado) o (None, None) si no hay conexión."""
    gc = get_gsheet_client()
    if gc is None:
        return None, None
    try:
        sh = gc.open_by_key(SHEET_ID)
        hojas = [w.title for w in sh.worksheets()]
        if "mensajes" not in hojas:
            sh.add_worksheet(title="mensajes", rows=5000, cols=15)
            sh.worksheet("mensajes").append_row([
                "fecha", "asesor", "remitente", "texto",
                "intencion", "razon_perdida", "tipo_negocio",
                "es_lead", "es_perdida", "es_venta"
            ])
        if "estado" not in hojas:
            sh.add_worksheet(title="estado", rows=10, cols=3)
            sh.worksheet("estado").append_row(["asesor_clave", "ultima_fecha"])
        return sh.worksheet("mensajes"), sh.worksheet("estado")
    except Exception:
        return None, None


def cargar_estado():
    """Carga el estado desde Google Sheets o archivo local."""
    ws_msgs, ws_estado = get_worksheets()
    if ws_estado:
        try:
            rows = ws_estado.get_all_records()
            return {r["asesor_clave"]: r["ultima_fecha"] for r in rows if r.get("asesor_clave")}
        except Exception:
            pass
    # Fallback local
    if ESTADO_PATH.exists():
        with open(ESTADO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_estado(estado):
    """Guarda el estado en Google Sheets y también localmente como backup."""
    ws_msgs, ws_estado = get_worksheets()
    if ws_estado:
        try:
            ws_estado.clear()
            ws_estado.append_row(["asesor_clave", "ultima_fecha"])
            for clave, fecha in estado.items():
                ws_estado.append_row([clave, str(fecha)])
        except Exception:
            pass
    # Backup local
    with open(ESTADO_PATH, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2, default=str)


def get_claves_existentes_sheets():
    """Retorna set de claves unicas ya guardadas en Sheets para control de duplicados."""
    ws_msgs, _ = get_worksheets()
    if ws_msgs is None:
        return set()
    try:
        records = ws_msgs.get_all_records()
        claves = set()
        for r in records:
            clave = f"{r.get('remitente','')}|{r.get('fecha','')}|{str(r.get('texto',''))[:50]}"
            claves.add(clave)
        return claves
    except Exception:
        return set()


def guardar_mensajes_en_sheets(msgs_por_asesor, claves_existentes=None):
    """Guarda solo mensajes nuevos (no duplicados) en Sheets."""
    ws_msgs, _ = get_worksheets()
    if ws_msgs is None:
        return 0
    if claves_existentes is None:
        claves_existentes = get_claves_existentes_sheets()
    try:
        filas = []
        for clave, msgs in msgs_por_asesor.items():
            asesor_nombre = next((a["nombre"] for a in ASESORES if a["clave"] == clave), clave)
            for m in msgs:
                clave_unica = f"{m.get('remitente','')}|{str(m.get('fecha',''))}|{str(m.get('texto',''))[:50]}"
                if clave_unica in claves_existentes:
                    continue
                filas.append([
                    str(m.get("fecha", "")),
                    asesor_nombre,
                    m.get("remitente", ""),
                    m.get("texto", "")[:300],
                    m.get("intencion", ""),
                    m.get("razon_perdida", "") or "",
                    m.get("tipo_negocio", ""),
                    str(m.get("es_lead", False)),
                    str(m.get("es_perdida", False)),
                    str(m.get("es_venta", False)),
                ])
                claves_existentes.add(clave_unica)
        if filas:
            ws_msgs.append_rows(filas)
        return len(filas)
    except Exception:
        return 0


def cargar_mensajes_desde_sheets():
    """Carga todos los mensajes guardados desde Google Sheets."""
    ws_msgs, _ = get_worksheets()
    if ws_msgs is None:
        return {}
    try:
        records = ws_msgs.get_all_records()
        msgs_por_asesor = defaultdict(list)
        for r in records:
            clave = next((a["clave"] for a in ASESORES if a["nombre"] == r.get("asesor", "")), None)
            if not clave:
                continue
            # Parsear fecha con múltiples formatos
            fecha_raw = str(r.get("fecha", ""))
            fecha_obj = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
                try:
                    fecha_obj = datetime.strptime(fecha_raw.strip(), fmt).date()
                    break
                except ValueError:
                    pass
            if fecha_obj is None:
                continue
            msgs_por_asesor[clave].append({
                "fecha": fecha_obj,
                "remitente": r.get("remitente", ""),
                "texto": r.get("texto", ""),
                "intencion": r.get("intencion", ""),
                "razon_perdida": r.get("razon_perdida", "") or None,
                "tipo_negocio": r.get("tipo_negocio", ""),
                "es_lead": str(r.get("es_lead", "False")) == "True",
                "es_perdida": str(r.get("es_perdida", "False")) == "True",
                "es_venta": str(r.get("es_venta", "False")) == "True",
            })
        return dict(msgs_por_asesor)
    except Exception:
        return {}


PATRONES_PERDIDA = {
    "Competencia": [r"ya lo hice con otro|con otra empresa|con otro proveedor|encontré otro"],
    "Precio":      [r"muy caro|más barato|no me alcanza|precio alto|fuera de mi presupuesto"],
    "Proyecto cancelado": [r"cancelado|suspendido|dejé el proyecto|por ahora no|ya no necesito"],
}

PATRONES_COT = [r"cotiz|presupuest|precio|a como sale|cuánto cuesta|proforma|cuánto vale"]
PATRONES_VEN = [r"confirmo|adelante|acepto|mándame la factura|forma de pago|ya realicé el pago"]
PATRONES_QUE = [r"dañado|no funciona|no estoy conforme|reclamo|nadie me responde|se demoran|mala atención"]
PATRONES_SEG = [r"en qué estado|cuándo despachan|ya llegó|sigo esperando|sin respuesta|"
               r"insisto|que insista|siguen sin|llevan días|hace días|segunda vez|"
               r"tercera vez|por qué no responden|cuándo me responden|"
               r"no me han respondido|seguimos esperando|aún no|todavía no|"
               r"cuándo me confirman|cuándo me dan|cuándo me envían|cuándo está listo"]
PALABRAS_PROYECTO = ["m2", "metros", "obra", "proyecto", "instalación", "área", "piso", "pared"]


def cargar_estado():
    if ESTADO_PATH.exists():
        with open(ESTADO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_estado(estado):
    with open(ESTADO_PATH, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2, default=str)


def detectar_numero_desde_nombre(nombre_archivo):
    m = re.search(r"(\+\d[\d\s\-]{8,15})", nombre_archivo)
    if m:
        return re.sub(r"[\s\-]", "", m.group(1))
    return None


def asesor_desde_numero(numero_limpio):
    for a in ASESORES:
        if re.sub(r"[\s\-]", "", a["numero"]) == numero_limpio:
            return a
    return None


def coincide_patron(texto, patrones):
    t = texto.lower()
    for pat in patrones:
        if re.search(pat, t):
            return True
    return False


def detectar_razon_perdida(texto):
    for razon, patrones in PATRONES_PERDIDA.items():
        if coincide_patron(texto, patrones):
            return razon
    return None


PATRONES_CUR = [r"curso|taller|capacitación|certificación|certificado|inscripción|inscribir|"
               r"cuándo es el.*curso|precio del curso|costo del curso|quiero aprender|"
               r"cupo.*curso|me interesa.*curso"]

def detectar_intencion(texto):
    if coincide_patron(texto, PATRONES_QUE): return "QUE"
    if coincide_patron(texto, PATRONES_VEN): return "VEN"
    if coincide_patron(texto, PATRONES_SEG): return "SEG"
    if coincide_patron(texto, PATRONES_CUR): return "CUR"
    if coincide_patron(texto, PATRONES_COT): return "COT"
    return "INF"


def detectar_tipo_negocio(texto):
    t = texto.lower()
    menciones = sum(1 for p in PALABRAS_PROYECTO if p in t)
    return "Proyecto" if menciones >= 2 else "Producto suelto"


def parsear_txt_whatsapp(contenido):
    """Parsea el .txt exportado de WhatsApp Business. Solo mensajes de clientes."""
    mensajes = []
    patron = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?\s?m\.?)?)\s*[-–]\s*([^:]+):\s*(.*)"
    )
    lineas = contenido.split("\n")
    msg_actual = None
    for linea in lineas:
        m = patron.match(linea.strip())
        if m:
            if msg_actual:
                mensajes.append(msg_actual)
            fecha_str, hora_str, remitente, texto = m.groups()
            if "cifrado de extremo a extremo" in texto.lower():
                msg_actual = None
                continue
            fecha_obj = None
            for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"):
                try:
                    fecha_obj = datetime.strptime(fecha_str.strip(), fmt).date()
                    break
                except ValueError:
                    pass
            msg_actual = {
                "fecha": fecha_obj,
                "remitente": remitente.strip(),
                "texto": texto.strip(),
                "es_rocktec": any(k in remitente.lower() for k in ["rocktec", "admin", "ventas", "gerencia"]),
            }
        elif msg_actual and linea.strip():
            msg_actual["texto"] += " " + linea.strip()
    if msg_actual:
        mensajes.append(msg_actual)
    return [m for m in mensajes if not m["es_rocktec"] and m.get("fecha") is not None]


def es_chat_grupal(contenido):
    """Detecta si el .txt exportado es un chat grupal de WhatsApp."""
    primeras_lineas = contenido[:2000].lower()
    return "creó el grupo" in primeras_lineas or "created group" in primeras_lineas
    mensajes = []
    patron = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?\s?m\.?)?)\s*[-–]\s*([^:]+):\s*(.*)"
    )
    lineas = contenido.split("\n")
    msg_actual = None
    for linea in lineas:
        m = patron.match(linea.strip())
        if m:
            if msg_actual:
                mensajes.append(msg_actual)
            fecha_str, hora_str, remitente, texto = m.groups()
            if "cifrado de extremo a extremo" in texto.lower():
                msg_actual = None
                continue
            fecha_obj = None
            for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"):
                try:
                    fecha_obj = datetime.strptime(fecha_str.strip(), fmt).date()
                    break
                except ValueError:
                    pass
            msg_actual = {
                "fecha": fecha_obj,
                "remitente": remitente.strip(),
                "texto": texto.strip(),
                "es_rocktec": any(k in remitente.lower() for k in ["rocktec", "admin", "ventas", "gerencia"]),
            }
        elif msg_actual and linea.strip():
            msg_actual["texto"] += " " + linea.strip()
    if msg_actual:
        mensajes.append(msg_actual)
    return [m for m in mensajes if not m["es_rocktec"] and m.get("fecha") is not None]


def procesar_mensajes(mensajes, ultima_fecha):
    if ultima_fecha:
        mensajes = [m for m in mensajes if m["fecha"] > ultima_fecha]
    resultado = []
    for m in mensajes:
        intencion = detectar_intencion(m["texto"])
        razon = detectar_razon_perdida(m["texto"]) if intencion in ("COT", "INF") else None
        tipo = detectar_tipo_negocio(m["texto"])
        resultado.append({
            "fecha": m["fecha"],
            "remitente": m["remitente"],
            "texto": m["texto"][:200],
            "intencion": intencion,
            "razon_perdida": razon,
            "tipo_negocio": tipo,
            "es_lead": intencion in ("COT", "TEC", "VEN", "SEG"),
            "es_perdida": razon is not None,
            "es_venta": intencion == "VEN",
        })
    return resultado


def limpiar_para_pdf(texto):
    """Elimina caracteres Unicode fuera del rango latin-1 que rompen fpdf."""
    return (str(texto)
        .replace('\u2014', '-')   # em dash —
        .replace('\u2013', '-')   # en dash –
        .replace('\u2019', "'")   # comilla curva '
        .replace('\u2018', "'")   # comilla curva '
        .replace('\u201c', '"')   # comilla " abre
        .replace('\u201d', '"')   # comilla " cierra
        .replace('\u00b7', '.')   # punto medio ·
        .encode('latin-1', errors='replace').decode('latin-1')
    )

def filtrar_por_periodo(df, periodo):
    hoy = date.today()
    if df.empty:
        return df
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df = df.dropna(subset=["fecha"])
    if periodo == "Hoy":
        return df[df["fecha"] == hoy]
    elif periodo == "Esta semana":
        inicio = hoy - timedelta(days=hoy.weekday())
        return df[df["fecha"] >= inicio]
    elif periodo == "Este mes":
        return df[df["fecha"] >= date(hoy.year, hoy.month, 1)]
    return df  # Histórico — sin filtro, todo


def render_tab5():
    st.markdown("## 📊 Panel de Inteligencia Comercial")
    st.caption("Carga los chats exportados de WhatsApp para ver leads, pérdidas y ventas por asesor.")

    # Período
    periodo = st.radio(
        "Período",
        ["Hoy", "Esta semana", "Este mes", "Histórico"],
        horizontal=True,
        index=2,
    )

    st.divider()

    # Subida de archivos
    with st.expander("📁 Subir chats de WhatsApp (.txt o .zip)", expanded=True):
        archivos_raw = st.file_uploader(
            "Sube uno o varios archivos .txt o .zip (WhatsApp exporta en zip)",
            type=["txt", "zip"],
            accept_multiple_files=True,
        )

        # Extraer .txt de los zips automáticamente
        archivos = []
        if archivos_raw:
            for f in archivos_raw:
                if f.name.lower().endswith(".zip"):
                    try:
                        z = zipfile.ZipFile(f)
                        txts = [e for e in z.namelist() if e.lower().endswith(".txt")]
                        if txts:
                            import io as _io
                            contenido = z.read(txts[0])
                            # Crear objeto file-like con el nombre del zip (sin .zip)
                            nombre_txt = f.name.replace(".zip", ".txt")
                            fake_file = _io.BytesIO(contenido)
                            fake_file.name = nombre_txt
                            archivos.append(fake_file)
                        else:
                            st.warning(f"⚠️ '{f.name}' no contiene archivo .txt")
                    except Exception as e:
                        st.warning(f"⚠️ Error leyendo '{f.name}': {e}")
                else:
                    archivos.append(f)

        if archivos:
            estado = cargar_estado()
            nuevos_msgs = defaultdict(list)
            asignaciones = {}

            # Paso 1: determinar asesor para cada archivo
            for archivo in archivos:
                numero_limpio = detectar_numero_desde_nombre(archivo.name)
                asesor = None
                if numero_limpio:
                    asesor = asesor_desde_numero(numero_limpio)
                if not asesor:
                    opciones_asesores = [f"{a['nombre']} ({a['numero']})" for a in ASESORES]
                    seleccion = st.selectbox(
                        f"¿A qué asesor pertenece '{archivo.name}'?",
                        opciones_asesores,
                        key=f"sel_{archivo.name}"
                    )
                    idx = opciones_asesores.index(seleccion)
                    asesor = ASESORES[idx]
                asignaciones[archivo.name] = asesor

            # Paso 2: botón para confirmar y procesar
            if st.button("✅ Procesar chats", type="primary", use_container_width=True):
                with st.spinner("Verificando duplicados..."):
                    claves_existentes = get_claves_existentes_sheets()

                for archivo in archivos:
                    asesor = asignaciones[archivo.name]
                    contenido = archivo.read().decode("utf-8", errors="ignore")

                    if es_chat_grupal(contenido):
                        st.warning(
                            f"⚠️ **'{archivo.name}'** es un chat grupal — no aplica para el panel comercial. "
                            f"Los grupos de aplicadores son canales internos, no de atención a clientes."
                        )
                        continue

                    mensajes = parsear_txt_whatsapp(contenido)
                    procesados = procesar_mensajes(mensajes, ultima_fecha=None)

                    # Filtrar duplicados por clave unica remitente+fecha+texto
                    nuevos = []
                    for m in procesados:
                        clave_unica = f"{m.get('remitente','')}|{str(m.get('fecha',''))}|{str(m.get('texto',''))[:50]}"
                        if clave_unica not in claves_existentes:
                            nuevos.append(m)
                            claves_existentes.add(clave_unica)

                    if nuevos:
                        nuevos_msgs[asesor["clave"]].extend(nuevos)
                        st.success(f"✅ {asesor['nombre']} — {archivo.name}: {len(nuevos)} mensajes nuevos")
                    else:
                        st.info(f"ℹ️ '{archivo.name}': todos los mensajes ya estaban registrados")

                if any(nuevos_msgs.values()):
                    for clave, msgs in nuevos_msgs.items():
                        key = f"msgs_{clave}"
                        existentes = st.session_state.get(key, [])
                        st.session_state[key] = existentes + msgs
                    guardar_mensajes_en_sheets(nuevos_msgs, claves_existentes)
                    st.session_state.pop("sesion_limpiada", None)
                    total = sum(len(v) for v in nuevos_msgs.values())
                    st.success(f"✅ {total} mensajes nuevos guardados en la base de datos.")

    # Cargar datos — primero de session_state, si está vacío carga desde Google Sheets
    datos_por_asesor = {}
    hay_datos_en_sesion = any(st.session_state.get(f"msgs_{a['clave']}") for a in ASESORES)

    if not hay_datos_en_sesion and not st.session_state.get("sesion_limpiada", False):
        # Intentar cargar desde Google Sheets (solo si no se acaba de limpiar)
        msgs_sheets = cargar_mensajes_desde_sheets()
        if msgs_sheets:
            for clave, msgs in msgs_sheets.items():
                st.session_state[f"msgs_{clave}"] = msgs

    for a in ASESORES:
        msgs = st.session_state.get(f"msgs_{a['clave']}", [])
        if msgs:
            df = pd.DataFrame(msgs)
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
            df = df.dropna(subset=["fecha"])
            datos_por_asesor[a["clave"]] = filtrar_por_periodo(df, periodo)
        else:
            datos_por_asesor[a["clave"]] = pd.DataFrame()

    # 3 tarjetas por asesor — diseño compacto con colores
    st.markdown("### Resumen por asesor")
    cols = st.columns(3)
    for i, asesor in enumerate(ASESORES):
        df = datos_por_asesor[asesor["clave"]]
        n_perdidas = int(df["es_perdida"].sum()) if not df.empty else 0
        n_leads    = int(df["es_lead"].sum())    if not df.empty else 0
        n_ventas   = int(df["es_venta"].sum())   if not df.empty else 0
        with cols[i]:
            st.markdown(f"""
            <div style='border:1px solid #E0E0E0; border-top:3px solid #C0392B;
                        border-radius:8px; padding:12px 14px; background:#FFFFFF;'>
                <div style='font-weight:700; font-size:13px; color:#1A1A1A; margin-bottom:2px;'>{asesor["nombre"]}</div>
                <div style='font-size:11px; color:#888; margin-bottom:10px;'>{asesor["numero"]}</div>
                <div style='display:flex; gap:6px;'>
                    <div style='flex:1; background:#FDEDEC; border-radius:6px; padding:7px 6px; text-align:center;'>
                        <div style='font-size:18px; font-weight:700; color:#C0392B;'>{n_perdidas}</div>
                        <div style='font-size:10px; color:#C0392B;'>Pérdidas</div>
                    </div>
                    <div style='flex:1; background:#EBF5FB; border-radius:6px; padding:7px 6px; text-align:center;'>
                        <div style='font-size:18px; font-weight:700; color:#1A5276;'>{n_leads}</div>
                        <div style='font-size:10px; color:#1A5276;'>Leads</div>
                    </div>
                    <div style='flex:1; background:#EAFAF1; border-radius:6px; padding:7px 6px; text-align:center;'>
                        <div style='font-size:18px; font-weight:700; color:#1E8449;'>{n_ventas}</div>
                        <div style='font-size:10px; color:#1E8449;'>Ventas</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── CLIMA DE CONVERSACIONES ─────────────────────────────────────────────
    st.markdown("### 🌡️ Clima de conversaciones")

    df_todos_clima = pd.concat(
        [datos_por_asesor[a["clave"]] for a in ASESORES if not datos_por_asesor[a["clave"]].empty],
        ignore_index=True
    ) if any(not datos_por_asesor[a["clave"]].empty for a in ASESORES) else pd.DataFrame()

    if not df_todos_clima.empty and "intencion" in df_todos_clima.columns:
        df_todos_clima["sentimiento"] = df_todos_clima["intencion"].apply(
            lambda x: inferir_sentimiento(x)[0]
        )
        total_clima = len(df_todos_clima)
        n_pos = int((df_todos_clima["sentimiento"] == "Positivo").sum())
        n_neu = int((df_todos_clima["sentimiento"] == "Neutro").sum())
        n_neg = int((df_todos_clima["sentimiento"] == "Negativo").sum())
        pct_pos = round(n_pos / total_clima * 100) if total_clima > 0 else 0
        pct_neu = round(n_neu / total_clima * 100) if total_clima > 0 else 0
        pct_neg = round(n_neg / total_clima * 100) if total_clima > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style='background:#EAFAF1; border-radius:8px; padding:12px; text-align:center;'>
                <div style='font-size:28px;'>😊</div>
                <div style='font-size:22px; font-weight:700; color:#1E8449;'>{n_pos}</div>
                <div style='font-size:12px; color:#1E8449;'>Positivo · {pct_pos}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='background:#FEF9E7; border-radius:8px; padding:12px; text-align:center;'>
                <div style='font-size:28px;'>😐</div>
                <div style='font-size:22px; font-weight:700; color:#B7770D;'>{n_neu}</div>
                <div style='font-size:12px; color:#B7770D;'>Neutro · {pct_neu}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style='background:#FDEDEC; border-radius:8px; padding:12px; text-align:center;'>
                <div style='font-size:28px;'>😠</div>
                <div style='font-size:22px; font-weight:700; color:#C0392B;'>{n_neg}</div>
                <div style='font-size:12px; color:#C0392B;'>Negativo · {pct_neg}%</div>
            </div>""", unsafe_allow_html=True)

        if pct_neg >= 20:
            st.markdown(f"""
            <div style='background:#FDEDEC; border:1px solid #FADBD8; border-left:4px solid #C0392B;
                        border-radius:6px; padding:10px 14px; margin-top:10px; font-size:13px;'>
                ⚠️ <strong style='color:#C0392B;'>{n_neg} clientes con tono negativo ({pct_neg}%)</strong>
                — Revisar urgente: quejas o seguimientos sin respuesta acumulados.
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("Sube chats para ver el clima de conversaciones.")

    st.divider()

    # ── LISTADO DE ACTIVIDADES PRIORIZADAS ──────────────────────────────────
    st.markdown("### 📋 Clientes priorizados — qué hacer hoy")

    hoy = date.today()

    todos_msgs = []
    for asesor in ASESORES:
        df = datos_por_asesor[asesor["clave"]]
        if not df.empty:
            df2 = df.copy()
            df2["asesor_nombre"] = asesor["nombre"]
            todos_msgs.append(df2)

    if todos_msgs:
        df_all = pd.concat(todos_msgs, ignore_index=True)
        df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce").dt.date
        df_all = df_all.dropna(subset=["fecha"])
        df_all["dias"] = df_all["fecha"].apply(lambda f: (hoy - f).days if f else 999)

        def clasificar(row):
            if row["intencion"] == "QUE":
                return "urgente"
            elif row["intencion"] == "VEN":
                return "cierre"
            elif row["intencion"] in ("COT", "SEG") and row["dias"] <= 2:
                return "interes_alto"
            elif row["intencion"] in ("COT", "SEG") and 3 <= row["dias"] <= 6:
                return "seguimiento"
            elif row["intencion"] in ("COT", "SEG") and row["dias"] >= 7:
                return "posible_perdida"
            return None

        df_all["categoria"] = df_all.apply(clasificar, axis=1)
        df_actividades = df_all[df_all["categoria"].notna()].copy()

        CATEGORIAS = [
            ("urgente",         "⚠️ URGENTES — ATENDER AHORA",        "#FDEDEC", "#C0392B", "#FADBD8", "Llamar ahora"),
            ("cierre",          "💲 LISTOS PARA CERRAR",               "#EAFAF1", "#1E8449", "#A9DFBF", "Enviar factura"),
            ("interes_alto",    "🔥 INTERÉS ALTO — COTIZAR HOY",       "#FEF9E7", "#B7770D", "#FAD7A0", "Cotizar hoy"),
            ("seguimiento",     "🕐 SEGUIMIENTO +3 DÍAS",              "#EBF5FB", "#1A5276", "#AED6F1", "Recontactar"),
            ("posible_perdida", "⚪ POSIBLE PÉRDIDA — ÚLTIMO INTENTO", "#F8F9FA", "#555555", "#D5DBDB", "Último intento"),
        ]

        BADGE_COLORS = {
            "QUE": ("#C0392B", "#FADBD8"),
            "VEN": ("#1E8449", "#A9DFBF"),
            "COT": ("#B7770D", "#FAD7A0"),
            "SEG": ("#1A5276", "#AED6F1"),
            "CUR": ("#6C3483", "#D7BDE2"),
            "TEC": ("#117A65", "#A2D9CE"),
            "INF": ("#555555", "#D5DBDB"),
        }

        hay_actividades = False
        for cat_clave, cat_titulo, bg_color, text_color, badge_bg, accion in CATEGORIAS:
            subset = df_actividades[df_actividades["categoria"] == cat_clave].sort_values("dias")
            if subset.empty:
                continue
            hay_actividades = True

            st.markdown(
                f"""<div style='background:{text_color}; padding:7px 14px;
                border-radius:6px 6px 0 0; margin-top:14px;'>
                <strong style='color:#FFFFFF; font-size:12px; letter-spacing:0.3px;'>
                {cat_titulo}</strong></div>""",
                unsafe_allow_html=True
            )

            filas_html = ""
            for _, row in subset.iterrows():
                dias = int(row["dias"])
                dias_txt = "hoy" if dias == 0 else f"hace {dias}d"
                intent = row["intencion"]
                texto = str(row["texto"])[:75] + "…" if len(str(row["texto"])) > 75 else str(row["texto"])
                remitente = str(row.get("remitente", "—"))
                asesor_n = str(row.get("asesor_nombre", "—"))
                iconos = {"urgente":"⚠️","cierre":"💲","interes_alto":"🔥","seguimiento":"🕐","posible_perdida":"↩️"}
                icono = iconos.get(cat_clave, "•")

                filas_html += f"""
                <div style='display:flex; align-items:center; padding:9px 14px;
                border-bottom:1px solid {badge_bg}; background:{bg_color}; gap:10px;'>
                    <span style='font-size:16px; flex-shrink:0;'>{icono}</span>
                    <div style='flex:1; min-width:0;'>
                        <div style='font-size:12px; color:#1A1A1A; font-style:italic; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"{texto}"</div>
                        <div style='font-size:11px; color:#555; margin-top:1px;'>{remitente} · {asesor_n} · <span style='color:#999;'>{dias_txt}</span></div>
                    </div>
                    <div style='text-align:right; flex-shrink:0;'>
                        <span style='background:{text_color}; color:#FFF; border-radius:3px; padding:2px 8px; font-size:11px; font-weight:700;'>{intent}</span>
                        <div style='font-size:11px; color:{text_color}; font-weight:600; margin-top:3px;'>{accion} →</div>
                    </div>
                </div>"""

            st.markdown(
                f"<div style='border:1px solid {badge_bg}; border-top:none; border-radius:0 0 6px 6px; overflow:hidden; margin-bottom:6px;'>{filas_html}</div>",
                unsafe_allow_html=True
            )

        if not hay_actividades:
            st.info("Sin actividades pendientes en este período. 🎉")
    else:
        st.info("Sube chats para ver las actividades priorizadas.")

    st.divider()

    # Tabla de pérdidas
    st.markdown("### Pérdidas confirmadas")
    dfs_perdidas = []
    for asesor in ASESORES:
        df = datos_por_asesor[asesor["clave"]]
        if not df.empty:
            sub = df[df["es_perdida"]].copy()
            sub["asesor"] = asesor["nombre"]
            dfs_perdidas.append(sub)

    if dfs_perdidas:
        df_p = pd.concat(dfs_perdidas, ignore_index=True).sort_values("fecha", ascending=False)
        df_p["texto"] = df_p["texto"].str[:80] + "…"
        st.dataframe(
            df_p[["fecha", "asesor", "texto", "razon_perdida", "tipo_negocio"]].rename(columns={
                "fecha": "Fecha", "asesor": "Asesor", "texto": "Mensaje",
                "razon_perdida": "Razón", "tipo_negocio": "Tipo"
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Sin pérdidas confirmadas en este período. 🎉")

    st.divider()

    # Gráficos simples
    col_iz, col_der = st.columns(2)

    with col_iz:
        st.markdown("**Razones de pérdida**")
        conteo = defaultdict(int)
        for a in ASESORES:
            df = datos_por_asesor[a["clave"]]
            if not df.empty:
                for r in df[df["es_perdida"]]["razon_perdida"]:
                    if r: conteo[r] += 1
        if conteo:
            st.bar_chart(dict(sorted(conteo.items(), key=lambda x: -x[1])))
        else:
            st.caption("Sin datos aún.")

    with col_der:
        st.markdown("**Leads por asesor**")
        leads = {a["nombre"]: int(datos_por_asesor[a["clave"]]["es_lead"].sum())
                 if not datos_por_asesor[a["clave"]].empty else 0
                 for a in ASESORES}
        if any(leads.values()):
            st.bar_chart(leads)
        else:
            st.caption("Sin datos aún.")

    st.divider()

    # Botones — limpiar + descargas
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("🗑️ Limpiar sesión", use_container_width=True):
            for a in ASESORES:
                st.session_state.pop(f"msgs_{a['clave']}", None)
            st.session_state["sesion_limpiada"] = True
            st.success("✅ Vista limpiada. El histórico en la base de datos se conserva.")
            st.rerun()

    # Preparar datos para descarga
    todos_exp = []
    for a in ASESORES:
        df = datos_por_asesor[a["clave"]]
        if not df.empty:
            df2 = df.copy()
            df2["asesor"] = a["nombre"]
            todos_exp.append(df2)

    if todos_exp:
        df_exp = pd.concat(todos_exp, ignore_index=True)

        # Renombrar columnas para exportación legible
        cols_export = {
            "fecha": "Fecha", "asesor": "Asesor", "remitente": "Cliente",
            "texto": "Mensaje", "intencion": "Intención",
            "es_lead": "Es Lead", "es_perdida": "Es Pérdida", "es_venta": "Es Venta",
            "razon_perdida": "Razón Pérdida", "tipo_negocio": "Tipo Negocio"
        }
        df_exp_clean = df_exp[[c for c in cols_export if c in df_exp.columns]].rename(columns=cols_export)
        # Agregar sentimiento
        if "Intención" in df_exp_clean.columns:
            df_exp_clean["Sentimiento"] = df_exp_clean["Intención"].apply(
                lambda x: inferir_sentimiento(x)[0]
            )

        fecha_hoy = date.today().strftime("%Y%m%d")

        with col_b:
            # CSV
            st.download_button(
                "⬇️ Descargar CSV",
                data=df_exp_clean.to_csv(index=False).encode("utf-8"),
                file_name=f"reporte_rocktec_{fecha_hoy}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_c:
            # Excel
            import io
            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                df_exp_clean.to_excel(writer, index=False, sheet_name="Reporte Rocktec")
                # Hoja resumen
                resumen = []
                for a in ASESORES:
                    df_a = datos_por_asesor[a["clave"]]
                    resumen.append({
                        "Asesor": a["nombre"],
                        "Leads": int(df_a["es_lead"].sum()) if not df_a.empty else 0,
                        "Ventas": int(df_a["es_venta"].sum()) if not df_a.empty else 0,
                        "Pérdidas": int(df_a["es_perdida"].sum()) if not df_a.empty else 0,
                    })
                pd.DataFrame(resumen).to_excel(writer, index=False, sheet_name="Resumen Asesores")
            buffer_excel.seek(0)
            st.download_button(
                "📊 Descargar Excel",
                data=buffer_excel,
                file_name=f"reporte_rocktec_{fecha_hoy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_d:
            # PDF — tabla HTML convertida a PDF con pdfkit o fpdf2
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, f"Reporte Comercial Rocktec - {date.today().strftime('%d/%m/%Y')}", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.ln(3)

                # Resumen por asesor
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, "Resumen por Asesor", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for a in ASESORES:
                    df_a = datos_por_asesor[a["clave"]]
                    leads = int(df_a["es_lead"].sum()) if not df_a.empty else 0
                    ventas = int(df_a["es_venta"].sum()) if not df_a.empty else 0
                    perdidas = int(df_a["es_perdida"].sum()) if not df_a.empty else 0
                    pdf.cell(0, 6, limpiar_para_pdf(f"  {a['nombre']}: Leads={leads}  Ventas={ventas}  Perdidas={perdidas}"), ln=True)

                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, "Detalle de Mensajes", ln=True)
                pdf.set_font("Helvetica", "", 8)

                cols_pdf = ["Fecha", "Asesor", "Cliente", "Intención", "Sentimiento"]
                col_widths = [22, 45, 40, 22, 22]
                # Header
                for col, w in zip(cols_pdf, col_widths):
                    pdf.cell(w, 6, col, border=1)
                pdf.ln()
                # Filas
                for _, row in df_exp_clean.head(50).iterrows():
                    for col, w in zip(cols_pdf, col_widths):
                        val = limpiar_para_pdf(str(row.get(col, "")))[:20]
                        pdf.cell(w, 5, val, border=1)
                    pdf.ln()

                if len(df_exp_clean) > 50:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.cell(0, 6, f"  ... y {len(df_exp_clean)-50} registros más. Descarga el Excel para ver todos.", ln=True)

                buffer_pdf = io.BytesIO(pdf.output())
                st.download_button(
                    "📄 Descargar PDF",
                    data=buffer_pdf,
                    file_name=f"reporte_rocktec_{fecha_hoy}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.button("📄 PDF (instalar fpdf2)", disabled=True, use_container_width=True)
                st.caption("Ejecuta: pip install fpdf2")
    else:
        with col_b:
            st.button("⬇️ CSV", disabled=True, use_container_width=True)
        with col_c:
            st.button("📊 Excel", disabled=True, use_container_width=True)
        with col_d:
            st.button("📄 PDF", disabled=True, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Panel Comercial Rocktec", layout="wide")
    render_tab5()
