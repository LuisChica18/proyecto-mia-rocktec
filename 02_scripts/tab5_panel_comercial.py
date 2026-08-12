"""
Tab 5: Panel de Inteligencia Comercial - Rocktec MIA 2026
"""
import re
import json
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


def guardar_mensajes_en_sheets(msgs_por_asesor):
    """Guarda los mensajes nuevos procesados en la pestaña 'mensajes' de Sheets."""
    ws_msgs, _ = get_worksheets()
    if ws_msgs is None:
        return
    try:
        filas = []
        for clave, msgs in msgs_por_asesor.items():
            asesor_nombre = next((a["nombre"] for a in ASESORES if a["clave"] == clave), clave)
            for m in msgs:
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
        if filas:
            ws_msgs.append_rows(filas)
    except Exception:
        pass


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
            msgs_por_asesor[clave].append({
                "fecha": r.get("fecha", ""),
                "remitente": r.get("remitente", ""),
                "texto": r.get("texto", ""),
                "intencion": r.get("intencion", ""),
                "razon_perdida": r.get("razon_perdida", "") or None,
                "tipo_negocio": r.get("tipo_negocio", ""),
                "es_lead": r.get("es_lead", "False") == "True",
                "es_perdida": r.get("es_perdida", "False") == "True",
                "es_venta": r.get("es_venta", "False") == "True",
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
PATRONES_SEG = [r"en qué estado|cuándo despachan|ya llegó|sigo esperando|sin respuesta"]
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


def detectar_intencion(texto):
    if coincide_patron(texto, PATRONES_QUE): return "QUE"
    if coincide_patron(texto, PATRONES_VEN): return "VEN"
    if coincide_patron(texto, PATRONES_COT): return "COT"
    if coincide_patron(texto, PATRONES_SEG): return "SEG"
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


def filtrar_por_periodo(df, periodo):
    hoy = date.today()
    if periodo == "Hoy":
        return df[df["fecha"] == hoy]
    elif periodo == "Esta semana":
        inicio = hoy - timedelta(days=hoy.weekday())
        return df[df["fecha"] >= inicio]
    elif periodo == "Este mes":
        return df[df["fecha"] >= date(hoy.year, hoy.month, 1)]
    return df


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
    with st.expander("📁 Subir chats de WhatsApp (.txt)", expanded=True):
        archivos = st.file_uploader(
            "Sube uno o varios archivos .txt",
            type=["txt"],
            accept_multiple_files=True,
        )
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
                for archivo in archivos:
                    asesor = asignaciones[archivo.name]
                    contenido = archivo.read().decode("utf-8", errors="ignore")

                    # Excluir chats grupales
                    if es_chat_grupal(contenido):
                        st.warning(
                            f"⚠️ **'{archivo.name}'** es un chat grupal — no aplica para el panel comercial. "
                            f"Los grupos de aplicadores son canales internos, no de atención a clientes."
                        )
                        continue

                    mensajes = parsear_txt_whatsapp(contenido)
                    ultima_str = estado.get(asesor["clave"])
                    ultima = datetime.strptime(ultima_str, "%Y-%m-%d").date() if ultima_str else None
                    procesados = procesar_mensajes(mensajes, ultima)
                    if procesados:
                        nuevos_msgs[asesor["clave"]].extend(procesados)
                        nueva_ultima = max(m["fecha"] for m in procesados)
                        if ultima is None or nueva_ultima > ultima:
                            estado[asesor["clave"]] = str(nueva_ultima)
                        st.success(f"✅ {asesor['nombre']}: {len(procesados)} mensajes nuevos procesados")
                    else:
                        st.info(f"ℹ️ {asesor['nombre']}: sin mensajes nuevos desde la última carga")

                if any(nuevos_msgs.values()):
                    for clave, msgs in nuevos_msgs.items():
                        key = f"msgs_{clave}"
                        existentes = st.session_state.get(key, [])
                        st.session_state[key] = existentes + msgs
                    guardar_estado(estado)
                    guardar_mensajes_en_sheets(nuevos_msgs)
                    st.info("✅ Listo. Cierra este panel para ver los totales actualizados.")

    # Cargar datos — primero de session_state, si está vacío carga desde Google Sheets
    datos_por_asesor = {}
    hay_datos_en_sesion = any(st.session_state.get(f"msgs_{a['clave']}") for a in ASESORES)

    if not hay_datos_en_sesion:
        # Intentar cargar desde Google Sheets
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

    # 3 tarjetas por asesor
    st.markdown("### Resumen por asesor")
    cols = st.columns(3)
    for i, asesor in enumerate(ASESORES):
        df = datos_por_asesor[asesor["clave"]]
        n_perdidas = int(df["es_perdida"].sum()) if not df.empty else 0
        n_leads    = int(df["es_lead"].sum())    if not df.empty else 0
        n_ventas   = int(df["es_venta"].sum())   if not df.empty else 0
        with cols[i]:
            st.metric(asesor["nombre"], "")
            st.caption(asesor["numero"])
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Pérdidas", n_perdidas)
            with c2: st.metric("Leads",    n_leads)
            with c3: st.metric("Ventas",   n_ventas)

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

    # Botones
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Limpiar sesión", use_container_width=True):
            for a in ASESORES:
                st.session_state.pop(f"msgs_{a['clave']}", None)
            st.rerun()
    with col_b:
        todos = []
        for a in ASESORES:
            df = datos_por_asesor[a["clave"]]
            if not df.empty:
                df2 = df.copy(); df2["asesor"] = a["nombre"]
                todos.append(df2)
        if todos:
            df_exp = pd.concat(todos, ignore_index=True)
            st.download_button(
                "⬇️ Descargar CSV",
                data=df_exp.to_csv(index=False).encode("utf-8"),
                file_name=f"reporte_rocktec_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Descargar CSV", disabled=True, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Panel Comercial Rocktec", layout="wide")
    render_tab5()
