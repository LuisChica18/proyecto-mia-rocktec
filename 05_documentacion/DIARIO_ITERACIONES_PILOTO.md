# Diario de Iteraciones — Piloto Panel Comercial Rocktec
## Proyecto MIA 2026 | Fase 5: Piloto y Validación

**Propósito:** Registrar decisiones de diseño, problemas detectados durante las pruebas y correcciones aplicadas. Este documento evidencia el ciclo de mejora continua del sistema durante la validación con la usuaria final (Martha Elena Andrade Egas, Gerente General de Rocktec).

---

## ITERACIÓN 1 — 11 agosto 2026

### Contexto
Primera sesión de pruebas con la Gerente General de Rocktec. Se procesaron los primeros chats reales de WhatsApp Business con el sistema deployado en Streamlit Community Cloud.

### Problemas detectados
| # | Problema | Impacto |
|---|---|---|
| 1 | Pocos chats de prueba disponibles — solo datos sintéticos | No se podía validar con datos reales |
| 2 | Sesión se cierra rápidamente — timeout corto | Martha perdía el trabajo al reabrir |
| 3 | Descarga de resultados en Excel/PDF no disponible | Martha no puede exportar reportes |
| 4 | Categorización incorrecta en SEG y CUR con ejemplos reales | Modelo clasificaba mal seguimientos y cursos |
| 5 | Filtros Hoy/Semana/Mes/Histórico no mostraban diferencia | Filtros de fecha no funcionaban |
| 6 | Luis Cruel (co-autor) no podía acceder al sistema | Configuración de Sharing incompleta en Streamlit Cloud |

### Correcciones aplicadas
| # | Corrección | Fecha |
|---|---|---|
| 4 | Correcciones de reglas léxicas para SEG y CUR | 11 ago 2026 |
| 5 | Lógica de filtros de fecha corregida | 11 ago 2026 |

### Pendientes
- Descarga Excel/PDF
- Timeout de sesión
- Sharing con 4 correos autorizados

---

## ITERACIÓN 2 — 13 agosto 2026

### Contexto
Segunda sesión de pruebas. Se generaron 20 chats sintéticos adicionales para pruebas. Se identificaron problemas estructurales de diseño y flujo de datos.

### Problemas detectados
| # | Problema | Causa raíz | Impacto |
|---|---|---|---|
| 1 | Tarjetas de asesor con mucho espacio en blanco | Uso de `st.metric` apilados sin color | Visual poco profesional para demo |
| 2 | Botón "Limpiar sesión" no funcionaba | `session_state` se limpiaba pero Sheets recargaba inmediatamente | Martha no podía subir chats frescos |
| 3 | Chats nuevos aparecían como "ya procesados" | Control de duplicados por `ultima_fecha` por asesor bloqueaba chats anteriores | No se podían procesar múltiples chats |
| 4 | Para limpiar datos había que salir chat por chat | Diseño de sesión no contemplaba carga masiva | No es práctico con 20-30 chats reales |
| 5 | Tab 1 (Clasificador) y Tab 3 (Lote) eran redundantes | Duplicación de funcionalidad | Interfaz innecesariamente compleja |
| 6 | Filas de clientes priorizados sin color de fondo | `background:#FFFFFF` hardcodeado en HTML | No coincidía con diseño esperado por Martha |
| 7 | Sistema no tenía análisis de sentimiento | Pendiente de implementar | Faltaba componente prometido en propuesta |

### Decisiones de diseño tomadas
| Decisión | Justificación |
|---|---|
| **Limpiar sesión = solo resetea vista, NO toca Sheets** | Sheets es la base de datos permanente acumulativa de clientes Rocktec. Borrarla significaría perder el histórico comercial. |
| **Control de duplicados por clave única** (remitente + fecha + texto[:50]) | Permite carga masiva de 20-30 chats de una vez. Martha puede subir todos los chats de sus asesores juntos sin que el sistema repita mensajes ya registrados. Cuando un cliente retoma conversación, solo se guardan los mensajes nuevos. |
| **Sentimiento inferido desde intención** (sin modelo externo) | Martha usa herramientas básicas sin GPU. pysentimiento requiere recursos computacionales no disponibles en el entorno de la usuaria. Solución: derivar sentimiento de la intención ya clasificada (QUE→Negativo, VEN→Positivo, resto→Neutro). Documentado como limitación técnica justificada. |
| **"Clima de conversaciones" en Panel Comercial** | Martha trabaja en Tab 4 (Panel Comercial). Poner estadísticas de sentimiento ahí —donde ella ya opera— evita que tenga que navegar a tabs técnicos. |
| **Tab 1 + Tab 3 fusionados → 4 tabs en lugar de 5** | Clasificador individual y por lote son la misma funcionalidad. Simplificación UX para reducir fricción en demo y uso diario. |
| **Colores de fondo por categoría en filas de clientes** | Diseño visual alineado con expectativa de Martha: urgentes en rojo, cierres en verde, leads en amarillo. Mejora legibilidad y priorización visual. |

### Correcciones aplicadas
| # | Corrección | Archivo modificado |
|---|---|---|
| 1 | Tarjetas de asesor rediseñadas con HTML compacto y colores | `tab5_panel_comercial.py` |
| 2 | Limpiar sesión corregido — conserva Sheets | `tab5_panel_comercial.py` |
| 3 | Control de duplicados reemplazado por clave única | `tab5_panel_comercial.py` |
| 5 | Tab 1 y Tab 3 fusionados | `19_dashboard_streamlit.py` |
| 6 | Filas de clientes con color de fondo por categoría | `tab5_panel_comercial.py` |
| 7 | Sentimiento inferido + "Clima de conversaciones" | `tab5_panel_comercial.py` (en progreso) |

### Chats de prueba generados
Se generaron 5 chats sintéticos para validar el pipeline completo. Están almacenados en `02_scripts/chat prueba/` con nota explícita de que son datos generados para pruebas, NO datos reales de Rocktec.

| Archivo | Asesor | Intenciones cubiertas |
|---|---|---|
| Chat con +593 99 060 9023.txt | Admin Rocktec | QUE + SEG → URGENTE |
| Chat con +593 99 060 9023 curso.txt | Admin Rocktec | CUR → interés alto |
| Chat con +593 99 561 8025.txt | Ventas Rocktec | VEN → cierre |
| Chat con +593 99 561 8025 perdida.txt | Ventas Rocktec | Pérdida + COT → posible pérdida |
| Chat con +593 99 380 2851.txt | Gerencia Rocktec | TEC + COT + SEG → seguimiento |

---

## PRÓXIMAS ITERACIONES

### Iteración 3 — prevista 13 agosto 2026 (tarde)
- Prueba con chats reales exportados por Martha Andrade
- Validación de clasificación con datos reales del negocio
- Verificación de "Clima de conversaciones" con sentimiento real

### Pendientes técnicos
- [ ] Descarga Excel y PDF de resultados
- [ ] Ajuste de timeout de sesión
- [ ] Acceso admin de Patty en Streamlit Cloud (independencia de Luis para reboots)

---

*Documento mantenido por: Alexandra Patricia Mosquera Castro (A1)*
*Última actualización: 13 agosto 2026*
