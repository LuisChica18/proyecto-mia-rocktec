# Reporte de Pruebas de Estrés — Plataforma Rocktec MIA 2026

**Fecha:** 18 de agosto de 2026  
**Sistema:** Dashboard Streamlit — Tab 1 Clasificador + Tab 2 Dashboard  
**Ejecutado por:** Alexandra Patricia Mosquera Castro  

---

## 1. Objetivo

Verificar la robustez del sistema ante entradas inesperadas, mensajes ambiguos, vocabulario fuera del dominio y condiciones de estrés antes de la defensa final.

---

## 2. Pruebas de robustez — Entradas inválidas o extremas

| # | Mensaje ingresado | Resultado | Confianza | Evaluación |
|---|-------------------|-----------|-----------|------------|
| 1 | `2559999` (solo números) | INF | 73% | ✅ No crashea |
| 2 | `2559999bdwlkjfl%5` (números + letras + símbolos) | INF | 76% | ✅ No crashea |
| 3 | Texto random con errores `,mafdnkdfjgirtmrfn trabajno mas hecho muc hsol. errores` | INF | 70% | ✅ No crashea |
| 4 | `Hello` (inglés) | INF | — | ✅ No crashea |
| 5 | `HOLIS` (coloquial) | INF | — | ✅ No crashea |
| 6 | `GRACIAS` (muy corto) | INF | — | ✅ No crashea |

**Conclusión:** El sistema maneja entradas inválidas sin crashear. Clasifica como INF por defecto cuando no detecta patrones reconocibles. ✅

---

## 3. Pruebas de clasificación — Mensajes reales ambiguos

| # | Mensaje | Resultado | Confianza | Evaluación |
|---|---------|-----------|-----------|------------|
| 1 | `quiero información sobre precios` | COT | — | ✅ Correcto |
| 2 | `me interesa el microcemento` | Revisión humana | <75% | ⚠️ Ambiguo — aceptable |
| 3 | `necesito ayuda` | INF | — | ✅ Correcto |
| 4 | `kiero saber cuanto sale el microcement` | COT | — | ✅ Correcto (typos) |
| 5 | `ya se me dañó lo q me instalaron` | Revisión humana | <75% | ⚠️ Debería ser QUE — limitación léxica documentada |
| 6 | `precio?` | COT | — | ✅ Correcto |
| 7 | `no quiero nada gracias` | INF | — | ⚠️ Pérdida no detectada en Tab 1 (normal — las pérdidas se detectan en Panel Comercial) |
| 8 | `ya compré en otro lado` | INF | — | ⚠️ Pérdida no detectada en Tab 1 (normal — las pérdidas se detectan en Panel Comercial) |

**Nota:** Los mensajes de pérdida (#7 y #8) clasifican como INF en el Tab 1 porque el clasificador ML no fue entrenado para detectar pérdidas — esa función es exclusiva del Panel Comercial mediante patrones léxicos explícitos.

---

## 4. Pruebas de drift — Vocabulario fuera del dominio

Mensajes con términos de productos no presentes en el dataset de entrenamiento:

| # | Mensaje | Intención | Confianza | Método |
|---|---------|-----------|-----------|--------|
| 1 | `microgravilla decorativa para fachada` | INF | 76% | ML |
| 2 | `hormigón impreso para estacionamiento` | INF | 67% | ML |
| 3 | `overlay de concreto para piscina` | INF | 68% | ML |
| 4 | `terrazo moderno para sala` | INF | 72% | ML |
| 5 | `resina epoxi para piso industrial` | INF | 69% | ML |

**Análisis de drift:**
- Con vocabulario conocido del dominio: confianza promedio **80-90%**
- Con vocabulario nuevo (fuera del dominio): confianza promedio **70-72%**
- Reducción de confianza: **~15-20 puntos porcentuales**

Esta degradación controlada confirma el comportamiento esperado del modelo ante drift de datos. Evidently AI (Script 20) monitorea esta distribución en producción y genera alertas cuando el cambio es sistemático.

---

## 5. Resumen ejecutivo

| Categoría | Estado |
|-----------|--------|
| Entradas inválidas (números, símbolos) | ✅ Robusto |
| Mensajes muy cortos | ✅ Funciona |
| Mensajes coloquiales con typos | ✅ Funciona |
| Mensajes en inglés | ✅ Clasifica como INF |
| Mensajes ambiguos | ⚠️ Va a revisión humana (comportamiento correcto) |
| Vocabulario fuera del dominio | ⚠️ Confianza baja — drift detectable |
| Crashes o errores críticos | ✅ Ninguno detectado |

**Conclusión general:** El sistema es robusto para uso en producción con Martha Andrade (Rocktec). Los casos límite están documentados como limitaciones conocidas y manejados correctamente mediante el umbral de confianza (0.75) que deriva mensajes dudosos a revisión humana.

---

## 6. Limitaciones documentadas

1. **Mensajes de pérdida en Tab 1** — el clasificador ML no detecta pérdidas; esta función es del Panel Comercial.
2. **Vocabulario muy nuevo** — confianza baja pero sin crash; Evidently AI detecta el drift.
3. **QUE con lenguaje coloquial** — "ya se me dañó" no activa los patrones de queja; requiere palabras más explícitas.
4. **SEG con lenguaje indirecto** — "para cuando estará listo" no activa SEG; pendiente de ajuste de patrones.

---

*Plataforma de Inteligencia Comercial Rocktec · MIA 2026 · UDLA Ecuador*  
*Mosquera Castro A.P. · Cruel Chang L.C. · Chica Moncayo L.M.*
