import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Reporte y Exportación", layout="wide")
st.title("📄 Reporte e Interpretación — Qwen2.5-VL + Exportación")

# Inicializar todas las claves necesarias
for _k, _v in [("images", []), ("descriptions", {}), ("detections", {}),
               ("segmentations", {}), ("stats_df", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

images = st.session_state["images"]
df = st.session_state.get("stats_df")

if not images:
    st.warning("⚠️ No hay imágenes cargadas.")
    st.stop()

# Check Ollama
from src.describer import ImageDescriber
describer = ImageDescriber()
if describer.is_available:
    st.success(f"✅ Qwen2.5-VL disponible via Ollama ({describer.model})")
else:
    st.warning("⚠️ Ollama no detectado — se usarán descripciones estadísticas automáticas. "
               "Para activar Qwen2.5-VL: instala Ollama y ejecuta `ollama pull qwen2.5-vl:7b`")

tab_desc, tab_excel, tab_pdf, tab_summary = st.tabs([
    "🤖 Descripciones IA", "📥 Excel", "📄 PDF", "📊 Resumen Global"
])

# ── Tab 1: AI Descriptions ────────────────────────────────────────────────────
with tab_desc:
    max_n = min(len(images), 20)
    if max_n > 1:
        n_desc = st.slider("Imágenes a describir", 1, max_n, min(5, max_n))
    else:
        n_desc = max_n

    if st.button("🤖 Generar Descripciones", type="primary"):
        bar = st.progress(0)
        status = st.empty()
        stats_list = None
        if df is not None:
            stats_list = [df.iloc[i].to_dict() if i < len(df) else None for i in range(n_desc)]

        for i, img_path in enumerate(images[:n_desc]):
            status.text(f"Describiendo {i+1}/{n_desc}: {Path(img_path).name}")
            stats = stats_list[i] if stats_list else None
            desc = describer.describe(img_path, stats=stats)
            st.session_state["descriptions"][img_path] = desc
            bar.progress((i + 1) / n_desc)

        status.success(f"✅ {n_desc} descripciones generadas")

    if st.session_state.get("descriptions"):
        st.divider()
        for img_path, desc in list(st.session_state["descriptions"].items())[:n_desc]:
            with st.expander(f"📷 {Path(img_path).name}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(Image.open(img_path), use_column_width=True)
                with c2:
                    st.write(desc)
                    if df is not None:
                        idx = images.index(img_path)
                        if idx < len(df):
                            row = df.iloc[idx]
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Intensidad", f"{row.get('mean_intensity', 0):.1f}")
                            sc2.metric("Entropía", f"{row.get('entropy', 0):.3f}")
                            sc3.metric("Colorfulness", f"{row.get('colorfulness', 0):.1f}")

# ── Tab 2: Excel export ───────────────────────────────────────────────────────
with tab_excel:
    st.subheader("Exportar análisis completo a Excel")

    if df is None:
        st.info("Extrae estadísticas primero para incluirlas.")
    else:
        st.write(f"Se exportarán {len(df)} filas × {len(df.columns)} columnas.")

    if st.button("📥 Generar Excel", type="primary"):
        if df is not None:
            from src.exporter import ReportExporter
            exporter = ReportExporter()
            descs = list(st.session_state["descriptions"].values()) or None
            dets = [
                {"imagen": Path(p).name,
                 "n_objetos": d["num_detections"],
                 "clases": ", ".join(set(o["class_name"] for o in d["detections"]))}
                for p, d in st.session_state.get("detections", {}).items()
            ] or None

            with st.spinner("Generando Excel..."):
                out = exporter.export_excel(df, descs, dets)

            with open(out, "rb") as f:
                st.download_button(
                    "📥 Descargar Excel",
                    f.read(), out.name,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.success(f"✅ {out.name}")
        else:
            st.error("No hay datos estadísticos. Ejecuta el análisis primero.")

# ── Tab 3: PDF export ─────────────────────────────────────────────────────────
with tab_pdf:
    st.subheader("Exportar reporte académico en PDF")
    include_auto = st.checkbox("Incluir resumen ejecutivo automático", value=True)

    if st.button("📄 Generar PDF", type="primary"):
        if df is not None:
            from src.exporter import ReportExporter
            exporter = ReportExporter()
            summary = ""
            if include_auto:
                n = len(df)
                avg_i = df.get("mean_intensity", pd.Series([0])).mean()
                avg_e = df.get("entropy", pd.Series([0])).mean()
                avg_c = df.get("colorfulness", pd.Series([0])).mean()
                summary = (
                    f"El presente análisis comprende {n} imágenes digitales de la empresa, "
                    f"procesadas de manera automatizada mediante el pipeline propuesto. "
                    f"La intensidad media fue {avg_i:.1f} unidades, con entropía promedio "
                    f"de {avg_e:.3f} y colorfulness de {avg_c:.2f}. "
                    f"Las variables estadísticas extraídas permiten caracterizar el universo "
                    f"de imágenes de forma objetiva, cuantificable y reproducible."
                )

            with st.spinner("Generando PDF..."):
                out = exporter.export_pdf(df, summary_text=summary)

            with open(out, "rb") as f:
                st.download_button("📄 Descargar PDF", f.read(), out.name, "application/pdf")
            st.success(f"✅ {out.name}")
        else:
            st.error("Extrae las estadísticas primero.")

# ── Tab 4: Global summary ─────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Resumen Global del Análisis")

    n_imgs = len(images)
    n_dets = sum(d["num_detections"] for d in st.session_state.get("detections", {}).values())
    n_segs = sum(s["num_masks"] for s in st.session_state.get("segmentations", {}).values())
    n_descs = len(st.session_state.get("descriptions", {}))
    has_emb = st.session_state.get("embeddings") is not None
    has_model = st.session_state.get("predictor") is not None

    c1, c2, c3 = st.columns(3)
    c1.metric("Imágenes analizadas", n_imgs)
    c1.metric("Objetos detectados", n_dets)
    c2.metric("Segmentos extraídos", n_segs)
    c2.metric("Descripciones IA", n_descs)
    c3.metric("Embeddings indexados", "✅" if has_emb else "—")
    c3.metric("Modelo predictivo", "✅" if has_model else "—")

    if df is not None:
        st.divider()
        st.subheader("Variables estadísticas clave del universo")
        key_vars = {
            "Intensidad media (brillo)": "mean_intensity",
            "Desviación estándar (variabilidad)": "std_intensity",
            "Entropía (complejidad visual)": "entropy",
            "Densidad de bordes (detalle)": "edge_density",
            "Saturación media (vivacidad)": "mean_saturation",
            "Colorfulness (riqueza de color)": "colorfulness",
            "Contraste de textura": "texture_contrast",
        }
        rows = []
        for label, col in key_vars.items():
            if col in df.columns:
                s = df[col]
                rows.append({
                    "Variable": label,
                    "Media": round(s.mean(), 3),
                    "Mediana": round(s.median(), 3),
                    "Desv. Est.": round(s.std(), 3),
                    "Min": round(s.min(), 3),
                    "Max": round(s.max(), 3),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
