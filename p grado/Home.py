import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Análisis Estadístico de Imágenes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "images": [],
    "detections": {},
    "segmentations": {},
    "embeddings": None,
    "cluster_labels": None,
    "indexer": None,
    "tracks": {},
    "stats_df": None,
    "descriptions": {},
    "predictor": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Auto-load images already in data/input
from config import INPUT_DIR
if not st.session_state["images"]:
    existing = sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.jpeg")) + sorted(INPUT_DIR.glob("*.png"))
    if existing:
        st.session_state["images"] = [str(p) for p in existing]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Sistema de Análisis Estadístico de Imágenes Digitales")
st.markdown(
    "**Proyecto de Grado** — Procesamiento automatizado · "
    "Estadística descriptiva · Modelos predictivos · Reportes"
)
st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.subheader("Pipeline de Procesamiento")
steps = [
    ("🎯", "YOLOv11", "Detección de objetos"),
    ("✂️", "SAM2", "Segmentación"),
    ("🔮", "SigLIP2", "Embeddings visuales"),
    ("🔍", "FAISS", "Similitud y clustering"),
    ("🏃", "ByteTrack", "Tracking multi-imagen"),
    ("📈", "Stats", "Variables estadísticas"),
    ("🌲", "XGBoost", "Modelo predictivo"),
    ("🧩", "SHAP", "Explicabilidad"),
    ("🤖", "Qwen2.5-VL", "Descripción IA"),
]
cols = st.columns(len(steps))
for col, (icon, name, desc) in zip(cols, steps):
    with col:
        st.markdown(f"### {icon}")
        st.markdown(f"**{name}**")
        st.caption(desc)

st.divider()

# ── Status dashboard ──────────────────────────────────────────────────────────
n_imgs = len(st.session_state["images"])
n_dets = len(st.session_state["detections"])
n_segs = len(st.session_state["segmentations"])
has_emb = st.session_state["embeddings"] is not None
has_stats = st.session_state["stats_df"] is not None
has_model = st.session_state["predictor"] is not None

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Imágenes", n_imgs)
c2.metric("Detectadas", n_dets)
c3.metric("Segmentadas", n_segs)
c4.metric("Embeddings", "✅" if has_emb else "—")
c5.metric("Estadísticas", "✅" if has_stats else "—")
c6.metric("Modelo", "✅" if has_model else "—")

st.divider()

# ── Instructions ──────────────────────────────────────────────────────────────
st.subheader("Flujo de trabajo")
st.markdown("""
| Paso | Página | Acción |
|------|--------|--------|
| 1 | **Carga de Imágenes** | Sube las fotos o carga las imágenes del proyecto |
| 2 | **Detección** | Detecta productos/objetos con YOLOv11 |
| 3 | **Segmentación** | Segmenta cada objeto con SAM2 |
| 4 | **Embeddings y Similitud** | Genera vectores SigLIP2 e indexa con FAISS |
| 5 | **Análisis Estadístico** | Extrae y visualiza variables cuantificables |
| 6 | **Modelo Predictivo** | Entrena XGBoost y explica con SHAP |
| 7 | **Reporte** | Genera descripciones con Qwen2.5-VL y exporta PDF/Excel |
""")

if n_imgs > 0:
    st.success(f"✅ {n_imgs} imágenes listas. Navega por el menú lateral para continuar.")
else:
    st.info("👈 Comienza en **Carga de Imágenes** en el menú lateral.")
