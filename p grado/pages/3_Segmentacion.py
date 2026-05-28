import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Segmentación SAM2", layout="wide")
st.title("✂️ Segmentación de Objetos — SAM2")

if not st.session_state.get("images"):
    st.warning("⚠️ Ve a **Carga de Imágenes** primero.")
    st.stop()

if "segmentations" not in st.session_state:
    st.session_state["segmentations"] = {}

images = st.session_state["images"]
MAX_IMGS = 15

st.info(
    f"⚡ SAM2 se ejecuta en CPU — se procesarán máximo {MAX_IMGS} imágenes. "
    "Los resultados se guardan en caché."
)

with st.expander("⚙️ Configuración", expanded=True):
    use_bboxes = st.checkbox(
        "Usar bounding boxes de YOLOv11 como prompts (más preciso y rápido)",
        value=True,
    )
    _max_seg = min(len(images), MAX_IMGS)
    if _max_seg > 1:
        n_process = st.slider("Imágenes a segmentar", 1, _max_seg, min(5, _max_seg))
    else:
        n_process = 1
        st.info(f"Solo hay {len(images)} imagen disponible.")

if st.button("🚀 Ejecutar Segmentación SAM2", type="primary", use_container_width=True):
    try:
        from src.segmentor import ImageSegmentor
        seg = ImageSegmentor()
        bar = st.progress(0)
        status = st.empty()

        for i, img_path in enumerate(images[:n_process]):
            bboxes = None
            if use_bboxes:
                det = st.session_state.get("detections", {}).get(img_path, {})
                bboxes = [d["bbox"] for d in det.get("detections", [])] or None

            status.text(f"Segmentando {i+1}/{n_process}: {Path(img_path).name}")
            result = seg.segment(img_path, bboxes=bboxes)
            st.session_state["segmentations"][img_path] = result
            bar.progress((i + 1) / n_process)

        status.success(f"✅ Segmentación completada para {n_process} imágenes")
    except ImportError:
        st.error("⏳ **ultralytics** aún se está instalando. Espera 1-2 minutos y vuelve a intentar.")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["segmentations"]:
    st.divider()
    segs = st.session_state["segmentations"]

    total_masks = sum(s["num_masks"] for s in segs.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Imágenes segmentadas", len(segs))
    c2.metric("Máscaras totales", total_masks)
    c3.metric("Promedio por imagen", f"{total_masks / max(len(segs), 1):.1f}")

    st.divider()
    st.subheader("Visualización")
    selected = st.selectbox(
        "Selecciona imagen",
        options=list(segs.keys()),
        format_func=lambda x: Path(x).name,
    )
    if selected:
        from src.segmentor import ImageSegmentor
        seg_data = segs[selected]
        img_bgr = cv2.imread(selected)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = ImageSegmentor.overlay_masks(img_rgb, seg_data["masks"])

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_column_width=True)
        with col2:
            st.image(overlay, caption=f"Segmentado — {seg_data['num_masks']} máscaras", use_column_width=True)

        # Per-mask stats
        if seg_data["masks"]:
            st.write("**Áreas de máscaras (píxeles):**")
            areas = [m["area"] for m in seg_data["masks"]]
            import pandas as pd
            st.dataframe(
                pd.DataFrame({"Máscara": range(1, len(areas)+1), "Área (px)": areas}),
                use_container_width=True
            )
