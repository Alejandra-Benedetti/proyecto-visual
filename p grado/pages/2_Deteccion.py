import streamlit as st
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Detección", layout="wide")
st.title("🎯 Detección de Objetos")

for _k, _v in [("images", []), ("detections", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state["images"]:
    st.warning("⚠️ Ve a **Carga de Imágenes** primero.")
    st.stop()

images = st.session_state["images"]

# ── Mode selector ─────────────────────────────────────────────────────────────
st.subheader("Modo de detección")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("### 🌐 YOLOv11")
    st.caption("Detecta clases COCO. Rápido, bueno para identificar *qué* objeto es.")
with col_m2:
    st.markdown("### 🎯 SAM2 Auto + Filtros")
    st.caption("Segmenta todo objeto visible. **Recomendado para conteo preciso.**")
with col_m3:
    st.markdown("### 🔬 SAHI + YOLOv11")
    st.caption("Divide imagen en tiles antes de inferir. **Máxima precisión en objetos pequeños.**")

mode = st.radio(
    "Selecciona modo",
    ["🌐 YOLOv11 (clases COCO)", "🎯 SAM2 Auto + Filtros + Watershed", "🔬 SAHI Alta Precisión"],
    label_visibility="collapsed",
)

# ── Config ────────────────────────────────────────────────────────────────────
with st.expander("⚙️ Parámetros", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        conf = st.slider("Confianza mín. (YOLO/SAHI)", 0.05, 0.50, 0.10, 0.01,
                         disabled="SAM2" in mode)
    with col2:
        iou = st.slider("IoU NMS (YOLOv11)", 0.10, 0.90, 0.40, 0.05,
                        disabled="SAM2" in mode)
    with col3:
        if "SAHI" in mode:
            slice_sz = st.select_slider("Tile size (SAHI)", [320, 512, 640], value=512)
        else:
            slice_sz = 512
    with col4:
        st.metric("Imágenes a procesar", len(images))

    if "SAM2" in mode:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_area = st.slider("Área mín. producto (%)", 0.01, 1.0, 0.08, 0.01,
                                 help="% del área total de la imagen")
        with col_f2:
            max_area = st.slider("Área máx. producto (%)", 2.0, 25.0, 18.0, 1.0)
        with col_f3:
            min_solid = st.slider("Solidez mínima", 0.1, 0.9, 0.30, 0.05,
                                  help="Solidez=área_máscara/área_convex_hull. Filtra formas irregulares.")

if st.button("🚀 Ejecutar Detección", type="primary", use_container_width=True):
    # Limpiar caché previo
    from config import CACHE_DIR
    for f in list(CACHE_DIR.glob("det_*.json")) + list(CACHE_DIR.glob("sahi_*.json")):
        f.unlink()

    bar = st.progress(0)
    status = st.empty()

    try:
        # ── Modo 1: YOLOv11 ──────────────────────────────────────────────────
        if "YOLOv11" in mode and "SAHI" not in mode:
            from src.detector import ObjectDetector
            detector = ObjectDetector(conf=conf, iou=iou)
            for i, path in enumerate(images):
                status.text(f"YOLOv11 [{i+1}/{len(images)}]: {Path(path).name}")
                det = detector.detect(path)
                st.session_state["detections"][path] = det
                bar.progress((i + 1) / len(images))

        # ── Modo 2: SAM2 Auto + Filtros + Watershed ───────────────────────────
        elif "SAM2" in mode:
            from ultralytics import SAM
            from src.precise_detector import filter_sam_masks

            @st.cache_resource
            def load_sam():
                return SAM("sam2.1_t.pt")

            sam = load_sam()

            for i, path in enumerate(images):
                status.text(f"SAM2 Auto [{i+1}/{len(images)}]: {Path(path).name}")
                results = sam(path, verbose=False)
                img_bgr = cv2.imread(path)
                ih, iw = img_bgr.shape[:2]

                raw_masks = []
                for r in results:
                    if r.masks is not None:
                        for m in r.masks.data:
                            arr = m.cpu().numpy().astype(bool)
                            raw_masks.append({"mask": arr, "area": int(arr.sum())})

                # Aplicar filtros geométricos + watershed
                filtered = filter_sam_masks(
                    raw_masks, (ih, iw),
                    min_area_ratio=min_area / 100,
                    max_area_ratio=max_area / 100,
                    min_solidity=min_solid,
                )

                detections = []
                for md in filtered:
                    mask = md["mask"]
                    ys, xs = np.where(mask)
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    detections.append({
                        "class_id": 0,
                        "class_name": "producto",
                        "confidence": 1.0,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "bbox_norm": [],
                        "area_px": md["area"],
                    })

                st.session_state["detections"][path] = {
                    "image": path,
                    "num_detections": len(detections),
                    "detections": detections,
                    "mode": "SAM2+Filtros+Watershed",
                }
                bar.progress((i + 1) / len(images))

        # ── Modo 3: SAHI ──────────────────────────────────────────────────────
        else:
            from src.precise_detector import SahiDetector
            detector = SahiDetector(conf=conf, slice_size=slice_sz, overlap=0.2)
            for i, path in enumerate(images):
                status.text(f"SAHI [{i+1}/{len(images)}]: {Path(path).name} "
                            f"(tiles de {slice_sz}px, 20% overlap)")
                det = detector.detect(path)
                st.session_state["detections"][path] = det
                bar.progress((i + 1) / len(images))

        total = sum(d["num_detections"] for d in st.session_state["detections"].values())
        status.success(f"✅ {len(images)} imágenes — **{total} objetos detectados**")

    except ImportError as e:
        st.error(f"⏳ Dependencia faltante: `{e}`")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["detections"]:
    st.divider()
    dets = st.session_state["detections"]

    total_objs = sum(d["num_detections"] for d in dets.values())
    per_img = [{"imagen": Path(p).name, "n_objetos": d["num_detections"]}
               for p, d in dets.items()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imágenes procesadas", len(dets))
    c2.metric("Objetos totales", total_objs)
    c3.metric("Promedio por imagen", f"{total_objs / max(len(dets), 1):.1f}")
    c4.metric("Modo", list(dets.values())[0].get("mode", "YOLOv11") if dets else "—")

    tab1, tab2, tab3 = st.tabs(["📊 Estadísticas", "🖼️ Inspector visual", "📋 Tabla"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            per_df = pd.DataFrame(per_img)
            fig = px.histogram(
                per_df, x="n_objetos", nbins=20,
                title="Distribución de objetos detectados por imagen",
                color_discrete_sequence=["#2196F3"],
                labels={"n_objetos": "N° objetos"},
                marginal="box",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(
                per_df.sort_values("n_objetos", ascending=False).head(25),
                x="imagen", y="n_objetos",
                title="Top 25 imágenes por productos detectados",
                color="n_objetos", color_continuous_scale="Blues",
            )
            fig2.update_xaxes(tickangle=45, tickfont_size=8)
            st.plotly_chart(fig2, use_container_width=True)

        # Stats del conteo
        counts = [d["num_detections"] for d in dets.values()]
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Mín. productos/imagen", int(np.min(counts)))
        s2.metric("Máx. productos/imagen", int(np.max(counts)))
        s3.metric("Media", f"{np.mean(counts):.1f}")
        s4.metric("Mediana", f"{np.median(counts):.0f}")
        s5.metric("Desv. Est.", f"{np.std(counts):.1f}")

    with tab2:
        selected = st.selectbox(
            "Imagen a inspeccionar",
            options=list(dets.keys()),
            format_func=lambda x: f"{Path(x).name}  ({dets[x]['num_detections']} objetos)",
        )
        if selected:
            det_data = dets[selected]
            img_bgr = cv2.imread(selected)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            drawn = img_rgb.copy()

            for obj in det_data["detections"]:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 200, 80), 2)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(drawn,
                         caption=f"{Path(selected).name} — {det_data['num_detections']} objetos detectados",
                         use_column_width=True)
            with col2:
                st.metric("Objetos", det_data["num_detections"])
                mode_lbl = det_data.get("mode", "YOLOv11")
                st.caption(f"Modo: {mode_lbl}")
                areas = [o.get("area_px", 0) for o in det_data["detections"]]
                if any(a > 0 for a in areas):
                    st.metric("Área media (px)", f"{int(np.mean([a for a in areas if a > 0])):,}")
                    st.metric("Área total cubierta",
                              f"{sum(areas) * 100 / (img_bgr.shape[0] * img_bgr.shape[1]):.1f}%")

    with tab3:
        rows = [{
            "imagen": Path(p).name,
            "n_objetos": d["num_detections"],
            "modo": d.get("mode", "YOLOv11"),
        } for p, d in dets.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("n_objetos", ascending=False),
                     use_container_width=True)

        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar CSV de detecciones", csv,
                           "detecciones.csv", "text/csv")
