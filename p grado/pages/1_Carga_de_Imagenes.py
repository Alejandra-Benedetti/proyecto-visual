import streamlit as st
from pathlib import Path
import shutil
from PIL import Image
from config import INPUT_DIR, CACHE_DIR

st.set_page_config(page_title="Carga de Imágenes", layout="wide")
st.title("📁 Carga y Gestión del Universo de Imágenes")

if "images" not in st.session_state:
    st.session_state["images"] = []

tab_upload, tab_project, tab_view = st.tabs(
    ["⬆️ Subir imágenes", "📂 Fotos del proyecto", "🖼️ Ver universo"]
)

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Selecciona imágenes (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Guardar imágenes subidas", type="primary"):
        added = 0
        for f in uploaded:
            dest = INPUT_DIR / f.name
            dest.write_bytes(f.read())
            added += 1
            if str(dest) not in st.session_state["images"]:
                st.session_state["images"].append(str(dest))
        st.success(f"✅ {added} imagen(es) guardada(s)")
        st.rerun()

# ── Tab 2: Project photos ─────────────────────────────────────────────────────
with tab_project:
    fotos_dir = Path("Fotos/Fotos")
    if not fotos_dir.exists():
        st.warning("No se encontró la carpeta Fotos/Fotos/")
    else:
        fotos = sorted(fotos_dir.glob("*.jpeg")) + sorted(fotos_dir.glob("*.jpg"))
        st.info(f"📂 {len(fotos)} imágenes encontradas en Fotos/Fotos/")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Cargar TODAS las fotos del proyecto", type="primary", use_container_width=True):
                added = 0
                for f in fotos:
                    dest = INPUT_DIR / f.name
                    if not dest.exists():
                        shutil.copy(f, dest)
                        added += 1
                    if str(dest) not in st.session_state["images"]:
                        st.session_state["images"].append(str(dest))
                st.success(f"✅ {added} nuevas copiadas — {len(st.session_state['images'])} en total")
                st.rerun()
        with col2:
            sample_n = st.number_input("O carga una muestra aleatoria de N imágenes", 5, len(fotos), min(20, len(fotos)))
            if st.button("Cargar muestra", use_container_width=True):
                import random
                sample = random.sample(fotos, int(sample_n))
                added = 0
                for f in sample:
                    dest = INPUT_DIR / f.name
                    if not dest.exists():
                        shutil.copy(f, dest)
                        added += 1
                    if str(dest) not in st.session_state["images"]:
                        st.session_state["images"].append(str(dest))
                st.success(f"✅ {added} imágenes de muestra cargadas")
                st.rerun()

# ── Tab 3: Gallery ────────────────────────────────────────────────────────────
with tab_view:
    images = st.session_state["images"]
    n = len(images)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total imágenes", n)

    if n > 0:
        # Quick stats
        from PIL import Image as PILImage
        sample_img = PILImage.open(images[0])
        col2.metric("Resolución muestra", f"{sample_img.width}×{sample_img.height}")
        col3.metric("Formato", Path(images[0]).suffix.upper())

        st.divider()

        # Management
        if st.button("🗑️ Limpiar caché de procesamiento", help="Elimina resultados guardados para reprocesar"):
            for f in CACHE_DIR.glob("*"):
                f.unlink()
            st.success("Caché limpiado")

        if st.button("🗑️ Eliminar todas las imágenes cargadas", type="secondary"):
            for f in INPUT_DIR.glob("*"):
                f.unlink()
            st.session_state["images"] = []
            for key in ["detections", "segmentations", "embeddings",
                        "cluster_labels", "indexer", "stats_df", "descriptions"]:
                if key in st.session_state:
                    st.session_state[key] = {} if key in ("detections", "segmentations", "descriptions") else None
            st.rerun()

        st.divider()
        st.subheader(f"Galería — {n} imágenes")
        cols_per_row = 4
        for row_start in range(0, n, cols_per_row):
            cols = st.columns(cols_per_row)
            for col, img_path in zip(cols, images[row_start:row_start + cols_per_row]):
                with col:
                    st.image(PILImage.open(img_path), caption=Path(img_path).name[:25], use_column_width=True)
    else:
        st.info("No hay imágenes cargadas. Usa las pestañas anteriores para cargar imágenes.")
