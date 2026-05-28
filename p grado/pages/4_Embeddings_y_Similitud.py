import streamlit as st
from pathlib import Path
import numpy as np
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Embeddings y Similitud", layout="wide")
st.title("🔮 Embeddings Visuales y Similitud — SigLIP2 + FAISS")

# Inicializar claves de session_state necesarias en esta página
for _k, _v in [("images", []), ("embeddings", None), ("cluster_labels", None),
               ("indexer", None), ("detections", {}), ("tracks", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state["images"]:
    st.warning("⚠️ Ve a **Carga de Imágenes** primero.")
    st.stop()

images = st.session_state["images"]

with st.expander("⚙️ Configuración", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Número de clusters FAISS", 2, 10, 5)
    with col2:
        viz_method = st.radio("Reducción de dimensionalidad", ["PCA", "t-SNE"])

if st.button("🚀 Generar Embeddings SigLIP2 e Indexar FAISS", type="primary", use_container_width=True):
    try:
        from src.embedder import VisualEmbedder
        from src.indexer import SimilarityIndexer

        embedder = VisualEmbedder()
        bar = st.progress(0)
        status = st.empty()
        status.info("Cargando modelo SigLIP2 (primera vez puede tardar ~30 s)...")

        def cb(cur, tot):
            bar.progress(cur / tot)
            status.text(f"Embeddings: {cur}/{tot} imágenes...")

        embeddings = embedder.embed_batch(images, progress_cb=cb)
        st.session_state["embeddings"] = embeddings

        status.text("Indexando en FAISS...")
        indexer = SimilarityIndexer()
        indexer.build(embeddings, images)
        st.session_state["indexer"] = indexer

        status.text("Clustering...")
        labels = indexer.cluster(embeddings, n_clusters=n_clusters)
        st.session_state["cluster_labels"] = labels

        status.success(f"✅ {len(embeddings)} imágenes indexadas en {n_clusters} clusters")
    except ImportError as e:
        st.error(f"⏳ Dependencia aún instalándose: `{e}`. Espera 1-2 min y reintenta.")
    except Exception as e:
        st.error(f"Error: {e}")

# ── ByteTrack tracking ────────────────────────────────────────────────────────
if st.session_state.get("detections") and st.session_state["embeddings"] is not None:
    with st.expander("🏃 ByteTrack — Tracking multi-imagen"):
        if st.button("Ejecutar ByteTrack sobre detecciones"):
            from src.tracker import ObjectTracker
            tracker = ObjectTracker()
            for img_path in images:
                det = st.session_state["detections"].get(img_path, {})
                if det.get("detections"):
                    tracker.update(det, img_path)
            st.session_state["tracks"] = {
                "history": tracker.history,
                "persistent": tracker.persistent_objects(min_appearances=2),
            }
            n_pers = len(st.session_state["tracks"]["persistent"])
            st.success(f"✅ {n_pers} objetos persistentes (aparecen en ≥2 imágenes)")

# ── Visualization ─────────────────────────────────────────────────────────────
if st.session_state["embeddings"] is not None:
    st.divider()
    embeddings = st.session_state["embeddings"]
    labels = st.session_state.get("cluster_labels")

    st.subheader("Espacio de Embeddings")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    with st.spinner(f"Calculando {viz_method}..."):
        if viz_method == "PCA":
            coords = PCA(n_components=2).fit_transform(embeddings)
        else:
            perp = min(30, len(embeddings) - 1)
            coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(embeddings)

    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=labels.astype(str) if labels is not None else None,
        hover_name=[Path(p).name for p in images],
        title=f"Espacio de embeddings ({viz_method}) — coloreado por cluster",
        labels={"x": f"{viz_method} 1", "y": f"{viz_method} 2", "color": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    # Similarity search
    st.divider()
    st.subheader("🔍 Búsqueda de Imágenes Similares")
    query_idx = st.selectbox(
        "Imagen de consulta",
        range(len(images)),
        format_func=lambda i: Path(images[i]).name,
    )
    if st.button("Buscar similares"):
        indexer = st.session_state["indexer"]
        results = indexer.search(embeddings[query_idx], k=6)
        cols = st.columns(6)
        for col, res in zip(cols, results):
            with col:
                st.image(Image.open(res["path"]), use_column_width=True)
                st.caption(f"Sim: {res['score']:.3f}")

    # Cluster distribution
    if labels is not None:
        st.divider()
        st.subheader("Distribución de Clusters")
        unique, counts = np.unique(labels, return_counts=True)
        fig2 = px.bar(
            x=[f"Cluster {c}" for c in unique], y=counts,
            title="Imágenes por cluster",
            color=unique.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"x": "Cluster", "y": "N° imágenes"},
        )
        st.plotly_chart(fig2, use_container_width=True)
