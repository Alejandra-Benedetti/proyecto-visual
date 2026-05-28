import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats as scipy_stats

st.set_page_config(page_title="Análisis Estadístico", layout="wide")
st.title("📊 Análisis Estadístico — Extracción de Variables Cuantificables")

for _k, _v in [("images", []), ("stats_df", None), ("cluster_labels", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state.get("images"):
    st.warning("⚠️ Ve a **Carga de Imágenes** primero.")
    st.stop()

images = st.session_state["images"]

if st.button("🚀 Extraer Variables Estadísticas de Todas las Imágenes", type="primary", use_container_width=True):
    from src.stats_extractor import StatisticalExtractor
    extractor = StatisticalExtractor()
    bar = st.progress(0)
    status = st.empty()

    def cb(cur, tot):
        bar.progress(cur / tot)
        status.text(f"Analizando {cur}/{tot}: {Path(images[cur-1]).name}")

    df = extractor.extract_batch(images, progress_cb=cb)
    st.session_state["stats_df"] = df
    status.success(f"✅ {len(df)} imágenes × {len(df.columns)} variables extraídas")

df = st.session_state.get("stats_df")
if df is None:
    st.info("Haz clic en el botón para comenzar la extracción.")
    st.stop()

num_df = df.select_dtypes(include=[np.number])

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Imágenes analizadas", len(df))
c2.metric("Variables extraídas", len(num_df.columns))
c3.metric("Intensidad media", f"{df.get('mean_intensity', pd.Series([0])).mean():.1f}")
c4.metric("Entropía media", f"{df.get('entropy', pd.Series([0])).mean():.3f}")
c5.metric("Densidad bordes media", f"{df.get('edge_density', pd.Series([0])).mean():.3f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Resumen descriptivo",
    "📈 Distribuciones",
    "🔗 Correlaciones",
    "📦 Box plots",
    "🗺️ Mapa de calor",
])

# ── Tab 1: Descriptive summary ────────────────────────────────────────────────
with tab1:
    key_cols = [c for c in [
        "mean_intensity", "std_intensity", "median_intensity",
        "entropy", "edge_density", "mean_saturation", "colorfulness",
        "texture_contrast", "texture_homogeneity", "cv_intensity",
    ] if c in df.columns]
    summary = df[key_cols].describe().T.round(4)
    summary.insert(0, "Variable", summary.index)
    st.dataframe(summary, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV completo", csv_bytes, "estadisticas.csv", "text/csv")

# ── Tab 2: Distributions ─────────────────────────────────────────────────────
with tab2:
    col_sel = st.selectbox("Variable", list(num_df.columns), key="dist_col")
    series = df[col_sel].dropna()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=col_sel, nbins=25, marginal="box",
                           title=f"Histograma — {col_sel}",
                           color_discrete_sequence=["#2196F3"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.violin(df, y=col_sel, box=True,
                         title=f"Violin — {col_sel}",
                         color_discrete_sequence=["#4CAF50"])
        st.plotly_chart(fig2, use_container_width=True)

    # Stat summary for selected variable
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Media", f"{series.mean():.3f}")
    s2.metric("Mediana", f"{series.median():.3f}")
    s3.metric("Desv. Est.", f"{series.std():.3f}")
    s4.metric("Asimetría", f"{scipy_stats.skew(series):.3f}")
    s5.metric("Curtosis", f"{scipy_stats.kurtosis(series):.3f}")
    s6.metric("CV", f"{series.std() / (series.mean() + 1e-9):.3f}")

# ── Tab 3: Correlations ───────────────────────────────────────────────────────
with tab3:
    key_cols2 = [c for c in [
        "mean_intensity", "std_intensity", "entropy",
        "edge_density", "mean_saturation", "colorfulness",
        "texture_contrast", "texture_homogeneity", "texture_energy",
        "mean_hue", "cv_intensity", "spatial_variance",
    ] if c in df.columns]
    corr = df[key_cols2].corr()
    fig = px.imshow(
        corr, title="Matriz de Correlación de Pearson",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Box plots ──────────────────────────────────────────────────────────
with tab4:
    multi = st.multiselect(
        "Variables a comparar (normalizadas 0-1)",
        list(num_df.columns),
        default=[c for c in ["mean_intensity", "entropy", "edge_density",
                              "mean_saturation", "colorfulness"] if c in num_df.columns],
    )
    if multi:
        norm = num_df[multi].copy()
        for c in multi:
            rng = norm[c].max() - norm[c].min()
            norm[c] = (norm[c] - norm[c].min()) / (rng + 1e-9)
        melted = norm.melt(var_name="Variable", value_name="Valor norm.")
        fig = px.box(melted, x="Variable", y="Valor norm.",
                     color="Variable", title="Box Plots de Variables Estadísticas (norm. 0-1)",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Heatmap ────────────────────────────────────────────────────────────
with tab5:
    top_n = st.slider("Top N variables (por varianza)", 5, min(40, len(num_df.columns)), 15)
    top_cols = num_df.var().sort_values(ascending=False).head(top_n).index.tolist()

    heat = num_df[top_cols].copy()
    for c in top_cols:
        rng = heat[c].max() - heat[c].min()
        heat[c] = (heat[c] - heat[c].min()) / (rng + 1e-9)

    img_names = df["image_name"].tolist() if "image_name" in df.columns else [f"img_{i}" for i in range(len(df))]
    # Truncate names for display
    short_names = [n[:20] for n in img_names]

    fig = px.imshow(
        heat.T, x=short_names, y=top_cols,
        title="Mapa de Calor — Variables por Imagen (normalizado)",
        color_continuous_scale="Viridis", aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ── Cluster coloring (if available) ──────────────────────────────────────────
if st.session_state.get("cluster_labels") is not None and len(st.session_state["cluster_labels"]) == len(df):
    st.divider()
    st.subheader("Variables por Cluster (SigLIP2 + FAISS)")
    labels = st.session_state["cluster_labels"]
    df_c = df.copy()
    df_c["cluster"] = labels.astype(str)

    var_cluster = st.selectbox("Variable a comparar por cluster", key_cols, key="cluster_var")
    fig = px.box(df_c, x="cluster", y=var_cluster, color="cluster",
                 title=f"{var_cluster} por cluster de similaridad",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
