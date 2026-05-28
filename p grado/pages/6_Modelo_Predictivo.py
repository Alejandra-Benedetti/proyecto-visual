import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modelo Predictivo", layout="wide")
st.title("🌲 Modelo Predictivo — XGBoost + SHAP")

for _k, _v in [("images", []), ("stats_df", None), ("predictor", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

df = st.session_state.get("stats_df")
if df is None:
    st.warning("⚠️ Primero extrae las estadísticas en **Análisis Estadístico**.")
    st.stop()

num_df = df.select_dtypes(include=[np.number])

tab_train, tab_shap, tab_pred = st.tabs(["🎯 Entrenamiento", "🧩 SHAP Explicabilidad", "🔮 Predicciones"])

# ── Tab 1: Training ───────────────────────────────────────────────────────────
with tab_train:
    st.subheader("Variable objetivo")
    target_opt = st.radio("Elige la variable objetivo", [
        "Entropía (proxy de complejidad visual)",
        "Intensidad media (luminosidad)",
        "Colorfulness (riqueza de color)",
        "Densidad de bordes (detalle)",
        "Cargar datos externos (CSV/Excel con ventas)",
    ])

    target_series = None
    target_name = ""

    if "Entropía" in target_opt and "entropy" in df.columns:
        target_series, target_name = df["entropy"], "entropy"
    elif "Intensidad" in target_opt and "mean_intensity" in df.columns:
        target_series, target_name = df["mean_intensity"], "mean_intensity"
    elif "Colorfulness" in target_opt and "colorfulness" in df.columns:
        target_series, target_name = df["colorfulness"], "colorfulness"
    elif "bordes" in target_opt and "edge_density" in df.columns:
        target_series, target_name = df["edge_density"], "edge_density"
    else:
        up = st.file_uploader("Sube CSV o Excel con la columna objetivo", type=["csv", "xlsx"])
        if up:
            ext_df = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up)
            st.dataframe(ext_df.head(), use_container_width=True)
            target_col = st.selectbox("Columna objetivo", ext_df.columns.tolist())
            if len(ext_df) >= len(df):
                target_series = ext_df[target_col].iloc[:len(df)].reset_index(drop=True)
                target_name = target_col
            else:
                st.error(f"El archivo tiene {len(ext_df)} filas pero hay {len(df)} imágenes")

    if target_series is not None:
        st.info(f"Objetivo: **{target_name}** — rango [{target_series.min():.3f}, {target_series.max():.3f}], media {target_series.mean():.3f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_est = st.slider("N° árboles", 50, 500, 100, 50)
        with col2:
            depth = st.slider("Profundidad", 2, 8, 4)
        with col3:
            lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

        if st.button("🚀 Entrenar XGBoost", type="primary"):
            try:
                from src.predictor import PredictiveModel
                from config import XGBOOST_PARAMS
                XGBOOST_PARAMS.update({"n_estimators": n_est, "max_depth": depth, "learning_rate": lr})

                predictor = PredictiveModel()
                with st.spinner("Entrenando..."):
                    metrics = predictor.train(df, target_series)

                st.session_state["predictor"] = predictor
            except ImportError as e:
                st.error(f"⏳ Dependencia pendiente: `{e}`. Espera y reintenta.")
                st.stop()
            except ValueError as e:
                st.error(f"⚠️ {e}")
                st.stop()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['rmse']:.4f}")
            c2.metric("MAE", f"{metrics['mae']:.4f}")
            c3.metric("R²", f"{metrics['r2']:.4f}")
            c4.metric("Train/Test", f"{metrics['n_train']}/{metrics['n_test']}")

            # Feature importance plot
            st.subheader("Importancia de Variables (XGBoost)")
            imp = predictor.feature_importance().head(20)
            fig = px.bar(
                x=imp.values, y=imp.index, orientation="h",
                title="Top 20 variables más importantes",
                labels={"x": "Importancia", "y": "Variable"},
                color=imp.values, color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: SHAP ───────────────────────────────────────────────────────────────
with tab_shap:
    if not st.session_state.get("predictor"):
        st.info("Entrena el modelo primero.")
    else:
        predictor = st.session_state["predictor"]

        if st.button("🧩 Calcular Valores SHAP", type="primary"):
            import shap
            feat_df = df[predictor.feature_names].fillna(0)
            X_scaled = predictor.scaler.transform(feat_df)

            with st.spinner("Calculando SHAP values..."):
                explainer = shap.TreeExplainer(predictor.model)
                shap_vals = explainer.shap_values(X_scaled)

            # Summary plot
            st.subheader("SHAP Summary Plot")
            fig1, _ = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_scaled,
                              feature_names=predictor.feature_names, show=False)
            st.pyplot(fig1)
            plt.close()

            # Bar plot
            st.subheader("SHAP Feature Importance (media |SHAP|)")
            fig2, _ = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, feature_names=predictor.feature_names,
                              plot_type="bar", show=False)
            st.pyplot(fig2)
            plt.close()

            # Table
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "Variable": predictor.feature_names,
                "SHAP medio |val|": mean_shap,
            }).sort_values("SHAP medio |val|", ascending=False).head(20)
            st.dataframe(shap_df.reset_index(drop=True), use_container_width=True)

# ── Tab 3: Predictions ────────────────────────────────────────────────────────
with tab_pred:
    if not st.session_state.get("predictor"):
        st.info("Entrena el modelo primero.")
    else:
        predictor = st.session_state["predictor"]
        feat_df = df[predictor.feature_names].fillna(0)
        preds = predictor.predict(feat_df)

        names = df["image_name"].tolist() if "image_name" in df.columns else list(range(len(df)))
        pred_df = pd.DataFrame({"imagen": names, "prediccion": preds})

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(pred_df, x="prediccion", nbins=20,
                               title="Distribución de Predicciones",
                               color_discrete_sequence=["#9C27B0"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(
                pred_df.sort_values("prediccion", ascending=False).head(20),
                x="imagen", y="prediccion",
                title="Top 20 imágenes por predicción",
                color="prediccion", color_continuous_scale="Purples",
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            pred_df.sort_values("prediccion", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
