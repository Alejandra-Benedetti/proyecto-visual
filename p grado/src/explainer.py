import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelExplainer:
    def __init__(self, xgb_model, feature_names: list[str]):
        self.model = xgb_model
        self.feature_names = feature_names
        self._explainer = None
        self.shap_values: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ModelExplainer":
        import shap
        self._explainer = shap.TreeExplainer(self.model)
        self.shap_values = self._explainer.shap_values(X)
        return self

    def importance_df(self) -> pd.DataFrame:
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def plot_summary(self, X: np.ndarray) -> plt.Figure:
        import shap
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            pd.DataFrame(X, columns=self.feature_names),
            show=False,
        )
        return fig

    def plot_bar(self) -> plt.Figure:
        import shap
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )
        return fig
