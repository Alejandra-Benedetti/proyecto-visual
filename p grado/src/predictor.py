import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import MODELS_DIR, XGBOOST_PARAMS


EXCLUDE_COLS = {"image_name", "width", "height"}


class PredictiveModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self._model_file = MODELS_DIR / "xgb_model.pkl"
        self._scaler_file = MODELS_DIR / "xgb_scaler.pkl"
        self._features_file = MODELS_DIR / "xgb_features.json"
        self.is_trained = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        num = df.select_dtypes(include=[np.number])
        drop = [c for c in EXCLUDE_COLS if c in num.columns]
        return num.drop(columns=drop).fillna(0)

    def train(self, features_df: pd.DataFrame, target: pd.Series) -> dict:
        import xgboost as xgb
        import json

        X = self._prepare(features_df)
        # Remove target column from features if it leaked in
        if target.name in X.columns:
            X = X.drop(columns=[target.name])
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        n = len(X_scaled)

        if n < 5:
            raise ValueError(
                f"Se necesitan al menos 5 imágenes con estadísticas para entrenar. "
                f"Actualmente hay {n}. Carga más imágenes y extrae sus estadísticas."
            )

        # Garantizar al menos 1 muestra en test
        test_size = max(1, int(n * 0.2))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, target.values, test_size=test_size, random_state=42
        )

        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        self.model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = self.model.predict(X_te)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "r2": float(r2_score(y_te, y_pred)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        self.is_trained = True
        with open(self._model_file, "wb") as f:
            pickle.dump(self.model, f)
        with open(self._scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)
        self._features_file.write_text(json.dumps(self.feature_names), encoding="utf-8")
        return metrics

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df[self.feature_names].fillna(0)
        return self.model.predict(self.scaler.transform(X))

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

    def load(self) -> bool:
        import json
        if self._model_file.exists():
            with open(self._model_file, "rb") as f:
                self.model = pickle.load(f)
            with open(self._scaler_file, "rb") as f:
                self.scaler = pickle.load(f)
            self.feature_names = json.loads(
                self._features_file.read_text(encoding="utf-8")
            )
            self.is_trained = True
            return True
        return False
