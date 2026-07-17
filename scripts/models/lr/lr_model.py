import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    def __init__(self, params: dict | None = None):
        default_params = {
            "fit_intercept": True,
            "copy_X": True,
            "n_jobs": -1,
            "positive": False
        }

        if params is not None:
            default_params.update(params)

        self.model = LinearRegression(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        complete_mask = X.notna().all(axis=1) & y.notna()

        X_clean = X.loc[complete_mask]
        y_clean = y.loc[complete_mask]

        if X_clean.empty:
            raise ValueError(
                "No complete observations available for training."
            )

        self.model.fit(X_clean, y_clean)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.replace([np.inf, -np.inf], np.nan)

        complete_mask = X.notna().all(axis=1)

        predictions = pd.Series(
            np.nan,
            index=X.index,
            name="prediction",
            dtype=float
        )

        if complete_mask.any():
            predictions.loc[complete_mask] = self.model.predict(
                X.loc[complete_mask]
            )

        return predictions