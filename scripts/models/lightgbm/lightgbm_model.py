from lightgbm import LGBMRegressor

import pandas as pd 
import numpy as np


class LightGBMRegressionModel: 
    def __init__(self, params: dict | None = None):
        default_params = {
            "num_leaves": 64, 
            "n_estimators": 200,
            "learning_rate": 0.2, 
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        
        if params is not None:
            default_params.update(params)

        self.model = LGBMRegressor(**default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series): 
        self.model.fit(X, y)
        return self
        
        
    def predict(self, X) -> pd.Series: 
        y_pred = self.model.predict(X)
        return y_pred