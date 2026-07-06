

import numpy as np 
import pandas as pd 

from xgboost import XGBRegressor


class XGBoostRegressionModel: 
    def __init__(self, params: dict | None = None):
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.1, 
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1, 
            "tree_method": "hist"
        }
        
        if params is not None:
            default_params.update(params)

        self.model = XGBRegressor(**default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series): 
        self.model.fit(X, y)
        return self
        
        
    def predict(self, X) -> pd.Series: 
        y_pred = self.model.predict(X)
        return y_pred
    
    
    