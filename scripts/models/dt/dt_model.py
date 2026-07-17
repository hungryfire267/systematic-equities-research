import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeRegressor

class DTRegressionModel: 
    def __init__(self, params: dict | None = None):
        default_params = {
            "criterion": "squared_error",
            "splitter": "best",
            "max_depth": 6,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": None,
            "random_state": 42
        }
        
        if params is not None:
            default_params.update(params)

        self.model = DecisionTreeRegressor(**default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series): 
        self.model.fit(X, y)
        return self
        
        
    def predict(self, X) -> pd.Series: 
        y_pred = self.model.predict(X)
        return y_pred