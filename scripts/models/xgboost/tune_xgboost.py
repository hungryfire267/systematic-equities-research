import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from scripts.models.xgboost.xgboost_model import XGBoostRegressionModel
from scripts.models.walk_forward import WalkForwardValidator

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

XGBOOST_DIR = RESULTS_DIR / "xgboost_model"
XGBOOST_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTuner: 
    def __init__(self, n_iter, random_state, feature_matrix_df, data_type): 
        self.rng = random.Random(random_state)
        self.n_iter = n_iter
        self.feature_matrix_df = feature_matrix_df
        self.data_type = data_type
        
        self.target_col = "future_return_5d"
        self.feature_cols = self.feature_matrix_df.columns[2:].drop("future_return_5d")
        
    def get_param_grids(self): 
        param_grids = { 
            "learning_rate": self.rng.choice([0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3]), 
            "max_depth": int(self.rng.choice([3, 4, 5, 6, 7])),
            "n_estimators": int(self.rng.choice([100, 200, 300, 500])),
            "min_child_weight": int(self.rng.choice([10, 20, 50, 100]))
        }
        
        return param_grids
    
    def mean_ic(self, preds):
        daily_ic = preds.groupby("Date").apply(
            lambda x: x["prediction"].corr(
                x[self.target_col],
                method="spearman"
            ), 
            include_groups = False
        )

        return daily_ic.dropna().mean()
    
    def run_data(self): 
        results = []
        for i in range(self.n_iter):
            params = self.get_param_grids()
            
            model = XGBoostRegressionModel(params=params)
            
            wf = WalkForwardValidator(
                self.feature_matrix_df, 
                model, 
                validation_start = "2023-07-01",
                validation_end="2025-06-30",
                rebalance_date=1,
                min_train_size=25000
            )
            
            X_test, prediction_outputs = wf.run_data()
            
            score = self.mean_ic(prediction_outputs)
            
            row = {
                **params, 
                "mean_ic": score
            }
            
            results.append(row)
            
            print(f"{i + 1}/{self.n_iter}: IC = {score:.5f}")
            
        results_df = pd.DataFrame(results).sort_values("mean_ic", ascending=False).reset_index(drop=True)
        
        best_params = results_df.iloc[0].drop("mean_ic").to_dict()
        
        results_df.to_csv(
            XGBOOST_DIR / f"random_search{self.data_type}.csv",
            index=False,
        )
        
        with open(os.path.join(XGBOOST_DIR, f"best_params_{self.data_type}.json"), "w") as f: 
            json.dump(best_params, f, indent=4)
        
        