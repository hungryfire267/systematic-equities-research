import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from scripts.models.lightgbm.lightgbm_model import LightGBMRegressionModel
from scripts.models.walk_forward import WalkForwardValidator

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LIGHTGBM_DIR = RESULTS_DIR / "lightgbm_model"
LIGHTGBM_DIR.mkdir(parents=True, exist_ok=True)

class LightGBMTuner: 
    def __init__(self, n_iter, random_state, feature_matrix_df): 
        self.rng = random.Random(random_state)
        self.n_iter = n_iter
        self.feature_matrix_df = feature_matrix_df
        
        self.feature_cols = self.feature_matrix_df.columns[2:-1]
        self.target_col = self.feature_matrix_df.columns[-1]
        
    def get_param_grids(self): 
        param_grids = { 
            "learning_rate": self.rng.choice([0.01, 0.02, 0.04, 0.08, 0.1]), 
            "num_leaves": self.rng.choice([31, 63, 127]),
            "max_depth": self.rng.choice([5, 7, 9, -1]),
            "n_estimators": self.rng.choice([100, 200, 300, 500]),
            "min_child_samples": self.rng.choice([10, 20, 50, 100])
        }
        
        return param_grids
    
    def mean_ic(self, preds):
        daily_ic = preds.groupby("Date").apply(
            lambda x: x["prediction"].corr(
                x[self.target_col],
                method="spearman"
            )
        )

        return daily_ic.dropna().mean()
    
    def run_data(self): 
        results = []
        for i in range(self.n_iter):
            params = self.get_param_grids()
            
            model = LightGBMRegressionModel(params=params)
            
            wf = WalkForwardValidator(
                self.feature_matrix_df, 
                model, 
                validation_start = "2024-01-01",
                validation_end="2025-12-31",
                rebalance_date=1,
                min_train_size=25000
            )
            
            prediction_outputs = wf.run_data()
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
            LIGHTGBM_DIR / "random_search.csv",
            index=False,
        )
        
        with open(os.path.join(LIGHTGBM_DIR, "best_params.json"), "w") as f: 
            json.dump(best_params, f, indent=4)
        
        