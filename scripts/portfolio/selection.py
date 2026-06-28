import json
import numpy as np
import os 
import pandas as pd
from pathlib import Path

from scripts.models.walk_forward import WalkForwardValidator

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" 
PROCESSED_FEATURE_DIR = PROCESSED_DIR / "features"

LIGHTGBM_MODEL_DIR = BASE_DIR / "results" / "lightgbm_model"

class TopBottom20Selector: 
    def __init__(self, model_class):
        with open(os.path.join(LIGHTGBM_MODEL_DIR, "best_params.json"), "r") as f:
            self.best_params = json.load(f)
        
        self.feature_matrix = pd.read_parquet(os.path.join(PROCESSED_FEATURE_DIR, "feature_matrix_stock.parquet"))
        self.model = model_class(params = self.best_params)
        
    def fit_model(self): 
        wf = WalkForwardValidator(
            feature_matrix=self.feature_matrix,
            model=self.model,
            validation_start="2025-07-01",
            validation_end="2026-12-31",
            rebalance_date=1,
            min_train_size=25000,
        )

        X_test, test_preds = wf.run_data()
        print(X_test)
        return X_test

