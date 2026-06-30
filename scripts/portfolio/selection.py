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
        
        int_params = [
            "num_leaves",
            "max_depth",
            "n_estimators",
            "min_child_samples",
        ]

        for p in int_params:
            self.best_params[p] = int(self.best_params[p])
        
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
        return test_preds
    
    def clean_preds(self, test_preds): 
        test_preds = test_preds[["Date", "Ticker", "prediction"]].copy()

        test_preds_wide = test_preds.pivot(
            index="Date", 
            columns="Ticker", 
            values="prediction"
        )

        # Higher prediction = better rank
        test_preds_rank = test_preds_wide.rank(
            axis=1,
            ascending=False,
            method="first"
        )

        return test_preds_rank


    def select_top_bottom(self, test_preds_rank, top_n=20):
        rows = []

        for date, row in test_preds_rank.iterrows():
            row = row.dropna()

            longs = row.nsmallest(top_n).index
            shorts = row.nlargest(top_n).index

            for ticker in longs:
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "side": "long",
                    "rank": row[ticker],
                })

            for ticker in shorts:
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "side": "short",
                    "rank": row[ticker],
                })

        return pd.DataFrame(rows)


    def run_data(self): 
        test_preds = self.fit_model()
        test_preds_rank = self.clean_preds(test_preds)

        selected_df = self.select_top_bottom(test_preds_rank, top_n=20)

        selected_df = selected_df.merge(
            test_preds[["Date", "Ticker", "prediction", "future_return_5d"]],
            on=["Date", "Ticker"],
            how="left"
        )

        return test_preds, test_preds_rank, selected_df

