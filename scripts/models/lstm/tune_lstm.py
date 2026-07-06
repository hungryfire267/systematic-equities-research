import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.models.lstm.lstm_model import LSTMRegressionModel
from scripts.models.walk_forward_lstm import WalkForwardLSTMValidator

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LSTM_DIR = RESULTS_DIR / "lstm_model"
LSTM_DIR.mkdir(parents=True, exist_ok=True)


class LSTMTuner:
    def __init__(self, n_iter, random_state, feature_matrix_df):
        self.rng = random.Random(random_state)
        self.random_state = random_state
        self.n_iter = n_iter
        self.feature_matrix_df = feature_matrix_df

        self.target_col = "future_return_5d"
        self.feature_cols = self.feature_matrix_df.columns[2:].drop("future_return_5d")

    def get_param_grids(self):
        param_grids = {
            "dropout_rate": self.rng.choice([0.1, 0.2, 0.3]),
            "hidden_dim_1": self.rng.choice([32, 64, 128]),
            "hidden_dim_2": self.rng.choice([16, 32, 64]),
            "learning_rate": self.rng.choice([0.0005, 0.001, 0.002]),
            "epochs": self.rng.choice([10, 20, 30]),
            "early_stopping_patience": self.rng.choice([2, 3]),
            "sequence_length": self.rng.choice([10, 20, 30]),
        }

        return param_grids

    def build_model(self, params):
        def model_factory():
            tf.keras.utils.set_random_seed(self.random_state)

            model = LSTMRegressionModel(
                dropout_rate=params["dropout_rate"],
                hidden_dim_1=params["hidden_dim_1"],
                hidden_dim_2=params["hidden_dim_2"],
                output_dim=1,
            )
            return model

        return model_factory

    def mean_ic(self, preds):
        if preds.empty:
            return np.nan

        daily_ic = preds.groupby("Date").apply(
            lambda x: x["prediction"].corr(
                x[self.target_col],
                method="spearman",
            )
        )

        return daily_ic.dropna().mean()

    def run_data(self):
        results = []
        for i in range(self.n_iter):
            params = self.get_param_grids()

            model = self.build_model(params)

            wf = WalkForwardLSTMValidator(
                self.feature_matrix_df,
                model,
                validation_start="2023-07-01",
                validation_end="2025-06-30",
                rebalance_date=1,
                min_train_size=25000,
                sequence_length=params["sequence_length"],
                training_mode="listnet",
                fit_kwargs={
                    "epochs": params["epochs"],
                    "learning_rate": params["learning_rate"],
                    "verbose": 0,
                    "early_stopping_patience": params["early_stopping_patience"],
                    "early_stopping_min_delta": 1e-4,
                    "restore_best_weights": True,
                },
                predict_kwargs={"verbose": 0},
            )

            X_test, prediction_outputs = wf.run_data()

            score = self.mean_ic(prediction_outputs)

            row = {
                **params,
                "mean_ic": score,
            }

            results.append(row)

            print(f"{i + 1}/{self.n_iter}: IC = {score:.5f}")

        results_df = (
            pd.DataFrame(results)
            .sort_values("mean_ic", ascending=False)
            .reset_index(drop=True)
        )

        best_params = results_df.iloc[0].drop("mean_ic").to_dict()

        results_df.to_csv(
            LSTM_DIR / "random_search.csv",
            index=False,
        )

        with open(os.path.join(LSTM_DIR, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
