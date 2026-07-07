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

MIN_TRAINED_FOLDS = 40
MIN_COVERAGE = 0.5


def configure_tensorflow_gpu(verbose: int = 1):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            print("TensorFlow GPU: none detected, using CPU")
        return []

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if verbose:
        gpu_names = ", ".join(gpu.name for gpu in gpus)
        print(f"TensorFlow GPU: using {gpu_names}")
    return gpus


class LSTMTuner:
    def __init__(self, n_iter, random_state, feature_matrix_df):
        self.gpus = configure_tensorflow_gpu()
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
            "fine_tune_epochs": self.rng.choice([2, 3, 5]),
            "early_stopping_patience": self.rng.choice([2, 3]),
            "listnet_temperature": self.rng.choice([0.5, 1.0, 2.0]),
            "sequence_length": self.rng.choice([5, 10, 20]),
            "min_train_size": self.rng.choice([3000, 5000, 8000]),
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
                min_train_size=params["min_train_size"],
                sequence_length=params["sequence_length"],
                training_mode="listnet",
                transfer_learning=True,
                reset_model_every_n_folds=26,
                verbose=1,
                fit_kwargs={
                    "epochs": params["epochs"],
                    "fine_tune_epochs": params["fine_tune_epochs"],
                    "learning_rate": params["learning_rate"],
                    "verbose": 1,
                    "early_stopping_patience": params["early_stopping_patience"],
                    "early_stopping_min_delta": 1e-4,
                    "restore_best_weights": True,
                    "listnet_target_transform": "zscore",
                    "listnet_temperature": params["listnet_temperature"],
                },
                predict_kwargs={"verbose": 0},
            )

            X_test, prediction_outputs = wf.run_data()

            raw_score = self.mean_ic(prediction_outputs)
            trained_folds = wf.trained_fold_count
            skipped_folds = wf.skipped_fold_count
            total_folds = trained_folds + skipped_folds
            coverage = trained_folds / total_folds if total_folds else 0.0
            score = raw_score

            if trained_folds < MIN_TRAINED_FOLDS or coverage < MIN_COVERAGE:
                score = np.nan
                print(
                    f"{i + 1}/{self.n_iter}: rejected low-coverage trial "
                    f"(raw_ic={raw_score:.5f}, trained_folds={trained_folds}, "
                    f"coverage={coverage:.1%})"
                )

            if prediction_outputs.empty:
                print(
                    f"{i + 1}/{self.n_iter}: no trained folds "
                    f"(min_train_size={params['min_train_size']}, "
                    f"sequence_length={params['sequence_length']}, "
                    f"skipped_folds={skipped_folds})"
                )

            row = {
                **params,
                "mean_ic": score,
                "raw_mean_ic": raw_score,
                "trained_folds": trained_folds,
                "skipped_folds": skipped_folds,
                "coverage": coverage,
                "skipped_train_too_small": wf.skipped_train_too_small_count,
                "skipped_empty_test": wf.skipped_empty_test_count,
            }

            results.append(row)

            print(
                f"{i + 1}/{self.n_iter}: IC = {score:.5f} "
                f"(raw_ic={raw_score:.5f}, trained_folds={trained_folds}, "
                f"skipped_folds={skipped_folds}, coverage={coverage:.1%})"
            )

        results_df = (
            pd.DataFrame(results)
            .sort_values("mean_ic", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

        valid_results_df = results_df.dropna(subset=["mean_ic"])
        if valid_results_df.empty:
            best_params = {}
        else:
            best_params = valid_results_df.iloc[0].drop("mean_ic").to_dict()

        results_df.to_csv(
            LSTM_DIR / "random_search.csv",
            index=False,
        )

        with open(os.path.join(LSTM_DIR, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
