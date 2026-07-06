import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd


class WalkForwardLSTMValidator:
    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        model: Any | Callable[[], Any],
        validation_start: str,
        validation_end: str,
        rebalance_date: int,
        min_train_size: int = 30000,
        sequence_length: int = 20,
        fit_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ):
        self.feature_matrix = feature_matrix.copy()
        self.feature_matrix["Date"] = pd.to_datetime(self.feature_matrix["Date"])

        self.target_col = "future_return_5d"
        self.feature_cols = self.feature_matrix.columns[2:].drop(self.target_col)
        self.model = model

        if rebalance_date not in range(1, 6):
            raise ValueError("Rebalance date must be a weekday")
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        self.rebalance_date = rebalance_date
        self.min_train_size = min_train_size
        self.sequence_length = sequence_length
        self.fit_kwargs = fit_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}

        self.validation_start = pd.to_datetime(validation_start)
        self.validation_end = pd.to_datetime(validation_end)

    def get_rebalance_dates(self):
        mask = (
            (self.feature_matrix["Date"].dt.weekday == self.rebalance_date)
            & (self.feature_matrix["Date"] >= self.validation_start)
            & (self.feature_matrix["Date"] <= self.validation_end)
        )

        dates = self.feature_matrix.loc[mask, "Date"].unique()
        return np.sort(dates)

    def _get_model(self):
        if callable(self.model) and not hasattr(self.model, "fit"):
            return self.model()
        return self.model

    def _make_sequences(
        self,
        df: pd.DataFrame,
        target_dates: set[pd.Timestamp] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        feature_cols = list(self.feature_cols)
        cols = ["Date", "Ticker", self.target_col] + feature_cols
        df = df[cols].replace([np.inf, -np.inf], np.nan).sort_values(["Ticker", "Date"])

        X, y, meta = [], [], []
        for ticker, ticker_df in df.groupby("Ticker", sort=False):
            ticker_df = ticker_df.reset_index(drop=True)
            features = ticker_df[feature_cols].to_numpy(dtype=np.float32)
            targets = ticker_df[self.target_col].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(ticker_df["Date"])

            for idx in range(self.sequence_length - 1, len(ticker_df)):
                date = pd.Timestamp(dates.iloc[idx])
                if target_dates is not None and date not in target_dates:
                    continue
                if np.isnan(targets[idx]):
                    continue

                window = features[idx - self.sequence_length + 1 : idx + 1]
                if np.isnan(window).any():
                    continue

                X.append(window)
                y.append(targets[idx])
                meta.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        self.target_col: targets[idx],
                    }
                )

        if not X:
            empty_X = np.empty((0, self.sequence_length, len(feature_cols)), dtype=np.float32)
            return empty_X, np.empty((0,), dtype=np.float32), pd.DataFrame(meta)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            pd.DataFrame(meta),
        )

    def run_data(self):
        start_date = self.feature_matrix["Date"].min()
        adjusted_start_date = start_date + datetime.timedelta(days=365)

        self.feature_matrix = self.feature_matrix[
            self.feature_matrix["Date"] >= adjusted_start_date
        ].copy()
        dates = self.get_rebalance_dates()

        unique_dates = np.sort(self.feature_matrix["Date"].unique())

        predictions = []
        last_X_test = np.empty((0, self.sequence_length, len(self.feature_cols)), dtype=np.float32)
        for date in dates:
            horizon = 5

            date_idx = np.where(unique_dates == date)[0][0]

            purge_cutoff_idx = max(0, date_idx - horizon)
            purge_cutoff_date = unique_dates[purge_cutoff_idx]

            train_df = self.feature_matrix[
                self.feature_matrix["Date"] <= purge_cutoff_date
            ].copy()

            test_history_df = self.feature_matrix[
                self.feature_matrix["Date"] <= date
            ].copy()

            X_train, y_train, _ = self._make_sequences(train_df)
            X_test, _, test_meta = self._make_sequences(
                test_history_df,
                target_dates={pd.Timestamp(date)},
            )

            if X_train.shape[0] < self.min_train_size or X_test.shape[0] == 0:
                continue

            model = self._get_model()
            model.fit(X_train, y_train, **self.fit_kwargs)

            output = test_meta[["Date", "Ticker", self.target_col]].copy()
            preds = model.predict(X_test, **self.predict_kwargs)
            output["prediction"] = np.asarray(preds).reshape(-1)
            output["model_name"] = model.__class__.__name__

            predictions.append(output)
            last_X_test = X_test

        if not predictions:
            empty_predictions = pd.DataFrame(
                columns=["Date", "Ticker", self.target_col, "prediction", "model_name"]
            )
            return last_X_test, empty_predictions

        final_df = pd.concat(predictions, ignore_index=True)
        return last_X_test, final_df
