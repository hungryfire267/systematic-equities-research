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
        training_mode: str = "pointwise",
        transfer_learning: bool = False,
        reset_model_every_n_folds: int | None = None,
        verbose: int = 0,
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
        self.training_mode = training_mode
        self.transfer_learning = transfer_learning
        self.reset_model_every_n_folds = reset_model_every_n_folds
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}

        self.validation_start = pd.to_datetime(validation_start)
        self.validation_end = pd.to_datetime(validation_end)

        if self.training_mode not in {"pointwise", "listnet"}:
            raise ValueError("training_mode must be either 'pointwise' or 'listnet'")
        if reset_model_every_n_folds is not None and reset_model_every_n_folds < 1:
            raise ValueError("reset_model_every_n_folds must be at least 1")

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

    def _fit_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_meta: pd.DataFrame,
        fit_kwargs: dict[str, Any],
    ):
        if self.training_mode == "listnet":
            return self._fit_listnet(model, X_train, y_train, train_meta, fit_kwargs)

        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def _fit_listnet(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_meta: pd.DataFrame,
        fit_kwargs: dict[str, Any],
    ):
        import tensorflow as tf

        from scripts.models.lstm.lstm_model import ListNetLoss

        fit_kwargs = fit_kwargs.copy()
        epochs = fit_kwargs.pop("epochs", 1)
        learning_rate = fit_kwargs.pop("learning_rate", 0.001)
        verbose = fit_kwargs.pop("verbose", 0)
        optimizer = fit_kwargs.pop("optimizer", None)
        early_stopping_patience = fit_kwargs.pop("early_stopping_patience", None)
        early_stopping_min_delta = fit_kwargs.pop("early_stopping_min_delta", 0.0)
        restore_best_weights = fit_kwargs.pop("restore_best_weights", True)
        target_transform = fit_kwargs.pop("listnet_target_transform", "raw")
        target_temperature = fit_kwargs.pop("listnet_temperature", 1.0)

        if fit_kwargs:
            unexpected = ", ".join(sorted(fit_kwargs))
            raise ValueError(f"Unsupported ListNet fit_kwargs: {unexpected}")
        if target_transform not in {"raw", "zscore", "rank"}:
            raise ValueError("listnet_target_transform must be 'raw', 'zscore', or 'rank'")
        if target_temperature <= 0:
            raise ValueError("listnet_temperature must be positive")

        optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = ListNetLoss()

        train_groups = (
            train_meta.assign(_row=np.arange(len(train_meta)))
            .groupby("Date", sort=True)["_row"]
            .apply(lambda rows: rows.to_numpy())
            .tolist()
        )
        train_groups = [rows for rows in train_groups if len(rows) > 1]

        best_loss = np.inf
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            epoch_losses = []
            for rows in train_groups:
                X_batch = tf.convert_to_tensor(X_train[rows], dtype=tf.float32)
                y_target = self._transform_listnet_target(
                    y_train[rows],
                    target_transform,
                    target_temperature,
                )
                y_batch = tf.convert_to_tensor(y_target, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    preds = tf.reshape(model(X_batch, training=True), (-1,))
                    loss = loss_fn(
                        tf.expand_dims(y_batch, axis=0),
                        tf.expand_dims(preds, axis=0),
                    )

                grads = tape.gradient(loss, model.trainable_variables)
                grads_and_vars = [
                    (grad, var)
                    for grad, var in zip(grads, model.trainable_variables)
                    if grad is not None
                ]
                optimizer.apply_gradients(grads_and_vars)
                epoch_losses.append(float(loss.numpy()))

            mean_loss = np.mean(epoch_losses) if epoch_losses else np.nan
            if verbose:
                print(f"ListNet epoch {epoch + 1}/{epochs}: loss = {mean_loss:.6f}")

            if early_stopping_patience is None or np.isnan(mean_loss):
                continue

            if mean_loss < best_loss - early_stopping_min_delta:
                best_loss = mean_loss
                best_weights = model.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                if restore_best_weights and best_weights is not None:
                    model.set_weights(best_weights)
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                break

        return model

    def _transform_listnet_target(
        self,
        y: np.ndarray,
        target_transform: str,
        target_temperature: float,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)

        if target_transform == "zscore":
            std = y.std()
            if std > 1e-8:
                y = (y - y.mean()) / std
            else:
                y = y - y.mean()
        elif target_transform == "rank":
            y = pd.Series(y).rank(method="average", pct=True).to_numpy(dtype=np.float32)
            y = y - 0.5

        return y / target_temperature

    def run_data(self):
        start_date = self.feature_matrix["Date"].min()
        adjusted_start_date = start_date + datetime.timedelta(days=365)

        self.feature_matrix = self.feature_matrix[
            self.feature_matrix["Date"] >= adjusted_start_date
        ].copy()
        dates = self.get_rebalance_dates()

        unique_dates = np.sort(self.feature_matrix["Date"].unique())

        predictions = []
        model = None
        trained_fold_count = 0
        last_X_test = np.empty((0, self.sequence_length, len(self.feature_cols)), dtype=np.float32)
        for fold_idx, date in enumerate(dates, start=1):
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

            X_train, y_train, train_meta = self._make_sequences(train_df)
            X_test, _, test_meta = self._make_sequences(
                test_history_df,
                target_dates={pd.Timestamp(date)},
            )

            if X_train.shape[0] < self.min_train_size or X_test.shape[0] == 0:
                if self.verbose:
                    print(
                        f"LSTM fold {fold_idx}/{len(dates)} "
                        f"{pd.Timestamp(date).date()}: skipped "
                        f"(train={X_train.shape[0]}, test={X_test.shape[0]})"
                    )
                continue

            reset_due = (
                not self.transfer_learning
                or model is None
                or (
                    self.reset_model_every_n_folds is not None
                    and trained_fold_count % self.reset_model_every_n_folds == 0
                )
            )
            fit_kwargs = self.fit_kwargs.copy()
            if self.transfer_learning and not reset_due:
                fine_tune_epochs = fit_kwargs.pop("fine_tune_epochs", None)
                if fine_tune_epochs is not None:
                    fit_kwargs["epochs"] = fine_tune_epochs

            if self.verbose:
                training_style = "fresh" if reset_due else "fine-tune"
                print(
                    f"LSTM fold {fold_idx}/{len(dates)} "
                    f"{pd.Timestamp(date).date()}: {training_style} training "
                    f"(train={X_train.shape[0]}, test={X_test.shape[0]})"
                )

            if reset_due:
                model = self._get_model()
            self._fit_model(model, X_train, y_train, train_meta, fit_kwargs)
            trained_fold_count += 1

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
