import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimiser:
    def __init__(
        self,
        selected_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        covariance_window: int = 63,
        minimum_observations: int = 40,
        weight_bound: float = 0.10,
        ridge: float = 1e-4,
    ):
        self.selected_df = selected_df.copy()
        self.returns_df = returns_df.copy()

        self.covariance_window = covariance_window
        self.minimum_observations = minimum_observations
        self.weight_bound = weight_bound
        self.ridge = ridge

        required_columns = {
            "Date",
            "Ticker",
            "side",
            "prediction",
        }

        missing = required_columns.difference(
            self.selected_df.columns
        )

        if missing:
            raise ValueError(
                f"selected_df is missing columns: {sorted(missing)}"
            )

        self.selected_df["Date"] = pd.to_datetime(
            self.selected_df["Date"]
        )

        if "Date" in self.returns_df.columns:
            self.returns_df["Date"] = pd.to_datetime(
                self.returns_df["Date"]
            )
            self.returns_df = self.returns_df.set_index("Date")

        self.returns_df.index = pd.to_datetime(
            self.returns_df.index
        )

        self.returns_df = (
            self.returns_df
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .sort_index()
        )

    @staticmethod
    def neg_sharpe(
        weights: np.ndarray,
        mu: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        expected_return = float(weights @ mu)

        variance = float(
            weights @ covariance @ weights
        )

        volatility = np.sqrt(max(variance, 1e-12))

        return -(expected_return / volatility)

    def prepare_inputs(
        self,
        date: pd.Timestamp,
        side_df: pd.DataFrame,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:

        requested_tickers = side_df["Ticker"].tolist()

        missing_tickers = [
            ticker
            for ticker in requested_tickers
            if ticker not in self.returns_df.columns
        ]

        if missing_tickers:
            print(
                f"{date.date()}: missing return history for "
                f"{missing_tickers}"
            )

        tickers = [
            ticker
            for ticker in requested_tickers
            if ticker in self.returns_df.columns
        ]

        history = (
            self.returns_df.loc[
                self.returns_df.index < date,
                tickers,
            ]
            .tail(self.covariance_window)
        )

        valid_tickers = history.columns[
            history.notna().sum()
            >= self.minimum_observations
        ].tolist()

        if not valid_tickers:
            raise ValueError(
                "No tickers have sufficient return history."
            )

        history = (
            history[valid_tickers]
            .dropna(axis=0, how="any")
        )

        if len(history) < self.minimum_observations:
            raise ValueError(
                "Insufficient common observations for covariance: "
                f"{len(history)}"
            )

        prediction_series = (
            side_df
            .drop_duplicates("Ticker", keep="last")
            .set_index("Ticker")["prediction"]
            .astype(float)
        )

        mu = prediction_series.loc[
            valid_tickers
        ].to_numpy()

        covariance = history.cov().to_numpy()

        covariance = (
            covariance + covariance.T
        ) / 2

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.maximum(
            eigenvalues,
            self.ridge,
        )

        covariance = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.T
        )

        return valid_tickers, mu, covariance

    def optimise_side(
        self,
        tickers: list[str],
        mu: np.ndarray,
        covariance: np.ndarray,
        side: str,
    ) -> pd.DataFrame:

        n = len(tickers)

        if n == 0:
            raise ValueError(f"No tickers selected for {side}.")

        if n * self.weight_bound < 1:
            raise ValueError(
                f"{side} has {n} tickers, which is insufficient "
                f"for a strict {self.weight_bound:.0%} weight cap. "
                f"At least {int(np.ceil(1 / self.weight_bound))} "
                "tickers are required."
            )

        objective_mu = (
            mu if side == "long" else -mu
        )

        initial_weights = np.full(n, 1 / n)

        bounds = [
            (0.0, self.weight_bound)
            for _ in range(n)
        ]

        constraints = [{
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
        }]

        result = minimize(
            self.neg_sharpe,
            x0=initial_weights,
            args=(objective_mu, covariance),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-9,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"{side} optimisation failed: "
                f"{result.message}"
            )

        weights = result.x

        if not np.isclose(weights.sum(), 1.0):
            raise RuntimeError(
                f"{side} weights do not sum to one: "
                f"{weights.sum():.8f}"
            )

        if side == "short":
            weights = -weights

        return pd.DataFrame({
            "Ticker": tickers,
            "weight": weights,
            "side": side,
        })

    def optimise_one_date(
        self,
        date: pd.Timestamp,
    ) -> pd.DataFrame:

        date = pd.Timestamp(date)

        day_df = self.selected_df.loc[
            self.selected_df["Date"] == date
        ].copy()

        portfolio_parts = []

        for side in ("long", "short"):
            side_df = day_df.loc[
                day_df["side"] == side
            ].copy()

            if side_df.empty:
                raise ValueError(
                    f"No {side} stocks selected."
                )

            tickers, mu, covariance = self.prepare_inputs(
                date=date,
                side_df=side_df,
            )

            weights_df = self.optimise_side(
                tickers=tickers,
                mu=mu,
                covariance=covariance,
                side=side,
            )

            weights_df["Date"] = date
            portfolio_parts.append(weights_df)

        result = pd.concat(
            portfolio_parts,
            ignore_index=True,
        )

        gross_exposure = result["weight"].abs().sum()
        net_exposure = result["weight"].sum()

        if not np.isclose(gross_exposure, 2.0):
            raise RuntimeError(
                f"Unexpected gross exposure: {gross_exposure:.6f}"
            )

        if not np.isclose(net_exposure, 0.0):
            raise RuntimeError(
                f"Unexpected net exposure: {net_exposure:.6f}"
            )

        return result

    def run_data(self) -> pd.DataFrame:
        portfolio_list = []
        failed_dates = []

        dates = sorted(
            self.selected_df["Date"]
            .dropna()
            .unique()
        )

        for date in dates:
            try:
                portfolio_list.append(
                    self.optimise_one_date(date)
                )
            except Exception as error:
                failed_dates.append((date, str(error)))
                print(
                    f"{pd.Timestamp(date).date()}: "
                    f"optimisation failed ({error})"
                )

        if not portfolio_list:
            raise RuntimeError(
                "No portfolio dates were successfully optimised."
            )

        if failed_dates:
            print(
                f"Warning: {len(failed_dates)} of "
                f"{len(dates)} rebalance dates failed."
            )

        return (
            pd.concat(portfolio_list, ignore_index=True)
            .sort_values(["Date", "side", "Ticker"])
            .reset_index(drop=True)
        )