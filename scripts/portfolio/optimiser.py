import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimiser:
    def __init__(
        self,
        selected_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        covariance_window: int = 63,
        weight_bound: float = 0.10,
        ridge: float = 1e-4,
    ):
        self.selected_df = selected_df.copy()
        self.returns_df = returns_df.copy()
        self.covariance_window = covariance_window
        self.weight_bound = weight_bound
        self.ridge = ridge

        self.selected_df["Date"] = pd.to_datetime(self.selected_df["Date"])

        if "Date" in self.returns_df.columns:
            self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
            self.returns_df = self.returns_df.set_index("Date")

        self.returns_df.index = pd.to_datetime(self.returns_df.index)

    def neg_sharpe(self, weights, mu, cov):
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        port_vol = max(port_vol, 1e-8)
        return -(port_return / port_vol)

    def optimise_side(self, tickers, mu, cov, side):
        n = len(tickers)

        if n == 0:
            raise ValueError(f"No tickers for {side}")

        min_required_bound = 1 / n
        effective_bound = max(self.weight_bound, min_required_bound)

        x0 = np.ones(n) / n
        bounds = [(0.0, effective_bound)] * n

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        if side == "short":
            mu = -mu

        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = cov + np.eye(n) * self.ridge

        result = minimize(
            self.neg_sharpe,
            x0=x0,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print(f"{side} optimisation failed: {result.message}")
            weights = x0
        else:
            weights = result.x

        weights = np.clip(weights, 0.0, effective_bound)

        if weights.sum() == 0:
            weights = x0
        else:
            weights = weights / weights.sum()

        if side == "short":
            weights = -weights

        return pd.DataFrame({
            "Ticker": tickers,
            "weight": weights,
            "side": side,
        })

    def optimise_one_date(self, date):
        date = pd.Timestamp(date)

        day_df = self.selected_df[self.selected_df["Date"] == date].copy()
        portfolio_parts = []

        for side in ["long", "short"]:
            side_df = day_df[day_df["side"] == side].copy()

            tickers = side_df["Ticker"].tolist()

            mu = (
                side_df
                .set_index("Ticker")
                .loc[tickers, "prediction"]
                .astype(float)
                .values
            )

            cov = (
                self.returns_df
                .loc[:date, tickers]
                .tail(self.covariance_window)
                .cov()
                .values
            )

            weights_df = self.optimise_side(
                tickers=tickers,
                mu=mu,
                cov=cov,
                side=side,
            )

            weights_df["Date"] = date
            portfolio_parts.append(weights_df)

        return pd.concat(portfolio_parts, ignore_index=True)

    def run_data(self):
        portfolio_list = []

        dates = sorted(self.selected_df["Date"].dropna().unique())

        for date in dates:
            try:
                weights_df = self.optimise_one_date(date)
                portfolio_list.append(weights_df)
            except Exception as e:
                print(f"{date}: optimisation skipped ({e})")

        return pd.concat(portfolio_list, ignore_index=True)