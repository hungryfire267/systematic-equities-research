import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimiser:
    def __init__(self, predicted_df, returns_df, covariance_window=63, weight_bound=0.10):
        self.selected_df = predicted_df.copy()
        self.returns_df = returns_df.copy()
        self.covariance_window = covariance_window
        self.weight_bound = weight_bound

        self.selected_df["Date"] = pd.to_datetime(self.selected_df["Date"])

        if "Date" in self.returns_df.columns:
            self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
            self.returns_df = self.returns_df.set_index("Date")

        self.returns_df.index = pd.to_datetime(self.returns_df.index)

    def get_mu_cov(self, date):
        day_df = self.selected_df[self.selected_df["Date"] == date].copy()
        tickers = day_df["Ticker"].tolist()

        mu = day_df.set_index("Ticker").loc[tickers, "prediction"].values

        cov = (
            self.returns_df.loc[:date, tickers]
            .tail(self.covariance_window)
            .cov()
            .values
        )

        return tickers, mu, cov

    def neg_sharpe(self, weights, mu, cov):
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)

        if port_vol == 0:
            return 1e9

        return -port_return / port_vol

    def optimise_one_date(self, date):
        tickers, mu, cov = self.get_mu_cov(date)
        n = len(tickers)

        x0 = np.ones(n) / n

        bounds = [(-self.weight_bound, self.weight_bound)] * n

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w)}
        ]

        result = minimize(
            self.neg_sharpe,
            x0=x0,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise ValueError(result.message)

        return pd.DataFrame({
            "Date": date,
            "Ticker": tickers,
            "weight": result.x
        })

    def run_data(self):
        rows = []

        for date in sorted(self.selected_df["Date"].unique()):
            try:
                weights = self.optimise_one_date(date)
                rows.append(weights)
            except Exception as e:
                print(f"{date}: optimisation failed: {e}")

        if not rows:
            raise ValueError("No successful optimisations.")

        return pd.concat(rows, ignore_index=True)