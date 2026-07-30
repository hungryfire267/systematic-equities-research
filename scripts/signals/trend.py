import numpy as np
import os
import pandas as pd
from pathlib import Path

from scripts.signals.utils import cross_sectional_ranking


BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

company_paths_dict = {
    "prices": os.path.join(COMPANIES_DIR, "prices.parquet")
}


class Trends:
    def __init__(self, rolling_window_list):
        self.rolling_window_list = rolling_window_list

        prices_df = (
            pd.read_parquet(company_paths_dict["prices"])
            .set_index("Date")
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        # Log prices are only defined for strictly positive prices.
        prices_df = prices_df.where(prices_df > 0)

        self.prices_df = prices_df
        self.log_prices_df = np.log(prices_df).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def rolling_trend_r2(
        self,
        window: int,
        annualize: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        n = len(self.log_prices_df)

        x = np.broadcast_to(
            np.arange(n, dtype=float)[:, None],
            self.log_prices_df.shape,
        )

        X = pd.DataFrame(
            x,
            index=self.log_prices_df.index,
            columns=self.log_prices_df.columns,
        )

        valid_mask = self.log_prices_df.notna()
        X = X.where(valid_mask)

        rolling_x = X.rolling(
            window=window,
            min_periods=window,
        )

        rolling_y = self.log_prices_df.rolling(
            window=window,
            min_periods=window,
        )

        mean_x = rolling_x.mean()
        mean_y = rolling_y.mean()

        covariance = (
            (X * self.log_prices_df)
            .rolling(window, min_periods=window)
            .mean()
            - mean_x * mean_y
        )

        variance_x = (
            (X ** 2)
            .rolling(window, min_periods=window)
            .mean()
            - mean_x ** 2
        )

        variance_y = (
            (self.log_prices_df ** 2)
            .rolling(window, min_periods=window)
            .mean()
            - mean_y ** 2
        )

        # Floating-point arithmetic can produce tiny negative variances.
        variance_x = variance_x.clip(lower=0)
        variance_y = variance_y.clip(lower=0)

        # Prevent division by zero or near-zero variance.
        variance_x = variance_x.mask(
            np.isclose(variance_x, 0.0),
            np.nan,
        )

        variance_y = variance_y.mask(
            np.isclose(variance_y, 0.0),
            np.nan,
        )

        slope = covariance.div(variance_x)

        r2 = covariance.pow(2).div(
            variance_x * variance_y
        )

        slope = slope.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        r2 = (
            r2
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0, upper=1)
        )

        if annualize:
            slope = slope * 252

        return slope, r2

    def run_data(self) -> dict[str, dict[int, pd.DataFrame]]:
        trend_dict = {}
        r2_dict = {}

        for window in self.rolling_window_list:
            slope, r2 = self.rolling_trend_r2(window)

            slope_rank_df = (
                cross_sectional_ranking(
                    slope,
                    higher_is_better=True,
                )
                .reset_index()
            )

            r2_rank_df = (
                cross_sectional_ranking(
                    r2,
                    higher_is_better=True,
                )
                .reset_index()
            )

            slope_rank_df["Date"] = pd.to_datetime(
                slope_rank_df["Date"]
            )

            r2_rank_df["Date"] = pd.to_datetime(
                r2_rank_df["Date"]
            )

            trend_dict[window] = slope_rank_df
            r2_dict[window] = r2_rank_df

        return {
            "trend": trend_dict,
            "r2": r2_dict,
        }