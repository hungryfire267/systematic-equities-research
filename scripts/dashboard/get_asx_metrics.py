import numpy as np
import pandas as pd


class ASXMetrics:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        date_col: str = "Date",
        price_col: str = "Close",
        periods_per_year: int = 52,
        risk_free_rate: float = 0.0
    ):
        self.date_col = date_col
        self.price_col = price_col
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

        self.prices_df = self._prepare_prices(prices_df)
        self.prices = self.prices_df[self.price_col]

    def _prepare_prices(
        self,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        required_cols = {self.date_col, self.price_col}

        if not required_cols.issubset(prices_df.columns):
            raise ValueError(
                f"prices_df must contain {required_cols}. "
                f"Received columns: {list(prices_df.columns)}"
            )

        prepared_df = prices_df[
            [self.date_col, self.price_col]
        ].copy()

        prepared_df[self.date_col] = pd.to_datetime(
            prepared_df[self.date_col]
        )

        prepared_df[self.price_col] = pd.to_numeric(
            prepared_df[self.price_col],
            errors="coerce"
        )

        prepared_df = (
            prepared_df
            .dropna(subset=[self.date_col, self.price_col])
            .drop_duplicates(subset=self.date_col, keep="last")
            .sort_values(self.date_col)
            .set_index(self.date_col)
        )

        if prepared_df.empty:
            raise ValueError("No valid ASX price observations were found.")

        return prepared_df

    def get_daily_returns(self) -> pd.Series:
        returns = self.prices.pct_change().dropna()
        returns.name = "ASX 200 Daily Return"

        return returns

    def get_holding_period_returns(
        self,
        rebalance_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Calculate benchmark returns between consecutive strategy
        rebalance dates.
        """
        dates = pd.DatetimeIndex(
            pd.to_datetime(rebalance_dates)
        ).sort_values().unique()

        if len(dates) < 2:
            raise ValueError(
                "At least two rebalance dates are required."
            )

        aligned_prices = (
            self.prices
            .reindex(self.prices.index.union(dates))
            .sort_index()
            .ffill()
            .reindex(dates)
        )

        returns = aligned_prices.pct_change().dropna()
        returns.name = "ASX 200 Benchmark Return"

        return returns

    def get_cumulative_returns(
        self,
        returns: pd.Series
    ) -> pd.Series:
        clean_returns = self._prepare_returns(returns)

        cumulative_returns = (
            1 + clean_returns
        ).cumprod() - 1

        cumulative_returns.name = "ASX 200 Cumulative Return"

        return cumulative_returns

    def get_annual_return(
        self,
        returns: pd.Series
    ) -> float:
        clean_returns = self._prepare_returns(returns)

        total_growth = (1 + clean_returns).prod()
        years = len(clean_returns) / self.periods_per_year

        if years <= 0:
            return np.nan

        return float(total_growth ** (1 / years) - 1)

    def get_annual_volatility(
        self,
        returns: pd.Series
    ) -> float:
        clean_returns = self._prepare_returns(returns)

        return float(
            clean_returns.std(ddof=1)
            * np.sqrt(self.periods_per_year)
        )

    def get_sharpe_ratio(
        self,
        returns: pd.Series
    ) -> float:
        clean_returns = self._prepare_returns(returns)

        periodic_risk_free_rate = (
            (1 + self.risk_free_rate)
            ** (1 / self.periods_per_year)
            - 1
        )

        excess_returns = (
            clean_returns - periodic_risk_free_rate
        )

        volatility = excess_returns.std(ddof=1)

        if volatility == 0 or np.isnan(volatility):
            return np.nan

        return float(
            excess_returns.mean()
            / volatility
            * np.sqrt(self.periods_per_year)
        )

    def get_sortino_ratio(
        self,
        returns: pd.Series
    ) -> float:
        clean_returns = self._prepare_returns(returns)

        periodic_risk_free_rate = (
            (1 + self.risk_free_rate)
            ** (1 / self.periods_per_year)
            - 1
        )

        excess_returns = (
            clean_returns - periodic_risk_free_rate
        )

        downside_returns = excess_returns[
            excess_returns < 0
        ]

        if downside_returns.empty:
            return np.nan

        downside_deviation = np.sqrt(
            np.mean(np.square(downside_returns))
        )

        if downside_deviation == 0:
            return np.nan

        return float(
            excess_returns.mean()
            / downside_deviation
            * np.sqrt(self.periods_per_year)
        )

    def get_drawdown_series(
        self,
        returns: pd.Series
    ) -> pd.Series:
        clean_returns = self._prepare_returns(returns)

        wealth_index = (1 + clean_returns).cumprod()
        running_peak = wealth_index.cummax()

        drawdown = wealth_index / running_peak - 1
        drawdown.name = "ASX 200 Drawdown"

        return drawdown

    def get_max_drawdown(
        self,
        returns: pd.Series
    ) -> float:
        drawdown = self.get_drawdown_series(returns)

        return float(drawdown.min())

    def get_calmar_ratio(
        self,
        returns: pd.Series
    ) -> float:
        annual_return = self.get_annual_return(returns)
        max_drawdown = self.get_max_drawdown(returns)

        if max_drawdown == 0 or np.isnan(max_drawdown):
            return np.nan

        return float(
            annual_return / abs(max_drawdown)
        )

    def get_win_rate(
        self,
        returns: pd.Series
    ) -> float:
        clean_returns = self._prepare_returns(returns)

        return float(
            (clean_returns > 0).mean()
        )

    def get_distribution_metrics(
        self,
        returns: pd.Series
    ) -> dict:
        clean_returns = self._prepare_returns(returns)

        return {
            "mean_return": float(clean_returns.mean()),
            "median_return": float(clean_returns.median()),
            "standard_deviation": float(
                clean_returns.std(ddof=1)
            ),
            "skewness": float(clean_returns.skew()),
            "excess_kurtosis": float(clean_returns.kurt()),
            "minimum_return": float(clean_returns.min()),
            "maximum_return": float(clean_returns.max()),
            "positive_return_rate": float(
                (clean_returns > 0).mean()
            ),
            "observations": int(len(clean_returns))
        }

    def get_metrics(
        self,
        returns: pd.Series
    ) -> dict:
        """
        Return all headline benchmark metrics.
        """
        clean_returns = self._prepare_returns(returns)

        return {
            "annual_return": self.get_annual_return(
                clean_returns
            ),
            "sharpe_ratio": self.get_sharpe_ratio(
                clean_returns
            ),
            "sortino_ratio": self.get_sortino_ratio(
                clean_returns
            ),
            "annual_volatility": self.get_annual_volatility(
                clean_returns
            ),
            "max_drawdown": self.get_max_drawdown(
                clean_returns
            ),
            "calmar_ratio": self.get_calmar_ratio(
                clean_returns
            ),
            "win_rate": self.get_win_rate(
                clean_returns
            )
        }

    @staticmethod
    def _prepare_returns(
        returns: pd.Series
    ) -> pd.Series:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series.")

        clean_returns = pd.to_numeric(
            returns.copy(),
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if clean_returns.empty:
            raise ValueError(
                "Return series contains no valid observations."
            )

        return clean_returns