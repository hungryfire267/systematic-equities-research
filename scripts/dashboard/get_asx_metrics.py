import numpy as np
import pandas as pd


class ASXMetrics:
    """
    Calculate ASX 200 benchmark returns and performance metrics over
    the same holding periods used by the strategy.

    The benchmark is aligned to the strategy's actual rebalance dates,
    rather than using arbitrary rolling five-day returns.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        rebalance_dates: pd.Index | pd.Series | pd.DatetimeIndex,
        date_col: str = "Date",
        price_col: str = "^AXJO",
        periods_per_year: int = 52,
        risk_free_rate: float = 0.0
    ):
        self.date_col = date_col
        self.price_col = price_col
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

        self.prices_df = self._prepare_prices(
            prices_df=prices_df
        )

        self.prices = self.prices_df[
            self.price_col
        ]

        self.rebalance_dates = (
            self._prepare_rebalance_dates(
                rebalance_dates=rebalance_dates
            )
        )

        # Core datasets used throughout the dashboard
        self.daily_returns = (
            self._calculate_daily_returns()
        )

        self.rebalance_prices = (
            self._calculate_rebalance_prices()
        )

        self.weekly_returns = (
            self._calculate_holding_period_returns()
        )

        self.cumulative_returns = (
            self._calculate_cumulative_returns()
        )

        self.drawdown = (
            self._calculate_drawdown_series()
        )

    def _prepare_prices(
        self,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Clean and index the ASX price DataFrame.
        """
        if not isinstance(prices_df, pd.DataFrame):
            raise TypeError(
                "prices_df must be a pandas DataFrame."
            )

        required_columns = {
            self.date_col,
            self.price_col
        }

        missing_columns = (
            required_columns
            - set(prices_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "prices_df is missing required columns: "
                f"{sorted(missing_columns)}. "
                f"Available columns: "
                f"{list(prices_df.columns)}"
            )

        prepared_df = prices_df[
            [
                self.date_col,
                self.price_col
            ]
        ].copy()

        prepared_df[self.date_col] = (
            pd.to_datetime(
                prepared_df[self.date_col],
                errors="coerce"
            )
        )

        prepared_df[self.price_col] = (
            pd.to_numeric(
                prepared_df[self.price_col],
                errors="coerce"
            )
        )

        prepared_df = (
            prepared_df
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna(
                subset=[
                    self.date_col,
                    self.price_col
                ]
            )
            .drop_duplicates(
                subset=self.date_col,
                keep="last"
            )
            .sort_values(
                self.date_col
            )
            .set_index(
                self.date_col
            )
        )

        if prepared_df.empty:
            raise ValueError(
                "No valid ASX price observations "
                "were found."
            )

        if len(prepared_df) < 2:
            raise ValueError(
                "At least two ASX price observations "
                "are required."
            )

        return prepared_df

    @staticmethod
    def _prepare_rebalance_dates(
        rebalance_dates:
        pd.Index | pd.Series | pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """
        Convert strategy rebalance dates into a clean,
        sorted DatetimeIndex.
        """
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                rebalance_dates,
                errors="coerce"
            )
        )

        dates = dates[
            ~dates.isna()
        ]

        dates = pd.DatetimeIndex(
            dates.unique()
        ).sort_values()

        if len(dates) < 2:
            raise ValueError(
                "At least two valid rebalance dates "
                "are required."
            )

        return dates

    @staticmethod
    def _prepare_returns(
        returns: pd.Series
    ) -> pd.Series:
        """
        Clean a return series before metric calculation.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError(
                "returns must be a pandas Series."
            )

        clean_returns = (
            pd.to_numeric(
                returns.copy(),
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
            .sort_index()
        )

        if clean_returns.empty:
            raise ValueError(
                "Return series contains no valid "
                "observations."
            )

        return clean_returns

    def _calculate_daily_returns(
        self
    ) -> pd.Series:
        """
        Calculate daily ASX 200 returns.
        """
        returns = (
            self.prices
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        returns.name = (
            "ASX 200 Daily Return"
        )

        return returns

    def _calculate_rebalance_prices(
        self
    ) -> pd.Series:
        """
        Align ASX prices to the strategy's actual
        rebalance dates.

        For a weekend or market holiday, the latest
        available price on or before that date is used.
        """
        combined_index = (
            self.prices.index.union(
                self.rebalance_dates
            )
        )

        aligned_prices = (
            self.prices
            .reindex(
                combined_index
            )
            .sort_index()
            .ffill()
            .reindex(
                self.rebalance_dates
            )
            .dropna()
        )

        if len(aligned_prices) < 2:
            raise ValueError(
                "Not enough ASX prices were available "
                "for the supplied rebalance dates."
            )

        aligned_prices.name = (
            "ASX 200 Rebalance Price"
        )

        return aligned_prices

    def _calculate_holding_period_returns(
        self
    ) -> pd.Series:
        """
        Calculate the ASX return between consecutive
        strategy rebalance dates.

        These are the benchmark returns used in the
        weekly backtest dashboard.
        """
        returns = (
            self.rebalance_prices
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        returns.name = (
            "ASX 200 Benchmark Return"
        )

        return returns

    def _calculate_cumulative_returns(
        self
    ) -> pd.Series:
        """
        Calculate cumulative return from weekly
        benchmark returns.
        """
        cumulative_returns = (
            1 + self.weekly_returns
        ).cumprod() - 1

        cumulative_returns.name = (
            "ASX 200 Cumulative Return"
        )

        return cumulative_returns

    def _calculate_drawdown_series(
        self
    ) -> pd.Series:
        """
        Calculate the ASX benchmark drawdown series.
        """
        wealth_index = (
            1 + self.weekly_returns
        ).cumprod()

        running_peak = (
            wealth_index.cummax()
        )

        drawdown = (
            wealth_index
            .div(running_peak)
            .sub(1)
        )

        drawdown.name = (
            "ASX 200 Drawdown"
        )

        return drawdown

    def get_daily_returns(
        self
    ) -> pd.Series:
        return self.daily_returns.copy()

    def get_rebalance_prices(
        self
    ) -> pd.Series:
        return self.rebalance_prices.copy()

    def get_holding_period_returns(
        self
    ) -> pd.Series:
        return self.weekly_returns.copy()

    def get_cumulative_returns(
        self
    ) -> pd.Series:
        return self.cumulative_returns.copy()

    def get_drawdown_series(
        self
    ) -> pd.Series:
        return self.drawdown.copy()

    def get_annual_return(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate the compounded annual return.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        total_growth = (
            1 + clean_returns
        ).prod()

        years = (
            len(clean_returns)
            / self.periods_per_year
        )

        if years <= 0:
            return np.nan

        annual_return = (
            total_growth ** (1 / years)
            - 1
        )

        return float(
            annual_return
        )

    def get_annual_volatility(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate annualised volatility.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        volatility = (
            clean_returns.std(ddof=1)
            * np.sqrt(
                self.periods_per_year
            )
        )

        return float(
            volatility
        )

    def get_sharpe_ratio(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate annualised Sharpe ratio.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        periodic_risk_free_rate = (
            (1 + self.risk_free_rate)
            ** (
                1 / self.periods_per_year
            )
            - 1
        )

        excess_returns = (
            clean_returns
            - periodic_risk_free_rate
        )

        excess_volatility = (
            excess_returns.std(ddof=1)
        )

        if (
            excess_volatility == 0
            or np.isnan(
                excess_volatility
            )
        ):
            return np.nan

        sharpe_ratio = (
            excess_returns.mean()
            / excess_volatility
            * np.sqrt(
                self.periods_per_year
            )
        )

        return float(
            sharpe_ratio
        )

    def get_sortino_ratio(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate annualised Sortino ratio.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        periodic_risk_free_rate = (
            (1 + self.risk_free_rate)
            ** (
                1 / self.periods_per_year
            )
            - 1
        )

        excess_returns = (
            clean_returns
            - periodic_risk_free_rate
        )

        downside_returns = (
            excess_returns[
                excess_returns < 0
            ]
        )

        if downside_returns.empty:
            return np.nan

        downside_deviation = np.sqrt(
            np.mean(
                np.square(
                    downside_returns
                )
            )
        )

        if (
            downside_deviation == 0
            or np.isnan(
                downside_deviation
            )
        ):
            return np.nan

        sortino_ratio = (
            excess_returns.mean()
            / downside_deviation
            * np.sqrt(
                self.periods_per_year
            )
        )

        return float(
            sortino_ratio
        )

    def get_max_drawdown(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate maximum drawdown.
        """
        if returns is None:
            drawdown = self.drawdown
        else:
            clean_returns = (
                self._prepare_returns(
                    returns
                )
            )

            wealth_index = (
                1 + clean_returns
            ).cumprod()

            running_peak = (
                wealth_index.cummax()
            )

            drawdown = (
                wealth_index
                .div(running_peak)
                .sub(1)
            )

        return float(
            drawdown.min()
        )

    def get_calmar_ratio(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate annual return divided by the
        absolute maximum drawdown.
        """
        selected_returns = (
            self.weekly_returns
            if returns is None
            else returns
        )

        annual_return = (
            self.get_annual_return(
                selected_returns
            )
        )

        max_drawdown = (
            self.get_max_drawdown(
                selected_returns
            )
        )

        if (
            max_drawdown == 0
            or np.isnan(
                max_drawdown
            )
        ):
            return np.nan

        calmar_ratio = (
            annual_return
            / abs(max_drawdown)
        )

        return float(
            calmar_ratio
        )

    def get_win_rate(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate the proportion of positive
        holding periods.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        win_rate = (
            clean_returns > 0
        ).mean()

        return float(
            win_rate
        )

    def get_best_period_return(
        self,
        returns: pd.Series | None = None
    ) -> float:
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        return float(
            clean_returns.max()
        )

    def get_worst_period_return(
        self,
        returns: pd.Series | None = None
    ) -> float:
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        return float(
            clean_returns.min()
        )

    def get_distribution_metrics(
        self,
        returns: pd.Series | None = None
    ) -> dict:
        """
        Return descriptive statistics for the
        weekly benchmark return distribution.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        return {
            "mean_return": float(
                clean_returns.mean()
            ),
            "median_return": float(
                clean_returns.median()
            ),
            "standard_deviation": float(
                clean_returns.std(ddof=1)
            ),
            "skewness": float(
                clean_returns.skew()
            ),
            "excess_kurtosis": float(
                clean_returns.kurt()
            ),
            "minimum_return": float(
                clean_returns.min()
            ),
            "maximum_return": float(
                clean_returns.max()
            ),
            "positive_return_rate": float(
                (clean_returns > 0).mean()
            ),
            "negative_return_rate": float(
                (clean_returns < 0).mean()
            ),
            "observations": int(
                len(clean_returns)
            )
        }
    
    def get_total_return(
        self,
        returns: pd.Series | None = None
    ) -> float:
        """
        Calculate the total compounded return over
        the complete benchmark period.
        """
        clean_returns = self._prepare_returns(
            self.weekly_returns
            if returns is None
            else returns
        )

        total_return = (
            (1 + clean_returns).prod()
            - 1
        )

        return float(total_return)

    def get_metrics(
        self,
        returns: pd.Series | None = None
    ) -> dict:
        """
        Return all headline ASX benchmark metrics
        used in the backtesting dashboard.
        """
        selected_returns = (
            self.weekly_returns
            if returns is None
            else self._prepare_returns(returns)
        )

        return {
            "annual_return": self.get_annual_return(
                selected_returns
            ),
            "total_return": self.get_total_return(
                selected_returns
            ),
            "sharpe_ratio": self.get_sharpe_ratio(
                selected_returns
            ),
            "sortino_ratio": self.get_sortino_ratio(
                selected_returns
            ),
            "annual_volatility": self.get_annual_volatility(
                selected_returns
            ),
            "max_drawdown": self.get_max_drawdown(
                selected_returns
            ),
            "win_rate": self.get_win_rate(
                selected_returns
            ),
            "calmar_ratio": self.get_calmar_ratio(
                selected_returns
            ),
            "worst_week": self.get_worst_period_return(
                selected_returns
            )
        }

    def get_dashboard_data(
        self
    ) -> dict:
        """
        Return the main benchmark datasets and
        metric dictionaries needed by the dashboard.
        """
        return {
            "daily_returns": (
                self.get_daily_returns()
            ),
            "rebalance_prices": (
                self.get_rebalance_prices()
            ),
            "weekly_returns": (
                self.get_holding_period_returns()
            ),
            "cumulative_returns": (
                self.get_cumulative_returns()
            ),
            "drawdown": (
                self.get_drawdown_series()
            ),
            "summary_metrics": (
                self.get_metrics()
            ),
            "distribution_metrics": (
                self.get_distribution_metrics()
            )
        }