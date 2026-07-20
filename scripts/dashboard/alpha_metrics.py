import numpy as np
import pandas as pd
import statsmodels.api as sm


class AlphaMetrics:
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 52,
        risk_free_rate: float = 0.0
    ):
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

        self.returns_df = pd.concat(
            [
                strategy_returns.rename("strategy_return"),
                benchmark_returns.rename("benchmark_return")
            ],
            axis=1,
            join="inner"
        ).dropna()

        if len(self.returns_df) < 10:
            raise ValueError(
                "Not enough aligned return observations for alpha analysis."
            )

        self.periodic_rf = (
            (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        )

        self.strategy_excess = (
            self.returns_df["strategy_return"] - self.periodic_rf
        )

        self.benchmark_excess = (
            self.returns_df["benchmark_return"] - self.periodic_rf
        )

        self.model = self._fit_capm()

    def _fit_capm(self):
        x = sm.add_constant(self.benchmark_excess)

        return sm.OLS(
            self.strategy_excess,
            x
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    def get_periodic_alpha(self) -> float:
        return float(self.model.params["const"])

    def get_annualised_alpha(self) -> float:
        periodic_alpha = self.get_periodic_alpha()

        return float(
            (1 + periodic_alpha) ** self.periods_per_year - 1
        )

    def get_beta(self) -> float:
        return float(
            self.model.params["benchmark_return"]
        )

    def get_alpha_t_stat(self) -> float:
        return float(self.model.tvalues["const"])

    def get_alpha_p_value(self) -> float:
        return float(self.model.pvalues["const"])

    def get_alpha_confidence_interval(
        self,
        alpha: float = 0.05
    ) -> tuple[float, float]:
        interval = self.model.conf_int(alpha=alpha).loc["const"]

        lower_periodic = float(interval.iloc[0])
        upper_periodic = float(interval.iloc[1])

        lower_annual = (
            (1 + lower_periodic) ** self.periods_per_year - 1
        )
        upper_annual = (
            (1 + upper_periodic) ** self.periods_per_year - 1
        )

        return float(lower_annual), float(upper_annual)

    def get_r_squared(self) -> float:
        return float(self.model.rsquared)

    def get_tracking_error(self) -> float:
        active_returns = (
            self.returns_df["strategy_return"]
            - self.returns_df["benchmark_return"]
        )

        return float(
            active_returns.std(ddof=1)
            * np.sqrt(self.periods_per_year)
        )

    def get_information_ratio(self) -> float:
        active_returns = (
            self.returns_df["strategy_return"]
            - self.returns_df["benchmark_return"]
        )

        tracking_error = active_returns.std(ddof=1)

        if np.isclose(tracking_error, 0):
            return np.nan

        return float(
            active_returns.mean()
            / tracking_error
            * np.sqrt(self.periods_per_year)
        )

    def get_metrics(self) -> dict:
        confidence_interval = (
            self.get_alpha_confidence_interval()
        )

        return {
            "annualised_alpha": self.get_annualised_alpha(),
            "beta": self.get_beta(),
            "alpha_t_stat": self.get_alpha_t_stat(),
            "alpha_p_value": self.get_alpha_p_value(),
            "alpha_ci_lower": confidence_interval[0],
            "alpha_ci_upper": confidence_interval[1],
            "r_squared": self.get_r_squared(),
            "information_ratio": self.get_information_ratio(),
            "tracking_error": self.get_tracking_error(),
            "observations": len(self.returns_df)
        }