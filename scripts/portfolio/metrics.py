
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

class GetMetrics:
    PERIODS_PER_YEAR = 52

    def __init__(self, portfolio_df: pd.DataFrame):
        self.portfolio_df = portfolio_df.copy()

    def get_strategy_returns(self) -> pd.DataFrame:
        required = {"Date", "portfolio_return"}
        missing = required.difference(self.portfolio_df.columns)

        if missing:
            raise KeyError(
                f"Portfolio data is missing columns: {sorted(missing)}"
            )

        strategy_returns = (
            self.portfolio_df
            .assign(
                Date=lambda df: pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                ),
                portfolio_return=lambda df: pd.to_numeric(
                    df["portfolio_return"],
                    errors="coerce"
                )
            )
            .dropna(subset=["Date", "portfolio_return"])
            .groupby("Date", as_index=False)["portfolio_return"]
            .sum()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return strategy_returns

    def get_sortino_ratio(
        self,
        returns: pd.Series,
        minimum_acceptable_return: float = 0.0
    ) -> float:
        downside = np.minimum(
            returns - minimum_acceptable_return,
            0.0
        )

        downside_deviation = (
            np.sqrt(np.mean(np.square(downside)))
            * np.sqrt(self.PERIODS_PER_YEAR)
        )

        if downside_deviation == 0:
            return np.nan

        annualised_excess_return = (
            returns.mean() - minimum_acceptable_return
        ) * self.PERIODS_PER_YEAR

        return annualised_excess_return / downside_deviation

    @staticmethod
    def get_max_drawdown(returns: pd.Series) -> float:
        wealth = (1 + returns).cumprod().to_numpy()
        wealth = np.insert(wealth, 0, 1.0)

        running_peak = np.maximum.accumulate(wealth)
        drawdown = wealth / running_peak - 1

        return float(drawdown.min())

    def run_data(self):
        strategy_returns = self.get_strategy_returns()

        r = (
            strategy_returns["portfolio_return"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )

        if len(r) < 2:
            raise ValueError(
                "At least two valid return observations are required."
            )

        if (r <= -1).any():
            raise ValueError(
                "Portfolio returns cannot be less than or equal to -100%."
            )

        total_return = (1 + r).prod() - 1

        annual_return = (
            (1 + total_return)
            ** (self.PERIODS_PER_YEAR / len(r))
            - 1
        )

        weekly_volatility = r.std(ddof=1)
        annual_volatility = (
            weekly_volatility
            * np.sqrt(self.PERIODS_PER_YEAR)
        )

        sharpe_ratio = (
            r.mean() / weekly_volatility
            * np.sqrt(self.PERIODS_PER_YEAR)
            if weekly_volatility > 0
            else np.nan
        )

        sortino_ratio = self.get_sortino_ratio(r)
        max_drawdown = self.get_max_drawdown(r)

        calmar_ratio = (
            annual_return / abs(max_drawdown)
            if max_drawdown < 0
            else np.nan
        )

        metrics = {
            "annual_return": annual_return,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "win_rate": (r > 0).mean(),
            "calmar_ratio": calmar_ratio,
            "worst_week": r.min(),
            "n_rebalances": len(r),
        }

        return metrics, strategy_returns
    
class GetPredictionMetrics: 
    def __init__(self, portfolio_df): 
        self.portfolio_df = portfolio_df
        
    def get_rmse_mae(self): 
        preds = self.portfolio_df["prediction"]
        actuals = self.portfolio_df["future_return_5d"]
        
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        return mae, rmse
    
    def get_ic(self): 
        daily_ic = self.portfolio_df.groupby("Date").apply(
            lambda x: x["prediction"].corr(
                x["future_return_5d"],
                method="spearman"
            ), 
            include_groups = False
        )
        
        mean_ic = daily_ic.dropna().mean()
        std_ic = daily_ic.dropna().std(ddof=1)
        
        icir = mean_ic / std_ic * np.sqrt(52)
        return mean_ic, icir, daily_ic
    
    def get_hit_rate(self): 
        preds = self.portfolio_df["prediction"]
        actuals = self.portfolio_df["future_return_5d"]
        
        hit_rate = (np.sign(preds) == np.sign(actuals)).mean()
        return hit_rate
    
    def get_hit_results(self): 
        df = self.portfolio_df.copy()
        df["hit"] = (
            np.sign(df["prediction"]) ==
            np.sign(df["future_return_5d"])
        )

        return df
    
    def run_data(self): 
        mean_ic, icir, daily_ic = self.get_ic()
        hit_rate = self.get_hit_rate()
        mae, rmse = self.get_rmse_mae()
        
        prediction_metrics_dict = {
            "mean_ic": mean_ic, 
            "annualised_icir": icir, 
            "hit_rate": hit_rate, 
            "mae": mae, 
            "rmse": rmse
        }
        
        hit_results_df = self.get_hit_results()
        
        return prediction_metrics_dict, daily_ic, hit_results_df
        
        
        
        