
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

class GetMetrics: 
    def __init__(self, portfolio_df): 
        self.portfolio_df = portfolio_df
    
    def get_strategy_returns(self): 
        strategy_returns = (
            self.portfolio_df
            .groupby("Date")["portfolio_return"]
            .sum()
            .reset_index()
        )
        
        return strategy_returns
    
    def get_sortino_ratio(self, r, annual_returns): 
        downside_returns = r[r < 0]
        downside_vol = downside_returns.std(ddof=1) * np.sqrt(52)
        sortino_ratio = annual_returns / downside_vol
        return sortino_ratio
    
    def run_data(self): 
        strategy_returns = self.get_strategy_returns()
        r = strategy_returns["portfolio_return"].dropna()
        
        annual_return = (1 + r).prod() ** (52 / len(r)) - 1
        
        annual_vol = r.std() * np.sqrt(52)
        
        sharpe = r.mean() / r.std() * np.sqrt(52)

        sortino_ratio = self.get_sortino_ratio(r, annual_return)
        
        
        equity = (1 + r).cumprod()
        total_return = equity.iloc[-1] - 1
        drawdown = equity / equity.cummax() - 1
        max_drawdown = drawdown.min()
        
        calmar_ratio = annual_return / np.abs(max_drawdown)
        win_rate = (r > 0).mean()
        worst_week_row = r.min()

        
        n_rebalances = len(r)
        backtest_metrics_dict = { 
            "annual_return": annual_return, 
            "total_return": total_return,
            "sharpe_ratio": sharpe, 
            "sortino_ratio": sortino_ratio,
            "annual_volatility": annual_vol,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar_ratio": calmar_ratio, 
            "worst_week": worst_week_row
        }
        
        return backtest_metrics_dict, strategy_returns
    
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
        
        
        
        