
import numpy as np
import pandas as pd

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
        
        
        
        