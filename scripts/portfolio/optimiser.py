
import numpy as np 
import os
import pandas as pd
from pathlib import Path 

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

class MeanVarianceOptimiser: 
    def __init__(self, predicted_df: pd.DataFrame, return_df: pd.DataFrame, covariance_window: int = 63, weight_bound: float = 0.10): 
        self.selected_df = predicted_df
        self.return_df = return_df
        self.covariance_window = covariance_window
        self.weight_bound = weight_bound
    
    def get_mu_cov(self, date): 
        day_df = (
            self.selected_df[self.selected_df["Date"] == date]
            .copy()
            .sort_values("Ticker")
        )

        tickers = day_df["Ticker"].tolist()

        mu = (
            day_df
            .set_index("Ticker")
            .loc[tickers, "prediction"]
        )

        cov = (
            self.returns_df
            .loc[:date, tickers]
            .tail(self.covariance_window)
            .cov()
        )
        
        return mu, cov
        
    def get_weights(self, mu, cov, date): 
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=cov,
            weight_bounds=(-self.weight_bound, self.weight_bound)
        )

        ef.add_objective(objective_functions.L2_reg)

        ef.max_sharpe()

        weights = ef.clean_weights()

        weights_df = (
            pd.DataFrame({
                "Ticker": list(weights.keys()),
                "weight": list(weights.values())
            })
        )

        weights_df["Date"] = date

        return weights_df
        
    def run_data(self): 
        dates = np.sort(
            self.selected_df["Date"].unique()
        )
        
        portfolio_list = []
        for date in dates: 
            try: 
                mu, cov = self.get_mu_cov(date)
                weights = self.get_weights(mu, cov, date)
                portfolio_list.append(weights)
                
            except Exception as e: 
                print(f"{date}: optimisation failed {e}")
                
        portfolio_df = pd.concat(portfolio_list).reset_index(drop=True)
        
        return portfolio_df
        
    
