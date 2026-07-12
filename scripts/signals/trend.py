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
        self.prices_df = pd.read_parquet(company_paths_dict["prices"]).set_index("Date")
        self.log_prices_df = np.log(self.prices_df)
    
    def rolling_trend_r2(self, window: int, annualize: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(self.log_prices_df)

        # global time index, same shape as prices, carrying the same NaN pattern
        x = np.broadcast_to(np.arange(n)[:, None], self.log_prices_df.shape).astype(float)
        X = pd.DataFrame(x, index=self.log_prices_df.index, columns=self.log_prices_df.columns)
        X = X.where(self.log_prices_df.notna())

        roll = lambda d: d.rolling(window, min_periods=window)
        mx, my = roll(X).mean(), roll(self.log_prices_df).mean()

        cov   = roll(X * self.log_prices_df).mean() - mx * my
        var_x = roll(X * X).mean() - mx ** 2
        var_y = roll(self.log_prices_df * self.log_prices_df).mean() - my ** 2

        slope = cov / var_x
        r2    = (cov ** 2) / (var_x * var_y)

        if annualize:
            slope = slope * 252  # log-price slope is per trading day

        return slope, r2
    
    def run_data(self) -> pd.DataFrame: 
        trend_dict, r2_dict = dict(), dict()
        for window in self.rolling_window_list: 
            slope, r2 = self.rolling_trend_r2(window)
            slope_rank_df = cross_sectional_ranking(slope, higher_is_better=True).reset_index()
            r2_rank_df = cross_sectional_ranking(r2, higher_is_better=True).reset_index()
            
            slope_rank_df["Date"] = pd.to_datetime(slope_rank_df["Date"])
            r2_rank_df["Date"] = pd.to_datetime(r2_rank_df["Date"])
            trend_dict[window] = slope_rank_df
            r2_dict[window] = r2_rank_df
        
        final_trend_dict = { 
            "trend": trend_dict, 
            "r2": r2_dict
        }
        

        return final_trend_dict
        
        

Trends([21, 63, 126]).run_data()
