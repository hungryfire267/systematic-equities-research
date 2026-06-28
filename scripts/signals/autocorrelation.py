import numpy as np 
import os
import pandas as pd
from pathlib import Path

from scripts.signals.utils import cross_sectional_ranking, date_parser


BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"


companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class Autocorrelation: 
    def __init__(self, rolling_window_list: list): 
        self.returns_df = date_parser(pd.read_parquet(companies_paths_dict["returns"]))
        self.volatility_df = self.returns_df ** 2
        print(self.volatility_df)
        
        self.rolling_window_list = rolling_window_list
        
    def calculate_autocorrelation(self, x, lag=1) -> pd.Series:
        return pd.Series(x).autocorr(lag=lag)
    
    def run_data(self): 
        autocorrelation_dict = dict()
        for window in self.rolling_window_list: 
            autocorr_df = self.volatility_df.rolling(window).apply(
                lambda x: self.calculate_autocorrelation(x, lag=1),
                raw=False
            )
            autocorr_rank_df = cross_sectional_ranking(autocorr_df)
            autocorrelation_dict[window] = autocorr_rank_df
            
        autocorr_final_dict = {
            "autocorr": autocorrelation_dict
        }
        
        return autocorr_final_dict