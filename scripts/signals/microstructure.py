import numpy as np
import os
import pandas as pd
from pathlib import Path
from scripts.signals.utils import date_parser, cross_sectional_ranking


BASE_DIR = Path(__file__).resolve().parents[2]

COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

companies_paths_dict = {
    "prices": os.path.join(COMPANIES_DIR, "prices.parquet"),
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet"),
    "volume": os.path.join(COMPANIES_DIR, "volume.parquet")
}


class Microstructure: 
    def __init__(self, window_list): 
        self.prices_df = date_parser(pd.read_parquet(companies_paths_dict["prices"]))
        self.returns_df = date_parser(pd.read_parquet(companies_paths_dict["returns"]))
        self.volume_df = date_parser(pd.read_parquet(companies_paths_dict["volume"]))   
        
        self.window_periods = dict()
        for window in window_list: 
            min_periods = max(3, int(0.5 * window))
            self.window_periods[window] = min_periods
        
    
    def get_dollar_volume(self):
        return self.prices_df * self.volume_df
    
    def dollar_volume_liquidity(self): 
        dv_liquidity_dict = dict()
        dollar_volume = self.get_dollar_volume()
        for window, min_periods in self.window_periods.items(): 
            liquidity = dollar_volume.rolling(window=window, min_periods=min_periods).mean()
            dv_rank = cross_sectional_ranking(liquidity, higher_is_better=False)
            dv_liquidity_dict[window] = dv_rank.reset_index()
        return dv_liquidity_dict

    def get_amihud(self):
        amihud_dict = dict() 
        dollar_volume = self.get_dollar_volume()
        amihud = self.returns_df.abs() / dollar_volume.replace(0, np.nan)
        
        for window, min_periods in self.window_periods.items():
            amihud_smoothed = amihud.rolling(window=window, min_periods=min_periods).mean()
            amihud_rank = cross_sectional_ranking(amihud_smoothed, higher_is_better=True)
            amihud_dict[window] = amihud_rank.reset_index()
        
        return amihud_dict
    
    def run_data(self): 
        dv_liquidity_dict = self.dollar_volume_liquidity() 
        amihud_illiquidity_dict = self.get_amihud()
        
        final_microstructure_dict = { 
            "dv_liquidity": dv_liquidity_dict,
            "amihud": amihud_illiquidity_dict 
        }
        return final_microstructure_dict
    
 