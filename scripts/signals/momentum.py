import numpy as np
import os
import pandas as pd
from pathlib import Path
from scripts.signals.utils import cross_sectional_ranking, date_parser

BASE_DIR = Path(__file__).resolve().parents[2]

COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class Momentum: 
    def __init__(self): 
        self.returns_df = date_parser(pd.read_parquet(companies_paths_dict["returns"]))
        
        
    def get_momentum(self, lookback: int, skip: int) -> pd.DataFrame:
        past_returns = self.returns_df.copy().shift(skip)
        cumulative_returns = past_returns.rolling(lookback).sum()
        
        return past_returns, cumulative_returns
        
    def information_discreteness(self, lookback=252, skip=21) -> pd.DataFrame: 
        past_returns, cumulative_returns = self.get_momentum(lookback=lookback, skip=skip)
        
        up_days = (past_returns > 0).rolling(lookback).sum()
        down_days = (past_returns < 0 ).rolling(lookback).sum()
        
        id_score = (up_days - down_days) / lookback
        id_score = id_score * np.sign(cumulative_returns)
        
        return id_score 
        
    def run_data(self) -> pd.DataFrame: 
        momentum_df_dict, id_df_dict = dict(), dict()
        
        lookback_list = [63, 126, 252]
        skip = 21

        for lookback in lookback_list: 
            key = str(lookback) + "_" + str(skip)
            _, momentum_score = self.get_momentum(lookback=lookback, skip=skip)
            momentum_df_dict[key] = cross_sectional_ranking(momentum_score, True).reset_index()
            
            id_score = self.information_discreteness(lookback=lookback, skip=skip)
            id_df_dict[key] = cross_sectional_ranking(id_score, True).reset_index()
        
        final_momentum_dict = {
            "momentum": momentum_df_dict, 
            "id": id_df_dict
        }
        
        return final_momentum_dict