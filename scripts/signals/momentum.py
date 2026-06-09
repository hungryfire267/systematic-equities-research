import numpy as np
import os
import pandas as pd
from pathlib import Path
from utils import cross_sectional_ranking


BASE_DIR = Path(__file__).resolve().parents[1]

COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class Momentum: 
    def __init__(self): 
        self.returns_df = pd.read_parquet(companies_paths_dict["returns"])     
        self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
        self.returns_df = self.returns_df.set_index("Date")
        
        
    def get_momentum(self, lookback=252, skip=21) -> pd.DataFrame:
        past_returns = self.returns_df.copy().shift(skip)
        cumulative_returns = past_returns.rolling(lookback).sum()
        
        return past_returns, cumulative_returns
        
    def information_discreteness(self, lookback=252, skip=21) -> pd.DataFrame: 
        past_returns, cumulative_returns = self.get_momentum()
        
        up_days = (past_returns > 0).rolling(lookback).sum()
        down_days = (past_returns < 0 ).rolling(lookback).sum()
        
        id_score = (up_days - down_days) / lookback
        id_score = id_score * np.sign(cumulative_returns)
        
        return id_score 
        
    def run_data(self) -> pd.DataFrame: 
        momentum_df_dict, id_df_dict = dict(), dict()
        
        _, momentum_score = self.get_momentum()
        momentum_df_dict["252_12"] = cross_sectional_ranking(momentum_score, True)
        
        id_score = self.information_discreteness()
        id_df_dict["252_12"] = cross_sectional_ranking(id_score, True)
        
        
        return momentum_df_dict, id_df_dict