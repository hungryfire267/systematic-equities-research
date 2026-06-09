import numpy as np
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

industry_paths_dict = { 
    "industry_returns": os.path.join(INDUSTRY_DIR, "returns.parquet")
}

class Momentum: 
    def __init__(self, weights: list, n: int): 
        self.returns_df = pd.read_parquet(Path(r"data/raw/companies/returns.parquet"))
        self.industry_returns = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/industry/industry_returns.parquet"))        
        self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
        self.returns_df = self.returns_df.set_index("Date")
        
        assert (n > 0)
        assert (len(weights) == n)
        
        self.factor_config = {
            "mom_12_1": {
                "score": None, 
                "higher_is_better": True
            }, 
            "id": {
                "score":None, 
                "higher_is_better": True
            }
        }
        
        self.weights = weights
        
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
    
    def get_momentum_ranks(self) -> pd.DataFrame:
        ranking_dict = dict()  
        for factor, config in self.factor_config.items(): 
            rank = config["score"].rank(axis=1, pct=True, ascending=config["higher_is_better"])
            ranking_dict[factor] = rank
        
        return ranking_dict
        
    def run_data(self) -> pd.DataFrame: 
        _, self.factor_config["mom_12_1"]["score"] = self.get_momentum()
        self.factor_config["id"]["score"] = self.information_discreteness()
        
        ranking_dict = self.get_momentum_ranks()
        
        final_score = sum(
            weight * rank for weight, rank in zip(self.weights, ranking_dict.values())
        )
        final_rank = final_score.rank(axis=1, pct=True)
        
        return final_rank