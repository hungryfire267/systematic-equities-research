
import numpy as np
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

companies_paths_dict = { 
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class ForwardReturns: 
    def __init__(self, horizon_list=np.array([5, 10, 21])): 
        self.df = pd.read_parquet(companies_paths_dict["returns"])
        self.horizon_list = horizon_list
        
    def melt_df(self) -> pd.DataFrame: 
        self.df["Date"] =  pd.to_datetime(self.df["Date"])
        returns_long = self.df.melt(
            id_vars = "Date", 
            var_name = "Ticker",
            value_name = "return"
        )
        
        returns_df = returns_long.sort_values(["Date", "Ticker"])
        
        return returns_df
    
    def calculate(self, returns_df: pd.DataFrame) -> pd.DataFrame: 
        df = returns_df.copy() 
        for horizon in self.horizon_list:
            df[f"future_return_{horizon}d"] = (
                df.groupby("Ticker")["return"]
                  .transform(lambda x: (1 + x).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1)
            )
        return df
        
    def run_data(self) -> pd.DataFrame: 
        returns_df = self.melt_df()
        final_df = self.calculate(returns_df)
        return final_df
    
    
        
    
        
        
        
        
        
        