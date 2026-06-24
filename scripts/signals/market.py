import numpy as np
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ASX_DIR = BASE_DIR / "data" / "raw" / "asx"


asx_paths_dict = { 
    "asx_index": os.path.join(ASX_DIR, "asx_index.parquet"),
    "asx_returns": os.path.join(ASX_DIR, "asx_returns.parquet")
}

class MarketSignals: 
    def __init__(self): 
        
        self.asx_index_df = pd.read_parquet(asx_paths_dict["asx_index"])
        self.asx_index_df["Date"] = pd.to_datetime(self.asx_index_df["Date"])
    
    def get_market_return(self): 
        market_return = self.asx_index_df.copy()
        market_return["market_return_1d"] = self.asx_index_df["^AXJO"].pct_change(1)
        market_return["market_return_5d"] = self.asx_index_df["^AXJO"].pct_change(5)
        market_return["market_return_21d"] = self.asx_index_df["^AXJO"].pct_change(21)
        market_return.drop(columns=["^AXJO"])
        return market_return
    
    def get
    
    def run_data(self): 
        market_return = self.get_market_return()
        print(market_return)
        
if __name__ == "__main__":
    MarketSignals().run_data()