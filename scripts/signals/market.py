from arch import arch_model
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
        market_return = market_return.drop(columns=["^AXJO"])
        return market_return
    
    def get_market_momentum(self): 
        market_momentum_df = self.asx_index_df.copy() 
        market_momentum_df["momentum_63_21"] = (
            self.market_momentum_df["^AXJO"].shift(21) / self.market_momentum_df["^AXJO"].shift(63) - 1
        )
        market_momentum_df["momentum_252_21"] = (
            self.market_momentum_df["^AXJO"].shift(21) / self.market_momentum_df["^AXJO"].shift(252) - 1
        )
        market_momentum_df = market_momentum_df.drop(columns = ["^AXJO"])
        
        return market_momentum_df
    
    def get_rebalance_dates(self, dates: pd.Series, rebalance_date: int): 
        weekdays = dates.dt.weekday
        rebalanced_dates = dates[weekdays == rebalance_date] 
        rebalanced_dates_np = rebalanced_dates.dt.date.to_numpy() 
        return rebalanced_dates_np
    
    def calculate_garch_volatility(self, market_log_returns: pd.DataFrame, min_train_size=25000, rebalance_date=0): 
        dates = market_log_returns["Date"]
        rebalanced_dates = self.get_rebalance_dates(dates, rebalance_date)
        
        returns = market_log_returns["log_market_return_1d"].dropna() * 100
        n_records = len(returns)
    
        oos_conditional_vol = np.full(n_records, np.nan)
        for t in range(min_train_size, n_records): 
            current_date = returns.iloc[t]
            
            was_yesterday_rebalance = 
            
        
        
        
        
        
        
        model = arch_model(
            returns, mean="Zero", vol="GARCH", p = 1, o = 1, q = 1, dist = "normal"
        )
        
        fitted = model.fit(disp="off")
        
        
        print(returns)
        pass
    
    def get_market_volatility(self, market_return_df: pd.DataFrame): 
        market_volatility_df = market_return_df.copy()
        market_volatility_df["market_volatility_21d"] = market_volatility_df["market_return_1d"].rolling(21).std()
        market_volatility_df["market_volatility_63d"] = market_volatility_df["market_return_1d"].rolling(63).std()
        
        ### GARCH Implementation 
        market_log_returns = self.asx_index_df.copy() 
        market_log_returns["Date"] = pd.to_datetime(market_log_returns["Date"])
        market_log_returns["log_market_return_1d"] = np.log(self.asx_index_df["^AXJO"]).diff()
        self.calculate_garch_volatility(market_log_returns)
        print(market_log_returns)
        
        
        market_volatility_df = market_volatility_df.drop(columns = ["market_return_1d", "market_return_5d", "market_return_21d"])
        return market_volatility_df
    
    def run_data(self): 
        market_return_df = self.get_market_return()
        market_volatility_df = self.get_market_volatility(market_return_df)
        print(market_volatility_df)
        
if __name__ == "__main__":
    MarketSignals().run_data()