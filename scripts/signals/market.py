from arch import arch_model
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

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
            market_momentum_df["^AXJO"].shift(21) / market_momentum_df["^AXJO"].shift(63) - 1
        )
        market_momentum_df["momentum_252_21"] = (
            market_momentum_df["^AXJO"].shift(21) / market_momentum_df["^AXJO"].shift(252) - 1
        )
        market_momentum_df = market_momentum_df.drop(columns = ["^AXJO"])
        
        return market_momentum_df
    
    def get_rebalance_dates(self, dates: pd.Series, rebalance_date: int): 
        weekdays = dates.dt.weekday
        rebalanced_dates = dates[weekdays == rebalance_date] 
        rebalanced_dates_np = rebalanced_dates.dt.date
        return rebalanced_dates_np
    
    def calculate_garch_volatility(self, market_log_returns: pd.DataFrame, min_train_size=252, rebalance_date=0): 
        dates = market_log_returns["Date"]
        rebalanced_dates = self.get_rebalance_dates(dates, rebalance_date)
        returns = market_log_returns["log_market_return_1d"].dropna() * 100
        is_rebalance_day = returns.index.isin(pd.to_datetime(rebalanced_dates))
        
        n_records = len(returns)
    
        oos_conditional_vol = np.full(n_records, np.nan)
        vol_score = np.full(n_records, np.nan)
        omega, alpha, gamma, beta = None, None, None, None
        for t in range(min_train_size, n_records): 
            expanding_history_returns = returns.iloc[0: t]
            
            was_yesterday_rebalance = is_rebalance_day[t - 1]
            
            if (t == min_train_size) or was_yesterday_rebalance: 
                ### Student T captures the volatility/ fat tail better than normal in financial market data
                model = arch_model(expanding_history_returns, p = 1, o = 1, q = 1, dist = "studentst", rescale=False)
                try: 
                    res = model.fit(disp = "off", show_warning=False)
                    omega = res.params.loc["omega"]
                    alpha = res.params.loc["alpha[1]"]
                    gamma = res.params.loc["gamma[1]"]
                    beta = res.params.loc["beta[1]"]
                except Exception: 
                    if omega is None: 
                        omega, alpha, gamma, beta = 0.05, 0.05, 0.05, 0.05
            
            
            last_return_shock = returns.iloc[t - 1]

            if t == min_train_size: 
                last_volatility = res.conditional_volatility.iloc[-1] ** 2
            else: 
                last_volatility = oos_conditional_vol[t-1] ** 2
            
            if last_return_shock < 0: 
                is_negative = 1.0 
            else: 
                is_negative = 0.0 
            
            
            variance_forecast = (
                omega + alpha * (last_return_shock ** 2) + 
                gamma * (last_return_shock ** 2) * is_negative +
                beta * last_volatility
            )
            
            current_raw_vol = np.sqrt(variance_forecast)
            oos_conditional_vol[t] = current_raw_vol
            
            if t == min_train_size: 
                vol_score[t] = 50.0
            else: 
                historical_vols = oos_conditional_vol[min_train_size:t]
                historical_vols = historical_vols[~np.isnan(historical_vols)] 
                vol_pool = np.append(historical_vols, current_raw_vol)
                
                ranks = rankdata(vol_pool)
                current_percentile = (ranks[-1] - 1) / (len(vol_pool) - 1) * 100
                vol_score[t] = current_percentile
            
                
        feature_df = pd.DataFrame(index=returns.index)
        
        
        feature_df["Date"] = pd.to_datetime(dates)
        feature_df['raw_garch_vol'] = oos_conditional_vol / 100
        feature_df['volatility_score'] = vol_score
        
        return feature_df.iloc[min_train_size:].dropna()
        
    def get_market_volatility(self, market_return_df: pd.DataFrame): 
        market_volatility_df = market_return_df.copy()
        market_volatility_df["market_volatility_21d"] = market_volatility_df["market_return_1d"].rolling(21).std()
        market_volatility_df["market_volatility_63d"] = market_volatility_df["market_return_1d"].rolling(63).std()
        
        ### GARCH Implementation 
        market_log_returns = self.asx_index_df.copy() 
        market_log_returns["Date"] = pd.to_datetime(market_log_returns["Date"])
        market_log_returns["log_market_return_1d"] = np.log(self.asx_index_df["^AXJO"]).diff()
        feature_df = self.calculate_garch_volatility(market_log_returns)
        market_volatility_df["raw_garch_vol"] = feature_df["raw_garch_vol"].copy()
        
        market_volatility_df = market_volatility_df.drop(columns = ["market_return_1d", "market_return_5d", "market_return_21d"])
        return market_volatility_df
    
    def get_market_drawdown(self, drawdown_windows): 
        market_drawdown_df = self.asx_index_df.copy() 
        market_drawdown_df["Date"] = pd.to_datetime(market_drawdown_df["Date"])
        
        for window in drawdown_windows: 
            market_drawdown_df[f"market_drawdown_{window}d"] = (
                market_drawdown_df["^AXJO"] / market_drawdown_df["^AXJO"].rolling(window).max()
            ) - 1
        
        market_drawdown_df = market_drawdown_df.drop(columns=["^AXJO"])
    
        return market_drawdown_df
        
    def run_data(self): 
        market_return_df = self.get_market_return()
        market_momentum_df = self.get_market_momentum()
        market_volatility_df = self.get_market_volatility(market_return_df)
        market_drawdown_df = self.get_market_drawdown([21, 63, 252])
        
        market_data_dict = {
            "returns": market_return_df, 
            "momentum": market_momentum_df,
            "volatility": market_volatility_df, 
            "drawdown": market_drawdown_df
        }
        return market_data_dict
        
if __name__ == "__main__":
    MarketSignals().run_data()