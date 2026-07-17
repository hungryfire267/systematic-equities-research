import numpy as np
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "asx_companies.csv"
ASX_DIR = BASE_DIR / "data" / "raw" / "asx"
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

class RelativeStrength:
    def __init__(self, industry_diff_list: list[int], market_diff_list: list[int]): 
        self.industry_diff_list = industry_diff_list
        self.market_diff_list = market_diff_list
        
        self.asx_df_wide = pd.read_parquet(os.path.join(ASX_DIR, "asx_index.parquet"))
        self.prices_df_wide =  pd.read_parquet(os.path.join(COMPANIES_DIR, "prices.parquet"))
        self.industry_df = pd.read_csv(UNIVERSE_PATH)
        
        self.index_df = pd.read_parquet(os.path.join(ASX_DIR, "asx_index.parquet"))

    def prepare_data(self): 
        self.prices_df = self.prices_df_wide.melt(
            id_vars="Date", var_name="Ticker", value_name="Close"
        )
        self.industry_df["asxCode"] = self.industry_df["asxCode"].astype(str).str.strip() + ".AX"
        df = self.prices_df.merge(
            self.industry_df[["asxCode", "industry"]], 
            left_on = "Ticker", right_on = "asxCode", how= "left"
        )
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        self.df = df.copy()
    
    def compute_industry_returns(self, window): 
        new_df = self.df.sort_values(["Date", "Ticker"]).copy() 
        
        return_col = f"return_{window}"
        industry_return_col = f"industry_return_{window}"
        
        new_df[return_col] = new_df.groupby('Ticker')['Close'].pct_change(window, fill_method=None)
        group_sum = new_df.groupby(['Date', 'industry'])[return_col].transform('sum')
        group_count = new_df.groupby(['Date', 'industry'])[return_col].transform('count')
        new_df[industry_return_col] = (group_sum - new_df[return_col]) / (group_count - 1)
        new_df[f"rs_industry_{window}"] = new_df[return_col] - new_df[industry_return_col]
        df_wide =  new_df[["Date", "Ticker", f"rs_industry_{window}"]].reset_index(drop=True).pivot(
            index='Date', columns='Ticker', values=f'rs_industry_{window}'
        )
        return df_wide
    

    def compute_market_relative(self, horizon: int):
        new_df = self.df.sort_values(['Ticker', 'Date']).copy()
        index_df = self.index_df.sort_values('Date').copy()

        return_col = f'return_{horizon}'
        new_df[return_col] = new_df.groupby('Ticker')['Close'].pct_change(horizon, fill_method=None)

        # index return, same horizon
        index_df[f'market_return_{horizon}'] = index_df['^AXJO'].pct_change(horizon, fill_method=None)

        
        new_df = new_df.merge(
            index_df[['Date', f'market_return_{horizon}']],
            on='Date',
            how='left'
        )

        # relative strength = stock return minus market return
        new_df[f'rs_market_{horizon}'] = new_df[return_col] - new_df[f'market_return_{horizon}']
        print(new_df)
        
        df_wide =  new_df[["Date", "Ticker", f"rs_market_{horizon}"]].reset_index(drop=True).pivot(
            index='Date', columns='Ticker', values=f'rs_market_{horizon}'
        )
        print(df_wide)
        return df_wide
        
    def run_data(self): 
        self.prepare_data()
        rs_industry_dict, rs_market_dict = {}, {} 
        for window in self.industry_diff_list: 
            rs_industry_dict[window] = self.compute_industry_relative(window)
        
        for window in self.market_diff_list: 
            rs_market_dict[window] = self.compute_market_relative(window)
        
        return {
            "rs_industry": rs_industry_dict, 
            "rs_market": rs_market_dict
        }
        
        
if __name__ == "__main__": 
    rs_pipeline = RelativeStrength([5, 63], [21]).run_data()
