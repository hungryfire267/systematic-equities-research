
import numpy as np 
import os
import pandas as pd
from pathlib import Path 

# BASE_DIR = Path(__file__).resolve().parents[1]
# COMPANIES_DIR = BASE_DIR / "data"/ "raw" / "companies"

# prices_df = pd.read_parquet(os.path.join(COMPANIES_DIR, "prices.parquet"))




class GetStockMetrics: 
    def __init__(self, prices_df: pd.DataFrame, company_code: str): 
        df = prices_df.copy() 
        df = df[["Date", company_code]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        self.company_code = company_code
        self.prices_df = df.dropna(subset=[self.company_code])
    
    def get_price_statistics(self): 
        start_date = self.prices_df.index[0]
        end_date = self.prices_df.index[-1]
        
        start_price = self.prices_df.iloc[0].values[0]
        latest_price = self.prices_df.iloc[-1].values[0]
        
        lowest_price = self.prices_df.min().values[0]
        highest_price = self.prices_df.max().values[0]
        
        total_return = latest_price / start_price - 1
        latest_21d_return = latest_price / self.prices_df.iloc[-21].values - 1 if len(self.prices_df) >= 21 else None
        latest_63d_return = latest_price / self.prices_df.iloc[-63].values - 1 if len(self.prices_df) >= 63 else None
        
        price_statistic_dict = { 
            "start_date": start_date,
            "end_date": end_date, 
            "start_price": start_price, 
            "latest_price": latest_price, 
            "lowest_price": lowest_price,
            "highest_price": highest_price,
            "total_return": total_return,
            "Latest 21d Return": latest_21d_return, 
            "Latest_63d Return": latest_63d_return
        }
        
        return price_statistic_dict
        
        


    

# stock_metrics_pipeline = GetStockMetrics(prices_df, "COL.AX")
# price_statistic_dict = stock_metrics_pipeline.get_price_statistics()
# print(price_statistic_dict)
