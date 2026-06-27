import datetime as dt
import numpy as np
import pandas as pd 
from pathlib import Path
import yfinance as yf 

class VIX: 
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_vix(self): 
        vix_data = yf.download(
            tickers="^AXVI", auto_adjust=True, start=self.start_date, end=self.end_date
        )
        
        final_df = vix_data["Close"].reset_index()
        final_df.index.name = None
        
        return final_df
    
    def run_data(self): 
        final_df = self.fetch_vix()
        return final_df