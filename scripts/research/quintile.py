import numpy as np
import os
import pandas as pd


class Quintile: 
    def __init__(self,  return_df: pd.DataFrame, signal_df: pd.DataFrame):
        self.return_df = return_df
        self.signal_df = signal_df
        
        self.return_df["Date"] = pd.to_datetime(self.return_df["Date"])
        self.signal_df["Date"] = pd.to_datetime(self.signal_df["Date"])
        
        self.factor_list = list(self.signal_df.columns[2:])
        
    def merge_dfs(self): 
        self.df = pd.merge(left = self.return_df, right = self.signal_df, how = "outer", on = ["Date", "Ticker"])
        
        print(self.df)
        
    def calculate(self, factor_col: str, target_col: str): 
        pass
        
    def run_data(self): 
        self.merge_dfs()
        
        for factor in self.factor_list: 
            
            
        print(self.factor_list)
            
