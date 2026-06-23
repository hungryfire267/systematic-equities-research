import datetime
import numpy as np
import os
import pandas as pd


class WalkForwardValidator: 
    def __init__(self, feature_matrix, rebalance_date: int, min_train_size = 30000): 
        self.feature_matrix = feature_matrix
        
        if (rebalance_date not in range(1, 6)): 
            raise ValueError("Rebalance date must be a weekday: Monday=0, ..., Friday=4")
        self.rebalance_date = rebalance_date
        self.min_train_size = min_train_size
    
    def get_rebalance_dates(self): 
        mask = self.feature_matrix["Date"].dt.weekday == self.rebalance_date
        dates = self.feature_matrix.loc[mask, "Date"].unique()
        return dates
        
    def run_data(self): 
        
        start_date = self.feature_matrix["Date"].min()
        adjusted_start_date = start_date + datetime.timedelta(days=365)
        
        print(adjusted_start_date)
        self.feature_matrix = self.feature_matrix[self.feature_matrix["Date"] == adjusted_start_date]
        dates = self.get_rebalance_dates()
        
        predictions = [] 
        for date in dates: 
            train_df = self.feature_matrix[self.feature_matrix["Date"] < date].copy()
            test_df = self.feature_matrix[self.feature_matrix["Date"] == date].copy()
            
            train_df = train_df.dropna(subset=[self.target_col])
            test_df = test_df.dropna(subset=[self.target_col])
            
            if train_df.shape[0] < self.min_train_size or test_df.empty: 
                continue
            
            


            