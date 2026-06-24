import datetime
import numpy as np
import os
import pandas as pd


class WalkForwardValidator: 
    def __init__(
        self, feature_matrix, model, validation_start: str, validation_end: str, rebalance_date: int, min_train_size = 30000
    ): 
        self.feature_matrix = feature_matrix
        self.feature_cols = self.feature_matrix.columns[2:-1]
        self.target_col = self.feature_matrix.columns[-1]
        
        self.model = model
        
        if (rebalance_date not in range(1, 6)): 
            raise ValueError("Rebalance date must be a weekday")
        self.rebalance_date = rebalance_date
        self.min_train_size = min_train_size
        
        self.validation_start = pd.to_datetime(validation_start)
        self.validation_end = pd.to_datetime(validation_end)
    
    def get_rebalance_dates(self): 
        mask = self.feature_matrix["Date"].dt.weekday == self.rebalance_date
        dates = self.feature_matrix.loc[mask, "Date"].unique()
        return dates
        
    def run_data(self): 
        
        start_date = self.feature_matrix["Date"].min()
        adjusted_start_date = start_date + datetime.timedelta(days=365)
        
        print(adjusted_start_date)
        self.feature_matrix = self.feature_matrix[self.feature_matrix["Date"] >= adjusted_start_date]
        dates = self.get_rebalance_dates()
        
        unique_dates = np.sort(self.feature_matrix["Date"].unique())
        
        predictions = [] 
        for date in dates:
            horizon = 5  # or infer from target_col

            date_idx = np.where(unique_dates == date)[0][0]

            purge_cutoff_idx = max(0, date_idx - horizon)
            purge_cutoff_date = unique_dates[purge_cutoff_idx]

            train_df = self.feature_matrix[
                self.feature_matrix["Date"] <= purge_cutoff_date
            ].copy()
            
            test_df = self.feature_matrix[self.feature_matrix["Date"] == date].copy()
            
            train_df = train_df.dropna(subset=[self.target_col])
            test_df = test_df.dropna(subset=[self.target_col])
            
            if train_df.shape[0] < self.min_train_size or test_df.empty: 
                continue
            
            X_train = train_df[self.feature_cols].copy() 
            y_train = train_df[self.target_col].copy() 
            
            X_test = test_df[self.feature_cols].copy() 
            
            self.model.fit(X_train, y_train)
            
            output = test_df[["Date", "Ticker", self.target_col]].copy()
            output["prediction"] = self.model.predict(X_test)
            output["model_name"] = self.model.__class__.__name__
            
            predictions.append(output)
        
        final_df = pd.concat(predictions, ignore_index=True)
        return final_df
            
            


            