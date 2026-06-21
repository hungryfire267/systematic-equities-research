
from functools import reduce
import numpy as np
import pandas as pd

class FeatureMatrixBuilder:
    def __init__(self, feature_dfs_dict: dict[str, pd.DataFrame], target_df): 
        self.feature_dfs_dict = feature_dfs_dict 
        self.target_df_dict = target_df
        
    def merge_features(self) -> pd.DataFrame: 
        df_list = []
        for feature, df in self.feature_dfs_dict.items(): 
            temp = df.copy() 
            temp["Date"] = pd.to_datetime(temp["Date"])
            df_list.append(df)
        
        feature_df = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["Date", "Ticker"],
                how="outer"
            ),
            df_list
        )
        
        return feature_df
    
    def add_target(self, feature_df: pd.DataFrame) -> pd.DataFrame: 
        target_df = self.target_df
        target_df["Date"] = pd.to_datetime(target_df["Date"]) 
        
        return pd.merge(
            feature_df,
            target_df,
            on=["Date", "Ticker"],
            how="inner"
        )
        
    def run_data(self): 
        feature_df = self.merge_features()
        final_df = self.add_target(feature_df)

        return (
            final_df
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )
        
        