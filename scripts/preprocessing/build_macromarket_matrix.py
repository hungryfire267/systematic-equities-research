from functools import reduce
import pandas as pd

class BuildMacroMarketMatrix: 
    def __init__(self, paths_dict: dict): 
        self.paths_dict = paths_dict
        
    def load_data(self) -> pd.DataFrame: 
        features_dict = dict()
        for feature in self.paths_dict.keys(): 
            feature_df = pd.read_parquet(self.paths_dict[feature])
            feature_df["Date"] = pd.to_datetime(feature_df["Date"])
            features_dict[feature] = feature_df 
        return features_dict
    
    def run_data(self) -> pd.DataFrame: 
        features_dict = self.load_data()
        final_df = reduce(
            lambda left, right: pd.merge(left, right, on="Date", how="outer"),
            features_dict.values()
        )
        return final_df