
import numpy as np
import os
import pandas as pd
from pathlib import Path

from scripts.preprocessing.get_feature_signals import GetFeatureSignals
from scripts.preprocessing.build_feature_matrix import FeatureMatrixBuilder

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


processed_paths_dict = {
    "beta": os.path.join(PROCESSED_DIR, "beta.parquet"),
    "mean_volatility": os.path.join(PROCESSED_DIR, "mean_volatility.parquet"),
    "microstructure": os.path.join(PROCESSED_DIR, "microstructure.parquet"),
    "momentum": os.path.join(PROCESSED_DIR, "momentum.parquet"), 
    "momentum_liquidity": os.path.join(PROCESSED_DIR, "momentum_liquidity.parquet"),
    "pvo": os.path.join(PROCESSED_DIR, "pvo.parquet"),
    "reversal": os.path.join(PROCESSED_DIR, "reversal.parquet"),
    "reversal_illiquidity": os.path.join(PROCESSED_DIR, "reversal_illiquidity.parquet")
}

predictive_factors_dict = {
    "beta": ["market_beta_63", "industry_beta_63", "market_resid_vol_126", "industry_resid_vol_126"],
    "mean_volatility": ["mean_volatility_10", "mean_volatility_21", "mean_volatility_63"],
    "microstructure": ["amihud_21"],
    "momentum": ["momentum_252_21"],
    "momentum_liquidity": ["momentum_liquidity_21"], 
    "reversal": ["reversal_5", "rsr_21"]
}

if __name__ == "__main__": 
    GetFeatureSignals(processed_paths_dict).run_data()
    
    feature_dfs_dict = {}
    for feature, feature_list in predictive_factors_dict.items(): 
        df = pd.read_parquet(processed_paths_dict[feature])[["Date", "Ticker"] + predictive_factors_dict[feature]].copy()
        df = df.set_index(["Date", "Ticker"])
        feature_dfs_dict[feature] = df
    
    print(feature_dfs_dict)
    
    
    
