from functools import reduce
import numpy as np
import os
import pandas as pd
from pathlib import Path

from scripts.preprocessing.build_feature_matrix import FeatureMatrixBuilder
from scripts.preprocessing.build_macromarket_matrix import BuildMacroMarketMatrix
from scripts.preprocessing.build_targets import ForwardReturns
from scripts.preprocessing.get_feature_signals import GetFeatureSignals


from scripts.signals.market import MarketSignals

BASE_DIR = Path(__file__).resolve().parents[1]
MACRO_DIR = BASE_DIR / "data" / "raw" / "macro"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_COMPANIES_DIR = PROCESSED_DIR / "companies"
PROCESSED_COMPANIES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_MARKETS_DIR = PROCESSED_DIR / "markets"
PROCESSED_MARKETS_DIR.mkdir(parents=True, exist_ok=True)

# ALPHA COMPANY SIGNALS

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

# MARKET SIGNALS 
market_paths_dict = {
    "returns": os.path.join(PROCESSED_MARKETS_DIR, "returns.parquet"),
    "momentum": os.path.join(PROCESSED_MARKETS_DIR, "momentum.parquet"),
    "volatility": os.path.join(PROCESSED_MARKETS_DIR, "volatility.parquet"),
    "drawdown": os.path.join(PROCESSED_MARKETS_DIR, "drawdown.parquet")
}


# MACRO SIGNALS 
macro_paths_dict = {
    "currency_rates": os.path.join(MACRO_DIR, "currency_rates.parquet"),
    "interest_rates": os.path.join(MACRO_DIR, "interest_rates.parquet"),
    "vix": os.path.join(MACRO_DIR, "vix.parquet")
}




feature_matrix_pipeline_dict = {
    "feature_matrix_first": os.path.join(PROCESSED_DIR, "feature_matrix_first.parquet")
}

if __name__ == "__main__": 
    
    # MARKET SIGNALS
    MarketSignals(market_paths_dict).run_data()
    market_df = BuildMacroMarketMatrix(market_paths_dict).run_data()
    
    # MACRO SIGNALS
    macro_df = BuildMacroMarketMatrix(macro_paths_dict).run_data()
    
    GetFeatureSignals(processed_paths_dict).run_data()
    
    feature_dfs_dict = {}
    for feature, feature_list in predictive_factors_dict.items(): 
        df = pd.read_parquet(processed_paths_dict[feature])[["Date", "Ticker"] + predictive_factors_dict[feature]].copy()
        feature_dfs_dict[feature] = df
    
    target_dfs = ForwardReturns().run_data()[["Date", "Ticker", "future_return_5d"]]
    
    feature_matrix_df = FeatureMatrixBuilder(feature_dfs_dict, target_dfs).run_data()
    
    feature_matrix_df = feature_matrix_df.merge(
        market_df,
        on="Date",
        how="left",
        validate="many_to_one"
    )

    feature_matrix_df = feature_matrix_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(feature_matrix_df)
    feature_matrix_df.to_parquet(feature_matrix_pipeline_dict["feature_matrix_first"], index=False, engine="pyarrow")
    
    
    feature_matrix_stock.to_parquet(
        feature_matrix_pipe
    )
    
    
    
