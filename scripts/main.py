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

PROCESSED_FEATURE_DIR = PROCESSED_DIR / "features"
PROCESSED_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ALPHA COMPANY SIGNALS

processed_paths_dict = {
    "autocorr": os.path.join(PROCESSED_COMPANIES_DIR, "autocorr.parquet"),
    "beta": os.path.join(PROCESSED_COMPANIES_DIR, "beta.parquet"),
    "mean_volatility": os.path.join(PROCESSED_COMPANIES_DIR, "mean_volatility.parquet"),
    "microstructure": os.path.join(PROCESSED_COMPANIES_DIR, "microstructure.parquet"),
    "momentum": os.path.join(PROCESSED_COMPANIES_DIR, "momentum.parquet"), 
    "momentum_liquidity": os.path.join(PROCESSED_COMPANIES_DIR, "momentum_liquidity.parquet"),
    "pvo": os.path.join(PROCESSED_COMPANIES_DIR, "pvo.parquet"),
    "reversal": os.path.join(PROCESSED_COMPANIES_DIR, "reversal.parquet"),
    "reversal_illiquidity": os.path.join(PROCESSED_COMPANIES_DIR, "reversal_illiquidity.parquet"), 
    "trend": os.path.join(PROCESSED_COMPANIES_DIR, "trend.parquet")
}

predictive_factors_dict = {
    "autocorr": ["autocorr_21", "autocorr_63"],
    "beta": ["market_beta_63", "industry_beta_63", "market_resid_vol_126", "industry_resid_vol_126"],
    "mean_volatility": ["mean_volatility_10", "mean_volatility_21", "mean_volatility_63"],
    "microstructure": ["amihud_21"],
    "momentum": ["momentum_252_21"],
    "momentum_liquidity": ["momentum_liquidity_21"], 
    "reversal": ["reversal_5", "rsr_21"], 
    "trend": ["trend_21", "trend_63", "trend_126", "r2_21", "r2_63", "r2_126"]
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


# PIPELINE DICT

feature_matrix_pipeline_dict = {
    "feature_matrix_stock": os.path.join(PROCESSED_FEATURE_DIR, "feature_matrix_stock.parquet"), 
    "feature_matrix_market": os.path.join(PROCESSED_FEATURE_DIR, "feature_matrix_market.parquet"), 
    "feature_matrix_macro_market": os.path.join(PROCESSED_FEATURE_DIR, "feature_matrix_macro_market.parquet")
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
    
    # 1. Stock Features Only
    feature_matrix_stock = feature_matrix_df.copy() 
    
    # 2. Stock + Market
    feature_matrix_market = feature_matrix_df.merge(
        market_df,
        on="Date",
        how="left",
        validate="many_to_one"
    )
    
    # 3. Stock + Market + Macro
    feature_matrix_macro_market = (
        feature_matrix_market
        .merge(
            macro_df,
            on="Date",
            how="left",
            validate="many_to_one"
        )
    )
    
    feature_matrix_stock = feature_matrix_stock.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    feature_matrix_market = feature_matrix_market.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    feature_matrix_macro_market = feature_matrix_macro_market.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    
    feature_matrix_stock.to_parquet(
        feature_matrix_pipeline_dict["feature_matrix_stock"], index=False, engine="pyarrow"
    )
    
    feature_matrix_market.to_parquet(
        feature_matrix_pipeline_dict["feature_matrix_market"], index=False, engine="pyarrow"
    )
    
    feature_matrix_macro_market.to_parquet(
        feature_matrix_pipeline_dict["feature_matrix_macro_market"], index=False, engine="pyarrow"
    )
    
    
    
