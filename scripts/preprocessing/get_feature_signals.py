from functools import reduce
import numpy as np
import os
import pandas as pd
from pathlib import Path

from scripts.signals.beta import BetaFeatures
from scripts.signals.mean_volatility import MeanVolatility
from scripts.signals.microstructure import Microstructure
from scripts.signals.momentum import Momentum
from scripts.signals.momentum_liquidity import MomentumLiquidity
from scripts.signals.pvo import PVO
from scripts.signals.reversal import Reversal
from scripts.signals.reversal_illiquidity import ReversalIlliquidity

BASE_DIR = Path(__file__).resolve().parents[2]
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

def reshape_feature_dict(df_dict, feature_name): 
    melted_dfs = [] 
    
    for key, df in df_dict.items(): 
        df = df.copy()
        long_df = df.melt(
            id_vars=['Date'], 
            var_name='Ticker', 
            value_name=f'{feature_name}_{key}'
        )
        long_df = long_df.set_index(["Date", "Ticker"])
        melted_dfs.append(long_df)
        
    ml_df = pd.concat(melted_dfs, axis=1).reset_index()
    ml_df = ml_df.sort_values(by=['Date', 'Ticker'], ascending=[True, True])
    ml_df = ml_df.reset_index()
    ml_df = ml_df.drop(columns=["index"])
    return ml_df

signal_configs = [ 
    {
        "name": "beta",
        "class": BetaFeatures, 
        "params": {
            "window_list": np.array([10, 21, 63, 126])
        }
    },
    {
        "name": "mean_volatility", 
        "class": MeanVolatility, 
        "params": {
            "windows_list": np.array([5, 10, 21, 63]), 
            "set_window": 21
        }
    }, 
    {
        "name": "microstructure",
        "class": Microstructure, 
        "params": {
            "window_list": np.array([21, 63, 126])
        }
    },
    {
        "name": "momentum",
        "class": Momentum,
        "params": {}
    }, 
    {
        "name": "momentum_liquidity",
        "class": MomentumLiquidity, 
        "params": {
            "liquidity_window_list":  np.array([21, 63, 126])
        }
    },
    {
        "name": "pvo",
        "class": PVO, 
        "params": {
            "span_list": np.array([(26, 12)]),
            "extreme_list": np.array([(0.01, 0.99)])
        }
    }, 
    {
        "name": "reversal",
        "class": Reversal, 
        "params": {
            "windows_list": np.array([5, 10, 21])
        }
    },
    {
        "name": "reversal_illiquidity",
        "class": ReversalIlliquidity, 
        "params": {
            "reversal_window_list": np.array([5, 10, 21]),
            "illiquidity_window_list": np.array([21, 63, 126])
        }
    }
]

final_features = {}
parameters_features = {}
for config in signal_configs:
    print(f"Running the data for {config['name']}")
    signal = config["class"](**config["params"])
    signal_dict = signal.run_data() 
    
    
    name = config["name"]
    
    summary_feature_dict = dict()
    for feature, feature_dict in signal_dict.items(): 
        if (name == "mean_volatility" and feature == "parameters"): 
            parameters_features[name] = feature_dict
            continue
        else:
            summary_feature_dict[feature] = reshape_feature_dict(feature_dict, feature)
            
    merged_features = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Date", "Ticker"], how="outer"
        ), summary_feature_dict.values()
    )
    final_features[name] = merged_features
    merged_features.to_parquet(processed_paths_dict[name], index=False, engine="pyarrow")

print(final_features)