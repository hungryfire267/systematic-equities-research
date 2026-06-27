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
from scripts.signals.trend import Trends

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class GetFeatureSignals: 
    def __init__(
        self, 
        processed_paths_dict: dict,
        beta_features=np.array([10, 21, 63, 126]), 
        mv_features=np.array([5, 10, 21, 63]),
        microstructure_features=np.array([21, 63, 126]),
        trend_features=np.array([21, 63, 126]),
        reversal_features=np.array([5, 10, 21])
    ):
        self.processed_paths_dict = processed_paths_dict
        self.signal_configs = [ 
            {
                "name": "beta",
                "class": BetaFeatures, 
                "params": {
                    "window_list": beta_features
                }
            },
            {
                "name": "mean_volatility", 
                "class": MeanVolatility, 
                "params": {
                    "windows_list": mv_features, 
                    "set_window": 21
                }
            }, 
            {
                "name": "microstructure",
                "class": Microstructure, 
                "params": {
                    "window_list": microstructure_features
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
                    "liquidity_window_list":  microstructure_features
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
                    "windows_list": reversal_features
                }
            },
            {
                "name": "reversal_illiquidity",
                "class": ReversalIlliquidity, 
                "params": {
                    "reversal_window_list": reversal_features,
                    "illiquidity_window_list": microstructure_features
                }
            }, 
            {
                "name": "trend", 
                "class": Trends, 
                "params": {
                    "rolling_window_list":trend_features
                }
            }
        ]
        
    def reshape_feature_dict(self, df_dict, feature_name): 
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
    
    def run_data(self):
        final_features = {}
        parameters_features = {}
        for config in self.signal_configs:
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
                    summary_feature_dict[feature] = self.reshape_feature_dict(feature_dict, feature)
                    
            merged_features = reduce(
                lambda left, right: pd.merge(
                    left, right, on=["Date", "Ticker"], how="outer"
                ), summary_feature_dict.values()
            )
            final_features[name] = merged_features
            merged_features.to_parquet(self.processed_paths_dict[name], index=False, engine="pyarrow")
            