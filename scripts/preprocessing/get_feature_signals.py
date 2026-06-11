import numpy as np
import pandas as pd

from scripts.signals.beta import BetaFeatures
from scripts.signals.microstructure import Microstructure
from scripts.signals.momentum import Momentum
from scripts.signals.pvo import PVO
from scripts.signals.reversal import Reversal
from scripts.signals.reversal_illiquidity import ReversalIlliquidity



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
    return ml_df

signal_configs = [ 
    {
        "name": "Beta",
        "class": BetaFeatures, 
        "params": {
            "window_list": np.array([10, 21, 63, 126])
        }
    },
    {
        "name": "Microstructure",
        "class": Microstructure, 
        "params": {
            "window_list": np.array([21, 63, 126])
        }
    },
    {
        "name": "Momentum",
        "class": Momentum,
        "params": {}
    }, 
    {
        "name": "PVO",
        "class": PVO, 
        "params": {
            "span_list": np.array([(26, 12)]),
            "extreme_list": np.array([(0.01, 0.99)])
        }
    }, 
    {
        "name": "Reversal",
        "class": Reversal, 
        "params": {
            "windows_list": np.array([5, 10, 21])
        }
    },
    {
        "name": "Reversal Illiquidity",
        "class": ReversalIlliquidity, 
        "params": {
            "reversal_window_list": np.array([5, 10, 21]),
            "illiquidity_window_list": np.array([21, 63, 126])
        }
    }
]


final_features = {}
for config in signal_configs:
    print(f"Running the data for {config['name']}")
    signal = config["class"](**config["params"])
    signal_dict = signal.run_data() 
    
    for feature, feature_dict in signal_dict.items(): 
        final_features[feature] = reshape_feature_dict(feature_dict, feature)

print(final_features)