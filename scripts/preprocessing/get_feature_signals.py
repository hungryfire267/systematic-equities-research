import numpy as np
import pandas as pd

from signals import BetaFeatures, Microstructure

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

beta_windows = np.array([10, 21, 63, 126])

market_beta_df_dict, market_vol_df_dict, industry_beta_df_dict, industry_vol_df_dict = BetaFeatures(beta_windows).get_data()

market_beta_features = reshape_feature_dict(market_beta_df_dict, "market_beta")
industry_beta_features = reshape_feature_dict(industry_beta_df_dict, "industry_beta")
market_resid_vol_features = reshape_feature_dict(market_vol_df_dict, "market_resid_vol")
industry_resid_vol_features = reshape_feature_dict(industry_vol_df_dict, "market_resid_vol")
