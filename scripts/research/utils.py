import pandas as pd

category_cols = { 
    "beta": ["market_beta", "industry_beta", "market_resid_vol", "industry_resid_vol"], 
    "mean_volatility": ["mean_volatility"],
    "microstructure": ["dv_liquidity", "amihud"],
    "momentum": ["momentum", "id"],
    "momentum_liquidity": ["momentum_liquidity"],
    "pvo": ["pvo"],
    "reversal": ["reversal", "rsr"],
    "reversal_illiquidity": ["reversal_illiquidity_(5", "reversal_illiquidity_(10,", "reversal_illiquidity_(21"]
}

def get_category_dicts(category_name: str): 
    category_dicts = {}
    valid_cols = category_cols[category_name]
    for col in valid_cols: 
        category_dicts[col] = []
        
    return category_dicts

def get_category_cols(signal_name: str, category_name: str) -> str | None:
    valid_cols = category_cols[category_name]
    for col in valid_cols: 
        if signal_name.startswith(col): 
            return col
        
    return None
    


def get_final_dict(ic_df_dict: dict, category_name: str) -> dict:
    category_dicts = get_category_dicts(category_name)
    for signal_name in ic_df_dict.keys(): 
        col = get_category_cols(signal_name, category_name)
        
        if col is not None:
            category_dicts[col].append(signal_name)
    
    return category_dicts

def summary_plot_final_df(summary_df: pd.DataFrame): 
    
    summary_plot_df = summary_df.T
    summary_plot_df.columns = summary_plot_df.iloc[0]
    summary_plot_df = summary_plot_df[1:].copy()
    
    return summary_plot_df

