

category_cols = { 
    "beta": ["market_beta", "industry_beta", "market_resid_vol", "industry_resid_vol"], 
    "mean_volatility": ["mean_volatility"],
    "microstructure": ["dv_liquidity", "amihud"],
    "momentum": ["momentum", "id"],
    "reversal": ["reversal", "rsr"]
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