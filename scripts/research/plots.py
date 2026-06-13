import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


category_cols = { 
    "beta": ["market_beta", "industry_beta", "market_resid_vol", "industry_resid_vol"], 
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

def plot_ic_timeseries(ic_df_dict: dict, category_name: str): 
    category_dicts = get_final_dict(ic_df_dict, category_name)
    n = len(category_dicts.keys())
    
    if (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    axs = axs.ravel() 
    
    i = 0 
    for category, category_signal_names in category_dicts.items(): 
        for signal_name in category_signal_names: 
            df = ic_df_dict[signal_name].copy()
            df["IC_63D"] = df["IC"].rolling(63).mean()
            axs[i].plot(df["Date"], df["IC_63D"], linewidth=2, label=f"{signal_name}_63D")
        axs[i].legend()
        i += 1
            
    plt.show()
    
