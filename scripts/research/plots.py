import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from 


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

def plot_ic_timeseries(ic_df_dict: dict, category_name: str): 
    category_dicts = get_final_dict(ic_df_dict, category_name)
    n = len(category_dicts.keys())
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    axs = np.atleast_1d(axs).ravel()
    fig.suptitle(f"Time Series of IC score across {category_name} signals")
    
    i = 0 
    for category, category_signal_names in category_dicts.items(): 
        for signal_name in category_signal_names: 
            df = ic_df_dict[signal_name].copy()
            df["IC_63D"] = df["IC"].rolling(63).mean()
            axs[i].plot(df["Date"], df["IC_63D"], linewidth=2, label=f"{signal_name}_63D")
        
        axs[i].legend()
        axs[i].set_title(f"Line plot of {category}", fontsize=13)
        axs[i].set_xlabel(f"Date", fontsize=12)
        axs[i].set_ylabel("IC Score", fontsize=12)
        i += 1
            
    plt.show()

def plot_ic_summary_bar(ic_df_dict: pd.DataFrame, summary_df: pd.DataFrame, category_name: str, metric: str): 
    category_dicts = get_final_dict(ic_df_dict, category_name)
    n = len(category_dicts.keys())
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    
    fig.suptitle(f"Summary statistics of {metric} across {category_name} signals", fontsize=14)
    
    axs = np.atleast_1d(axs).ravel()
    
    i = 0
    for category, category_signal_names in category_dicts.items(): 
        df = summary_df[category_signal_names].copy() 
        
        values = df.loc[metric].sort_values()
        
        axs[i].set_title(f"Bar plot of {category}", fontsize=13)
        axs[i].set_xlabel(f"Signal name", fontsize=12)
        axs[i].set_ylabel(f"{metric}", fontsize=12)
        axs[i].barh(values.index, values.values)
        axs[i].axvline(0, linestyle="--")
        
        i += 1
    plt.tight_layout()
    plt.show()
    
def plot_quintiles(summary_dict: dict, category_type: str | None = None ) -> tuple[plt.Figure, plt.Axes]:
    n = len(category_cols[category_type])
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    axs = np.atleast_1d(axs).ravel()
    
    i = 0
    for signal, summary_df in summary_dict.items():
        plot_df = summary_df[
            summary_df["quintile"].apply(lambda x: isinstance(x, (int, np.integer)))
        ]

        axs[i].bar(
            plot_df["quintile"].astype(str),
            plot_df["mean_forward_return"]
        )
        axs[i].axhline(0, linestyle="--", linewidth=1)
        axs[i].set_xlabel("Quintile")
        axs[i].set_ylabel("Mean forward return")
        axs[i].set_title(f"Mean Forward Return by {signal} Signal Quintile")
        
        i += 1
    plt.tight_layout()
    plt.show()