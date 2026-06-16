import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from scripts.research import utils

def plot_ic_timeseries(ic_df_dict: dict, category_name: str): 
    category_dicts = utils.get_final_dict(ic_df_dict, category_name)
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
    category_dicts = utils.get_final_dict(ic_df_dict, category_name)
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
        
        axs[i].set_title(f"Bar plot of {category}", fontsize=12)
        axs[i].set_xlabel(f"Signal name", fontsize=11)
        axs[i].set_ylabel(f"{metric}", fontsize=11)
        sns.barplot(
            x=values.values,
            y=values.index,
            ax=axs[i], 
            palette="mako"
        )

        axs[i].axvline(0, linestyle="--")
        
        i += 1
    plt.tight_layout()
    plt.show()
    
def plot_quintiles(quintile_df_dict: dict, summary_dict: dict, category_name):
    category_dicts = utils.get_final_dict(quintile_df_dict, category_name)
    
    n = len(category_dicts.keys())
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    axs = np.atleast_1d(axs).ravel()
    
    i = 0
    for category, category_signal_names in category_dicts.items(): 
        axs[i].set_title(f"Mean Forward Returns by {category} category signal quantile")
        summary_category_list = []
        for signal in category_signal_names: 
            summary_df = summary_dict[signal]
            plot_df = summary_df[
                summary_df["quintile"].apply(lambda x: isinstance(x, (int, np.integer)))
            ].copy() 
            summary_category_list.append(plot_df)
        summary_category_df = pd.concat(summary_category_list, ignore_index=True)
        sns.lineplot(summary_category_df, x="quintile", y = "mean_forward_return", hue="factor", ax=axs[i])
        i += 1
    plt.tight_layout()
    plt.show()