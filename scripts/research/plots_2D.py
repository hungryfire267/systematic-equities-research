
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import re
import seaborn as sns

def plot_ic_timeseries_2D(ic_df_dict: dict, category_name: str): 
    name_dict = {
        "reversal_illiquidity": "Reversal Illiquidty"
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    
    axs = np.atleast_1d(axs).ravel()
    fig.suptitle(f"Time Series of IC score across {name_dict[category_name]} signals")
    
    i = 0 
    for category_window, category_dict in ic_df_dict.items(): 
        for signal_name, signal_df in category_dict.items(): 
            df = signal_df.copy()
            df["IC_63D"] = df["IC"].rolling(63).mean()
            axs[i].plot(df["Date"], df["IC_63D"], linewidth=2, label=f"{signal_name}_63D")
        
        axs[i].legend()
        axs[i].set_title(f"Line plot of Reversal {category_window}", fontsize=13)
        axs[i].set_xlabel(f"Date", fontsize=12)
        axs[i].set_ylabel("IC Score", fontsize=12)
        i += 1
            
    plt.show()

def plot_ic_summary_bar_2D(final_summary_dict: dict, category_name:str, metric="mean_ic"): 
    metric_dict = { 
        "mean_ic": "Mean IC", 
        "ic_ir": "IC Information Ratio"
    }
    
    name_dict = {
        "reversal_illiquidity": "Reversal Illiquidty"
    }
    
    title, y_label = category_name.split("_")
    
    # new_labels = [re.search(r',\s*(\d+)\)', s).group(1) for s in labels]
    
    i = 0
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f"Summary statistics of {metric_dict[metric]} across {name_dict[category_name]} signals", fontsize=13)
    for key, df in final_summary_dict.items(): 
        values = df.loc[metric].sort_values()
        labels = list(values.index)
        new_labels = [re.search(r',\s*(\d+)\)', s).group(1) for s in labels]
        axs[i].set_title(f"Bar plot of {name_dict[category_name]} signal with {title} of rolling window {metric_dict[metric]}", fontsize=12)
        axs[i].set_xlabel(f"{metric_dict[metric]}", fontsize=11)
        axs[i].set_ylabel(f"{metric}", fontsize=11)
        sns.barplot(
            x=values.values,
            y=values.index,
            ax=axs[i], 
            palette="mako"
        )
        
        axs[i].set_xlabel(f"{metric_dict[metric]}", fontsize=11)
        axs[i].set_ylabel(f"{y_label} rolling window", fontsize=11)
        axs[i].tick_params(axis="x", labelrotation=45)
        axs[i].set_yticks(range(len(labels)))
        axs[i].set_yticklabels(new_labels)
        axs[i].axvline(0, linestyle="--")
        
        i += 1
    plt.tight_layout()
    plt.show()