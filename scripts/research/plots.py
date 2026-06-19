import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns


from scripts.research import utils

def plot_ic_timeseries(ic_df_dict: dict, category_name: str): 
    category_dicts = utils.get_final_dict(ic_df_dict, category_name)
    n = len(category_dicts.keys())
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 3): 
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
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

def plot_ic_summary_bar(
    ic_df_dict: pd.DataFrame, 
    summary_df: pd.DataFrame, 
    name_dict: pd.DataFrame,
    category_name: str,
    metric: str
): 
    metric_dict = { 
        "mean_ic": "Mean IC", 
        "ic_ir": "IC Information Ratio"
    }
    
    category_dicts = utils.get_final_dict(ic_df_dict, category_name)
    n = len(category_dicts.keys())
    
    if (n == 1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
    elif (n == 3): 
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    elif (n == 4): 
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))       
    
    
    fig.suptitle(f"Summary statistics of {metric_dict[metric]} across {category_name} signals", fontsize=14)
    
    axs = np.atleast_1d(axs).ravel()
    
    i = 0
    for category, category_signal_names in category_dicts.items(): 
        df = summary_df[category_signal_names].copy() 
        
        values = df.loc[metric].sort_values()
        
        print(category)
        print(name_dict)
        
        axs[i].set_title(f"Bar plot of {name_dict[category]} signal with {metric_dict[metric]}", fontsize=12)
        axs[i].set_xlabel(f"{metric_dict[metric]}", fontsize=11)
        axs[i].set_ylabel(f"{name_dict[category]}", fontsize=11)
        sns.barplot(
            x=values.values,
            y=values.index,
            ax=axs[i], 
            palette="mako"
        )

        axs[i].axvline(0, linestyle="--")
        
        axs[i].tick_params(axis="x", labelrotation=45)
        
        yticks = axs[i].get_yticklabels()
        factor_names = [tick.get_text() for tick in yticks]
        new_labels = [
            re.search(r"\d+$", name).group() if re.search(r"\d+$", name) else name
            for name in factor_names
        ]

        axs[i].set_yticklabels(new_labels)
        
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
    
    fig.suptitle(f"Mean Forwards Returns against {category_name}")
    i = 0
    for category, category_signal_names in category_dicts.items(): 
        axs[i].set_title(f"Mean Forward Returns against {category} category signal quantile")
        summary_category_list = []
        for signal in category_signal_names: 
            summary_df = summary_dict[signal]
            plot_df = summary_df[
                summary_df["quintile"].apply(lambda x: isinstance(x, (int, np.integer)))
            ].copy() 
            summary_category_list.append(plot_df)
        summary_category_df = pd.concat(summary_category_list, ignore_index=True)
        sns.lineplot(summary_category_df, x="quintile", y = "mean_forward_return", hue="factor", ax=axs[i], marker="o")
        axs[i].set_xlabel("Quintile", fontsize=11)
        axs[i].set_ylabel("Mean Forward Return", fontsize=11)
        i += 1
    plt.tight_layout()
    plt.show()
    
def plot_correlation(correlation_dfs_dict: dict, category_type: str): 
    n = len(correlation_dfs_dict.items())
    
    if (n == 1): 
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    elif (n == 2): 
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    elif (n == 3): 
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    fig.suptitle(f"Correlation of {category_type} category signals", fontsize=14)
    axs = np.atleast_1d(axs).ravel()
    i = 0 
    for key, df in correlation_dfs_dict.items(): 
        df.index = [x.split('_')[-1] for x in df.index]
        df.columns = [x.split('_')[-1] for x in df.columns]
        sns.heatmap(
            df, ax=axs[i], cmap="crest", annot=True, linewidths=1
        )
        axs[i].set_title(f"{key} Correlation signal", fontsize=12)
        axs[i].set_xlabel(f"{key} Window", fontsize=11)
        axs[i].set_ylabel(f"{key} Window", fontsize=11)
        i += 1
    plt.tight_layout()
    plt.show()