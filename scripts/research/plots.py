import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_ic_timeseries(ic_df: pd.DataFrame, title: str):
    print(ic_df.columns)
    ic_df = ic_df.copy()
    ic_df["IC_63D"] = ic_df["IC"].rolling(63).mean()
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(ic_df["Date"], ic_df["IC"], label="Daily IC")
    plt.plot(ic_df["Date"], ic_df["IC_63D"], linewidth=2, label="63D Rolling Mean")
    plt.legend()
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("IC")
    plt.tight_layout()
    plt.show()