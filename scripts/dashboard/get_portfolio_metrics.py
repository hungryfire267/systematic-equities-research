import numpy as np
import os 
import pandas as pd
from pathlib import Path


class GetPortfolioMetrics: 
    def __init__(self, )
    
    
    def get_overall_metrics(self): 
        n_positions = portfolio_df.shape[0]
        n_long = long_portfolio_df.shape[0]
        n_short = short_portfolio_df.shape[0]

        net_exposure = portfolio_df["Weight"].sum()
        if abs(net_exposure) < 1e-10:
            net_exposure = 0
        gross_exposure = portfolio_df["Weight"].abs().sum()
        long_exposure = portfolio_df.loc[
            portfolio_df["Weight"] > 0,
            "Weight"
        ].sum()
        short_exposure = portfolio_df.loc[
            portfolio_df["Weight"] < 0,
            "Weight"
        ].sum()
        
        
        
        
        
        