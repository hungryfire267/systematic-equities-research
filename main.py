import datetime as dt
import pandas as pd
from pathlib import Path

import os


from scripts.run_fetch import ASXPipeline
# from scripts.signals import Fundamentals, Microstructure, BetaFeatures, Momentum, MomentumLiquidity

UNIVERSE_PATH = Path("data/asx_companies.csv")


if __name__ == "__main__": 
    companies_df = pd.read_csv(UNIVERSE_PATH)
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=1461)
    
    pipeline = ASXPipeline(companies_df, start_date, end_date)
    pipeline.get_data()
    
    
    
    # Different script
    # pipeline = PairsTrading(60).run()
    # pipeline = Momentum(0.25, 0.75).get_Momentum()
    
    # Reversal(0.25, 0.75, [5, 10, 21])
    
    # pipeline = Microstructure([5, 10, 21, 63])
    
    
    
    
    
    
    
    