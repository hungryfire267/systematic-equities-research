import datetime as dt
import os
import pandas as pd
from pathlib import Path

from scripts.run_fetch import ASXPipeline

BASE_DIR = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = Path(os.path.join(BASE_DIR, "data/asx_companies.csv"))

if __name__ == "__main__": 
    companies_df = pd.read_csv(UNIVERSE_PATH)
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=1461)
    
    pipeline = ASXPipeline(companies_df, start_date, end_date)
    pipeline.get_data()
    
    
    
    
    
    
    
    
    