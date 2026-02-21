import pandas as pd
import requests
import yfinance as yf
from pathlib import Path
import time

API_COMPANIES_PATH = "https://asx.api.markitdigital.com/asx-research/1.0/home/moversAndShakers?numberOfRows=250"
resp = requests.get(API_COMPANIES_PATH)
if resp.status_code != 200: 
    raise Exception("Failed to fetch data from ASX API")
data = resp.json()
companies = data.get("data", [])
companies_df = pd.DataFrame(companies)
companies_df_v2 = companies_df[companies_df["industry"] != "Not Applic"].copy()[:200]
companies_df_final = companies_df_v2[["asxCode", "companyName", "industry"]]


universe_path = Path("data/asx_companies.csv")
companies_df_final.to_csv(universe_path, index=False)




