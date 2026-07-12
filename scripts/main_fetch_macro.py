import datetime as dt
import numpy as np 
import os
import pandas as pd
from pathlib import Path 

from scripts.macro.fetch_cr import CurrencyRates
from scripts.macro.fetch_ir import InterestRates
from scripts.macro.fetch_vix import VIX


BASE_DIR = Path(__file__).resolve().parents[1]
MACRO_DIR = BASE_DIR / "data" / "raw" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

macro_paths_dict = { 
    "currency_rates": os.path.join(MACRO_DIR, "currency_rates.parquet"), 
    "interest_rates": os.path.join(MACRO_DIR, "interest_rates.parquet"),
    "vix": os.path.join(MACRO_DIR, "vix.parquet")
}

end_date = dt.date(2026, 6, 14)
start_date = end_date - dt.timedelta(days=1461)

cr_data = CurrencyRates(start_date, end_date).run_data()
ir_data = InterestRates(start_date, end_date).run_data()
vix_data = VIX(start_date, end_date).run_data()

cr_data.to_parquet(macro_paths_dict["currency_rates"], index=False, engine="pyarrow")
ir_data.to_parquet(macro_paths_dict["interest_rates"], index=False, engine="pyarrow")
vix_data.to_parquet(macro_paths_dict["vix"], index=False, engine="pyarrow")

