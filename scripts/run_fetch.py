import datetime as dt
import numpy as np
import os 
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import yfinance as yf


UNIVERSE_PATH = Path("data/asx_companies.csv")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES_DIR = RAW_DIR/"companies"
COMPANIES_DIR.mkdir(parents=True, exist_ok=True)

ASX_DIR = RAW_DIR/"asx"
ASX_DIR.mkdir(parents=True, exist_ok=True)


def get_companies_list(universe_path): 
    companies_df = pd.read_csv(universe_path)
    companies_codes = [str(company) + ".AX" for company in companies_df["asxCode"].tolist()]
    return companies_codes
    
class ASXPipeline: 
    def __init__(self, companies_df, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        self.company_codes = [str(company) + ".AX" for company in companies_df["asxCode"].tolist()]
        self.company_paths_dict = {   
            "prices": os.path.join(COMPANIES_DIR, "prices.parquet"),
            "log_prices": os.path.join(COMPANIES_DIR, "log_prices.parquet"),
            "volume": os.path.join(COMPANIES_DIR, "volume.parquet"),
            "returns": os.path.join(COMPANIES_DIR, "returns.parquet"), 
            "log_returns": os.path.join(COMPANIES_DIR, "log_returns.parquet"),
            "market_cap": os.path.join(COMPANIES_DIR, "market_cap.parquet")
        }
        self.asx_paths_dict = { 
            "index": os.path.join(ASX_DIR, "asx_index.parquet"), 
            "returns": os.path.join(ASX_DIR, "asx_returns.parquet"), 
            "log_returns": os.path.join(ASX_DIR, "asx_log_returns.parquet")
        }
    
    def GetData(self, market_cap: pd.DataFrame | None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: 
        data = yf.download(
            self.company_codes, auto_adjust=True, start=self.start_date, end=self.end_date, progress=False
        )
        data = data.reset_index()
        prices = self.DataframeParser(data[["Date", "Close"]])
        temp_prices = prices.set_index("Date")
        log_prices = np.log(temp_prices).reset_index()
        volume = self.DataframeParser(data[["Date", "Volume"]])
        returns = self.ReturnsParser(prices, "returns")
        log_returns = self.ReturnsParser(prices, "log_returns")
        
        
        asx_index = yf.download(
            "^AXJO", auto_adjust=True, start=self.start_date, end=self.end_date, progress=False
        )
        asx_index = asx_index.reset_index() 
        asx_prices = self.DataframeParser(asx_index[["Date", "Close"]])
        asx_returns = self.ReturnsParser(asx_prices, "returns")
        asx_log_returns = self.ReturnsParser(asx_prices, "log_returns")
        
        date_condition_one = prices["Date"] >= dt.datetime(2026, 1, 11)
        date_condition_two = prices["Date"] <= dt.datetime(2026, 1, 25)
        print(asx_returns.loc[(date_condition_one) & (date_condition_two), :])
        print(asx_log_returns.loc[(date_condition_one) & (date_condition_two), :])
        
        
        prices.to_parquet(self.company_paths_dict["prices"], index=False, engine="pyarrow")
        log_prices.to_parquet(self.company_paths_dict["log_prices"], index=False, engine="pyarrow")
        volume.to_parquet(self.company_paths_dict["volume"], index=False, engine="pyarrow")
        returns.to_parquet(self.company_paths_dict["returns"], index=False, engine="pyarrow")
        log_returns.to_parquet(self.company_paths_dict["log_returns"], index=False, engine="pyarrow")
        market_cap.to_parquet(self.company_paths_dict["market_cap"], index=False, engine="pyarrow")
        
        company_data_dict = {
            "prices": prices, "volume": volume, "returns": returns, "log_returns": log_returns, "market_cap": market_cap
        }
        
        asx_data_dict = { 
            "prices": asx_prices, "returns": asx_returns, "log_returns": asx_log_returns
        }
        
        return company_data_dict, asx_data_dict
    
    
    def DataframeParser(self, df: pd.DataFrame) -> pd.DataFrame: 
        new_df = df.rename(columns={("Date", ""): "Date"})
        new_cols = []
        for a, b in new_df.columns:
            if a == "Date":
                new_cols.append("Date")
            else:
                new_cols.append(b)
        new_df.columns = new_cols
        return new_df
    
    def ReturnsParser(self, df: pd.DataFrame, types) -> pd.DataFrame: 
        df = df.set_index("Date").copy() 
        
        if (types == "returns"): 
            new_df = df.pct_change()
            
        elif (types == "log_returns"): 
            new_df = np.log(df).diff()
        
        new_df = new_df.reset_index() 
        return new_df
    
    
    def GetMarketCap(self): 
        market_list = [] 
        i = 1
        for company in self.company_codes:
            try:
                company_ticker = yf.Ticker(company)
                shares_outstanding = company_ticker.info.get("sharesOutstanding")
                ticker_history = company_ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                market_cap = shares_outstanding * ticker_history["Close"]
                market_cap.index = pd.to_datetime(market_cap.index.date)
                market_cap.rename(company, inplace=True)
                market_list.append(market_cap)
                if (i % 40 == 0):
                    print(f"Successfully fetched market cap {i}/200")
                i += 1
            except Exception as e: 
                print(f"Failed for {company}: {e}")
        market_cap_df = pd.concat(market_list, axis=1, ignore_index=False)
        market_cap_df = market_cap_df.reset_index().rename(columns={"index": "Date"})
        return market_cap_df
    
    def getSectorReturns(self, market_cap_path, companies_df):
        market_cap_df = pd.read_parquet(market_cap_path)
        returns_df = pd.read_parquet(self.company_paths_dict["returns"])
        sector_list = companies_df["industry"].unique().tolist()
        industry_return_dict = {}
        for industry in sector_list: 
            print(f"Processing industry: {industry}")
            industry_df = companies_df[companies_df["industry"] == industry]
            industry_companies_list = [str(company) + ".AX" for company in industry_df["asxCode"].unique().tolist()]
            industry_market_cap_df = market_cap_df[industry_companies_list]
            weights = industry_market_cap_df.div(industry_market_cap_df.sum(axis=1), axis=0)
            company_returns = returns_df[industry_companies_list]
            industry_returns = (weights * company_returns).sum(axis=1)
            """         
            date_series = pd.Series(pd.date_range(start=self.start_date, end=self.end_date), name="Date")
            date_series = date_series[date_series.dt.dayofweek < 5]
            industry_returns.index = date_series
            """
            industry_return_dict[industry] = industry_returns
            break

        
        
    
    def FetchData(self, file_name: str) -> pd.DataFrame | None: 
        try: 
            path = self.company_paths_dict[file_name]
            df = pd.read_parquet(path)
            return df
            
        except KeyError as e: 
            valid_keys = list(self.company_paths_dict.keys())
            raise KeyError(
                f"Invalid file name {file_name}. Please choose from the following:", valid_keys
            ) from e    
    