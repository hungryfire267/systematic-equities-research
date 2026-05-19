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

INDUSTRY_DIR = RAW_DIR/"industry"
INDUSTRY_DIR.mkdir(parents=True, exist_ok=True)


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
        
        self.industry_paths_dict = { 
            "returns": os.path.join(INDUSTRY_DIR, "industry_returns.parquet")
        }
    
    def GetData(self, market_cap: pd.DataFrame | None, industry_returns: pd.DataFrame | None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: 
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
        
        self.get_fundamental_metrics(prices, market_cap)
        
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
        
        asx_prices.to_parquet(self.asx_paths_dict["index"], index=False, engine="pyarrow")
        asx_returns.to_parquet(self.asx_paths_dict["returns"], index=False, engine="pyarrow")
        asx_log_returns.to_parquet(self.asx_paths_dict["log_returns"], index=False, engine="pyarrow")
        
        industry_returns.to_parquet(self.industry_paths_dict["returns"], index=False, engine="pyarrow")
        
        company_data_dict = {
            "prices": prices, "volume": volume, "returns": returns, "log_returns": log_returns, "market_cap": market_cap
        }
        
        asx_data_dict = { 
            "prices": asx_prices, "returns": asx_returns, "log_returns": asx_log_returns
        }
        
        industry_data_dict = { 
            "returns": industry_returns
        }
        
        return company_data_dict, asx_data_dict, industry_data_dict
    
    
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
            industry_df = companies_df[companies_df["industry"] == industry]
            industry_companies_list = [str(company) + ".AX" for company in industry_df["asxCode"].unique().tolist()]
            industry_market_cap_df = market_cap_df[industry_companies_list]
            weights = industry_market_cap_df.div(industry_market_cap_df.sum(axis=1), axis=0)
            company_returns = returns_df[industry_companies_list]
            industry_returns = (weights * company_returns).sum(axis=1)
            industry_returns.index = market_cap_df["Date"]
            industry_returns.name = industry
            industry_returns.iloc[0] = np.nan     
            industry_return_dict[industry] = industry_returns
        industry_return_df = pd.DataFrame(industry_return_dict).reset_index()
        return industry_return_df
    
    def get_equity(self, bs: pd.DataFrame) -> pd.Series | None:
        if "Total Stockholder Equity" in list(bs.index):
            equity = bs.loc["Total Stockholder Equity"]
        elif "Total Equity Gross Minority Interest" in list(bs.index):
            minority = bs.loc["Minority Interest"].fillna(0) if "Minority Interest" in bs.index else 0
            equity = bs.loc["Total Equity Gross Minority Interest"] - minority
        elif "Common Stock" in list(bs.index): 
            equity = bs.loc["Common Stock"]
        else: 
            equity = None
        return equity
    
    def get_assets(self, bs: pd.DataFrame, equity: pd.Series) -> pd.Series | None: 
        if "Total Assets" in list(bs.index): 
            assets = bs.loc["Total Assets"]
        elif "Total Liabilities Net Minority Interest" in list(bs.index): 
            liabilities = bs.loc["Total Liabilities Net Minority Interest"].fillna(0)
            assets = equity + liabilities  
        else: 
            assets = None
        return assets
    
    def get_income(self, income_stmt: pd.DataFrame) -> pd.Series | None:
        if "Net Income" in list(income_stmt.index): 
            net_income = income_stmt.loc["Net Income"]
        else: 
            net_income = None
        return net_income
    
    def get_fundamentals(self, prices_df: pd.DataFrame) -> None: 
        company_fundamentals_dict = dict() 
        for company in self.company_codes: 
            try: 
                company_ticker = yf.Ticker(company)
                bs = company_ticker.balance_sheet.copy()
                income_stmt = company_ticker.income_stmt.copy() 
                
                shares = bs.loc["Ordinary Shares Number"]
                
                net_income = self.get_income(income_stmt)
                equity = self.get_equity(bs)
                assets = self.get_assets(bs, equity)
                
                
                fundamentals = pd.DataFrame({
                    "Shares": shares,
                    "Equity": equity, 
                    "Assets": assets, 
                    "Net Income": net_income
                })
                
                for key in fundamentals.keys(): 
                    fundamentals[key] = fundamentals[key].fillna(method="ffill")
                
                roa = net_income / assets
                roe = net_income / equity
                
                fundamentals["ROA"] = roa
                fundamentals["ROE"] = roe
            
            company_fundamentals_dict[company] = fundamentals
            break
        return company_fundamentals_dict
    
    def get_ptb(self, prices_df: pd.DataFrame) -> None: 
        ptb_dict = dict()
        for company in self.company_codes: 
            fundamentals = self.fundamentals_dict[company].copy()
            fundamentals = fundamentals.sort_index(ascending=False)
            
            ptb_df = pd.DataFrame({
                    "Date": prices_df["Date"].copy(),
                    "Price": prices_df[company].copy()
            })
            
            fund_tmp = fundamentals.reset_index().rename(columns={"index": "Date"})
            fund_tmp["Date"] = pd.to_datetime(fund_tmp["Date"])
            
            ptb_df = ptb_df.sort_values("Date")
            fund_tmp = fund_tmp.sort_values("Date")

            ptb_df = pd.merge_asof(
                ptb_df,
                fund_tmp[["Date", "bvps"]],
                on="Date",
                direction="backward"
            )
            
            na_condition = ptb_df["Price"].isna()
            ptb_df.loc[na_condition, "bvps"] = np.nan

            ptb_df[company] = ptb_df["Price"] / ptb_df["bvps"]
            ptb_df = ptb_df[["Date", company]]
            ptb_df = ptb_df.set_index("Date")
            ptb_dict[company] = ptb_df[company]
            
        ptb_final_df = pd.DataFrame(ptb_dict)
        
        return ptb_final_df
                
                
    def get_dividend_yield(self, prices_df: pd.DataFrame) -> None: 
        dividend_yield_dict = dict()
        
        earliest = prices_df["Date"].min()
        latest = prices_df["Date"].max() 
        
        zero_dividend_companies = []
        i = 0        
        for company in self.company_codes: 
            ptd_df = pd.DataFrame({
                "Date": prices_df["Date"].copy(), 
                "Price": prices_df[company].copy()
            })
            try: 
                ticker = yf.Ticker(company)
                dividends = ticker.dividends
                dividends.index = pd.to_datetime(dividends.index).tz_localize(None)

                dividends = dividends.reset_index().sort_index(ascending=True)
                earliest_date = dividends.loc[dividends["Date"] < earliest, "Date"].max()
                dividends_filtered = dividends[dividends["Date"] >= earliest_date].reset_index(drop=True)
                
                
                ptd_df = pd.merge_asof(
                    ptd_df,
                    dividends_filtered[["Date", "Dividends"]],
                    on="Date",
                    direction="backward"
                )
                
                div_events = (
                    dividends.set_index("Date")["Dividends"].sort_index()
                )

                calendar = pd.date_range(
                    start=min(div_events.index.min(), ptd_df["Date"].min()),
                    end=ptd_df["Date"].max(),
                    freq="D"
                )
                
                div_daily = div_events.reindex(calendar, fill_value=0)
                
                trailing_div = div_daily.rolling("365D").sum()
                
                ptd_df["trailing_annual_dividend"] = (
                    trailing_div
                    .reindex(ptd_df["Date"])
                    .to_numpy()
                )

                ptd_df["trailing_dividend_yield"] = (
                    ptd_df["trailing_annual_dividend"] / ptd_df["Price"]
                )

                ptd_df[company] = (
                    100 * ptd_df["trailing_dividend_yield"]
                )
                
                
            except Exception as e: 
                zero_dividend_companies.append(company)
                
                print(f"Failed to fetch dividend yield for {company}")
                ptd_df[company] = 0.0
                
            
            ptd_df = ptd_df[["Date", company]]
            ptd_df = ptd_df.set_index("Date")
            dividend_yield_dict[company] = ptd_df[company]
            
        ptd_final_df = pd.DataFrame(dividend_yield_dict)
        
        return ptd_final_df
        
    def get_earnings_yield(self, market_cap_df: pd.DataFrame) -> dict:
        earnings_yield_dict = dict()
        for company in self.company_codes:
            fundamentals_company = self.fundamentals_dict[company].copy()
            net_income_tmp = fundamentals_company["Net Income"]
            net_income_tmp = net_income_tmp.reset_index().rename(columns={"index": "Date"})
            net_income_tmp["Date"] = pd.to_datetime(net_income_tmp["Date"])
            
            earning_yields_df = pd.DataFrame({
                "Date": market_cap_df["Date"].copy(),
                "Market Cap": market_cap_df[company].copy()
            })
            
            net_income_tmp = net_income_tmp.sort_values("Date") 
            earning_yields_df = earning_yields_df.sort_values("Date")
            
            earning_yields_df = pd.merge_asof(
                earning_yields_df,
                net_income_tmp[["Date", "Net Income"]],
                on="Date",
                direction="backward"
            )
            
            earning_yields_df[company] = earning_yields_df["Net Income"] / earning_yields_df["Market Cap"]
            earning_yields_df = earning_yields_df[["Date", company]]
            earning_yields_df = earning_yields_df.set_index("Date")
            earnings_yield_dict[company] = earning_yields_df[company]
        
        earnings_yield_df = pd.DataFrame(earnings_yield_dict)
        return earnings_yield_df
    
    def get_ROA_ROE(self, prices_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
        roa_dict, roe_dict = dict(), dict()
        
        for company in self.company_codes: 
            fundamentals_company = self.fundamentals_dict[company].copy()
            roa_roe_tmp_df = fundamentals_company[["Net Income", "Equity", "Assets"]].copy()
            
            roa_tmp_df = roa_roe_tmp_df[["Net Income", "Assets"]].copy()
            roa_tmp_df = roa_tmp_df.reset_index().rename(columns={"index": "Date"})
            roa_tmp_df["Date"] = pd.to_datetime(roa_tmp_df["Date"])
            roa_tmp_df = roa_tmp_df.sort_values("Date")
            
            
            roe_tmp_df = roa_roe_tmp_df[["Net Income", "Equity"]].copy() 
            roe_tmp_df = roe_tmp_df.reset_index().rename(columns={"index": "Date"})
            roe_tmp_df["Date"] = pd.to_datetime(roe_tmp_df["Date"])
            roe_tmp_df = roe_tmp_df.sort_values("Date")
            
            roa_roe_df = pd.DataFrame({
                "Date": prices_df["Date"].copy(), 
            })
            
            roa_df = pd.merge_asof(
                roa_roe_df, 
                roa_tmp_df,
                on="Date", 
                direction="backward"
            )
            
            roe_df = pd.merge_asof(
                roa_roe_df,
                roe_tmp_df,
                on="Date",
                direction="backward"
            )
            
            roa_df[company] = roa_df["Net Income"] / roa_df["Assets"]
            roa_df = roa_df[["Date", company]]
            roa_df = roa_df.set_index("Date")
            roa_dict[company] = roa_df[company]
            
            roe_df[company] = roe_df["Net Income"] / roe_df["Equity"]
            roe_df = roe_df[["Date", company]]
            roe_df = roe_df.set_index("Date")
            roe_dict[company] = roe_df[company]
        
        roa_final_df = pd.DataFrame(roa_dict)
        roe_final_df = pd.DataFrame(roe_dict)
        
        print(roa_final_df)
        
        print(roe_final_df)
        
        return roa_final_df, roe_final_df
    
    def get_fundamental_metrics(self, prices_df: pd.DataFrame, market_cap_df: pd.DataFrame) -> None:
        self.fundamentals_dict = self.get_fundamentals(prices_df)
        self.ptb_df = self.get_ptb(prices_df)
        self.dividend_yield_df = self.get_dividend_yield(prices_df)
        self.earnings_yield_df = self.get_earnings_yield(market_cap_df)
        self.roa_df, self.roe_df = self.get_ROA_ROE(prices_df)
    
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