from abc import ABC, abstractmethod

import datetime as dt
from dbnomics import fetch_series
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os
import pandas as pd
from pathlib import Path
import yfinance as yf 



from pydantic import BaseModel, Field


RAW_DIR = Path("data/raw")
MACRO_DIR = RAW_DIR /"macro"
MACRO_DIR.mkdir(parents = True, exist_ok = True)




class ReleaseEntry(BaseModel):
    month_of_release: dt.date = Field(
        description="The calendar month when the data was released, e.g., 'May 2026'"
    )
    release_date: dt.date = Field(
        description="The precise release date formatted as YYYY-MM-DD"
    )


class ABSReleaseCalendar(BaseModel):
    releases: list[ReleaseEntry] = Field(description="Chronological list of all release entries")
    
 

class MacroPipeline(ABC):
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        
        self.macro_paths_dict = { 
            "CPI": os.path.join(MACRO_DIR, "cpi.parquet"), 
            "CurrencyRate": os.path.join(MACRO_DIR, "currency_rates.parquet"),
            "UnemploymentRate": os.path.join(MACRO_DIR, "unemployment_rates.parquet"), 
            "InterestRate": os.path.join(MACRO_DIR, "interest_rates.parquet"), 
            "YieldCurve": os.path.join(MACRO_DIR, "yield_curves.parquet")
        }
    
    @abstractmethod
    def get_raw_data(self) -> pd.DataFrame | dict[str, pd.DataFrame]: 
        pass
    
    @abstractmethod
    def clean_data(self, raw_data: pd.DataFrame | dict[str, pd.DataFrame]) -> pd.DataFrame: 
        pass
    
    def run_data(self) -> pd.DataFrame: 
        raw_data = self.get_raw_data() 
         
        final_data = self.clean_data(raw_data)
    
        class_name = self.__class__.__name__
        final_data.to_parquet(self.macro_paths_dict[class_name], index = False, engine="pyarrow")
    
        return final_data

class CPI(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        
        self.ai_client = genai.Client()
        self.series_mapping = {
            "monthly": "A130393721F", 
            "quarterly": "A2325847F"
        }
        
        
    def get_cpi_dates(self, cpi_type: str): 
        if cpi_type not in ("monthly", "quarterly"): 
            raise ValueError("cpi_type must be either 'monthly' or 'quarterly'")
        
        threshold_date = dt.date(2025, 10, 1)
        if self.start_date >= threshold_date: 
            release_dates_df = self.get_abs_release_dates(cpi_type="monthly")
        elif self.start_date <= dt.date(2025, 10, 1) and self.end_date >= dt.date(2025, 10, 1):
            release_dates_df_quarter = self.get_abs_release_dates(cpi_type = "quarterly")
            release_dates_df_month = self.get_abs_release_dates(cpi_type = "monthly")
        else: 
            release_dates_df = self.get_abs_release_dates(cpi_type="quarterly")
        return release_dates_df
        
    def get_abs_release_dates(self, cpi_type: str) -> pd.DataFrame:
        start_date_month = self.start_date.strftime("%B")
        start_date_year = self.start_date.year
        
        try: 
            prompt_instruction = f"""
                Generate a clean historical release schedule for the Australian Bureau of Statistics (ABS) 
                Labour Force CPI rate from {start_date_month} {start_date_year} to {end_date_month} {end_date_year}. 

                For each entry, provide the month in which the release actually occurred and its exact release date. 
                Keep the entries in strict chronological order.
            """
            
            response = self.ai_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents = prompt_instruction, 
                config = types.GenerateContentConfig(
                    response_mime_type = "application/json", 
                    response_schema = ABSReleaseCalendar, 
                    temperature = 0.05
                )
            )
        
            response_json = json.loads(response.text)
            release_date_df = pd.DataFrame(response_json["releases"])
            return release_date_df
        
        except Exception as e:
            print()
        
        
        pass
        
        
    def load_cpi_data(self, url) -> pd.DataFrame: 
        cpi_df = pd.read_excel(url, sheet_name="Data1", header=0)
        cpi_df = cpi_df.set_index("Unnamed: 0")
        cpi_df.index.name = "Date"
        return cpi_df 
    
    def get_raw_data(self) -> pd.DataFrame: 
        url_type_dict = { 
            "monthly": r"https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/apr-2026/640101.xlsx",
            "quarterly": r"https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/sep-quarter-2025/640101.xlsx"
        }
        
        df_monthly = self.load_cpi_data(url_type_dict["monthly"])
        df_quarterly = self.load_cpi_data(url_type_dict["quarterly"])
        
        df_dict_mapping = { 
            "monthly": df_monthly, 
            "quarterly": df_quarterly
        }
        
        return df_dict_mapping
    
    def wrangle_data(self, df: pd.DataFrame, df_type: str) -> pd.DataFrame:
        series_id = self.series_mapping[df_type]
        print(series_id)
        df = df[df.columns[df.loc["Series ID"].eq(series_id)]]
        df.columns = ["CPI"]
        series_id_index = df.index.get_loc("Series ID")
        df = df.iloc[series_id_index + 1: ]
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        if df_type == "monthly": 
            df = df[df["Date"] >= dt.date(2025, 11, 1)]
        return df
        
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame: 
        df_monthly, df_quarterly = raw_data.copy()
        
        df_monthly = self.wrangle_data(df_monthly, "monthly")
        df_quarterly = self.wrangle_data(df_quarterly, "quarterly")

        df = pd.concat([df_quarterly, df_monthly], axis=0).reset_index(drop=True)
        return df
        
    
class CurrencyRates(MacroPipeline): 
    def __init__(self, start_date, end_date):
        super().__init__(start_date, end_date)
        self.currency_list = ["CNY", "EUR", "GBP", "HKD", "JPY", "NZD", "USD"]
        self.exchange_2022_url = r"https://www.rba.gov.au/statistics/tables/xls-hist/2018-2022.xls"
        self.exchange_current_url = r"https://www.rba.gov.au/statistics/tables/xls-hist/2023-current.xls"
    
    def load_exchange_data(self, url) -> pd.DataFrame:
        exchange_data = pd.read_excel(url, sheet_name="Data", header=1)
        exchange_data = exchange_data.set_index("Title")
        exchange_data.index.name = "Date"
        return exchange_data
    
    def get_raw_data(self) -> pd.DataFrame: 
        currency_2022 = self.load_exchange_data(self.exchange_2022_url)
        currency_current = self.load_exchange_data(self.exchange_current_url)
        
        df_dict_mapping = { 
            "2022": currency_2022, 
            "current": currency_current
        }
        return df_dict_mapping
    
    def clean_data(self, raw_data: list[pd.DataFrame]) -> pd.DataFrame:
        df_2022, df_current = raw_data[0], raw_data[1]
        
        relevant_cols = []
        for currency in self.currency_list: 
            currency_col = "A$1=" + currency
            relevant_cols.append(currency_col) 
        
        df_2022 = df_2022[relevant_cols]
        df_current = df_current[relevant_cols]
        
        series_id_index_2022 = df_2022.index.get_loc("Series ID")
        series_id_index_current = df_current.index.get_loc("Series ID")
        assert(series_id_index_2022 == series_id_index_current)
        
        df_2022 = df_2022.iloc[series_id_index_2022 + 1:].copy()
        df_current = df_current.iloc[series_id_index_current + 1:, :].copy()
        
        df = pd.concat([df_2022, df_current], axis=0)
        df = df[relevant_cols]
        df["Date"] = pd.to_datetime(df.index).date
        start_condition = df["Date"] >= self.start_date
        end_condition = df["Date"] <= self.end_date
        df = df[start_condition & end_condition].reset_index(drop=True)
        df = df[["Date"] + relevant_cols]
        return df
    
        
        
class UnemploymentRate(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        
        self.series_id_mapping = { 
            "Trend": "trend", 
            "Seasonally Adjusted": "seasonal"
        }
        
        self.ai_client = genai.Client()
    
    def get_unemployment_dates(self) -> pd.DataFrame: 
        start_date_month = self.start_date.strftime("%B")
        start_date_year = self.start_date.year
        
        end_date_month = self.end_date.strftime("%B")
        end_date_year = self.end_date.year
        
        try: 
            prompt_instruction = f"""
                Generate a clean historical release schedule for the Australian Bureau of Statistics (ABS) 
                Labour Force unemployment rate from {start_date_month} {start_date_year} to {end_date_month} {end_date_year}. 

                For each entry, provide the month in which the release actually occurred and its exact release date. 
                Keep the entries in strict chronological order.
            """
            response = self.ai_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents = prompt_instruction, 
                config = types.GenerateContentConfig(
                    response_mime_type = "application/json", 
                    response_schema = ABSReleaseCalendar, 
                    temperature = 0.05
                )
            )
        
            response_json = json.loads(response.text)
            release_date_df = pd.DataFrame(response_json["releases"])
            return release_date_df
    
        except Exception as e: 
            print(f"Failed to get ABS release dates using Gemini API. Error: {e}")
            return None
            
        
    def get_raw_data(self) -> pd.DataFrame:
        unemployment_url = r"https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/apr-2026/62020001.xlsx"
        unemployment = pd.read_excel(unemployment_url, sheet_name="Data1", header=0)
        unemployment = unemployment.set_index("Unnamed: 0")
        unemployment.index.name = "Date"
        return unemployment 
    
    def clean_data(self, unemployment_df: pd.DataFrame) -> pd.DataFrame: 
        df = unemployment_df.copy()
        relevant_cols = df.columns[df.loc["Series ID"].isin(["A84423134K","A84423050A"])]
        df = df[relevant_cols]
        series_type = list(df.loc["Series Type", :])
        
        feature_rename_dict = dict()
        for col_feature, col_name in zip(series_type, df.columns): 
            feature_rename_dict[col_name] = "unemployment_" + self.series_id_mapping[col_feature]
        
        df = df.rename(columns=feature_rename_dict).reset_index() 
        series_id_index = df[df["Date"] == "Series ID"].index[0]
        df =df.drop(index=range(series_id_index + 1))
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        start_condition = df["Date"] >= self.start_date
        end_condition = df["Date"] <= self.end_date
        df = df[start_condition & end_condition].reset_index(drop=True)
        
        release_dates_df = self.get_unemployment_dates() 
            
        return df
    
    

class InterestRate(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        self.interest_rate_url = r"https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"
        
    def get_raw_data(self): 
        df = pd.read_excel(self.interest_rate_url, sheet_name = "Data", header=1)
        print(df)
        df = df.set_index("Title")
        df.index.name = "Date"
        return df

    
    def clean_data(self, raw_data: pd.DataFrame):
        df = raw_data.copy()
        series_id_index = df.index.get_loc("Series ID")
        df = df.iloc[series_id_index + 1:, :].copy()
        
        print(df)
        
        # Cash Rate Target - Official RBA Policy Stance
        # Interbank Overnight Cash Rate - actual overnight funding rate
        # EOD 3-month BABS - Bank funding / credit conditions

        relevant_cols = [
            "Cash Rate Target", 
            "Interbank Overnight Cash Rate", 
            "EOD 3-month BABs/NCDs"
        ]
        
        df = df[relevant_cols]
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        
        start_condition = df["Date"] >= self.start_date
        end_condition = df["Date"] <= self.end_date
        df = df[start_condition & end_condition].reset_index(drop=True)
        return df
        

    
class YieldCurve(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        self.yield_rba_series_mapping = { 
            "2Y" : "RBA/F2/FCMYGBAG2D", 
            "5Y": "RBA/F2/FCMYGBAG5D",
            "10Y": "RBA/F2/FCMYGBAG10D"
        }
        
        
        
    def get_raw_data(self): 
        data_df_dict = dict()
        for time, series in self.yield_rba_series_mapping.items(): 
            df = fetch_series(series)[["period", "value"]].copy()
            df["Date"] = pd.to_datetime(df["period"]).dt.date
            df = df[["Date", "value"]].copy() 
            data_df_dict[time] = df
        
        return data_df_dict
        
    def clean_data(self, raw_data_dict: dict) -> pd.DataFrame: 
        final_df_dict = dict()
        for time, df in raw_data_dict.items(): 
            start_condition = (df["Date"] >= self.start_date)
            end_condition = (df["Date"] <= self.end_date)
            final_df = df[start_condition & end_condition].set_index("Date").rename(
                columns = {"value": f"yield_{time}"}
            )
            final_df_dict[time] = final_df 
        df = pd.concat(final_df_dict.values(), axis=1)
        print(df)
        return df


if __name__ == "__main__": 
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=200)
    print(start_date)
    print(end_date)
    load_dotenv()
    
    interest_rate_data = InterestRate(start_date, end_date).run_data()
    yield_curve_data = YieldCurve(start_date, end_date).run_data()
    
    print(interest_rate_data)
    
    

    
    