
import datetime as dt
import pandas as pd
import yfinance as yf 


from abc import ABC, abstractmethod

API_KEY = 

class MacroPipeline(ABC):
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
    
    @abstractmethod
    def get_raw_data(self) -> pd.DataFrame | list[pd.DataFrame]: 
        pass
    
    @abstractmethod
    def clean_data(self, raw_data: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame: 
        pass
    
    def run_data(self) -> pd.DataFrame: 
        raw_data = self.get_raw_data() 
        clean_data = self.clean_data(raw_data)
        return clean_data
    
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
        return [currency_2022, currency_current] 
    
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
        df["Date"] = pd.to_datetime(df.index).date
        start_condition = df["Date"] >= self.start_date
        end_condition = df["Date"] <= self.end_date
        df = df[start_condition & end_condition].reset_index(drop=True)
        df = df[["Date"] + relevant_cols]
        return df
    
class CIP
        
        
        
        
        
class UnemploymentRate(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        
        self.series_id_mapping = { 
            "Trend": "trend", 
            "Seasonally Adjusted": "seasonal"
        }
        
        self.ai_client = genai.Client()
    
    def get_unemployment_dates(self) -> pd.DataFrame: 
        
        
        
        prompt = f"""
                Generate a clean historical release schedule for the Australia Bureau of Statistics (ABS) 
                Labour Force unemployment rate from January 2022 to April 2026. Keep the entries in 
                chronological order.
            """
        response = 
            
        
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
        return df
    
class RerefenceMonth: 
    
        
class yield_curve: 
    pass

class interest_rates: 
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        
    def get_interest_rates(self):
        cash_ocr_url = r"https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"

        ocr = pd.read_excel(cash_ocr_url, sheet_name="Data", header=1)
        title_series = ocr["Title"]
        series_id_index = title_series[title_series.str.contains("Series ID", na=False)].index[0]

        ocr_final = ocr.iloc[series_id_index + 1 :].copy()
        return ocr_final
    
    def clean_interest_rates(self, ocr_final: pd.DataFrame) -> pd.DataFrame:
        df = ocr_final.copy() 
        df["Date"] = pd.to_datetime(df["Title"]).dt.date
        
        # Cash Rate Target - Official RBA Policy Stance
        # Interbank Overnight Cash Rate - actual overnight funding rate
        # EOD 3-month BABS - Bank funding / credit conditions
        # 1 month OIS - Market Expectations of cash rate in 1 month (short term)
        # 6 month OIS - Market Expectations of cash rate in 6 months (medium term)
        # 3 month treasury note - Risk-free short-end yield
        
        desirable_columns = [
            "Cash Rate Target", 
            "Interbank Overnight Cash Rate", 
            "EOD 3-month BABs/NCDs", 
            "1-month OIS",
            "6-month OIS",
            "3-month Treasury Note"
        ]
        df = df[["Date"] + desirable_columns]
        start_condition = df["Date"] >= self.start_date
        end_condition = df["Date"] <= self.end_date
        df = df[start_condition & end_condition].reset_index(drop=True)
        return df
    
    def run_data(self) -> pd.DataFrame: 
        ocr_final = self.get_interest_rates()
        ocr_final_df = self.clean_interest_rates(ocr_final)
        return ocr_final_df

def get_day_prefix(day): 
    if day >= 11 and day <= 13:
        return "th"
    return {1: "st", 2: "2nd", 3: "rd"}.get(day % 10, "th")


if __name__ == "__main__": 
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=1461)
    
    end_date_year = end_date.year
    
    
    currency_data = CurrencyRates(start_date, end_date).run_data()
    print(currency_data.head())
    unemployment_data = UnemploymentRate(start_date, end_date).run_data()
    print(unemployment_data.head())
    

    
    