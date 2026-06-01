
import datetime as dt
import pandas as pd
import yfinance as yf 


from abc import ABC, abstractmethod

class MacroPipeline(ABC):
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
    
    @abstractmethod
    def get_raw_data(self): 
        pass
    
    @abstractmethod
    def clean_data(self, raw_data) -> pd.DataFrame: 
        pass
    
    def run_data(self) -> pd.DataFrame: 
        raw_data = self.get_raw_data() 
        clean_data = self.clean_data(raw_data)
        return clean_data
    
class CurrencyRates(MacroPipeline): 
    def __init__(self, start_date, end_date):
        super().__init__(start_date, end_date)
        self.currency_list = ["CNY", "EUR", "GBP", "HKD", "JPY", "NZD", "USD"]
        
        
        
        
class UnemploymentRate(MacroPipeline): 
    def __init__(self, start_date, end_date): 
        super().__init__(start_date, end_date)
        
        self.series_id_mapping = { 
            "Trend": "trend", 
            "Seasonally Adjusted": "seasonal"
        }
        
        
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
    

if __name__ == "__main__": 
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=1461)
    unemployment_data = UnemploymentRate(start_date, end_date).run_data()
    print(unemployment_data.head())
    
    
    
    
    