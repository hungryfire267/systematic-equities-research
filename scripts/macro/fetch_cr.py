import numpy as np
import pandas as pd

class CurrencyRates: 
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
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
    
    def clean_data(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_2022 = raw_data["2022"]
        df_current = raw_data["current"]
        
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
        return df
    
    def run_data(self):
        currency_df_dict_mapping = self.get_raw_data() 
        cr_data = self.clean_data(currency_df_dict_mapping)
        cr_data = cr_data[["Date",  "A$1=CNY", "A$1=EUR", "A$1=GBP", "A$1=HKD", "A$1=JPY", "A$1=NZD", "A$1=USD"]]
        
        return cr_data