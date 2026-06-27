import numpy as np 
import pandas as pd

class InterestRates: 
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        self.interest_rate_url = r"https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"
    
    def fetch_interest_rate_data(self): 
        ir_data = pd.read_excel(self.interest_rate_url, sheet_name = "Data", header=1)
        ir_data = ir_data.set_index("Title")
        ir_data.index.name = "Date"
        return ir_data
    
    def clean_data(self, ir_data: pd.DataFrame): 
        ir_data = ir_data.copy()
        series_id_index = ir_data.index.get_loc("Series ID")
        ir_data = ir_data.iloc[series_id_index + 1:, :].copy()
        
        # Cash Rate Target - Official RBA Policy Stance
        # Interbank Overnight Cash Rate - actual overnight funding rate
        # EOD 3-month BABS - Bank funding / credit conditions

        relevant_cols = [
            "Cash Rate Target", 
            "Interbank Overnight Cash Rate", 
            "EOD 3-month BABs/NCDs"
        ]
        
        ir_data = ir_data[relevant_cols]
        ir_data = ir_data.reset_index()
        ir_data["Date"] = pd.to_datetime(ir_data["Date"]).dt.date
        
        start_condition = ir_data["Date"] >= self.start_date
        end_condition = ir_data["Date"] <= self.end_date
        ir_data = ir_data[start_condition & end_condition].reset_index(drop=True)
        return ir_data
    
    def run_data(self): 
        ir_data = self.fetch_interest_rate_data()
        ir_data = self.clean_data(ir_data)
        return ir_data