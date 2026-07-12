import numpy as np
import pandas as pd

import yfinance as yf


class GetCompanyData: 
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        
    def get_data(self): 
        data = yf.download(
            self.company_codes, auto_adjust=True, start=self.start_date, end=self.end_date, progress=False
        )