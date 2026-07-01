import numpy as np
import pandas as pd


class GetCompanyData: 
    def __init__(self, start_date, end_date): 
        self.start_date = start_date
        self.end_date = end_date
        
    def get_data(self): 
        