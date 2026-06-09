import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from utils import date_parser, cross_sectional_ranking

BASE_DIR = Path(__file__).resolve().parents[2]

UNIVERSE_PATH = BASE_DIR / "data" / "asx_companies.csv"
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
ASX_DIR = BASE_DIR / "data" / "raw" / "asx"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

returns_paths_dict = {
    "company_returns": os.path.join(COMPANIES_DIR, "returns.parquet"),
    "asx_returns": os.path.join(ASX_DIR, "asx_returns.parquet"), 
    "industry_returns": os.path.join(INDUSTRY_DIR, "industry_returns.parquet")
}





class Reversal: 
    def __init__(self, windows_list: list[int]): 
        self.returns_df = date_parser(pd.read_parquet(returns_paths_dict["company_returns"]))
        self.asx_returns_df = date_parser(pd.read_parquet(returns_paths_dict["asx_returns"]))
        self.industry_returns_df = date_parser(pd.read_parquet(returns_paths_dict["industry_returns"]))
        
        self.companies_df = pd.read_csv(UNIVERSE_PATH)
        
        self.industry_dict = self.companies_df.set_index("asxCode")["industry"].to_dict()
        self.companies_df["code"] =  [str(code) + ".AX" for code in list(self.companies_df["asxCode"])]
        self.total_days = self.returns_df.shape[0]

        self.windows_list = windows_list
        
    def get_reversal(self, window: int) -> pd.DataFrame: 
        cumulative_returns = (
            1 + self.returns_df.rolling(window=window).apply(np.prod, raw=True)
        ) - 1
        reversal_score = - cumulative_returns.reset_index()
        return reversal_score
        

    def get_rsr(self, window: int) -> pd.DataFrame: 
        market_returns = self.asx_returns_df["^AXJO"]
        company_list = list(self.returns_df.columns[1:])
        
        rsr_dict = dict()
        for company in company_list: 
            company_returns = self.returns_df[company]
            industry = self.industry_dict[company]
            industry_returns = self.industry_returns_df[industry]
            total_returns = pd.concat([industry_returns, market_returns, company_returns], axis=1).dropna()
            null_days = self.total_days - total_returns.shape[0]
            y_returns = total_returns[company]
            X_returns = total_returns.drop(columns=[company])
            linear_model = LinearRegression().fit(X_returns, y_returns)
            residuals = y_returns - linear_model.predict(X_returns)
            residuals = residuals.reindex(range(0, residuals.index.max() + 1))
            residuals = residuals.sort_index()
            residuals.index = self.returns_df["Date"]
            
            rsr_dict[company] = - ((1 + residuals)).rolling(window=window).apply(np.prod, raw=True) -1
        
        rsr_score =  pd.DataFrame(rsr_dict).reset_index()
        return rsr_score

    
    def get_reversal_ranks(self, window: int) -> tuple[pd.DataFrame, pd.DataFrame]: 
        reversal_score = self.get_reversal(window)
        rsr_score = self.get_rsr(window)
        
        reversal_rank_df = cross_sectional_ranking(reversal_score, higher_is_better=True)
        rsr_rank_df = cross_sectional_ranking(rsr_score, higher_is_better=True)
        
        return reversal_rank_df, rsr_rank_df
             
    def run_data(self) -> tuple[dict, dict]: 
        reversal_df_dict, rsr_df_dict = dict(), dict()
        for window in self.windows_list: 
            reversal_df_dict[window], rsr_df_dict[window] = self.get_reversal_ranks(window)  
        
        return reversal_df_dict, rsr_df_dict