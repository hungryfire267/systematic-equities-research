import os
import pandas as pd
from pathlib import Path
from scripts.signals.utils import date_parser, cross_sectional_ranking

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

class BetaFeatures: 
    def __init__(self, window_list: list): 
        self.companies_df = pd.read_csv(UNIVERSE_PATH)
        self.returns_df = date_parser(pd.read_parquet(returns_paths_dict["company_returns"]))
        self.asx_returns_df = date_parser(pd.read_parquet(returns_paths_dict["asx_returns"]))
        self.industry_returns_df = date_parser(pd.read_parquet(returns_paths_dict["industry_returns"]))
        
        self.window_list = window_list
        
    
    @staticmethod
    def beta_calculation(combined_df: pd.DataFrame, beta_type: str, window: int) -> pd.Series: 
        cov = combined_df["company"].rolling(window=window).cov(combined_df[beta_type])
        var = combined_df[beta_type].rolling(window=window).var()
        return cov/var
    
    @staticmethod
    def vol_calculation(combined_df: pd.DataFrame, beta_type: str, window: int, beta: pd.Series) -> pd.Series:
        residuals = combined_df["company"] - beta * combined_df[beta_type]
        vol = residuals.rolling(window=window).std()
        return vol
    
    def get_market_rolling_beta_vol(self, window: int) -> tuple[pd.DataFrame, pd.DataFrame]: 
        market_beta_df_dict, market_vol_df_dict = dict(), dict()

        market_returns = self.asx_returns_df["^AXJO"].copy()
        for company in self.returns_df.columns: 
            company_returns = self.returns_df[company]
            combined_df = pd.concat([market_returns, company_returns], axis=1)
            combined_df.columns = ["market", "company"]
            beta = self.beta_calculation(combined_df, "market", window)
            vol = self.vol_calculation(combined_df, "market", window, beta)
            
            market_beta_df_dict[company] = beta
            market_vol_df_dict[company] = vol
        
        market_beta_df = pd.DataFrame(market_beta_df_dict)
        market_vol_df = pd.DataFrame(market_vol_df_dict)
            
        return market_beta_df, market_vol_df
    
    def get_industry_company_return(self, company: str) -> pd.Series: 
        company_final = company.split(".")[0].upper()
        condition = (self.companies_df["asxCode"] == company_final)
        company_industry = self.companies_df.loc[condition, "industry"].iloc[0]

        industry_returns = self.industry_returns_df[company_industry]
        return industry_returns
    
    
    def get_industry_rolling_beta_vol(self, window: int) -> tuple[pd.DataFrame, pd.DataFrame]: 
        industry_beta_df_dict, industry_vol_df_dict = dict(), dict()
        for company in self.returns_df.columns: 
            company_returns = self.returns_df[company]
            industry_returns = self.get_industry_company_return(company)
            combined_df = pd.concat([industry_returns, company_returns], axis = 1)
            combined_df.columns = ["industry", "company"]

            beta = self.beta_calculation(combined_df, "industry", window)
            vol = self.vol_calculation(combined_df, "industry", window, beta)
            
            industry_beta_df_dict[company] = beta
            industry_vol_df_dict[company] = vol
        
        industry_beta_df = pd.DataFrame(industry_beta_df_dict)
        industry_vol_df = pd.DataFrame(industry_vol_df_dict)
            
        return industry_beta_df, industry_vol_df

    
    def run_data(self) -> tuple[dict, dict, dict, dict]: 
        
        market_beta_df_dict, market_vol_df_dict = dict(), dict() 
        industry_beta_df_dict, industry_vol_df_dict = dict(), dict()
        
        for window in self.window_list: 
            market_beta_df, market_vol_df = self.get_market_rolling_beta_vol(window)
            industry_beta_df, industry_vol_df = self.get_industry_rolling_beta_vol(window)
            
            market_beta_df_dict[window] = cross_sectional_ranking(market_beta_df, higher_is_better = True).reset_index()
            industry_beta_df_dict[window] = cross_sectional_ranking(industry_beta_df, higher_is_better = True).reset_index()
            market_vol_df_dict[window] = cross_sectional_ranking(market_vol_df, higher_is_better = True).reset_index()
            industry_vol_df_dict[window] = cross_sectional_ranking(industry_vol_df, higher_is_better = True).reset_index()

        final_betafeatures_dict = {
            "market_beta": market_beta_df_dict,
            "industry_beta": industry_beta_df_dict,
            "market_resid_vol": market_vol_df_dict,
            "industry_resid_vol": industry_vol_df_dict
        }
            
        return final_betafeatures_dict