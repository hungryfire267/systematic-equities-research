import numpy as np
import os
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

UNIVERSE_PATH = BASE_DIR / "data" / "asx_companies.csv"
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"


companies_paths_dict = {
    "prices": os.path.join(COMPANIES_DIR, "prices.parquet"),
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class PairsTrading: 
    def __init__(self, window): 
        self.company_df = pd.read_csv(UNIVERSE_PATH)
        self.returns_df = pd.read_parquet(Path(r"data/raw/companies/returns.parquet"))
        self.prices_df = pd.read_parquet(Path("data/raw/companies/prices.parquet"))
        self.window = window
        self.sector_dict = dict()
        self.similar_companies = dict()
        self.coint_validation = dict() 
        self.pair_list = []
    
    def find_sector(self, company_code: str) -> str: 
        sector = self.company_df.loc[self.company_df["asxCode"] == company_code, "industry"].values[0]
        return sector
        
    def get_sector_df(self, company_code: str) -> None:
        sector = self.find_sector(company_code)
        if sector not in self.sector_dict.keys(): 
            sector_companies = self.company_df.loc[self.company_df["industry"] == sector]["asxCode"].values
            sector_companies_final = [company + ".AX" for company in sector_companies]
            self.sector_dict[sector] =  sector_companies_final
    
    def calculate_distances(self, returns_sector_df) -> pd.DataFrame: 
        X = returns_sector_df.fillna(0).values
        diff = X[:, :, None] - X[:, None, :]
        D = (diff ** 2).mean(axis = 0)
        distance_matrix = pd.DataFrame(D, index=returns_sector_df.columns, columns=returns_sector_df.columns)
        return distance_matrix
    
    def get_pairs(self) -> None:
        for sector, tickers in self.sector_dict.items(): 
            if len(tickers) < 2: 
                continue
            
            returns_sector_df = self.returns_df[tickers]
            distance_matrix = self.calculate_distances(returns_sector_df)
            
            paired = set() 
            
            for company in tickers: 
                if company in paired: 
                    continue
                
                candidates = [t for t in tickers if (t not in paired and t != company)]
                if not candidates: 
                    continue
                
                partner = distance_matrix.loc[candidates, company].idxmin()
                self.similar_companies[company] = partner
                self.similar_companies[partner] = company
                paired.add(company)
                paired.add(partner)
                
            leftovers = [t for t in tickers if t not in paired]
            if leftovers:
                last = leftovers[0]
                ranked = distance_matrix[last].drop(index=last).sort_values()
                if len(ranked) >= 2:
                    fallback = ranked.index[1]
                else:
                    fallback = ranked.index[0]

                self.similar_companies[last] = fallback
                
    def run_cointegration_tests(self): 
        count = 0
        for company in self.similar_companies.keys(): 
            partner = self.similar_companies[company]
            df = self.returns_df[[company, partner]].dropna()
            x = df[partner]
            y = df[company]
            _, p_value, _ = coint(y, x, trend = "c")
            if p_value <= 0.05: 
                self.coint_validation[company] = partner
                count += 1
        print(count)
        
    def simplify_coint_validation(self): 
        for company, partner in self.coint_validation: 
            if (company, partner) not in self.pair_list and (partner, company) not in self.pair_list: 
                self.pair_list.append(company, partner)
    
    def run_model(self):
        for company, partner in self.pair_list:
            df = self.prices_df[[company, partner]].dropna() 
            S = df[company] - df[partner]
            mu_hat, kappa_hat, sigma_hat = self.AR_OLS(S)
            Z = (S - mu_hat)/ sigma_hat
            self.z_score_dict[f"{company}_{partner}"] = Z
            
    def run(self): 
        companies_list = list(self.returns_df.columns[1:])
        for company in companies_list: 
            company_code = company.split(".")[0]
            self.get_sector_df(company_code)    
            
            
            
        self.get_pairs()
        print(self.similar_companies)
        self.run_cointegration_tests()
 