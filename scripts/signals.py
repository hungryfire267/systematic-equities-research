import numpy as np
import pandas as pd
from pathlib import Path
from scripts.run_fetch import ASXPipeline
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint

UNIVERSE_PATH = Path("data/asx_companies.csv")

def percentile_check(lower_percentile: float, upper_percentile: float) -> None: 
    if not (0 <= lower_percentile < upper_percentile <= 100): 
        raise ValueError("Require 0 <= lower_percentile < upper_percentile <= 100") 


class Momentum: 
    def __init__(self, lower_percentile: float, upper_percentile: float): 
        self.returns_df = pd.read_parquet(Path(r"data/raw/companies/returns.parquet"))
        percentile_check(lower_percentile, upper_percentile)
        self.lower = lower_percentile
        self.upper = upper_percentile
        
        self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
        self.returns_df = self.returns_df.set_index("Date")
        
    def get_Momentum(self): 
        momentum_score = self.returns_df.shift(21).rolling(252).sum()
        ranks = momentum_score.rank(axis=1, pct=True)
        momentum_signal_df = pd.DataFrame(0, index=ranks.index, columns=list(ranks.columns))
        momentum_signal_df[ranks >= self.upper] = 1
        momentum_signal_df[ranks <= self.lower] = -1
        self.momentum_signal_df = momentum_signal_df.reset_index()
        print(self.momentum_signal_df)
        
class PVO: 
    def __init__(
        self, extreme_list: tuple[float, float], signal_percentile: tuple[float, float], span_list: tuple[int, int]
    ): 
        self.volume_df = pd.read_parquet(Path(r"data/raw/companies/volume.parquet"))
        print("Checking PVO extremity check percentiles:")
        percentile_check(extreme_list[0], extreme_list[1])
        print("Checking valid signal percentiles:")
        percentile_check(signal_percentile[0], signal_percentile[1])
        
        self.slow, self.fast = span_list[0], span_list[1]
        self.lower, self.upper = signal_percentile[0], signal_percentile[1]
        
        
    def compute_ema(self, df: pd.DataFrame, span: int) -> pd.DataFrame:
        return df.ewm(span=span, adjust=False).mean()
    
    def calculate_pvo(self, df: pd.DataFrame): 
        ema_slow = self.compute_ema(df, span=self.slow_span)
        ema_fast = self.compute_ema(df, span=self.fast_span)
        pvo_df = (ema_fast - ema_slow)/ema_slow
        
        pvo_df = pvo_df.clip(lower=pvo_df.quantile(self.lower), upper=pvo_df.quantile(self.upper)) # Capping the extremes
        
    
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
        
            
    def AR_OLS(self, X: pd.DataFrame): 
        X_curr = X[1:]
        X_prev = X[:-1]
        
        regression_model = LinearRegression()
        regression_model.fit(X_prev.reshape(-1, 1), X_curr)
        a_hat = regression_model.intercept_
        phi_hat = regression_model.coef_[0]
        
        mu_hat = a_hat / (1 - phi_hat)
        kappa_hat = -np.log(phi_hat)
        errors = X_curr - (a_hat + phi_hat * X_prev)
        var_eps = np.var(errors, ddof=2)
        sigma_hat = np.sqrt(var_eps * (2.0 * kappa_hat) / (1.0 - np.exp(-2.0 * kappa_hat)))
        
        return mu_hat, kappa_hat, sigma_hat  
            
    def run(self): 
        companies_list = list(self.returns_df.columns[1:])
        for company in companies_list: 
            company_code = company.split(".")[0]
            self.get_sector_df(company_code)
        self.get_pairs()
        print(self.similar_companies)
        self.run_cointegration_tests()
            
class MeanVolatility: 
    
    def __init__(self, windows) -> None: 
        # returns_df = pipeline.FetchData("returns")
        self.windows = windows
        returns_df = pd.read_parquet(Path(r"data/raw/companies/returns.parquet"))
        self.df = returns_df
    
    def get_rolling_realised_volatility(self, X: np.ndarray) -> np.ndarray:
        rv = np.log(np.sqrt((X ** 2).rolling(self.windows).sum())) 
        rv = rv.replace([np.inf, -np.inf], np.nan).dropna()
        rv = rv.reset_index() 
        rv = rv.drop(columns=["index"])
        return rv.to_numpy().flatten()
    
    def AR_OLS(self, X: np.ndarray): 
        X_curr = X[1:]
        X_prev = X[:-1]
        
        regression_model = LinearRegression()
        regression_model.fit(X_prev.reshape(-1, 1), X_curr)
        a_hat = regression_model.intercept_
        phi_hat = regression_model.coef_[0]
        
        mu_hat = a_hat / (1 - phi_hat)
        kappa_hat = -np.log(phi_hat)
        errors = X_curr - (a_hat + phi_hat * X_prev)
        var_eps = np.var(errors, ddof=2)
        sigma_hat = np.sqrt(var_eps * (2.0 * kappa_hat) / (1.0 - np.exp(-2.0 * kappa_hat)))
        
        return mu_hat, kappa_hat, sigma_hat
    
    
    def run(self): 
        companies_list = list(self.df.columns[1:])
        kappa_dict, mu_dict, sigma_dict = dict(), dict(), dict()
        i = 0 
        for company in companies_list: 
            company_returns = self.df[company]
            rv = self.get_rolling_realised_volatility(company_returns)
            mu_hat, kappa_hat, sigma_hat = self.AR_OLS(rv)
            kappa_dict[company] = kappa_hat
            mu_dict[company] = mu_hat
            sigma_dict[company] = sigma_hat
            i += 1
            print(mu_hat)
            if (i == 5):
                break
        theta_df = pd.DataFrame([theta_dict])
        mu_df = pd.DataFrame([mu_dict])
        sigma_df = pd.DataFrame([sigma_dict])   
        return theta_df, mu_df, sigma_df
    

        
    

if __name__ == "__init__": 
    pipeline = MeanVolatility()
    theta_df, mu_df, sigma_df = MeanVolatility().run()
    print(mu_df)
    print(theta_df)