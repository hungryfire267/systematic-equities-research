import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from utils import cross_sectional_ranking

BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class MeanVolatility: 
    def __init__(self) -> None: 
        self.returns_df = pd.read_parquet(companies_paths_dict["returns"])

    def get_rolling_realised_volatility(self, X: np.ndarray, window: int) -> np.ndarray:
        rv = np.log(np.sqrt((X ** 2).rolling(window).sum())) 
        rv = rv.replace([np.inf, -np.inf], np.nan).dropna()
        rv = rv.reset_index() 
        rv = rv.drop(columns=["index"])
        return rv.to_numpy().flatten()
    
    def AR_OLS(self, X: np.ndarray) -> tuple[float, float, float]: 
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
    
    
    def run(self, windows_list: list, set_window: int) -> tuple[pd.DataFrame, dict]: 
        assert(set_window in windows_list)
        
        parameters_dict = dict()
        final_rank_dict = dict()
        companies_list = list(self.returns_df.columns[1:])
        for window in windows_list: 
            mean_vol_dict = dict()
            for company in companies_list: 
                company_returns = self.returns_df[company]
                rv = self.get_rolling_realised_volatility(company_returns, window)
                mu_hat, kappa_hat, sigma_hat = self.AR_OLS(rv)
                expected_change = kappa_hat * (rv - mu_hat)
                scaled_score = expected_change / sigma_hat
                
                mean_vol_dict[company] = scaled_score
                
                if (window == set_window): 
                    parameters_dict[company] = [mu_hat, kappa_hat, sigma_hat]
            
            mean_vol_score_df = pd.DataFrame(mean_vol_dict)
            mean_vol_rank_df = cross_sectional_ranking(mean_vol_score_df, mean_higher_is_better=True)
            final_rank_dict[window] = mean_vol_rank_df
            
        parameters_df = pd.DataFrame(parameters_dict, index = ["mu", "kappa", "sigma"])
        
        return parameters_df, final_rank_dict
    