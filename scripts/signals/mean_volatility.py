import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scripts.signals.utils import cross_sectional_ranking, date_parser

BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

class MeanVolatility: 
    def __init__(self, windows_list: list, set_window: int) -> None: 
        self.returns_df = date_parser(pd.read_parquet(companies_paths_dict["returns"]))
        
        assert(set_window in windows_list)
        self.windows_list = windows_list
        self.set_window = set_window

    def get_rolling_realised_volatility(self, X: pd.Series, window: int) -> pd.Series:
        rolling_sum = (X ** 2).rolling(window, min_periods=window).sum()
        rolling_vol = np.sqrt(rolling_sum)

        rolling_vol = rolling_vol.where(rolling_vol > 0, np.nan)

        rv = np.log(rolling_vol)
        return rv
    
    def AR_OLS(self, X: pd.Series) -> tuple[float, float, float]:
        X_fit = X.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if len(X_fit) < 3:
            return np.nan, np.nan, np.nan

        X_curr = X_fit[1:]
        X_prev = X_fit[:-1]

        regression_model = LinearRegression()
        regression_model.fit(X_prev.reshape(-1, 1), X_curr)

        a_hat = regression_model.intercept_
        phi_hat = regression_model.coef_[0]

        if phi_hat <= 0 or phi_hat >= 1:
            return np.nan, np.nan, np.nan

        mu_hat = a_hat / (1 - phi_hat)
        kappa_hat = -np.log(phi_hat)

        errors = X_curr - (a_hat + phi_hat * X_prev)
        var_eps = np.var(errors, ddof=2)

        sigma_hat = np.sqrt(
            var_eps * (2.0 * kappa_hat) / (1.0 - np.exp(-2.0 * kappa_hat))
        )

        return mu_hat, kappa_hat, sigma_hat
    
    
    def run_data(self) -> dict: 
        
        parameters_dict = dict()
        final_rank_dict = dict()
        companies_list = list(self.returns_df.columns)
        for window in self.windows_list: 
            mean_vol_dict = dict()
            for company in companies_list: 
                company_returns = self.returns_df[company]
                rv = self.get_rolling_realised_volatility(company_returns, window)
                mu_hat, kappa_hat, sigma_hat = self.AR_OLS(rv)
                expected_change = kappa_hat * (rv - mu_hat)
                scaled_score = expected_change / sigma_hat
                
                mean_vol_dict[company] = scaled_score
                
                if (window == self.set_window): 
                    parameters_dict[company] = [mu_hat, kappa_hat, sigma_hat]
            
            mean_vol_score_df = pd.DataFrame(mean_vol_dict)
            mean_vol_rank_df = cross_sectional_ranking(mean_vol_score_df, higher_is_better=True).reset_index()
            final_rank_dict[window] = mean_vol_rank_df
            
        parameters_df = pd.DataFrame(parameters_dict, index = ["mu", "kappa", "sigma"])
        
        final_mean_volatility_dict = {
            "parameters": parameters_df,
            "mean_volatility": final_rank_dict
        }
        
        return final_mean_volatility_dict
    