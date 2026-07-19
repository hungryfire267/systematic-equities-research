import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar

class ModelHypothesisTest: 
    def __init__(self, 
        alpha, 
        dt_ic,
        xgboost_ic,
        lgbm_ic,
        hit_contingency_table, 
        dt_returns,
        lightgbm_returns, 
        xgboost_returns
    ): 
        self.alpha = alpha
        
        self.dt_ic = dt_ic
        self.xgboost_ic = xgboost_ic
        self.lgbm_ic = lgbm_ic
        
        self.contingency_table = hit_contingency_table
        
        self.dt_returns = dt_returns
        self.lightgbm_returns = lightgbm_returns
        self.xgboost_returns = xgboost_returns
        
    def mean_weekly_ic(self, type_ic_1: str, type_ic_2: str): 
        ic_mapping = {
            "Decision Trees": self.dt_ic, 
            "LightGBM": self.lgbm_ic,
            "XGBoost": self.xgboost_ic
        }
        
        t_stat, p_value = stats.ttest_rel(ic_mapping[type_ic_1], ic_mapping[type_ic_2], alternative="greater")
        
        statement = None
        if (p_value > self.alpha): 
            statement = f"""
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude there is no evidence that {type_ic_1} produce a higher mean
                weekly IC than {type_ic_2}.
            """
        else: 
            statement = f"""
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that {type_ic_1} produce a higher
                mean weekly IC than {type_ic_2}.
            """
        return t_stat, p_value, statement
    
    def mcnemar_test(self):
        result = mcnemar(self.contingency_table, exact=False, correction=True)
        chi_squared_stat = result.statistic
        p_value = result.pvalue
        
        statement = None
        if (p_value > self.alpha): 
            statement = f"""
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude that there is no evidence that XGBoost and LightGBM have different hit rates.
            """
        else: 
            statement = f"""
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that XGBoost and LightGBM have different hit rates. 
            """
        
        return chi_squared_stat, p_value, statement
    
    def portfolio_returns_test(self, type_returns_1, type_returns_2): 
        returns_mapping = {
            "Decision Trees": self.dt_ic, 
            "LightGBM": self.lgbm_ic,
            "XGBoost": self.xgboost_ic
        }
        t_stat, p_value = stats.ttest_rel(returns_mapping[type_returns_1], returns_mapping[type_returns_2], alternative="two-sided")
        
        statement = None
        if (p_value > self.alpha): 
            statement = f"""
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude there is no evidence that {type_returns_1} and {type_returns_2} produce different mean
                weekly portfolio returns.
            """
        else: 
            statement = f"""
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that {type_returns_1} and {type_returns_2} produce different mean
                weekly portfolio returns.
            """
        return t_stat, p_value, statement
    
    def sharpe_ratio_test(self, periods_per_year=52):
        aligned = pd.concat(
            [
                self.lightgbm_returns.rename("LightGBM"),
                self.xgboost_returns.rename("XGBoost")
            ],
            axis=1
        ).dropna()

        lgbm = aligned["LightGBM"].astype(float)
        xgb = aligned["XGBoost"].astype(float)

        n = len(aligned)

        if n < 3:
            raise ValueError(
                "At least three aligned return observations are required."
            )

        if lgbm.std(ddof=1) == 0 or xgb.std(ddof=1) == 0:
            raise ValueError(
                "Sharpe ratio cannot be calculated when return volatility is zero."
            )

        # Use non-annualised Sharpe ratios in the hypothesis test
        lgbm_sharpe = lgbm.mean() / lgbm.std(ddof=1)
        xgb_sharpe = xgb.mean() / xgb.std(ddof=1)

        correlation = lgbm.corr(xgb)

        # Memmel-corrected asymptotic variance
        variance = (
            2 * (1 - correlation)
            + 0.5 * (
                lgbm_sharpe**2
                + xgb_sharpe**2
                - 2
                * correlation**2
                * lgbm_sharpe
                * xgb_sharpe
            )
        ) / n

        if variance <= 0:
            raise ValueError(
                "The estimated variance of the Sharpe-ratio difference "
                "is not positive."
            )

        test_statistic = (
            lgbm_sharpe - xgb_sharpe
        ) / np.sqrt(variance)

        # H1: Sharpe_LGBM > Sharpe_XGB
        p_value = stats.norm.sf(test_statistic)

        annualisation_factor = np.sqrt(periods_per_year)

        return {
            "test": (
                "One-sided Jobson–Korkie Sharpe-ratio test "
                "with Memmel correction"
            ),
            "n_observations": n,
            "lgbm_sharpe": lgbm_sharpe * annualisation_factor,
            "xgb_sharpe": xgb_sharpe * annualisation_factor,
            "sharpe_difference": (
                lgbm_sharpe - xgb_sharpe
            ) * annualisation_factor,
            "correlation": correlation,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "reject_null": p_value < self.alpha
        }