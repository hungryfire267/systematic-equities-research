import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar

class ModelHypothesisTest: 
    def __init__(self, alpha, xgboost_ic, lgbm_ic, hit_contingency_table, lightgbm_returns, xgboost_returns): 
        self.alpha = alpha
        
        self.xgboost_ic = xgboost_ic
        self.lgbm_ic = lgbm_ic
        
        self.contingency_table = hit_contingency_table
        
        self.lightgbm_returns = lightgbm_returns
        self.xgboost_returns = xgboost_returns
        
    def mean_weekly_ic(self): 
        t_stat, p_value = stats.ttest_rel(self.xgboost_ic, self.lgbm_ic, alternative="greater")
        
        statement = None
        if (p_value > self.alpha): 
            statement = f"""
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude there is no evidence that XGBoost produce a higher mean
                weekly IC than LightGBM.
            """
        else: 
            statement = f"""
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that XGBoost produce a higher
                mean weekly IC than LightGBM.
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
    
    def portfolio_returns_test(self): 
        t_stat, p_value = stats.ttest_rel(self.xgboost_returns, self.lightgbm_returns, alternative="two-sided")
        
        statement = None
        if (p_value > self.alpha): 
            statement = f"""
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude there is no evidence that XGBoost and LightGBM produce different mean
                weekly portfolio returns.
            """
        else: 
            statement = f"""
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that XGBoost and LightGBM produce different mean
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