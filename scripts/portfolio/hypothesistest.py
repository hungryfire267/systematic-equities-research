import pandas as pd
import scipy.stats as stats

class ModelHypothesisTest: 
    def __init__(self, alpha, xgboost_ic, lgbm_ic): 
        self.alpha = alpha
        
        self.xgboost_ic = xgboost_ic
        self.lgbm_ic = lgbm_ic
        
    def mean_weekly_ic(self): 
        t_stat, p_value = stats.ttest_rel(self.xgboost_ic, self.lgbm_ic, alternative="greater")
        
        statement = None
        if (p_value > self.alpha): 
            statement = """
                Since the p-value is greater than the critical value of {self.alpha}, we do not reject the 
                null hypothesis. We can conclude there is no evidence that XGBoost produce a higher mean
                weekly IC than LightGBM.
            """
        else: 
            statement = """
                Since the p-value is less than the critical value of {self.alpha}, we have significant evidence to reject
                the null hypothesis. Therefore we can conclude that there is evidence that XGBoost produce a higher
                mean weekly IC than LightGBM.
            """
        return t_stat, p_value, statement
        