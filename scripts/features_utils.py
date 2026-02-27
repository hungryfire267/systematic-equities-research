import numpy as np
import pandas as pd
import statsmodels.api as sm



def rolling_slope(series: pd.Series, window:int) -> np.ndarray: 
    n = series.shape[0]
    rolling_trend = np.full(n, np.nan)
    for i in range(window - 1, n): 
        t = np.arange(i - window + 1, i + 1, dtype=int)
        y_w = series.iloc[i - window + 1: i + 1]
        t_w = sm.add_constant(t).values
        
        model = sm.OLS(y_w, t_w).fit()
        rolling_trend[i] = model.params[1]
    return rolling_trend

def 
        
        