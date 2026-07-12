import numpy as np
import pandas as pd

import statsmodels.api as sm

def get_lag_future_returns(return_type: str) -> int: 
    returns_list = return_type.split("_")
    returns_window = returns_list[-1].replace("d","")
    lags = int(returns_window) - 1
    return lags
    
def newey_west_ttest(ic: pd.Series, lag: int) -> pd.DataFrame:         
    ic = ic.dropna()
    
    y = ic.values
    X = np.ones(len(y))
    
    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag}
    )

    return {
        "nw_t_stat": model.tvalues[0],
        "nw_p_value": model.pvalues[0]
    }
