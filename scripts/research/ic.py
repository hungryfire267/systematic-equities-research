import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

class IC: 
    def __init__(self, return_df: pd.DataFrame, signal_df: pd.DataFrame): 
        self.return_df = return_df
        self.signal_df = signal_df
        
        self.return_df["Date"] = pd.to_datetime(self.return_df["Date"])
        self.signal_df["Date"] = pd.to_datetime(self.signal_df["Date"])
        
        self.factor_list = list(self.signal_df.columns[2:])
        
        
        
    def merge_dfs(self): 
        self.df = pd.merge(left = self.return_df, right = self.signal_df, how = "outer", on = ["Date", "Ticker"])
        
    def calculate(self, factor_col, target_col: str) -> pd.DataFrame: 
        ic_rows = []
        
        for date, group in self.df.groupby("Date"): 
            valid = group[[factor_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

            if len(valid) < 10:
                continue

            ic = spearmanr(valid[factor_col], valid[target_col]).correlation

            ic_rows.append({
                "Date": date,
                "factor": factor_col,
                "target": target_col,
                "IC": ic
            })

        ic_df = pd.DataFrame(ic_rows)
        return ic_df
    
    def get_lag_future_returns(self, return_type: str) -> int: 
        returns_list = return_type.split("_")
        returns_window = returns_list[-1].replace("d","")
        lags = int(returns_window) - 1
        return lags
    
    def newey_west_ttest(self, ic: pd.Series, lag: int) -> pd.DataFrame:         
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

    def run_data(self) -> tuple[dict, pd.DataFrame]:
        self.merge_dfs()
        
        ic_dict = {}
        results_dict = {}
        
        
        for factor in self.factor_list: 
            ic_df = self.calculate(factor, "future_return_5d")
            ic_dict[factor] = ic_df
            ic = ic_df["IC"].dropna()
            
            lag = self.get_lag_future_returns("future_return_5d")
            newey_test_dict = self.newey_west_ttest(ic, lag)
        
            results_dict[factor] = {
                "mean_ic": ic.mean(),
                "std_ic": ic.std(),
                "ic_ir": ic.mean() / ic.std() if ic.std() != 0 else np.nan,
                "positive_ic_rate": (ic > 0).mean(),
                "nw_t_stat": newey_test_dict["nw_t_stat"],
                "nw_p_value": newey_test_dict["nw_p_value"],
                "n_periods": len(ic)
            }
        
        results_df = pd.DataFrame(results_dict).T.reset_index().rename(columns={
            "index": "Factors"
        })
        
        results_df["fdr_p_value"] = multipletests(
            results_df["nw_p_value"],
            method="fdr_bh"
        )[1]

        results_df["significant_5pct"] = results_df["fdr_p_value"] < 0.05

        return ic_dict, results_df