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

def get_innovation(self, y_obs, state_mean, state_cov, R_t, Q_t): 
        print(y_obs)
        T = y_obs.shape[0]
        x_pred = np.zeros_like(state_mean)
        x_pred[0] = self.initial_state_mean
        for i in range(1, T): 
            x_pred[i] = self.A @ state_mean[i-1]
        y_pred = (self.H @ x_pred.T).T
        P_pred = np.zeros_like(state_cov)
        P_pred[0] = self.initial_state_covariance
        for t in range(1, T):
            P_pred[t] = self.A @ state_cov[t - 1] @ self.A.T + Q_t[t]

        S_t = np.zeros(T)
        for t in range(T):
            S_t[t] = (self.H @ P_pred[t] @ self.H.T + R_t[t]).item()
            
        innovation = y_obs.squeeze() - y_pred.squeeze()    
        std_innovation = innovation/np.sqrt(S_t)
        return std_innovation
    
def get_R2(self, y_obs, x_long, stock): 
    trend_r2 = np.full(len(y_obs))
    for t in range(self.r2_window, len(y_obs)): 
        X[t:t+self.r2_window] = x_long[t:t+r]
        X = sm.add_constant(X.values)
        
        y = y_obs[t - self.r2_window:t]
        model = sm.OLS(y.values, X).fit() 
        
        trend_r2[t] = model.rsquared
        
    if stock not in self.trend_r2_dict.keys(): 
        self.trend_r2_dict[stock] = trend_r2
        
