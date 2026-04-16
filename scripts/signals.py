from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

UNIVERSE_PATH = Path("data/asx_companies.csv")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def percentile_check(lower_percentile: float, upper_percentile: float) -> None: 
    if not (0 <= lower_percentile < upper_percentile <= 100): 
        raise ValueError("Require 0 <= lower_percentile < upper_percentile <= 100") 



class Kalman: 
    def __init__(self, a11, a12, a22, h1, h2, window, r2_window): 
        self.log_prices_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}data/raw/companies/log_prices.parquet"))
        self.log_returns_df = pd.read_parquet(Path(r"data/raw/companies/log_returns.parquet"))
        
        if "Date" in list(self.log_prices_df.columns): 
            self.log_prices_df = self.log_prices_df.set_index("Date")
        if "Date" in list(self.log_returns_df.columns): 
            self.log_returns_df = self.log_returns_df.set_index("Date")
            
        self.window = window
        self.A = np.array([ 
            [a11, a12], 
            [0, a22]
        ])
        self.H = np.array([[h1, h2]])
        self.initial_state_covariance = np.eye(2)
        
        self.r2_window = r2_window
        self.r2_trend_dict = dict()
        
        
    def get_kalman_filter(self, stock): 
        p = self.log_prices_df[stock].astype(float)
        r = self.log_returns_df[stock].astype(float)
        
        df = pd.concat({"p": p, "r": r}, axis=1).dropna()
        p = df["p"]
        r = df["r"]
        
        var_r = r.rolling(window=self.window).var().bfill().ffill()
        R_series = var_r
        Q_series = (R_series/100).bfill().ffill()
        
        
        T = len(p)
        
        R_t = R_series.to_numpy().reshape(T, 1, 1)
        Q_t = np.zeros((T, 2, 2))
        Q_t[:, 0, 0] = 0.05 * R_series.bfill().ffill().to_numpy()
        Q_t[:, 1, 1] = 0.05 * R_series.bfill().ffill().to_numpy()
    
        y_obs = p.to_numpy().reshape(T, 1) 
        y0 = y_obs[0, 0]
        self.initial_state_mean = np.array([0.5 * y0, 0.5 * y0])
        
        kf = KalmanFilter(
            transition_matrices=self.A,
            observation_matrices=self.H,
            transition_covariance=Q_t,
            observation_covariance=R_t,
            initial_state_mean = self.initial_state_mean, 
            initial_state_covariance=self.initial_state_covariance
        )
        
        state_mean, state_cov = kf.filter(y_obs)
        std_innovation = self.get_innovation(y_obs, state_mean, state_cov, R_t, Q_t)
        std_innovation
        x_short = pd.Series(state_mean[:, 0], index=p.index, name=f"{stock}_short")
        x_long = pd.Series(state_mean[:, 1], index=p.index, name=f"{stock}_long")
        return y_obs, x_short, x_long, state_cov    
    
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
    
    def get_features(self, stock): 
        p, x_short, x_long, state_cov = self.get_kalman_filter(stock)
        self.trend_spread_dict[stock] = x_short - x_long
        self.get_R2(y_obs, x_long, stock)
        dx_long = x_long.diff(self.r2_window)
        dx_long.name = f"dx_long{self.r2_window}"
        self.dx_long_dict[stock] = dx_long
        return x_short
        
    def run_kalman_filter(self): 
        stocks = list(self.log_prices_df.columns)
        
        for stock in stocks: 
            print(stock)
            x_short = self.get_features(stock)
            print(x_short)
            break
    
    def plot_kalman_comparison(self, stocks: tuple[str, str]): 
        kalman_dict = dict()
        for stock in stocks: 
            kalman_dict[stock] = self.get_kalman_filter(stock)
        kalman_df = pd.DataFrame(kalman_dict)
        kalman_df = kalman_df.reset_index()
        
        prices_df_filtered = self.prices_df.loc[self.prices_df["Date"].dt.year == 2024, ["Date"] + stocks]
        self.prices_df = self.prices_df.set_index("Date")

        mean_1m = self.prices_df[stocks].rolling(window=21).mean()
        mean_1m = mean_1m.reset_index()
        mean_1m_filtered = mean_1m.loc[mean_1m["Date"].dt.year == 2024]
        
        mean_2m = self.prices_df[stocks].rolling(window=42).mean() 
        mean_2m = mean_2m.reset_index()
        mean_2m_filtered = mean_2m.loc[mean_2m["Date"].dt.year == 2024]
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        kalman_df_filtered = kalman_df.loc[kalman_df["Date"].dt.year == 2024, ["Date"] + stocks]
        
        for i, stock in enumerate(stocks):
            print(i)
            sns.lineplot(x=prices_df_filtered["Date"], y=prices_df_filtered[stock], ax=axs[i])
            sns.lineplot(x=mean_1m_filtered["Date"], y=mean_1m_filtered[stock], ax=axs[i])
            sns.lineplot(x=mean_2m_filtered["Date"], y=mean_2m_filtered[stock], ax=axs[i])
            sns.lineplot(x=kalman_df_filtered["Date"], y=kalman_df_filtered[stock], ax=axs[i])  
        plt.show()
            
        
class KalmanFilterBuilder: 
    def __init__(self, window: int): 
        self.window = window 



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
    
        
class Reversal: 
    def __init__(self, lower_percentile: float, upper_percentile: float, windows_list: list[int]): 
        self.returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/returns.parquet"))
        self.asx_returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/asx/asx_returns.parquet"))
        self.industry_dict = pd.read_csv(Path(rf"{PROJECT_ROOT}/data/asx_companies.csv")).set_index("asxCode")["industry"].to_dict()
        self.lower = lower_percentile
        self.upper = upper_percentile
        
        
    def get_Reversal(self): 
        reversal_dict = dict()
        for w in windows_list: 
            cumulative_returns = (1 + self.returns_df.rolling(window=w).apply(np.prod, raw=True))- 1
            reversal_dict[w] = - cumulative_returns
            
    def get_RSR(self): 
        print(self.returns_df.columns)
        
            
            
              
        
        
        

        
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
        self.lower_extreme, self.upper_extreme = extreme_list[0], extreme_list[1]
        self.lower_signal, self.upper_signal = signal_percentile[0], signal_percentile[1]
        
        
    def compute_ema(self, df: pd.DataFrame, span: int) -> pd.DataFrame:
        return self.volume_df.ewm(span=span, adjust=False).mean()
    
    def calculate_pvo(self) -> None: 
        ema_slow = self.compute_ema(self.volume_df, span=self.slow_span)
        ema_fast = self.compute_ema(self.volume_df, span=self.fast_span)
        pvo_df = (ema_fast - ema_slow)/ema_slow
        
        pvo_df = pvo_df.clip(lower=pvo_df.quantile(self.lower), upper=pvo_df.quantile(self.upper)) # Capping the extremes
        self.pvo_df = pvo_df
    
    def get_pvo_signals(self): 
        ranks = self.pvo_df.rank(axis=1, pct=True)
        pvo_signal_df = pd.DataFrame(0, index=self.pvo_df.index, columns=self.pvo_df.columns)
        pvo_signal_df[ranks >= self.upper_signal] = 1
        pvo_signal_df[ranks <= self.lower_signal] = -1
        self.pvo_signal_df = pvo_signal_df
    
    def run(self): 
        self.calculate_pvo()
        self.get_pvo_signals()   

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
    

        
if __name__ == "__main__": 
    Reversal(0.25, 0.75, [5, 10, 21]).get_RSR()
    # pipeline = Kalman(a11=0.95, a12=0.05, a22=0.995, h1=1.0, h2=1.0, window=20).run_kalman_filter()
    #pipeline.plot_kalman_comparison(["CBA.AX", "ZIP.AX"])

