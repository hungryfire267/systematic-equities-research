from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from signals_functions import date_parser
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

UNIVERSE_PATH = Path("data/asx_companies.csv")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPANIES_DIR = PROJECT_ROOT / "data" / "raw" / "companies"

companies_paths_dict = {
    ""
    "market_cap": os.path.join(COMPANIES_DIR, "market_cap.parquet"),
    "prices": os.path.join(COMPANIES_DIR, "prices.parquet"),
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet"),                 
    "volume": os.path.join(COMPANIES_DIR, "volume.parquet")
}







def cross_sectional_ranking(df: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame: 
    mean = df.mean(axis=1)
    std = df.std(axis=1, skipna=True)
    new_df = df.sub(mean, axis=0).div(std, axis=0)
    rank = new_df.rank(axis=1, pct=True, ascending=higher_is_better)
    return rank

class Fundamentals: 
    def __init__(self): 
        self.ptb_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/fundamentals/price_to_book.parquet"))
        self.dividend_yield_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/fundamentals/dividend_yield.parquet"))
        self.earnings_yield_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/fundamentals/earnings_yield.parquet"))
        self.roa_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/fundamentals/roa.parquet"))
        self.roe_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/fundamentals/roe.parquet"))
        
        self.factor_config = { 
            "ptb": {
                "df": self.ptb_df, 
                "higher_is_better": False,
                "group": "value"
            }, 
            "dividend_yield": {
                "df": self.dividend_yield_df, 
                "higher_is_better": True, 
                "group": "value"
            },  
            "earnings_yield": {
                "df": self.earnings_yield_df, 
                "higher_is_better": True, 
                "group": "value"
            }, 
            "roa": {
                "df": self.roa_df, 
                "higher_is_better": True, 
                "group": "quality"
            },
            "roe": {
                "df": self.roe_df, 
                "higher_is_better": True, 
                "group": "quality"
            }
        }
            
    def cross_sectional_ranking(self, df: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame: 
        mean = df.mean(axis=1)
        std = df.std(axis=1, skipna=True)
        new_df = df.sub(mean, axis=0).div(std, axis=0)
        rank = new_df.rank(axis=1, pct=True, ascending=higher_is_better)
        return rank
    
    def build_factor_ranks(self) -> dict:
        ranked_factors = dict()
        
        for factor_name, config in self.factor_config.items(): 
            ranked_factors[factor_name] = self.cross_sectional_ranking(
                config["df"], 
                higher_is_better = config["higher_is_better"]
            )
        
        return ranked_factors
    
    def build_group_signal(self, group_name: str): 
        ranked_factors = self.build_factor_ranks() 
        
        group_dfs = []
        for factor_name, config in self.factor_config.items(): 
            if (config["group"] == group_name): 
                group_dfs.append(ranked_factors[factor_name])
            
        group_signal = sum(group_dfs) / len(group_dfs)

        group_signal = group_signal.rank(axis=1, pct=True)
        return group_signal

    def run_data(self) -> pd.DataFrame: 
        value_signal = self.build_group_signal("value")
        quality_signal = self.build_group_signal("quality")
        fundamental_signals = (value_signal + quality_signal)/2
        
        fundamental_signals = fundamental_signals.rank(axis=1, pct=True)
        
        return fundamental_signals
    
class Momentum: 
    def __init__(self, weights: list, n: int): 
        self.returns_df = pd.read_parquet(Path(r"data/raw/companies/returns.parquet"))
        self.industry_returns = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/industry/industry_returns.parquet"))        
        self.returns_df["Date"] = pd.to_datetime(self.returns_df["Date"])
        self.returns_df = self.returns_df.set_index("Date")
        
        assert (n > 0)
        assert (len(weights) == n)
        
        self.factor_config = {
            "mom_12_1": {
                "score": None, 
                "higher_is_better": True
            }, 
            "id": {
                "score":None, 
                "higher_is_better": True
            }
        }
        
        self.weights = weights
        
    def get_momentum(self, lookback=252, skip=21) -> pd.DataFrame:
        past_returns = self.returns_df.copy().shift(skip)
        cumulative_returns = past_returns.rolling(lookback).sum()
        
        return past_returns, cumulative_returns
        
    def information_discreteness(self, lookback=252, skip=21) -> pd.DataFrame: 
        past_returns, cumulative_returns = self.get_momentum()
        
        up_days = (past_returns > 0).rolling(lookback).sum()
        down_days = (past_returns < 0 ).rolling(lookback).sum()
        
        id_score = (up_days - down_days) / lookback
        id_score = id_score * np.sign(cumulative_returns)
        
        
        return id_score 
    
    def get_momentum_ranks(self) -> pd.DataFrame:
        ranking_dict = dict()  
        for factor, config in self.factor_config.items(): 
            rank = config["score"].rank(axis=1, pct=True, ascending=config["higher_is_better"])
            ranking_dict[factor] = rank
        
        return ranking_dict
        
    def run_data(self) -> pd.DataFrame: 
        _, self.factor_config["mom_12_1"]["score"] = self.get_momentum()
        self.factor_config["id"]["score"] = self.information_discreteness()
        
        ranking_dict = self.get_momentum_ranks()
        
        final_score = sum(
            weight * rank for weight, rank in zip(self.weights, ranking_dict.values())
        )
        final_rank = final_score.rank(axis=1, pct=True)
        
        return final_rank

class Reversal: 
    def __init__(self, windows_list: list[int], weights: list[float]): 
        self.returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/returns.parquet"))
        self.asx_returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/asx/asx_returns.parquet"))
        self.industry_dict = pd.read_csv(Path(rf"{PROJECT_ROOT}/data/asx_companies.csv")).set_index("asxCode")["industry"].to_dict()
        self.industry_returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/industry/industry_returns.parquet"))
        self.industry_df = pd.read_csv(Path(rf"{PROJECT_ROOT}/data/asx_companies.csv"))
        self.industry_df["code"] =  [str(code) + ".AX" for code in list(self.industry_df["asxCode"])]
        self.industry_dict = self.industry_df.set_index("code")["industry"].to_dict()
        self.total_days = self.returns_df.shape[0]

        self.rsr5_dict, self.rsr10_dict, self.rsr21_dict = dict(), dict(), dict()
    
        self.reversal_config, self.rsr_config = dict(), dict()
        for window in windows_list: 
            self.reversal_config[window] = {
                "df": None, 
                "higher_is_better": True
            }
            self.rsr_config[window] = {
                "df": None, 
                "higher_is_better": True
            }
        assert(weights.sum() == 1)
        assert(len(weights) == len(windows_list))
        
        self.weights_dict = dict(zip(windows_list, weights))
        
        
        
        
        

    def get_Reversal(self) -> None: 
        for window in self.reversal_config.keys(): 
            cumulative_returns = (
                1 + self.returns_df.rolling(window=window).apply(np.prod, raw=True)
            ) - 1
            self.reversal_config[window]["df"] = - cumulative_returns.reset_index()
        

    def get_rsr(self) -> None: 
        rsr_dict = dict(), dict()
        rsr_dict = dict()
        print(self.returns_df.columns)
        market_returns = self.asx_returns_df["^AXJO"]
        company_list = list(self.returns_df.columns[1:])
        for company in company_list: 
            company_returns = self.returns_df[company]
            industry = self.industry_dict[company]
            industry_returns = self.industry_returns_df[industry]
            total_returns = pd.concat([industry_returns, market_returns, company_returns], axis=1).dropna()
            null_days = self.total_days - total_returns.shape[0]
            y_returns = total_returns[company]
            X_returns = total_returns.drop(columns=[company])
            linear_model = LinearRegression().fit(X_returns, y_returns)
            residuals = y_returns - linear_model.predict(X_returns)
            residuals = residuals.reindex(range(0, residuals.index.max() + 1))
            residuals = residuals.sort_index()
            residuals.index = self.returns_df["Date"]
            
            
            for window in self.rsr_config.keys(): 
                self.rsr_dict[window][company] = - ((1 + residuals)).rolling(window=window).apply(np.prod, raw=True) -1
        
        for window in self.rsr_config.keys(): 
            self.rsr_config[window]["df"] =  pd.DataFrame(rsr_dict[window]).reset_index()

    
    def get_reversal_ranks(self, reversal_type: str): 
        reversal_dict = None
        if reversal_type == "reversal": 
            reversal_dict = self.reversal_config
        elif reversal_type == "rsr": 
            reversal_dict = self.rsr_config
        else: 
            raise ValueError("reversal_type must be either 'reversal' or 'rsr'")
        
        
        for window, config in reversal_dict.items(): 
            rank_score = cross_sectional_ranking(config["df"], config["higher_is_better"])
            if (reversal_type == "reversal"): 
                self.reversal_config[window]["rank"] = rank_score
            else: 
                self.rsr_config[window]["rank"] = rank_score
             
    def run_data(self) -> pd.DataFrame: 
        self.get_rsr()
        self.get_reversal_ranks("reversal")
        self.get_reversal_ranks("rsr")
        
        # Combine ranks through ridge (after backtesting different combinations of weights, we found that giving more weight to shorter-term reversal signals yields better results)
        
        # For now we will use simple weights
        
        total_reversal_score = sum(
            self.weights_dict[window] * self.reversal_config[window]["rank"] for window in self.weights_dict.keys()
        )
        total_rsr_score = sum(
            self.weights_dict[window] * self.rsr_config[window]["rank"] for window in self.weights_dict.keys()
        )
            
        final_score = 0.5 * total_reversal_score + 0.5 * total_rsr_score
        final_rank = final_score.rank(axis=1, pct=True)
        
        return final_rank
    
    
            
        
        
        
        
        

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


     
class Microstructure: 
    def __init__(self, window_list): 
        prices_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/prices.parquet")) 
        volume_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/volume.parquet"))    
        returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/returns.parquet"))
        
        self.prices_df = date_parser(prices_df) 
        self.volume_df = date_parser(volume_df)
        self.returns_df = date_parser(returns_df)
        self.window_periods = dict()
        for window in window_list: 
            min_periods = max(3, int(0.5 * window))
            self.window_periods[window] = min_periods
        
        self.factor_config = dict()
    
    def get_dollar_volume(self):
        return self.prices_df * self.volume_df
    
    def dollar_volume_liquidity(self): 
        dv_liquidity_dict = dict()
        dollar_volume = self.get_dollar_volume()
        for window, min_periods in self.window_periods.items(): 
            liquidity = dollar_volume.rolling(window=window, min_periods=min_periods).mean()
            rank = liquidity.rank(axis=1, pct=True, ascending=True)
            dv_liquidity_dict[window] = rank.reset_index()
        return dv_liquidity_dict

    def get_amihud(self):
        amihud_dict = dict() 
        dollar_volume = self.get_dollar_volume()
        amihud = self.returns_df.abs() / dollar_volume
        
        for window, min_periods in self.window_periods.items():
            amihud_smoothed = amihud.rolling(window=window, min_periods=min_periods).mean()
            amihud_rank = cross_sectional_ranking(amihud_smoothed, higher_is_better=False)
            amihud_dict[window] = amihud_rank.reset_index()
        
        return amihud_dict
    
    def get_data(self): 
        dv_liquidity_dict = self.dollar_volume_liquidity() 
        amihud_illiqudity_dict = self.get_amihud()
        
        return dv_liquidity_dict, amihud_illiqudity_dict
        
class BetaFeatures: 
    def __init__(self, window_list: list, weights_list: list): 
        companies_df = pd.read_csv(rf"{PROJECT_ROOT}/data/asx_companies.csv")
        
        returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/companies/returns.parquet"))
        
        print(returns_df)
        asx_returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/asx/asx_returns.parquet"))
        industry_returns_df = pd.read_parquet(Path(rf"{PROJECT_ROOT}/data/raw/industry/industry_returns.parquet"))
        
        self.returns_df = date_parser(returns_df) 
        self.asx_returns_df = date_parser(asx_returns_df)
        self.industry_returns_df = date_parser(industry_returns_df)
        self.companies_df = companies_df
        
        self.window_list = window_list
        
        self.weights_list = weights_list
    
    @staticmethod
    def beta_calculation(combined_df: pd.DataFrame, beta_type, window) -> pd.Series: 
        cov = combined_df["company"].rolling(window=window).cov(combined_df[beta_type])
        var = combined_df[beta_type].rolling(window=window).var()
        return cov/var
    
    @staticmethod
    def vol_calculation(combined_df: pd.DataFrame, beta_type: str, window: int, beta: pd.Series) -> pd.Series:
        residuals = combined_df["company"] - beta * combined_df[beta_type]
        vol = residuals.rolling(window=window).std()
        return vol
    
    def get_market_rolling_beta_vol(self, window: int) -> dict: 
        market_beta_df_dict, market_vol_df_dict = dict(), dict()

        market_returns = self.asx_returns_df["^AXJO"].copy()
        for company in self.returns_df.columns: 
            company_returns = self.returns_df[company]
            combined_df = pd.concat([market_returns, company_returns], axis=1)
            combined_df.columns = ["market", "company"]
            beta = self.beta_calculation(combined_df, "market", window)
            vol = self.vol_calculation(combined_df, "market", window, beta)
            
            market_beta_df_dict[company] = beta
            market_vol_df_dict[company] = vol
        
        market_beta_df = pd.DataFrame(market_beta_df_dict)
        market_vol_df = pd.DataFrame(market_vol_df_dict)
            
        return market_beta_df, market_vol_df
    
    def get_industry_company_return(self, company: str) -> pd.Series: 
        company_final = company.split(".")[0].upper()
        condition = (self.companies_df["asxCode"] == company_final)
        company_industry = self.companies_df.loc[condition, "industry"].iloc[0]

        industry_returns = self.industry_returns_df[company_industry]
        return industry_returns
    
    
    def get_industry_rolling_beta_vol(self, window: int) -> dict: 
        industry_beta_df_dict, industry_vol_df_dict = dict(), dict()
        for company in self.returns_df.columns: 
            company_returns = self.returns_df[company]
            industry_returns = self.get_industry_company_return(company)
            combined_df = pd.concat([industry_returns, company_returns], axis = 1)
            combined_df.columns = ["industry", "company"]

            beta = self.beta_calculation(combined_df, "industry", window)
            vol = self.vol_calculation(combined_df, "industry", window, beta)
            
            industry_beta_df_dict[company] = beta
            industry_vol_df_dict[company] = vol
        
        industry_beta_df = pd.DataFrame(industry_beta_df_dict)
        industry_vol_df = pd.DataFrame(industry_vol_df_dict)
            
        return industry_beta_df, industry_vol_df

            
        
    
    def get_data(self) -> tuple[dict, dict]: 
        
        market_beta_df_dict, market_vol_df_dict = dict(), dict() 
        industry_beta_df_dict, industry_vol_df_dict = dict(), dict()
        
        for window in self.window_list: 
            market_beta_df, market_vol_df = self.get_market_rolling_beta_vol(window)
            industry_beta_df, industry_vol_df = self.get_industry_rolling_beta_vol(window)
            
            market_beta_df_dict[window] = cross_sectional_ranking(market_beta_df, higher_is_better = False).reset_index()
            industry_beta_df_dict[window] = cross_sectional_ranking(industry_beta_df, higher_is_better = False).reset_index()
            market_vol_df_dict[window] = cross_sectional_ranking(market_vol_df, higher_is_better = False).reset_index()
            industry_vol_df_dict[window] = cross_sectional_ranking(industry_vol_df, higher_is_better = False).reset_index()
    
            
        return market_beta_df_dict, market_vol_df_dict, industry_beta_df_dict, industry_vol_df_dict
         
         
        
        
        

        
        
        

        
class PVO: 
    def __init__(
        self, extreme_list: tuple[float, float], signal_percentile: tuple[float, float], span_list: tuple[int, int]
    ): 
        self.volume_df = pd.read_parquet(Path(r"data/raw/companies/volume.parquet"))
        
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
    
class MomentumLiquidity: 
    def __init__(self, momentum_weights: list, momentum_n, liquidity_window_list): 
        self.momentum_weights = momentum_weights 
        self.momentum_n = momentum_n 
        self.liquidity_window_list = liquidity_window_list
    
    def build_momentum_liquidity_rank(self, momentum_rank, dv_rank) -> pd.DataFrame: 
        momentum_dv_score = momentum_rank * dv_rank
        momentum_dv_rank = momentum_dv_score.rank(axis = 1, pct = True, ascending = True)
    
        return momentum_dv_rank
    
    
    def run_data(self) -> dict: 
        momentum_rank = Momentum(self.momentum_weights, self.momentum_n).run_data()
        dv_liquidity_dict, _ = Microstructure(self.liquidity_window_list)
        
        momentum_liquidity_dict = dict()
        for window in self.liquidity_window_list: 
            momentum_liquidity_dict[window] = self.build_momentum_liquidity_rank(
                window, momentum_rank, dv_liquidity_dict[window]
            )
        return momentum_liquidity_dict

class ReversalIlliquidity:
    def __init__(self, reversal_window_list, reversal_weight_list, liquidity_window_list):
        assertion_match_list = f"Window mismatch: reversal_window_list - {reversal_window_list} does \
            not match with illiquidity_window_list {liquidity_window_list}"
        assert(
            reversal_window_list == liquidity_window_list, assertion_match_list
        )
        
        self.window_list = reversal_weight_list 
        self.reversal_weight_list = reversal_weight_list
        
    def build_reversal_illiquidity_rank(self, reversal_rank, amihud_rank): 
        reversal_amihud_score = reversal_rank * amihud_rank
        reversal_amihud_rank = cross_sectional_ranking(reversal_amihud_score, higher_is_better=True)
        
        return reversal_amihud_rank
                 
                 
    def run_data(self) -> dict:
        reversal_dict = Reversal(self.window_list, self.reversal_weight_list)
        _, amihud_dict = Microstructure(self.window_list)
        
        reversal_illiquidity_dict = dict()
        for window in self.window_list: 
            reversal_rank = reversal_dict[window]
            amihud_rank = amihud_dict[window]
            reversal_illiquidity_dict[window] = self.build_reversal_illiquidity(reversal_rank, amihud_rank)
        
        return reversal_illiquidity_dict
                
        
    
        
        
        
        
        
if __name__ == "__main__": 
    momentum = Momentum([0.5, 0.5, 2])
    Reversal()
    Microstructure()

