import numpy as np
import os
from pathlib import Path
import pandas as pd

from scripts.models.lightgbm.lightgbm_model import LightGBMRegressionModel
from scripts.models.xgboost.xgboost_model import XGBoostRegressionModel
from scripts.portfolio.metrics import GetMetrics
from scripts.portfolio.optimiser import MeanVarianceOptimiser
from scripts.portfolio.selection import TopBottom20Selector


BASE_DIR = Path(__file__).resolve().parents[1]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

DT_MODEL_DIR = BASE_DIR / "results" / "dt_model" 
LIGHTGBM_MODEL_DIR = BASE_DIR / "results" / "lightgbm_model"
XGBOOST_MODEL_DIR = BASE_DIR / "results" / "xgboost_model"

BACKTEST_RESULTS_DIR = BASE_DIR / "results" /  "backtest"
BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_DT_DIR = BACKTEST_RESULTS_DIR / "dt"
BACKTEST_DT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_LIGHTGBM_DIR = BACKTEST_RESULTS_DIR / "lightgbm"
BACKTEST_LIGHTGBM_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_XGBOOST_DIR = BACKTEST_RESULTS_DIR / "xgboost"
BACKTEST_XGBOOST_DIR.mkdir(parents=True, exist_ok=True)

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}


class GetPortfolioReturns: 
    def __init__(self, model_dir, backtest_results_dir, data_type): 
        self.model_dir = model_dir
        self.backtest_results_dir = backtest_results_dir
        self.data_type = data_type
        self.returns_df = pd.read_parquet(companies_paths_dict["returns"])
    
    def run_portfolio(self):
        test_preds, test_preds_rank, topbottom20 = TopBottom20Selector(self.model_dir).run_data()
        portfolio_df = MeanVarianceOptimiser(topbottom20, self.returns_df).run_data()
        final_portfolio_df = portfolio_df.merge(
            topbottom20,
            on=["Date", "Ticker"],
            how="left"
        )

        final_portfolio_df["portfolio_return"] = (
            final_portfolio_df["weight"]
            * final_portfolio_df["future_return_5d"]
        )
        
        final_portfolio_df = final_portfolio_df[
            ["Date", "Ticker", "weight", "side_x", "prediction", "future_return_5d", "portfolio_return"]
        ].rename(columns={"side_x": "side"})
        
        final_datasets_df_dict = { 
            "final_portfolio": final_portfolio_df,
            "preds": test_preds, 
            "rank": test_preds_rank
        }
        
        return final_datasets_df_dict
    
    def get_backtest_paths_dict(self): 
        self.backtest_paths_dict = {
            "final_portfolio": os.path.join(self.backtest_results_dir, f"final_portfolio_{self.data_type}.parquet"),
            "preds": os.path.join(self.backtest_results_dir, f"test_preds_{self.data_type}.parquet"),
            "rank": os.path.join(self.backtest_results_dir, f"test_preds_rank_{self.data_type}.parquet")
        }
        
    
    def save_portfolio(self, final_datasets_df_dict):
        for key, df in final_datasets_df_dict.items():
            df.to_parquet(self.backtest_paths_dict[key], index=False, engine="pyarrow")
    
    def run_data(self): 
        final_datasets_df_dict = self.run_portfolio()
        self.get_backtest_paths_dict()
        self.save_portfolio(final_datasets_df_dict)
        
if __name__ == "__main__": 
    dt_stock_dir = os.path.join(DT_MODEL_DIR, "dt_model_stock.pkl")
    dt_market_dir = os.path.join(DT_MODEL_DIR, "dt_model_market.pkl")
    dt_macro_market_dir = os.path.join(DT_MODEL_DIR, "dt_model_macro_market.pkl")
    
    lgbm_stock_dir = os.path.join(LIGHTGBM_MODEL_DIR, "lightgbm_model_stock.pkl")
    lgbm_market_dir = os.path.join(LIGHTGBM_MODEL_DIR, "lightgbm_model_market.pkl")
    lgbm_macro_market_dir = os.path.join(LIGHTGBM_MODEL_DIR, "lightgbm_model_macro_market.pkl")
    
    xgboost_stock_dir = os.path.join(XGBOOST_MODEL_DIR, "xgboost_model_stock.pkl")
    xgboost_market_dir = os.path.join(XGBOOST_MODEL_DIR, "xgboost_model_market.pkl")
    xgboost_macro_market_dir = os.path.join(XGBOOST_MODEL_DIR, "xgboost_model_macro_market.pkl")
    
    print("Running Decision Tree Portfolio...")
    print("Running Stock...")
    dt_portfolio_stock = GetPortfolioReturns(dt_stock_dir, BACKTEST_DT_DIR, "stock").run_data() 
    print("Running Stock + Market...")
    dt_portfolio_market = GetPortfolioReturns(dt_market_dir, BACKTEST_DT_DIR, "market").run_data()
    print("Running Stock + Market + Macro...")
    dt_portfolio_macro_market = GetPortfolioReturns(dt_macro_market_dir, BACKTEST_DT_DIR, "macro_market").run_data()
    
    
    print("Running LightGBM Portfolio...")
    print("Running Stock...")
    lgbm_portfolio_stock = GetPortfolioReturns(lgbm_stock_dir, BACKTEST_LIGHTGBM_DIR, "stock").run_data() 
    print("Running Stock + Market...")
    lgbm_portfolio_market = GetPortfolioReturns(lgbm_market_dir, BACKTEST_LIGHTGBM_DIR, "market").run_data()
    print("Running Stock + Market + Macro...")
    lgbm_portfolio_macro_market = GetPortfolioReturns(lgbm_macro_market_dir, BACKTEST_LIGHTGBM_DIR, "macro_market").run_data()
    
    print("Running XGBoost Portfolio...")
    print("Running Stock...")
    xgboost_portfolio_stock = GetPortfolioReturns(xgboost_stock_dir, BACKTEST_XGBOOST_DIR, "stock").run_data()
    print("Running Stock + Market...")
    xgboost_portfolio_market = GetPortfolioReturns(xgboost_market_dir, BACKTEST_XGBOOST_DIR, "market").run_data()
    print("Running Stock + Market + Macro...")
    xgboost_portfolio_macro_market = GetPortfolioReturns(xgboost_macro_market_dir, BACKTEST_XGBOOST_DIR, "macro_market").run_data()
