import numpy as np
import os
from pathlib import Path
import pandas as pd

from scripts.models.lightgbm.lightgbm_model import LightGBMRegressionModel
from scripts.portfolio.metrics import GetMetrics
from scripts.portfolio.optimiser import MeanVarianceOptimiser
from scripts.portfolio.selection import TopBottom20Selector


BASE_DIR = Path(__file__).resolve().parents[1]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

BACKTEST_RESULTS_DIR = BASE_DIR / "results" /  "backtest"
BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

companies_paths_dict = {
    "returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}

returns_df = pd.read_parquet(companies_paths_dict["returns"])

model_class = LightGBMRegressionModel
topbottom20 = TopBottom20Selector(LightGBMRegressionModel).run_data()
portfolio_df = MeanVarianceOptimiser(topbottom20, returns_df).run_data()
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

final_portfolio_df_path = os.path.join(BACKTEST_RESULTS_DIR, "final_portfolio.parquet")
final_portfolio_df.to_parquet(final_portfolio_df_path, index=False, engine="pyarrow")
print(final_portfolio_df)

strategy_returns = (
    final_portfolio_df
    .groupby("Date")["portfolio_return"]
    .sum()
    .reset_index()
)

print(strategy_returns)

checks = final_portfolio_df.groupby("Date")["weight"].agg(
    net_weight="sum",
    gross_weight=lambda x: x.abs().sum(),
    n_positions="count",
    max_weight="max",
    min_weight="min"
)

print(checks)

r = strategy_returns["portfolio_return"]

sharpe = r.mean() / r.std() * np.sqrt(52)

equity = (1 + r).cumprod()
drawdown = equity / equity.cummax() - 1

print("Sharpe:", sharpe)
print("Total return:", equity.iloc[-1] - 1)
print("Max drawdown:", drawdown.min())

print(final_portfolio_df.groupby("Date")["weight"].agg(
    net_weight="sum",
    gross_weight=lambda x: x.abs().sum(),
    n_positions="count"
))