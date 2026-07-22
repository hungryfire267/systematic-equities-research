import pandas as pd
import os
from pathlib import Path
import streamlit as st
from components.sidebar import render_sidebar
from views.overview import render_overview
from views.portfolio import render_portfolio
from views.backtest_performance import render_backtesting
from views.model_comparison import render_model_comparison
from views.methodology_lessons import render_methodology_lessons
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from scripts.portfolio.metrics import GetMetrics

st.set_page_config(
    page_title="ASX Alpha System",
    page_icon="📈",
    layout="wide"
)


BACKTEST_RESULTS_DT_DIR = (
    BASE_DIR / "results" / "backtest" / "dt"
)

ASX_DIR = (
    BASE_DIR / "data" / "raw" / "asx"
)


dt_market_paths = {
    "final_portfolio_market": os.path.join(
        BACKTEST_RESULTS_DT_DIR,
        "final_portfolio_market.parquet"
    ),
    "test_preds_market": os.path.join(
        BACKTEST_RESULTS_DT_DIR,
        "test_preds_market.parquet"
    ),
    "test_preds_rank_market": os.path.join(
        BACKTEST_RESULTS_DT_DIR,
        "test_preds_rank_market.parquet"
    )
}


# ---------------------------------------------------------
# 1. Load the final strategy portfolio
# ---------------------------------------------------------

final_portfolio_df = pd.read_parquet(
    dt_market_paths["final_portfolio_market"]
)


# ---------------------------------------------------------
# 2. Calculate strategy metrics and realised returns
# ---------------------------------------------------------

portfolio_metrics, portfolio_returns = (
    GetMetrics(final_portfolio_df).run_data()
)


# ---------------------------------------------------------
# 3. Prepare the strategy return series
# ---------------------------------------------------------

portfolio_returns["Date"] = pd.to_datetime(
    portfolio_returns["Date"]
)

portfolio_returns = (
    portfolio_returns
    .dropna(subset=["Date", "portfolio_return"])
    .drop_duplicates(subset="Date", keep="last")
    .sort_values("Date")
)


selected_page = render_sidebar()

if selected_page == "🏠  Overview":
    render_overview()
elif selected_page == "📊  Portfolio":
    render_portfolio()
elif selected_page == "📈  Backtest Performance": 
    render_backtesting()
elif selected_page == "⚖️  Model Comparison":
    render_model_comparison()
elif selected_page == "✨  Methodology":
    render_methodology_lessons()
    
