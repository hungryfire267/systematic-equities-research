import os
import pandas as pd
from pathlib import Path
import streamlit as st
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from scripts.portfolio.metrics import GetMetrics
BACKTEST_RESULTS_DIR = BASE_DIR / "results" /  "backtest"

st.set_page_config(
    page_title="ASX Systematic Equities Dashboard",
    layout="wide"
)

st.title("ASX Systematic Equities Dashboard")
st.write("As of 1")

st.markdown("""
Use the sidebar to explore:

- Data Overview
- Signals
- Backtest
- Portfolio
- Model Diagnostics
""")

col1, col2 = st.columns(2)

with col1: 
    st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis")
    
with col2: 
    st.page_link("pages/2_Economic_Analysis.py", label="Economic Analysis")

final_portfolio_df = pd.read_parquet(os.path.join(BACKTEST_RESULTS_DIR, "final_portfolio.parquet"))    
backtest_metrics_dict = GetMetrics(final_portfolio_df).run_data()

print(backtest_metrics_dict)
    
col1, col2, col3 = st.columns(3)

annual_return = backtest_metrics_dict["annual_return"]
sharpe = backtest_metrics_dict["sharpe_ratio"]
max_drawdown = backtest_metrics_dict["max_drawdown"]
win_rate = backtest_metrics_dict["win_rate"]

print(win_rate)

with col1:
    st.metric("Annual Return", f"{annual_return:.2%}")

with col2:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

with col3:
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

win_rate_col, ic_col, stocks_traded_col = st.columns(3)

with win_rate_col: 
    st.metric("Win Rate", f"{win_rate:.2%}")