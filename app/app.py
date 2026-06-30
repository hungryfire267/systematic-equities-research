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
This dashboard provides an interactive overview of the ASX Systematic Equities project. 
It enables exploration of the underlying data, engineered features, 
machine learning model outputs, portfolio construction, and backtesting results through a 
collection of interactive visualisations and performance metrics.
""")

col1, col2 = st.columns(2)

with col1: 
    st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis")
    
with col2: 
    st.page_link("pages/2_Economic_Analysis.py", label="Economic Analysis")

final_portfolio_df = pd.read_parquet(os.path.join(BACKTEST_RESULTS_DIR, "final_portfolio.parquet"))    
backtest_metrics_dict, strategy_returns = GetMetrics(final_portfolio_df).run_data()
    
col1, col2, col3 = st.columns(3)

annual_return = backtest_metrics_dict["annual_return"]
total_return = backtest_metrics_dict["total_return"]
sharpe = backtest_metrics_dict["sharpe_ratio"]

sortino_ratio = backtest_metrics_dict["sortino_ratio"]
annual_volatility = backtest_metrics_dict["annual_volatility"]
max_drawdown = backtest_metrics_dict["max_drawdown"]

calmar_ratio = backtest_metrics_dict["calmar_ratio"]
win_rate = backtest_metrics_dict["win_rate"]
worst_week = backtest_metrics_dict["worst_week"]

col1, col2, col3 = st.columns(3)

col1.metric("Annual Return", f"{annual_return:.2%}")
col2.metric("Total Return", f"{total_return:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")


col4, col5, col6 = st.columns(3)
col4.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
col5.metric("Annual Volatility", f"{annual_volatility:.2%}")
col6.metric("Max Drawdown", f"{max_drawdown:.2%}")

col7, col8, col9 = st.columns(3)

col7.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
col8.metric("Win Rate", f"{win_rate:.2%}")
col9.metric("Worst Week", f"{worst_week:.2%}")

st.subheader("Performance View")
plot_choice = st.segmented_control(
    "Performance View",
    ["Equity Curve", "Drawdown", "Weekly Returns"],
    default="Equity Curve"
)

chart_df = strategy_returns.copy()
chart_df["equity_curve"] = (1 + chart_df["portfolio_return"]).cumprod()
chart_df["drawdown"] = chart_df["equity_curve"] / chart_df["equity_curve"].cummax() - 1

st.markdown("### Strategy Performance")

if plot_choice == "Equity Curve":
    st.line_chart(chart_df, x="Date", y="equity_curve", height=420)

elif plot_choice == "Drawdown":
    st.line_chart(chart_df, x="Date", y="drawdown", height=420)

else:
    st.bar_chart(chart_df, x="Date", y="portfolio_return", height=420)