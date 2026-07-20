import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import streamlit as st
import sys

from components.render_alpha import render_alpha_metric_cards
from components.render_backtest_metrics import render_backtest_metric_cards
from components.render_cumulative_returns import render_cumulative_returns
from components.render_drawdown import render_drawdown
from components.render_return_distribution import (
    render_return_distribution
)
from components.render_return_summary import (
    render_return_summary
)


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


from scripts.portfolio.metrics import GetMetrics, GetPredictionMetrics
from scripts.portfolio.hypothesistest import ModelHypothesisTest
from scripts.dashboard.get_asx_metrics import ASXMetrics
from scripts.dashboard.alpha_metrics import AlphaMetrics


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

strategy_returns = (
    portfolio_returns
    .set_index("Date")["portfolio_return"]
)

strategy_returns.name = "Decision Tree Strategy"


# ---------------------------------------------------------
# 4. Load the daily ASX 200 index prices
# ---------------------------------------------------------

asx_index_df = pd.read_parquet(
    os.path.join(
        ASX_DIR,
        "asx_index.parquet"
    )
)


# ---------------------------------------------------------
# 5. Calculate ASX returns using the exact same
#    strategy rebalance dates
# ---------------------------------------------------------

asx_metrics = ASXMetrics(
    prices_df=asx_index_df,
    rebalance_dates=strategy_returns.index,
    date_col="Date",
    price_col="^AXJO",
    periods_per_year=52,
    risk_free_rate=0.0
)

asx_returns = (
    asx_metrics
    .get_holding_period_returns()
    .rename("ASX 200")
)

asx_metrics_full = asx_metrics.get_metrics()

print(asx_metrics_full)

def render_backtesting():
    st.markdown("## Backtest Performance")

    st.caption(
        "A comparison between the ASX 200 benchmark "
        "and the Decision Tree strategy."
    )
    
    st.markdown("### Performance Highlights")

    st.caption(
        "Headline Decision Tree results compared with the ASX 200."
    )
    
    render_backtest_metric_cards(
        strategy_metrics=portfolio_metrics,
        benchmark_metrics=asx_metrics_full,
        strategy_name="Decision Tree",
        benchmark_name="ASX 200"
    )
    
    st.markdown("### Portfolio Performance")

    st.caption(
        "Growth, downside risk and return characteristics "
        "of the Decision Tree strategy relative to the ASX 200."
    )
    
    performance_col, drawdown_col = st.columns(
        2,
        gap="large"
    )
    
    selected_model = "Decision Tree"
    with performance_col:
        render_cumulative_returns(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name=selected_model,
            benchmark_name="ASX 200"
        )
    
    with drawdown_col:
        render_drawdown(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name=selected_model,
            benchmark_name="ASX 200"
        )
    
    left_column, right_column = st.columns(
        2,
        gap="large"
    )
    
    print(portfolio_returns)

    with left_column:
        render_return_summary(
            strategy_metrics=portfolio_metrics,
            benchmark_metrics=asx_metrics_full,
            strategy_name="Decision Tree",
            benchmark_name="ASX 200"
        )

    with right_column:
        render_return_distribution(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name="Decision Tree",
            benchmark_name="ASX 200"
        )
    
    
    alpha_analysis = AlphaMetrics(
        strategy_returns=strategy_returns,
        benchmark_returns=asx_returns,
        periods_per_year=52,
        risk_free_rate=0.0
    )

    alpha_metrics = alpha_analysis.get_metrics()

    st.write("## Alpha Analysis")

    render_alpha_metric_cards(alpha_metrics)