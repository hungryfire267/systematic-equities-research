import os
import pandas as pd
from pathlib import Path
import streamlit as st
import sys

from components.feature_comparison import render_feature_comparison, render_hypothesis_card

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


from scripts.portfolio.metrics import GetMetrics, GetPredictionMetrics
BACKTEST_RESULTS_LIGHTGBM_DIR = BASE_DIR / "results" /  "backtest" / "lightgbm"
BACKTEST_RESULTS_XGBOOST_DIR = BASE_DIR / "results" /  "backtest" / "xgboost"

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.markdown("""
<h1 style="
    font-size:2.4rem;
    font-weight:800;
    color:#0F172A;
    margin-bottom:0;
">
Model Comparison
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
    font-size:1.02rem;
    color:#64748B;
    line-height:1.6;
    margin-top:0.35rem;
    margin-bottom:1.5rem;
">
Evaluating the incremental value of stock, market and macroeconomic features
using walk-forward validation and portfolio backtesting.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:#F8FAFC;
border:1px solid #CBD5E1;
padding:14px 18px;
border-radius:10px;
font-size:0.95rem;
color:#475569;
">
ℹ️ All feature sets were evaluated using identical walk-forward splits and
portfolio construction rules.
</div>
""", unsafe_allow_html=True)

stock_tab, market_tab, macro_tab = st.tabs(
    [
        "📈 Stock Features",
        "🌐 Stock + Market",
        "🏦 Stock + Market + Macro"
    ]
)

final_portfolio_lightgbm = pd.read_parquet(os.path.join(BACKTEST_RESULTS_LIGHTGBM_DIR, "final_portfolio_stock.parquet"))
final_portfolio_xgboost = pd.read_parquet(os.path.join(BACKTEST_RESULTS_XGBOOST_DIR, "final_portfolio_stock.parquet"))

portfolio_stock_metrics_lightgbm_dict, lightgbm_returns = GetMetrics(final_portfolio_lightgbm).run_data()
portfolio_stock_metrics_xgboost_dict, xgboost_returns = GetMetrics(final_portfolio_xgboost).run_data()

prediction_stock_metrics_lightgbm_dict, lightgbm_ic = GetPredictionMetrics(final_portfolio_lightgbm).run_data()
prediction_stock_metrics_xgboost_dict, xgboost_ic = GetPredictionMetrics(final_portfolio_xgboost).run_data()

lightgbm_stock_metrics = {
    "prediction": prediction_stock_metrics_lightgbm_dict, 
    "portfolio": portfolio_stock_metrics_lightgbm_dict
}

xgboost_stock_metrics = {
    "prediction": prediction_stock_metrics_xgboost_dict,
    "portfolio": portfolio_stock_metrics_xgboost_dict
}

with stock_tab:
    render_feature_comparison(
        feature_title="Stock-Specific Features",
        feature_description=(
            "Price, volume, momentum and volatility predictors."
        ),
        lightgbm_results=lightgbm_stock_metrics,
        xgboost_results=xgboost_stock_metrics,
        lightgbm_ic=lightgbm_ic,
        xgboost_ic=xgboost_ic,
        lightgbm_returns=lightgbm_returns,
        xgboost_returns=xgboost_returns
    )

    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="
                font-size:2.4rem;
                font-weight:800;
                color:#0F172A;
                margin-bottom:0.25rem;
            ">
                Hypothesis Testing
            </h1>

            <p style="
                font-size:1.02rem;
                color:#64748B;
                line-height:1.6;
                margin-top:0;
                margin-bottom:0;
            ">
                Statistical tests assessing whether differences in predictive quality,
                portfolio performance and feature-set value are greater than expected
                from random variation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    render_hypothesis_card(
        number=1,
        question="Does XGBoost produce a higher mean weekly IC than LightGBM?",
        null_hypothesis=(
            r"H_0:\ \mu_{IC,\mathrm{XGB}}"
            r"\leq"
            r"\mu_{IC,\mathrm{LGBM}}"
        ),
        alternative_hypothesis=(
            r"H_1:\ \mu_{IC,\mathrm{XGB}}"
            r">"
            r"\mu_{IC,\mathrm{LGBM}}"
        ),
        test_name="One-sided paired t-test on aligned weekly IC observations.",
        accent_colour="#2563EB",
        background_colour="#EFF6FF"
    )

    render_hypothesis_card(
        number=2,
        question="Do LightGBM and XGBoost have different directional hit rates?",
        null_hypothesis=(
            r"H_0:\ p_{\mathrm{hit,LGBM}}"
            r"="
            r"p_{\mathrm{hit,XGB}}"
        ),
        alternative_hypothesis=(
            r"H_1:\ p_{\mathrm{hit,LGBM}}"
            r"\neq"
            r"p_{\mathrm{hit,XGB}}"
        ),
        test_name="McNemar test on paired correct and incorrect predictions.",
        accent_colour="#7C3AED",
        background_colour="#F5F3FF"
    )

    render_hypothesis_card(
        number=3,
        question="Do LightGBM and XGBoost produce different mean weekly portfolio returns?",
        null_hypothesis=(
            r"H_0:\ \mu_{r,\mathrm{LGBM}}"
            r"="
            r"\mu_{r,\mathrm{XGB}}"
        ),
        alternative_hypothesis=(
            r"H_1:\ \mu_{r,\mathrm{LGBM}}"
            r"\neq"
            r"\mu_{r,\mathrm{XGB}}"
        ),
        test_name="Two-sided paired t-test on aligned weekly portfolio returns.",
        accent_colour="#10B981",
        background_colour="#F0FDF4"
    )

    render_hypothesis_card(
        number=4,
        question="Does LightGBM achieve a higher Sharpe ratio than XGBoost?",
        null_hypothesis=(
            r"H_0:\ SR_{\mathrm{LGBM}}"
            r"\leq"
            r"SR_{\mathrm{XGB}}"
        ),
        alternative_hypothesis=(
            r"H_1:\ SR_{\mathrm{LGBM}}"
            r">"
            r"SR_{\mathrm{XGB}}"
        ),
        test_name=(
            "One-sided Sharpe-ratio difference test using the "
            "Jobson–Korkie test with Memmel correction."
        ),
        accent_colour="#EA580C",
        background_colour="#FFF7ED"
    )

with market_tab:
    render_feature_comparison(
        feature_title="Stock + Market Features",
        feature_description=(
            "Stock-specific variables combined with ASX 200, "
            "sector and market-wide information."
        ),
        lightgbm_metrics=market_lightgbm_metrics,
        xgboost_metrics=market_xgboost_metrics
    )
    


with macro_tab:
    render_feature_comparison(
        feature_title="Stock + Market + Macro Features",
        feature_description=(
            "Stock and market information combined with rates, "
            "inflation, exchange rates and other macroeconomic variables."
        ),
        lightgbm_metrics=macro_lightgbm_metrics,
        xgboost_metrics=macro_xgboost_metrics
    )




#### Get Data for each model

final_portfolio_lightgbm = pd.read_parquet(os.path.join(BACKTEST_RESULTS_LIGHTGBM_DIR, "final_portfolio_stock.parquet"))
final_portfolio_xgboost = pd.read_parquet(os.path.join(BACKTEST_RESULTS_XGBOOST_DIR, "final_portfolio_stock.parquet"))

test_preds_lightgbm = pd.read_parquet(os.path.join(BACKTEST_RESULTS_LIGHTGBM_DIR, "test_preds_stock.parquet"))





metrics_lightgbm, _ = GetMetrics(final_portfolio_lightgbm).run_data()
metrics_xgboost, _ = GetMetrics(final_portfolio_xgboost).run_data()


prediction_metrics_lightgbm_dict = GetPredictionMetrics(final_portfolio_lightgbm).run_data()
prediction_metrics_xgboost_dict = GetPredictionMetrics(final_portfolio_xgboost).run_data()

prediction_metrics_df = pd.DataFrame({
    "Metric": list(prediction_metrics_lightgbm_dict.keys()),
    "LightGBM": list(prediction_metrics_lightgbm_dict.values()),
    "XGBoost": list(prediction_metrics_xgboost_dict.values())
})


st.table(prediction_metrics_df)
st.table(final_portfolio_lightgbm.head(10))
st.table(final_portfolio_xgboost.head(10))



st.write(metrics_lightgbm)
st.write(metrics_xgboost)

