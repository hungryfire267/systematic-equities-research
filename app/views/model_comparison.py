import os
import pandas as pd
from pathlib import Path
import streamlit as st
import sys

from components.feature_comparison import (
    render_feature_comparison, 
    render_forecast_error_section,
    render_performance_comparison,
    render_hypothesis_card
)
from components.utils import get_hit_contingency_table

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


from scripts.portfolio.metrics import GetMetrics, GetPredictionMetrics
from scripts.portfolio.hypothesistest import ModelHypothesisTest

BACKTEST_RESULTS_DT_DIR = BASE_DIR / "results" / "backtest" / "dt"
BACKTEST_RESULTS_LIGHTGBM_DIR = BASE_DIR / "results" /  "backtest" / "lightgbm"
BACKTEST_RESULTS_XGBOOST_DIR = BASE_DIR / "results" /  "backtest" / "xgboost"

def render_model_comparison():
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

    model_paths = {
        "dt": BACKTEST_RESULTS_DT_DIR,
        "lightgbm": BACKTEST_RESULTS_LIGHTGBM_DIR, 
        "xgboost": BACKTEST_RESULTS_XGBOOST_DIR
    }

    feature_sets = {
        "stock": "final_portfolio_stock.parquet",
        "market": "final_portfolio_market.parquet",
        "macro_market": "final_portfolio_macro_market.parquet",
    }

    results = {}
    for model_name, model_dir in model_paths.items(): 
        results[model_name] = {}
        for feature_name, filename in feature_sets.items():
            portfolio = pd.read_parquet(os.path.join(model_dir, filename))
            portfolio_metrics, returns = GetMetrics(portfolio).run_data()
            prediction_metrics, ic, hit = GetPredictionMetrics(portfolio).run_data()
            
            results[model_name][feature_name] = {
                "portfolio_data": portfolio,
                "metrics": {
                    "prediction": prediction_metrics,
                    "portfolio": portfolio_metrics
                },
                "returns": returns,
                "ic": ic, 
                "hit": hit
            }

    print(results["dt"]["macro_market"]["ic"])

    hit_contingency_tables = {}
    for feature_name in feature_sets:
        lightgbm_hit = results["lightgbm"][feature_name]["hit"]
        xgboost_hit = results["xgboost"][feature_name]["hit"]

        hit_contingency_tables[feature_name] = (
            get_hit_contingency_table(lightgbm_hit, xgboost_hit).to_numpy()
        )
        

    pipeline_stock = ModelHypothesisTest(
        alpha=0.05,
        dt_ic=results["dt"]["stock"]["ic"],
        xgboost_ic=results["xgboost"]["stock"]["ic"],
        lgbm_ic=results["lightgbm"]["stock"]["ic"], 
        hit_contingency_table=hit_contingency_tables["stock"], 
        dt_returns=results["dt"]["stock"]["returns"]["portfolio_return"],
        lightgbm_returns=results["lightgbm"]["stock"]["returns"]["portfolio_return"], 
        xgboost_returns=results["xgboost"]["stock"]["returns"]["portfolio_return"]
    )

    pipeline_market = ModelHypothesisTest(
        alpha=0.05,
        dt_ic=results["dt"]["stock"]["ic"],
        xgboost_ic=results["xgboost"]["market"]["ic"],
        lgbm_ic=results["lightgbm"]["market"]["ic"], 
        hit_contingency_table=hit_contingency_tables["market"],
        dt_returns=results["dt"]["stock"]["returns"]["portfolio_return"],
        lightgbm_returns=results["lightgbm"]["market"]["returns"]["portfolio_return"], 
        xgboost_returns=results["xgboost"]["market"]["returns"]["portfolio_return"]
    )

    with stock_tab:
        render_feature_comparison(
            feature_title="Stock-Specific Features",
            feature_description=(
                "Price, volume, momentum and volatility predictors."
            ),
            dt_results=results["dt"]["stock"]["metrics"],
            lightgbm_results=results["lightgbm"]["stock"]["metrics"],
            xgboost_results=results["xgboost"]["stock"]["metrics"],
            dt_ic=results["dt"]["stock"]["ic"],
            lightgbm_ic=results["lightgbm"]["stock"]["ic"],
            xgboost_ic=results["xgboost"]["stock"]["ic"],
        )
        
        print(results["dt"]["stock"]["metrics"])
        
        render_performance_comparison(
            dt_results=results["dt"]["stock"]["metrics"],
            lightgbm_results=results["lightgbm"]["stock"]["metrics"],
            xgboost_results=results["xgboost"]["stock"]["metrics"],
            dt_returns=results["dt"]["stock"]["returns"],
            lightgbm_returns=results["lightgbm"]["stock"]["returns"],
            xgboost_returns=results["xgboost"]["stock"]["returns"]
        )
        
        st.write("#### Portfolio Performance")
        st.caption(
            """
            Evaluate the performance of the selected strategy using cumulative returns,
            risk-adjusted performance metrics, and benchmark comparisons over the
            backtesting period.
            """
        )
        
        st.write("#### Hypothesis Testing")
        st.caption(
        """
            Statistical tests assessing whether differences in predictive quality,
            portfolio performance and feature-set value are greater than expected
            from random variation.
        """
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
        
        t_stat, p_value, statement = pipeline_stock.mean_weekly_ic("Decision Trees", "LightGBM")
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{t_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_stock.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        chi_squared_stat, p_value, statement = pipeline_stock.mcnemar_test()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{chi_squared_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_stock.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        t_stat, p_value, statement = pipeline_stock.portfolio_returns_test()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{t_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_stock.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        results_dict = pipeline_stock.sharpe_ratio_test()
        print(results_dict)

    with market_tab:
        render_feature_comparison(
            feature_title="Stock + Market Features",
            feature_description=(
                "Stock-specific variables combined with ASX 200, "
                "sector and market-wide information."
            ),
            dt_results=results["dt"]["market"]["metrics"],
            lightgbm_results=results["lightgbm"]["market"]["metrics"],
            xgboost_results=results["xgboost"]["market"]["metrics"],
            dt_ic=results["dt"]["market"]["ic"],
            lightgbm_ic=results["lightgbm"]["market"]["ic"],
            xgboost_ic=results["xgboost"]["market"]["ic"]
        )
        
        render_performance_comparison(
            dt_results = results["dt"]["market"]["metrics"],
            lightgbm_results=results["lightgbm"]["market"]["metrics"],
            xgboost_results=results["xgboost"]["market"]["metrics"],
            dt_returns=results["dt"]["market"]["returns"],
            lightgbm_returns=results["lightgbm"]["market"]["returns"],
            xgboost_returns=results["xgboost"]["market"]["returns"]
        )
        
        st.write("#### Hypothesis Testing")
        st.caption(
        """
            Statistical tests assessing whether differences in predictive quality,
            portfolio performance and feature-set value are greater than expected
            from random variation.
        """
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
        
        t_stat, p_value, statement = pipeline_market.mean_weekly_ic()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{t_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_market.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        chi_squared_stat, p_value, statement = pipeline_market.mcnemar_test()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{chi_squared_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_market.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        t_stat, p_value, statement = pipeline_market.portfolio_returns_test()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="t-statistic",
                value=f"{t_stat:.4f}"
            )

        with col2:
            st.metric(
                label="p-value",
                value=f"{p_value:.4f}"
            )

        if p_value < pipeline_market.alpha:
            st.success(statement)
        else:
            st.info(statement)

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
        
        results_dict = pipeline_market.sharpe_ratio_test()
        print(results_dict)

    with macro_tab:
        st.write("hello")