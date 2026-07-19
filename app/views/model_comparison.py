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
    
    # ------------------------------------------------------------
    # OVERALL RECOMMENDATION
    # ------------------------------------------------------------

    overall_feature_set = "Stock + Market"
    overall_model = "Decision Tree"


    # ------------------------------------------------------------
    # COLOURS
    # ------------------------------------------------------------

    FEATURE_SET_COLOURS = {
        "Stock Features": {
            "accent": "#64748B",
            "dark": "#334155",
            "background": "#F8FAFC",
            "border": "#CBD5E1"
        },
        "Stock + Market": {
            "accent": "#EF4444",
            "dark": "#B91C1C",
            "background": "#FEF2F2",
            "border": "#FECACA"
        },
        "Stock + Market + Macro": {
            "accent": "#7C3AED",
            "dark": "#5B21B6",
            "background": "#F5F3FF",
            "border": "#DDD6FE"
        }
    }

    MODEL_COLOURS = {
        "Decision Tree": {
            "accent": "#F59E0B",
            "dark": "#92400E",
            "background": "#FFFBEB",
            "border": "#FDE68A"
        },
        "LightGBM": {
            "accent": "#10B981",
            "dark": "#047857",
            "background": "#ECFDF5",
            "border": "#A7F3D0"
        },
        "XGBoost": {
            "accent": "#2563EB",
            "dark": "#1D4ED8",
            "background": "#EFF6FF",
            "border": "#BFDBFE"
        }
    }

    feature_colours = FEATURE_SET_COLOURS[overall_feature_set]
    model_colours = MODEL_COLOURS[overall_model]


    # ------------------------------------------------------------
    # CONTENT
    # ------------------------------------------------------------

    feature_reasons = [
        (
            "Incremental market information",
            "ASX 200, sector and market-wide variables added useful context "
            "beyond stock-specific features alone."
        ),
        (
            "Strongest overall configuration",
            "The Stock + Market feature set produced the best combined prediction "
            "and realised portfolio outcome when paired with the Decision Tree."
        ),
        (
            "Efficient complexity",
            "It captured broader market conditions without the additional "
            "dimensionality and noise introduced by macroeconomic variables."
        )
    ]

    model_reasons = [
        (
            "Forecast accuracy",
            "Recorded the lowest MAE and RMSE, providing the strongest "
            "point-forecast accuracy across the candidate models."
        ),
        (
            "Portfolio performance",
            "Generated the highest annual return and the strongest overall "
            "realised portfolio outcome for the selected feature set."
        ),
        (
            "Simplicity and interpretability",
            "Delivered competitive risk-adjusted performance using a simpler "
            "and more transparent model than the boosting alternatives."
        )
    ]


    # ------------------------------------------------------------
    # STYLING
    # ------------------------------------------------------------

    st.markdown(
        f"""
        <style>
        .overall-summary-wrapper {{
            margin: 1rem 0 1.3rem 0;
        }}

        .overall-summary-heading {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.75rem;
            color: #0F172A;
            font-size: 1.05rem;
            font-weight: 800;
        }}

        .overall-summary-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.9rem;
        }}

        .overall-summary-card {{
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            min-height: 245px;
            box-shadow: 0 2px 7px rgba(15, 23, 42, 0.04);
        }}

        .overall-feature-card {{
            border-left: 5px solid {feature_colours["accent"]} !important;
            background: {feature_colours["background"]} !important;
        }}

        .overall-model-card {{
            border-left: 5px solid {model_colours["accent"]} !important;
            background: {model_colours["background"]} !important;
        }}

        .overall-card-title {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.7rem;
            font-size: 0.96rem;
            font-weight: 800;
        }}

        .overall-feature-title {{
            color: {feature_colours["dark"]} !important;
        }}

        .overall-model-title {{
            color: {model_colours["dark"]} !important;
        }}

        .overall-card-label {{
            color: #64748B;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }}

        .overall-card-value {{
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }}

        .overall-feature-value {{
            color: {feature_colours["accent"]} !important;
        }}

        .overall-model-value {{
            color: {model_colours["accent"]} !important;
        }}

        .overall-reason {{
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            margin-bottom: 0.58rem;
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.48;
        }}

        .overall-reason strong {{
            color: #1E293B;
        }}

        .overall-feature-check {{
            color: {feature_colours["accent"]} !important;
            font-weight: 900;
            line-height: 1.45;
        }}

        .overall-model-check {{
            color: {model_colours["accent"]} !important;
            font-weight: 900;
            line-height: 1.45;
        }}

        .overall-recommendation-card {{
            margin-top: 0.9rem;
            border: 1px solid #CBD5E1;
            border-left: 5px solid #334155;
            border-radius: 9px;
            background: #F8FAFC;
            padding: 0.9rem 1rem;
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.5;
        }}

        .overall-recommendation-title {{
            color: #0F172A;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}

        .overall-recommendation-name {{
            color: #0F172A;
            font-weight: 800;
        }}

        @media (max-width: 900px) {{
            .overall-summary-grid {{
                grid-template-columns: 1fr;
            }}

            .overall-summary-card {{
                min-height: auto;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------

    overall_summary_html = (
        '<div class="overall-summary-wrapper">'

            '<div class="overall-summary-heading">'
                '<span>◆</span>'
                '<span>Executive Summary</span>'
            '</div>'

            '<div class="overall-summary-grid">'

                '<div class="overall-summary-card overall-feature-card">'
                    '<div class="overall-card-title overall-feature-title">'
                        '<span>◉</span>'
                        '<span>Recommended Feature Set</span>'
                    '</div>'

                    '<div class="overall-card-label">Selected feature set</div>'
                    f'<div class="overall-card-value overall-feature-value">'
                        f'{overall_feature_set}'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-feature-check">✓</span>'
                        '<span>'
                            f'<strong>{feature_reasons[0][0]}:</strong> '
                            f'{feature_reasons[0][1]}'
                        '</span>'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-feature-check">✓</span>'
                        '<span>'
                            f'<strong>{feature_reasons[1][0]}:</strong> '
                            f'{feature_reasons[1][1]}'
                        '</span>'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-feature-check">✓</span>'
                        '<span>'
                            f'<strong>{feature_reasons[2][0]}:</strong> '
                            f'{feature_reasons[2][1]}'
                        '</span>'
                    '</div>'
                '</div>'

                '<div class="overall-summary-card overall-model-card">'
                    '<div class="overall-card-title overall-model-title">'
                        '<span>★</span>'
                        '<span>Recommended Model</span>'
                    '</div>'

                    '<div class="overall-card-label">Selected model</div>'
                    f'<div class="overall-card-value overall-model-value">'
                        f'{overall_model}'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-model-check">✓</span>'
                        '<span>'
                            f'<strong>{model_reasons[0][0]}:</strong> '
                            f'{model_reasons[0][1]}'
                        '</span>'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-model-check">✓</span>'
                        '<span>'
                            f'<strong>{model_reasons[1][0]}:</strong> '
                            f'{model_reasons[1][1]}'
                        '</span>'
                    '</div>'

                    '<div class="overall-reason">'
                        '<span class="overall-model-check">✓</span>'
                        '<span>'
                            f'<strong>{model_reasons[2][0]}:</strong> '
                            f'{model_reasons[2][1]}'
                        '</span>'
                    '</div>'
                '</div>'

            '</div>'

            '<div class="overall-recommendation-card">'
                '<div class="overall-recommendation-title">'
                    'Overall recommended configuration'
                '</div>'

                '<span class="overall-recommendation-name">'
                    f'{overall_model} using {overall_feature_set} features'
                '</span>'

                ' provides the strongest observed balance of forecast accuracy, '
                'realised portfolio performance, risk-adjusted returns and model '
                'interpretability. Market and sector features added useful information '
                'beyond stock-specific inputs, while the Decision Tree converted those '
                'signals into the strongest overall portfolio outcome. The detailed '
                'feature-set tabs below provide the supporting prediction, performance '
                'and hypothesis-testing evidence.'
            '</div>'

        '</div>'
    )

    st.markdown(
        overall_summary_html,
        unsafe_allow_html=True
    )

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
        lightgbm_hit = results["dt"][feature_name]["hit"]
        xgboost_hit = results["lightgbm"][feature_name]["hit"]

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
        
        t_stat, p_value, statement = pipeline_stock.portfolio_returns_test("Decision Trees", "LightGBM")
        
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
                r"\leq "
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
        
        selected_model = "LightGBM"

        MODEL_COLOURS = {
            "Decision Tree": {
                "accent": "#F59E0B",
                "dark": "#92400E",
                "background": "#FFFBEB",
                "border": "#FDE68A"
            },
            "LightGBM": {
                "accent": "#10B981",
                "dark": "#047857",
                "background": "#ECFDF5",
                "border": "#A7F3D0"
            },
            "XGBoost": {
                "accent": "#2563EB",
                "dark": "#1D4ED8",
                "background": "#EFF6FF",
                "border": "#BFDBFE"
            }
        }

        selected_colours = MODEL_COLOURS[selected_model]

        st.markdown(
            f"""
            <style>
            .model-summary-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.9rem;
                margin: 1rem 0 1.4rem 0;
            }}

            .model-summary-card {{
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 1rem 1.15rem;
                min-height: 270px;
                box-shadow: 0 2px 7px rgba(15, 23, 42, 0.04);
            }}

            .model-summary-findings-card {{
                border-left: 5px solid #2563EB;
                background: #F8FAFC;
            }}

            .model-summary-selection-card {{
                border-left: 5px solid {selected_colours["accent"]} !important;
                background: {selected_colours["background"]} !important;
            }}

            .model-summary-title {{
                display: flex;
                align-items: center;
                gap: 0.45rem;
                margin-bottom: 0.8rem;
                font-size: 1rem;
                font-weight: 800;
            }}

            .model-summary-findings-title {{
                color: #1D4ED8;
            }}

            .model-summary-selection-title {{
                color: {selected_colours["dark"]} !important;
            }}

            .model-summary-list {{
                margin: 0;
                padding-left: 1.15rem;
                color: #334155;
                font-size: 0.9rem;
                line-height: 1.58;
            }}

            .model-summary-list li {{
                margin-bottom: 0.5rem;
            }}

            .model-summary-label {{
                color: #64748B;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 0.2rem;
            }}

            .model-summary-name {{
                color: {selected_colours["accent"]} !important;
                font-size: 1.35rem;
                font-weight: 800;
                margin-bottom: 0.75rem;
            }}

            .model-summary-reason {{
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
                margin-bottom: 0.58rem;
                color: #334155;
                font-size: 0.9rem;
                line-height: 1.48;
            }}

            .model-summary-check {{
                color: {selected_colours["accent"]} !important;
                font-weight: 900;
                line-height: 1.45;
            }}

            .model-summary-note {{
                border-top: 1px solid {selected_colours["border"]} !important;
                margin-top: 0.8rem;
                padding-top: 0.7rem;
                color: #64748B;
                font-size: 0.84rem;
                line-height: 1.48;
            }}

            @media (max-width: 900px) {{
                .model-summary-grid {{
                    grid-template-columns: 1fr;
                }}

                .model-summary-card {{
                    min-height: auto;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        conclusion_html = (
            '<div class="model-summary-grid">'

                '<div class="model-summary-card model-summary-findings-card">'
                    '<div class="model-summary-title model-summary-findings-title">'
                        '<span>◆</span>'
                        '<span>Overall Results</span>'
                    '</div>'

                    '<ul class="model-summary-list">'
                        '<li>'
                            '<strong>Decision Tree delivered the highest annual return</strong> '
                            'and the strongest point-forecast accuracy, recording the lowest '
                            'MAE and RMSE.'
                        '</li>'

                        '<li>'
                            '<strong>LightGBM produced the strongest risk-adjusted performance</strong>, '
                            'with the highest Sharpe, Sortino and Calmar ratios, the smallest '
                            'maximum drawdown and the highest weekly win rate.'
                        '</li>'

                        '<li>'
                            '<strong>XGBoost achieved the strongest ranking performance</strong>, '
                            'recording the highest mean and annualised Information Coefficients '
                            'together with the highest directional hit rate.'
                        '</li>'

                        '<li>'
                            'The hypothesis tests found <strong>no statistically significant '
                            'differences</strong> in weekly IC, directional accuracy or mean '
                            'weekly portfolio returns at the 5% significance level.'
                        '</li>'
                    '</ul>'
                '</div>'

                '<div class="model-summary-card model-summary-selection-card">'
                    '<div class="model-summary-title model-summary-selection-title">'
                        '<span>★</span>'
                        '<span>Final Model Selection</span>'
                    '</div>'

                    '<div class="model-summary-label">Selected model</div>'
                    f'<div class="model-summary-name">{selected_model}</div>'

                    '<div class="model-summary-reason">'
                        '<span class="model-summary-check">✓</span>'
                        '<span><strong>Risk-adjusted performance:</strong> highest Sharpe, '
                        'Sortino and Calmar ratios.</span>'
                    '</div>'

                    '<div class="model-summary-reason">'
                        '<span class="model-summary-check">✓</span>'
                        '<span><strong>Downside protection:</strong> smallest maximum '
                        'drawdown and lowest annual volatility.</span>'
                    '</div>'

                    '<div class="model-summary-reason">'
                        '<span class="model-summary-check">✓</span>'
                        '<span><strong>Consistency:</strong> highest weekly win rate and '
                        'the strongest rolling Sharpe profile toward the end of the '
                        'evaluation period.</span>'
                    '</div>'

                    '<div class="model-summary-note">'
                        'Decision Tree generated the highest annual return and lowest forecast '
                        'error, while XGBoost achieved the strongest ranking metrics. However, '
                        f'{selected_model} is selected because it provides the strongest overall '
                        'balance of risk-adjusted performance, downside control and consistency. '
                        'This represents an economic preference rather than statistically proven '
                        'model superiority.'
                    '</div>'
                '</div>'

            '</div>'
        )

        st.markdown(conclusion_html, unsafe_allow_html=True)

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
                r"\leq "
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
        
        t_stat, p_value, statement = pipeline_market.mean_weekly_ic("Decision Trees", "LightGBM")
        
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
                r"\neq "
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
                r"\neq "
                r"\mu_{r,\mathrm{XGB}}"
            ),
            test_name="Two-sided paired t-test on aligned weekly portfolio returns.",
            accent_colour="#10B981",
            background_colour="#F0FDF4"
        )
        
        t_stat, p_value, statement = pipeline_market.portfolio_returns_test("Decision Trees", "LightGBM")
        
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
                r"\leq "
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
        
        st.markdown(
            """
            <style>
            .conclusion-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.9rem;
                margin: 1rem 0 1.4rem 0;
            }

            .conclusion-card {
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                background: #FFFFFF;
                padding: 1rem 1.15rem;
                min-height: 245px;
                box-shadow: 0 2px 7px rgba(15, 23, 42, 0.04);
            }

            .findings-card {
                border-left: 5px solid #2563EB;
                background: #F8FAFC;
            }

            .selection-card {
                border-left: 5px solid #F59E0B;
                background: #FFFBEB;
            }

            .conclusion-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                margin-bottom: 0.8rem;
                font-size: 1rem;
                font-weight: 800;
            }

            .findings-title {
                color: #1D4ED8;
            }

            .selection-title {
                color: #92400E;
            }

            .conclusion-list {
                margin: 0;
                padding-left: 1.15rem;
                color: #334155;
                font-size: 0.9rem;
                line-height: 1.58;
            }

            .conclusion-list li {
                margin-bottom: 0.5rem;
            }

            .selected-model-label {
                color: #64748B;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 0.2rem;
            }

            .selected-model-name {
                color: #D97706;
                font-size: 1.35rem;
                font-weight: 800;
                margin-bottom: 0.75rem;
            }

            .selection-reason {
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
                margin-bottom: 0.58rem;
                color: #334155;
                font-size: 0.9rem;
                line-height: 1.48;
            }

            .selection-check {
                color: #D97706;
                font-weight: 900;
                line-height: 1.45;
            }

            .selection-note {
                border-top: 1px solid #FDE68A;
                margin-top: 0.8rem;
                padding-top: 0.7rem;
                color: #64748B;
                font-size: 0.84rem;
                line-height: 1.48;
            }

            @media (max-width: 900px) {
                .conclusion-grid {
                    grid-template-columns: 1fr;
                }

                .conclusion-card {
                    min-height: auto;
                }
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        conclusion_html = (
            '<div class="conclusion-grid">'

                '<div class="conclusion-card findings-card">'
                    '<div class="conclusion-title findings-title">'
                        '<span>◆</span>'
                        '<span>Overall Results</span>'
                    '</div>'

                    '<ul class="conclusion-list">'
                        '<li>'
                            '<strong>Decision Tree</strong> achieved the strongest realised '
                            'portfolio performance, including the highest annual return and '
                            'the best Sharpe, Sortino and Calmar ratios.'
                        '</li>'

                        '<li>'
                            '<strong>XGBoost</strong> recorded the strongest ranking results, '
                            'with the highest Information Coefficient and directional hit rate.'
                        '</li>'

                        '<li>'
                            'The three models followed broadly similar performance patterns, '
                            'suggesting they captured many of the same relationships in the data.'
                        '</li>'

                        '<li>'
                            'The hypothesis tests found no statistically significant evidence '
                            'that one model consistently outperformed the others at the 5% level.'
                        '</li>'
                    '</ul>'
                '</div>'

                '<div class="conclusion-card selection-card">'
                    '<div class="conclusion-title selection-title">'
                        '<span>★</span>'
                        '<span>Final Model Selection</span>'
                    '</div>'

                    '<div class="selected-model-label">Selected model</div>'
                    '<div class="selected-model-name">Decision Tree</div>'

                    '<div class="selection-reason">'
                        '<span class="selection-check">✓</span>'
                        '<span><strong>Forecast accuracy:</strong> lowest MAE and RMSE.</span>'
                    '</div>'

                    '<div class="selection-reason">'
                        '<span class="selection-check">✓</span>'
                        '<span><strong>Portfolio performance:</strong> highest annual return, '
                        'Sharpe, Sortino and Calmar ratios.</span>'
                    '</div>'

                    '<div class="selection-reason">'
                        '<span class="selection-check">✓</span>'
                        '<span><strong>Risk and simplicity:</strong> smallest maximum drawdown '
                        'with lower model complexity and greater interpretability.</span>'
                    '</div>'

                    '<div class="selection-note">'
                        'XGBoost produced slightly stronger ranking metrics, but this advantage '
                        'did not translate into better realised portfolio performance and was '
                        'not statistically significant.'
                    '</div>'
                '</div>'

            '</div>'
        )

        st.markdown(conclusion_html, unsafe_allow_html=True)

    with macro_tab:
        st.write("hello")