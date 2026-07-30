import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from google import genai
import os
import json
import html
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


from scripts.portfolio.metrics import GetMetrics, GetPredictionMetrics

BACKTEST_RESULTS_DT_DIR = BASE_DIR / "results" / "backtest" / "dt"
BACKTEST_RESULTS_LIGHTGBM_DIR = BASE_DIR / "results" /  "backtest" / "lightgbm"
BACKTEST_RESULTS_XGBOOST_DIR = BASE_DIR / "results" /  "backtest" / "xgboost"

# ============================================================
# Shared model-comparison utilities
# ============================================================

def get_hit_contingency_table(
    decision_tree_hit_df: pd.DataFrame,
    lightgbm_hit_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the paired hit/miss table for Decision Tree vs LightGBM."""

    required_columns = {"Date", "Ticker", "hit"}

    for name, frame in {
        "Decision Tree": decision_tree_hit_df,
        "LightGBM": lightgbm_hit_df
    }.items():
        missing = required_columns.difference(frame.columns)

        if missing:
            raise KeyError(
                f"{name} hit data is missing columns: {sorted(missing)}"
            )

    comparison = (
        decision_tree_hit_df[["Date", "Ticker", "hit"]]
        .rename(columns={"hit": "decision_tree_hit"})
        .merge(
            lightgbm_hit_df[["Date", "Ticker", "hit"]]
            .rename(columns={"hit": "lightgbm_hit"}),
            on=["Date", "Ticker"],
            how="inner",
            validate="one_to_one"
        )
        .dropna(subset=["decision_tree_hit", "lightgbm_hit"])
    )

    comparison["decision_tree_hit"] = comparison[
        "decision_tree_hit"
    ].astype(bool)

    comparison["lightgbm_hit"] = comparison[
        "lightgbm_hit"
    ].astype(bool)

    table = pd.crosstab(
        comparison["decision_tree_hit"],
        comparison["lightgbm_hit"]
    ).reindex(
        index=[True, False],
        columns=[True, False],
        fill_value=0
    )

    table.index = [
        "Decision Tree hit",
        "Decision Tree miss"
    ]

    table.columns = [
        "LightGBM hit",
        "LightGBM miss"
    ]

    return table


class ModelHypothesisTest:
    """
    Compare LightGBM against the Decision Tree baseline.

    Tests
    -----
    1. Mean weekly IC:
       H1: mean(IC_LGBM) > mean(IC_DT)
    2. Directional hit rate:
       H1: hit rates differ between LGBM and DT
    3. Mean weekly portfolio return:
       H1: mean(return_LGBM) > mean(return_DT)
    4. Sharpe ratio:
       H1: Sharpe_LGBM > Sharpe_DT
    """

    def __init__(
        self,
        alpha: float,
        dt_ic: pd.Series,
        xgboost_ic: pd.Series,
        lgbm_ic: pd.Series,
        hit_contingency_table,
        dt_returns: pd.Series,
        lightgbm_returns: pd.Series,
        xgboost_returns: pd.Series
    ) -> None:
        self.alpha = alpha

        self.dt_ic = dt_ic
        self.xgboost_ic = xgboost_ic
        self.lgbm_ic = lgbm_ic

        self.contingency_table = np.asarray(
            hit_contingency_table,
            dtype=float
        )

        self.dt_returns = dt_returns
        self.lightgbm_returns = lightgbm_returns
        self.xgboost_returns = xgboost_returns

    @staticmethod
    def _aligned_pair(
        first: pd.Series,
        second: pd.Series,
        first_name: str,
        second_name: str
    ) -> pd.DataFrame:
        aligned = pd.concat(
            [
                pd.Series(first).rename(first_name),
                pd.Series(second).rename(second_name)
            ],
            axis=1
        ).dropna()

        if len(aligned) < 3:
            raise ValueError(
                "At least three aligned observations are required."
            )

        return aligned.astype(float)

    def mean_weekly_ic(self):
        """One-sided paired t-test: LightGBM mean IC > Decision Tree mean IC."""

        aligned = self._aligned_pair(
            self.dt_ic,
            self.lgbm_ic,
            "Decision Tree",
            "LightGBM"
        )

        t_statistic, p_value = stats.ttest_rel(
            aligned["LightGBM"],
            aligned["Decision Tree"],
            alternative="greater"
        )

        if p_value < self.alpha:
            statement = (
                "Reject the null hypothesis. The aligned observations provide "
                "evidence that LightGBM produces a higher mean weekly IC than "
                "the Decision Tree baseline."
            )
        else:
            statement = (
                "Do not reject the null hypothesis. The aligned observations "
                "do not provide sufficient evidence that LightGBM produces a "
                "higher mean weekly IC than the Decision Tree baseline."
            )

        return float(t_statistic), float(p_value), statement

    def mcnemar_test(self):
        """Two-sided McNemar test: DT and LightGBM hit rates are different."""

        if self.contingency_table.shape != (2, 2):
            raise ValueError(
                "McNemar's test requires a 2x2 paired contingency table."
            )

        result = mcnemar(
            self.contingency_table,
            exact=False,
            correction=True
        )

        chi_squared_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        if p_value < self.alpha:
            statement = (
                "Reject the null hypothesis. Decision Tree and LightGBM have "
                "statistically different directional hit rates."
            )
        else:
            statement = (
                "Do not reject the null hypothesis. There is insufficient "
                "evidence that Decision Tree and LightGBM have different "
                "directional hit rates."
            )

        return chi_squared_statistic, p_value, statement

    def portfolio_returns_test(self):
        """
        One-sided paired t-test:
        LightGBM mean weekly return > Decision Tree mean weekly return.
        """

        aligned = self._aligned_pair(
            self.dt_returns,
            self.lightgbm_returns,
            "Decision Tree",
            "LightGBM"
        )

        t_statistic, p_value = stats.ttest_rel(
            aligned["LightGBM"],
            aligned["Decision Tree"],
            alternative="greater"
        )

        if p_value < self.alpha:
            statement = (
                "Reject the null hypothesis. LightGBM produces a higher mean "
                "weekly portfolio return than the Decision Tree baseline."
            )
        else:
            statement = (
                "Do not reject the null hypothesis. There is insufficient "
                "evidence that LightGBM produces a higher mean weekly "
                "portfolio return than the Decision Tree baseline."
            )

        return float(t_statistic), float(p_value), statement

    def sharpe_ratio_test(self, periods_per_year: int = 52) -> dict:
        """
        One-sided Jobson-Korkie Sharpe-ratio test with Memmel correction.

        H1: Sharpe_LGBM > Sharpe_DT.
        Non-annualised Sharpe ratios are used in the test statistic.
        """

        aligned = self._aligned_pair(
            self.dt_returns,
            self.lightgbm_returns,
            "Decision Tree",
            "LightGBM"
        )

        dt = aligned["Decision Tree"]
        lgbm = aligned["LightGBM"]
        n_observations = len(aligned)

        dt_std = dt.std(ddof=1)
        lgbm_std = lgbm.std(ddof=1)

        if dt_std == 0 or lgbm_std == 0:
            raise ValueError(
                "Sharpe ratios cannot be tested when return volatility is zero."
            )

        dt_sharpe = dt.mean() / dt_std
        lgbm_sharpe = lgbm.mean() / lgbm_std
        correlation = dt.corr(lgbm)

        variance = (
            2 * (1 - correlation)
            + 0.5 * (
                dt_sharpe**2
                + lgbm_sharpe**2
                - 2
                * correlation**2
                * dt_sharpe
                * lgbm_sharpe
            )
        ) / n_observations

        if not np.isfinite(variance) or variance <= 0:
            raise ValueError(
                "The estimated variance of the Sharpe-ratio difference "
                "must be positive."
            )

        test_statistic = (
            lgbm_sharpe - dt_sharpe
        ) / np.sqrt(variance)

        p_value = float(stats.norm.sf(test_statistic))
        annualisation_factor = np.sqrt(periods_per_year)

        if p_value < self.alpha:
            statement = (
                "Reject the null hypothesis. LightGBM has a statistically "
                "higher Sharpe ratio than the Decision Tree baseline."
            )
        else:
            statement = (
                "Do not reject the null hypothesis. There is insufficient "
                "evidence that LightGBM has a higher Sharpe ratio than the "
                "Decision Tree baseline."
            )

        return {
            "test": (
                "One-sided Jobson-Korkie Sharpe-ratio test "
                "with Memmel correction"
            ),
            "n_observations": n_observations,
            "dt_sharpe": dt_sharpe * annualisation_factor,
            "lgbm_sharpe": lgbm_sharpe * annualisation_factor,
            "sharpe_difference": (
                lgbm_sharpe - dt_sharpe
            ) * annualisation_factor,
            "correlation": correlation,
            "test_statistic": float(test_statistic),
            "z_statistic": float(test_statistic),
            "p_value": p_value,
            "reject_null": p_value < self.alpha,
            "statement": statement
        }


# ============================================================
# Prediction, performance and hypothesis-rendering components
# ============================================================

MODEL_COLOURS = {
    "Decision Tree": "#F59E0B",
    "LightGBM": "#10B981",
    "XGBoost": "#2563EB"
}



def render_section_header(
    icon: str,
    title: str,
    description: str
) -> None:
    """Render a dashboard section heading consistent with the other pages."""
    st.markdown(
        f"""
        <div class="comparison-section-header">
            <div class="comparison-section-icon">{icon}</div>
            <div>
                <div class="comparison-section-title">{title}</div>
                <div class="comparison-section-description">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_set_banner(
    feature_title: str,
    feature_description: str
) -> None:
    """Render the selected feature-set heading inside each tab."""
    feature_class = (
        "feature-market-banner"
        if "Market" in feature_title
        else "feature-stock-banner"
    )

    icon = "🌐" if "Market" in feature_title else "📈"

    st.markdown(
        f"""
        <div class="feature-set-banner {feature_class}">
            <div class="feature-set-icon">{icon}</div>
            <div>
                <div class="feature-set-title">{feature_title}</div>
                <div class="feature-set-description">
                    {feature_description}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_comparison(
    feature_title: str,
    feature_description: str,
    dt_results: dict,
    lightgbm_results: dict,
    xgboost_results: dict,
    dt_ic: pd.Series,
    lightgbm_ic: pd.Series,
    xgboost_ic: pd.Series
) -> None:
    dt_pred = dt_results["prediction"]
    lgbm_pred = lightgbm_results["prediction"]
    xgb_pred = xgboost_results["prediction"]

    render_feature_set_banner(
        feature_title=feature_title,
        feature_description=feature_description
    )

    render_section_header(
        icon="↗",
        title="Prediction Analytics",
        description=(
            "Compare ranking quality, directional accuracy and forecast "
            "error before portfolio construction."
        )
    )

    top_left, top_right = st.columns(2, gap="large")

    with top_left:
        with st.container(border=True, height=445):
            st.plotly_chart(
                create_prediction_metric_chart(
                    dt_pred,
                    lgbm_pred,
                    xgb_pred
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"prediction_metrics_{feature_title}"
            )

    with top_right:
        with st.container(border=True, height=445):
            st.plotly_chart(
                create_ic_chart(
                    dt_ic,
                    lightgbm_ic,
                    xgboost_ic
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ic_chart_{feature_title}"
            )

    hit_col, error_col = st.columns(2, gap="large")

    with hit_col:
        with st.container(border=True, height=245):
            render_hit_rate_cards(
                dt_pred,
                lgbm_pred,
                xgb_pred
            )

    with error_col:
        with st.container(border=False, height=245):
            render_forecast_error_section(
                dt_pred,
                lgbm_pred,
                xgb_pred
            )

def render_performance_comparison(
    dt_results: dict,
    lightgbm_results: dict,
    xgboost_results: dict,
    dt_returns: pd.DataFrame,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
) -> None:
    dt_port = dt_results["portfolio"]
    lgbm_port = lightgbm_results["portfolio"]
    xgb_port = xgboost_results["portfolio"]

    render_section_header(
        icon="▥",
        title="Performance Analytics",
        description=(
            "Translate model forecasts into cumulative returns, drawdowns "
            "and risk-adjusted portfolio outcomes."
        )
    )

    top_left, top_right = st.columns(2, gap="large")

    with top_left:
        with st.container(border=True, height=455):
            st.plotly_chart(
                create_equity_curve(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"equity_{id(dt_returns)}"
            )

    with top_right:
        with st.container(border=True, height=455):
            st.plotly_chart(
                create_drawdown_curve(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"drawdown_{id(dt_returns)}"
            )

    bottom_left, bottom_right = st.columns(2, gap="large")

    with bottom_left:
        with st.container(border=True, height=430):
            render_portfolio_performance_table(
                lgbm_port,
                xgb_port,
                dt_port
            )

    with bottom_right:
        with st.container(border=True, height=430):
            st.plotly_chart(
                create_rolling_sharpe_chart(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns,
                    window=13
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"rolling_sharpe_{id(dt_returns)}"
            )

def render_portfolio_performance_table(
    lightgbm_portfolio: dict,
    xgboost_portfolio: dict,
    decision_tree_portfolio: dict
) -> None:
    metrics = [
        ("Annual Return", "annual_return", True, True),
        ("Sharpe Ratio", "sharpe_ratio", False, True),
        ("Sortino Ratio", "sortino_ratio", False, True),
        ("Annual Volatility", "annual_volatility", True, False),
        ("Maximum Drawdown", "max_drawdown", True, True),
        ("Calmar Ratio", "calmar_ratio", False, True),
        ("Weekly Win Rate", "win_rate", True, True)
    ]

    portfolios = {
        "Decision Tree": decision_tree_portfolio,
        "LightGBM": lightgbm_portfolio,
        "XGBoost": xgboost_portfolio
    }

    badge_colours = {
        "Decision Tree": {
            "background": "#FEF3C7",
            "colour": "#D97706"
        },
        "LightGBM": {
            "background": "#DCFCE7",
            "colour": "#059669"
        },
        "XGBoost": {
            "background": "#DBEAFE",
            "colour": "#2563EB"
        }
    }

    rows = ""

    for label, key, percentage, higher_is_better in metrics:
        values = {
            model_name: portfolio[key]
            for model_name, portfolio in portfolios.items()
        }

        if higher_is_better:
            winner = max(values, key=values.get)
        else:
            winner = min(values, key=values.get)

        displays = {
            model_name: (
                f"{value:.1%}"
                if percentage
                else f"{value:.2f}"
            )
            for model_name, value in values.items()
        }

        badge_background = badge_colours[winner]["background"]
        badge_colour = badge_colours[winner]["colour"]

        rows += (
            "<tr>"
            f'<td style="text-align:center;">{label}</td>'
            f"<td>{displays['Decision Tree']}</td>"
            f"<td>{displays['LightGBM']}</td>"
            f"<td>{displays['XGBoost']}</td>"
            "<td>"
            f'<span style="background:{badge_background};'
            f'color:{badge_colour};padding:0.2rem 0.6rem;'
            'border-radius:999px;font-size:0.72rem;'
            f'font-weight:700;white-space:nowrap;">{winner}</span>'
            "</td>"
            "</tr>"
        )

    table_html = (
        '<div style="border:0;'
        'border-radius:0;overflow:hidden;background:#FFFFFF;'
        'height:360px;">'
        '<div style="padding:0.8rem 0.9rem;'
        'font-size:0.9rem;font-weight:750;color:#0F172A;">'
        "Portfolio Performance Summary"
        "</div>"
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.8rem;">'
        "<thead>"
        '<tr style="background:#F8FAFC;">'
        '<th style="text-align:left;">Metric</th>'
        "<th>Decision Tree</th>"
        "<th>LightGBM</th>"
        "<th>XGBoost</th>"
        "<th>Better</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
        "<style>"
        "table th, table td {"
        "padding:0.58rem 0.7rem;"
        "border-top:1px solid #E2E8F0;"
        "text-align:center;"
        "vertical-align:middle;"
        "}"
        "table th {"
        "font-weight:700;"
        "color:#334155;"
        "}"
        "table td {"
        "color:#0F172A;"
        "}"
        "</style>"
    )

    st.markdown(table_html, unsafe_allow_html=True)
    
def render_sharpe_comparison(
    dt_portfolio: dict,
    lightgbm_portfolio: dict,
    xgboost_portfolio: dict
) -> None:
    sharpe_ratios = {
        "Decision Tree": dt_portfolio["sharpe_ratio"],
        "LightGBM": lightgbm_portfolio["sharpe_ratio"],
        "XGBoost": xgboost_portfolio["sharpe_ratio"]
    }

    model_colours = {
        "Decision Tree": "#F59E0B",
        "LightGBM": "#10B981",
        "XGBoost": "#2563EB"
    }

    winner = max(sharpe_ratios, key=sharpe_ratios.get)
    winner_sharpe = sharpe_ratios[winner]

    sorted_sharpes = sorted(
        sharpe_ratios.values(),
        reverse=True
    )
    difference = sorted_sharpes[0] - sorted_sharpes[1]

    model_cards = ""

    for model_name, sharpe_ratio in sharpe_ratios.items():
        model_cards += (
            '<div style="'
            'background:#FFFFFF;'
            f'border:1px solid {model_colours[model_name]}55;'
            'border-radius:10px;'
            'padding:0.85rem;'
            '">'
            '<p style="'
            'margin:0;'
            'color:#64748B;'
            'font-size:0.72rem;'
            'font-weight:800;'
            '">'
            f'{model_name.upper()}'
            '</p>'
            '<p style="'
            'margin:0.2rem 0 0 0;'
            'font-size:1.7rem;'
            'font-weight:800;'
            f'color:{model_colours[model_name]};'
            '">'
            f'{sharpe_ratio:.2f}'
            '</p>'
            '</div>'
        )

    card_html = (
        '<div style="'
        'border:1px solid #BBF7D0;'
        'border-radius:12px;'
        'background:linear-gradient(135deg,#F0FDF4,#ECFDF5);'
        'padding:1.1rem 1.2rem;'
        'min-height:260px;'
        '">'
        '<p style="'
        'margin:0 0 0.35rem 0;'
        'font-size:0.76rem;'
        'font-weight:800;'
        'letter-spacing:0.08em;'
        'color:#059669;'
        '">'
        'RISK-ADJUSTED PERFORMANCE'
        '</p>'
        '<p style="'
        'margin:0 0 0.8rem 0;'
        'font-size:1rem;'
        'font-weight:750;'
        'color:#0F172A;'
        '">'
        'Sharpe Ratio Comparison'
        '</p>'
        '<div style="'
        'display:grid;'
        'grid-template-columns:repeat(3, minmax(0, 1fr));'
        'gap:0.8rem;'
        '">'
        f'{model_cards}'
        '</div>'
        '<p style="'
        'margin:0.9rem 0 0 0;'
        'color:#334155;'
        'font-size:0.84rem;'
        'line-height:1.5;'
        '">'
        f'<strong>{winner}</strong> achieved the highest Sharpe ratio of '
        f'{winner_sharpe:.2f}, exceeding the next-best model by {difference:.2f}.'
        '</p>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )
    


def create_prediction_metric_chart(
    dt_prediction: dict,
    lightgbm_prediction: dict,
    xgboost_prediction: dict
):
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean IC",
                "Annualised ICIR"
            ],
            "Decision Tree": [
                dt_prediction["mean_ic"],
                dt_prediction["annualised_icir"]
            ],
            "LightGBM": [
                lightgbm_prediction["mean_ic"],
                lightgbm_prediction["annualised_icir"]
            ],
            "XGBoost": [
                xgboost_prediction["mean_ic"],
                xgboost_prediction["annualised_icir"]
            ]
        }
    )

    long_df = metrics_df.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Value"
    )

    fig = px.bar(
        long_df,
        x="Value",
        y="Metric",
        color="Model",
        barmode="group",
        orientation="h",
        text="Value",
        color_discrete_map=MODEL_COLOURS,
        title="Ranking & Correlation Metrics"
    )

    fig.update_traces(
        texttemplate="%{x:.4f}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=70, t=55, b=20),
        legend_title_text="",
        yaxis_title="",
        xaxis_title="",
        hovermode="y unified",
        bargap=0.30,
        bargroupgap=0.08,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=False
    )

    return fig


def render_hit_rate_cards(
    dt_prediction: dict,
    lightgbm_prediction: dict,
    xgboost_prediction: dict
) -> None:
    prediction_metrics = {
        "Decision Tree": dt_prediction,
        "LightGBM": lightgbm_prediction,
        "XGBoost": xgboost_prediction
    }

    card_styles = {
        "Decision Tree": {
            "border": "#FCD34D",
            "background": "#FFFBEB"
        },
        "LightGBM": {
            "border": "#A7F3D0",
            "background": "#F0FDF4"
        },
        "XGBoost": {
            "border": "#BFDBFE",
            "background": "#EFF6FF"
        }
    }

    winner = max(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["hit_rate"]
    )

    st.markdown(
        """
        <p style="
            font-size:0.95rem;
            font-weight:700;
            color:#0F172A;
            margin:0 0 0.6rem 0;
        ">
            Directional Hit Rate
            <span style="
                color:#64748B;
                font-size:0.78rem;
                font-weight:500;
            ">
                (Higher is Better)
            </span>
        </p>
        """,
        unsafe_allow_html=True
    )

    columns = st.columns(3)

    for column, (model_name, prediction) in zip(
        columns,
        prediction_metrics.items()
    ):
        style = card_styles[model_name]
        is_winner = model_name == winner

        winner_badge = (
            '<span style="background:#F1F5F9;color:#475569;'
            'padding:0.12rem 0.38rem;border-radius:999px;'
            'font-size:0.62rem;font-weight:800;margin-left:0.3rem;">'
            'BEST</span>'
            if is_winner
            else ""
        )

        card_html = (
            '<div style="'
            f'border:1px solid {style["border"]};'
            f'border-left:6px solid {MODEL_COLOURS[model_name]};'
            'border-radius:12px;'
            'padding:14px 13px;'
            f'background:{style["background"]};'
            'min-height:92px;'
            '">'
            '<p style="color:#64748B;font-size:0.68rem;'
            'font-weight:800;letter-spacing:0.06em;margin:0;'
            'white-space:nowrap;">'
            f'{model_name.upper()}'
            f'{winner_badge}'
            '</p>'
            '<p style="'
            f'color:{MODEL_COLOURS[model_name]};'
            'font-size:1.55rem;font-weight:800;margin:4px 0 0 0;">'
            f'{prediction["hit_rate"]:.2%}'
            '</p>'
            '</div>'
        )

        with column:
            st.markdown(card_html, unsafe_allow_html=True)

def create_ic_chart(
    dt_ic: pd.Series,
    lightgbm_ic: pd.Series,
    xgboost_ic: pd.Series
):
    ic_df = pd.concat(
        [
            dt_ic.rename("Decision Tree"),
            lightgbm_ic.rename("LightGBM"),
            xgboost_ic.rename("XGBoost")
        ],
        axis=1
    ).reset_index()

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=ic_df["Date"],
            y=ic_df["Decision Tree"],
            mode="lines",
            name="Decision Tree",
            line=dict(
                color=MODEL_COLOURS["Decision Tree"],
                width=2.5
            ),
            hovertemplate=(
                "<b>Decision Tree</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "IC: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ic_df["Date"],
            y=ic_df["LightGBM"],
            mode="lines",
            name="LightGBM",
            line=dict(
                color=MODEL_COLOURS["LightGBM"],
                width=2.5
            ),
            hovertemplate=(
                "<b>LightGBM</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "IC: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ic_df["Date"],
            y=ic_df["XGBoost"],
            mode="lines",
            name="XGBoost",
            line=dict(
                color=MODEL_COLOURS["XGBoost"],
                width=2.5
            ),
            hovertemplate=(
                "<b>XGBoost</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "IC: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )


    fig.update_layout(
        title="IC Through Time",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Weekly IC",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False
    )

    return fig


def create_equity_curve(
    dt_returns: pd.DataFrame,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    dt = dt_returns.copy()
    lgbm = lightgbm_returns.copy()
    xgb = xgboost_returns.copy()

    model_data = {
        "Decision Tree": dt,
        "LightGBM": lgbm,
        "XGBoost": xgb
    }

    for returns_df in model_data.values():
        returns_df["Date"] = pd.to_datetime(
            returns_df["Date"]
        )
        returns_df.sort_values(
            "Date",
            inplace=True
        )

        returns_df["Equity"] = (
            1 + returns_df["portfolio_return"]
        ).cumprod() - 1

    fig = go.Figure()

    for model_name, returns_df in model_data.items():
        fig.add_trace(
            go.Scatter(
                x=returns_df["Date"],
                y=returns_df["Equity"],
                mode="lines",
                name=model_name,
                line=dict(
                    color=MODEL_COLOURS[model_name],
                    width=2.8
                ),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Cumulative return: %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        line_color="#94A3B8"
    )

    fig.update_layout(
        title="Equity Curve (Cumulative Net Return)",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False
    )

    return fig


def create_drawdown_curve(
    dt_returns: pd.DataFrame,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    dt = dt_returns.copy()
    lgbm = lightgbm_returns.copy()
    xgb = xgboost_returns.copy()

    for df in (dt, lgbm, xgb):
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

    def calculate_drawdown(df: pd.DataFrame) -> pd.Series:
        equity = (1 + df["portfolio_return"]).cumprod()
        return equity / equity.cummax() - 1

    dt["Drawdown"] = calculate_drawdown(dt)
    lgbm["Drawdown"] = calculate_drawdown(lgbm)
    xgb["Drawdown"] = calculate_drawdown(xgb)

    fig = go.Figure()

    model_data = {
        "Decision Tree": dt,
        "LightGBM": lgbm,
        "XGBoost": xgb
    }

    for model_name, returns_df in model_data.items():
        fig.add_trace(
            go.Scatter(
                x=returns_df["Date"],
                y=returns_df["Drawdown"],
                mode="lines",
                name=model_name,
                line=dict(
                    color=MODEL_COLOURS[model_name],
                    width=2.8
                ),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Drawdown: %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        line_color="#94A3B8"
    )

    fig.update_layout(
        title="Drawdown Curve",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False
    )

    return fig

def create_rolling_sharpe_chart(
    dt_returns: pd.DataFrame,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame,
    window: int = 26
):
    dt = dt_returns.copy()
    lgbm = lightgbm_returns.copy()
    xgb = xgboost_returns.copy()

    def rolling_sharpe(returns):
        rolling_mean = returns.rolling(
            window=window,
            min_periods=window
        ).mean()

        rolling_std = returns.rolling(
            window=window,
            min_periods=window
        ).std(ddof=1)

        return (
            rolling_mean
            / rolling_std
            * np.sqrt(52)
        )

    dt["Rolling Sharpe"] = rolling_sharpe(
        dt["portfolio_return"]
    )

    lgbm["Rolling Sharpe"] = rolling_sharpe(
        lgbm["portfolio_return"]
    )

    xgb["Rolling Sharpe"] = rolling_sharpe(
        xgb["portfolio_return"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dt["Date"],
            y=dt["Rolling Sharpe"],
            mode="lines",
            name="Decision Tree",
            line=dict(
                color=MODEL_COLOURS["Decision Tree"],
                width=2.8
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lgbm["Date"],
            y=lgbm["Rolling Sharpe"],
            mode="lines",
            name="LightGBM",
            line=dict(
                color=MODEL_COLOURS["LightGBM"],
                width=2.8
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xgb["Date"],
            y=xgb["Rolling Sharpe"],
            mode="lines",
            name="XGBoost",
            line=dict(
                color=MODEL_COLOURS["XGBoost"],
                width=2.8
            )
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94A3B8"
    )

    fig.update_layout(
        title=f"{window}-Week Rolling Sharpe Ratio",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.2)"
    )

    return fig

def render_forecast_error_section(
    dt_prediction: dict,
    lightgbm_prediction: dict,
    xgboost_prediction: dict
) -> None:
    prediction_metrics = {
        "Decision Tree": dict(dt_prediction),
        "LightGBM": dict(lightgbm_prediction),
        "XGBoost": dict(xgboost_prediction)
    }

    model_styles = {
        "Decision Tree": {
            "background": "#FFFBEB",
            "header_background": "#FEF3C7",
            "border": "#FCD34D",
            "colour": "#D97706"
        },
        "LightGBM": {
            "background": "#F0FDF4",
            "header_background": "#DCFCE7",
            "border": "#A7F3D0",
            "colour": "#059669"
        },
        "XGBoost": {
            "background": "#EFF6FF",
            "header_background": "#DBEAFE",
            "border": "#BFDBFE",
            "colour": "#2563EB"
        }
    }

    # MSE may not be saved directly by GetPredictionMetrics.
    # It is exactly RMSE squared, so derive it when necessary.
    for metrics in prediction_metrics.values():
        metrics.setdefault("mse", metrics["rmse"] ** 2)

    mae_winner = min(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["mae"]
    )
    mse_winner = min(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["mse"]
    )
    rmse_winner = min(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["rmse"]
    )

    def badge(model: str) -> str:
        style = model_styles[model]

        return (
            f'<span style="background:{style["header_background"]};'
            f'color:{style["colour"]};padding:0.18rem 0.5rem;'
            'border-radius:999px;font-size:0.66rem;font-weight:750;'
            'white-space:nowrap;">'
            f'{model}'
            '</span>'
        )

    winners = {
        "MAE": mae_winner,
        "MSE": mse_winner,
        "RMSE": rmse_winner
    }

    unique_winners = set(winners.values())

    if len(unique_winners) == 1:
        overall_winner = next(iter(unique_winners))
        takeaway = (
            f"<strong>{overall_winner}</strong> records the lowest MAE, "
            "MSE and RMSE, indicating the strongest overall point-forecast "
            "accuracy."
        )
        takeaway_style = model_styles[overall_winner]
    else:
        takeaway = (
            f"<strong>MAE:</strong> {mae_winner}; "
            f"<strong>MSE:</strong> {mse_winner}; "
            f"<strong>RMSE:</strong> {rmse_winner}. "
            "MAE measures average absolute error, while MSE and RMSE place "
            "greater weight on larger forecast misses."
        )
        takeaway_style = model_styles[mae_winner]

    rows = [
        (
            "MAE",
            "mae",
            ".4f",
            mae_winner
        ),
        (
            "MSE",
            "mse",
            ".6f",
            mse_winner
        ),
        (
            "RMSE",
            "rmse",
            ".4f",
            rmse_winner
        )
    ]

    row_html = ""

    for label, key, number_format, winner in rows:
        dt_value = format(prediction_metrics["Decision Tree"][key], number_format)
        lgbm_value = format(prediction_metrics["LightGBM"][key], number_format)
        xgb_value = format(prediction_metrics["XGBoost"][key], number_format)

        row_html += (
            "<tr>"
            f'<td class="forecast-metric-name">{label}</td>'
            f"<td>{dt_value}</td>"
            f"<td>{lgbm_value}</td>"
            f"<td>{xgb_value}</td>"
            f"<td>{badge(winner)}</td>"
            "</tr>"
        )

    table_html = f"""
    <div class="forecast-error-card">
        <div class="forecast-error-header">
            Forecast Error
            <span>(Lower is Better)</span>
        </div>

        <table class="forecast-error-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>DT</th>
                    <th>LightGBM</th>
                    <th>XGBoost</th>
                    <th>Better</th>
                </tr>
            </thead>

            <tbody>
                {row_html}
            </tbody>
        </table>
    </div>
    """

    takeaway_html = f"""
    <div class="forecast-takeaway-card" style="
        --takeaway-border:{takeaway_style["border"]};
        --takeaway-background:{takeaway_style["background"]};
        --takeaway-header:{takeaway_style["header_background"]};
        --takeaway-colour:{takeaway_style["colour"]};
    ">
        <div class="forecast-takeaway-header">★ Key Takeaway</div>
        <div class="forecast-takeaway-text">{takeaway}</div>
    </div>
    """

    st.html(
        """
        <style>
        .forecast-error-card,
        .forecast-takeaway-card {
            height:225px;
            box-sizing:border-box;
            border-radius:14px;
            overflow:hidden;
        }

        .forecast-error-card {
            border:1px solid #E2E8F0;
            background:#FFFFFF;
        }

        .forecast-error-header,
        .forecast-takeaway-header {
            min-height:48px;
            display:flex;
            align-items:center;
            padding:0.72rem 0.85rem;
            box-sizing:border-box;
            font-size:0.88rem;
            font-weight:800;
        }

        .forecast-error-header {
            color:#0F172A;
            border-bottom:1px solid #E2E8F0;
        }

        .forecast-error-header span {
            color:#64748B;
            font-size:0.74rem;
            font-weight:500;
            margin-left:0.25rem;
        }

        .forecast-error-table {
            width:100%;
            border-collapse:collapse;
            table-layout:fixed;
            font-size:0.73rem;
        }

        .forecast-error-table th,
        .forecast-error-table td {
            padding:0.48rem 0.4rem;
            border-top:1px solid #E2E8F0;
            text-align:center;
            font-variant-numeric:tabular-nums;
            overflow-wrap:normal;
            word-break:normal;
        }

        .forecast-error-table th {
            color:#475569;
            background:#F8FAFC;
            font-weight:750;
            white-space:nowrap;
        }

        .forecast-error-table th:first-child,
        .forecast-error-table td:first-child {
            width:12%;
            text-align:left;
            padding-left:0.65rem;
        }

        .forecast-error-table th:nth-child(2) {
            width:13%;
        }

        .forecast-error-table th:nth-child(3) {
            width:19%;
        }

        .forecast-error-table th:nth-child(4) {
            width:17%;
        }

        .forecast-error-table th:last-child {
            width:29%;
        }

        .forecast-metric-name {
            color:#0F172A;
            font-weight:750;
        }

        .forecast-takeaway-card {
            border:1px solid var(--takeaway-border);
            background:var(--takeaway-background);
        }

        .forecast-takeaway-header {
            color:var(--takeaway-colour);
            background:var(--takeaway-header);
            border-bottom:1px solid var(--takeaway-border);
        }

        .forecast-takeaway-text {
            padding:0.95rem 1rem;
            color:#334155;
            font-size:0.80rem;
            line-height:1.58;
        }
        </style>
        """
    )

    table_col, takeaway_col = st.columns(2, gap="small")

    with table_col:
        st.html(table_html)

    with takeaway_col:
        st.html(takeaway_html)

def render_hypothesis_result(
    statistic_label: str,
    statistic_value: float,
    p_value: float,
    statement: str,
    alpha: float = 0.05
) -> None:
    """Render the test statistic, p-value and inference in one aligned card."""

    significant = p_value < alpha
    status_text = (
        "Statistically Significant"
        if significant
        else "Not Significant"
    )
    status_colour = "#DC2626" if significant else "#2563EB"
    status_background = "#FEF2F2" if significant else "#EFF6FF"
    status_border = "#FECACA" if significant else "#BFDBFE"

    st.html(
        f"""
        <style>
        .hypothesis-result-card {{
            display:grid;
            grid-template-columns:minmax(120px,0.55fr) 1px
                                  minmax(120px,0.55fr) 1px
                                  minmax(260px,1.7fr);
            align-items:stretch;
            gap:1rem;
            background:#FFFFFF;
            border:1px solid #E2E8F0;
            border-radius:14px;
            padding:0.95rem 1rem;
            margin:-0.45rem 0 1.2rem 0;
            box-shadow:0 5px 16px rgba(15,23,42,0.04);
        }}

        .hypothesis-result-metric {{
            display:flex;
            flex-direction:column;
            justify-content:center;
        }}

        .hypothesis-result-label {{
            color:#64748B;
            font-size:0.72rem;
            font-weight:800;
            letter-spacing:0.04em;
            text-transform:uppercase;
        }}

        .hypothesis-result-value {{
            color:#0F172A;
            font-size:1.35rem;
            font-weight:850;
            margin-top:0.22rem;
            font-variant-numeric:tabular-nums;
        }}

        .hypothesis-result-divider {{
            width:1px;
            background:#E2E8F0;
        }}

        .hypothesis-result-conclusion {{
            display:flex;
            flex-direction:column;
            justify-content:center;
        }}

        .hypothesis-result-statement {{
            color:#475569;
            font-size:0.80rem;
            line-height:1.5;
        }}

        @media (max-width:800px) {{
            .hypothesis-result-card {{
                grid-template-columns:1fr 1fr;
            }}

            .hypothesis-result-divider {{
                display:none;
            }}

            .hypothesis-result-conclusion {{
                grid-column:1 / -1;
            }}
        }}
        </style>

        <div class="hypothesis-result-card">
            <div class="hypothesis-result-metric">
                <div class="hypothesis-result-label">{statistic_label}</div>
                <div class="hypothesis-result-value">{statistic_value:.4f}</div>
            </div>

            <div class="hypothesis-result-divider"></div>

            <div class="hypothesis-result-metric">
                <div class="hypothesis-result-label">p-value</div>
                <div class="hypothesis-result-value">{p_value:.4f}</div>
            </div>

            <div class="hypothesis-result-divider"></div>

            <div class="hypothesis-result-conclusion">
                <span style="
                    display:inline-flex;
                    width:max-content;
                    align-items:center;
                    background:{status_background};
                    border:1px solid {status_border};
                    color:{status_colour};
                    border-radius:999px;
                    padding:0.24rem 0.62rem;
                    font-size:0.70rem;
                    font-weight:800;
                    margin-bottom:0.38rem;
                ">
                    {status_text}
                </span>
                <div class="hypothesis-result-statement">{statement}</div>
            </div>
        </div>
        """
    )

def render_hypothesis_card(
    number: int,
    question: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    test_name: str,
    accent_colour: str = "#2563EB",
    background_colour: str = "#EFF6FF"
) -> None:
    """Render one compact, aligned hypothesis definition."""

    st.html(
        f"""
        <div style="
            background:{background_colour};
            border:1px solid #E2E8F0;
            border-left:5px solid {accent_colour};
            border-radius:14px;
            padding:0.85rem 1rem;
            margin-top:0.8rem;
            box-shadow:0 4px 14px rgba(15,23,42,0.04);
        ">
            <div style="
                color:{accent_colour};
                font-size:0.67rem;
                font-weight:850;
                letter-spacing:0.08em;
                margin-bottom:0.22rem;
            ">
                TEST {number}
            </div>

            <div style="
                color:#0F172A;
                font-size:0.96rem;
                font-weight:800;
                line-height:1.4;
            ">
                {question}
            </div>

            <div style="
                color:#64748B;
                font-size:0.74rem;
                line-height:1.45;
                margin-top:0.28rem;
            ">
                {test_name}
            </div>
        </div>
        """
    )

    null_col, alternative_col = st.columns(2, gap="large")

    with null_col:
        st.caption("NULL HYPOTHESIS")
        st.latex(null_hypothesis)

    with alternative_col:
        st.caption("ALTERNATIVE HYPOTHESIS")
        st.latex(alternative_hypothesis)


MODEL_DISPLAY_NAMES = {
    "dt": "Decision Tree",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost"
}

FEATURE_DISPLAY_NAMES = {
    "stock": "Stock Features",
    "market": "Stock + Market"
}


def load_model_comparison_results() -> dict:
    """Load every model-feature backtest and calculate its metrics."""

    model_paths = {
        "dt": BACKTEST_RESULTS_DT_DIR,
        "lightgbm": BACKTEST_RESULTS_LIGHTGBM_DIR,
        "xgboost": BACKTEST_RESULTS_XGBOOST_DIR
    }

    feature_sets = {
        "stock": "final_portfolio_stock.parquet",
        "market": "final_portfolio_market.parquet"
    }

    results = {}

    for model_key, model_dir in model_paths.items():
        results[model_key] = {}

        for feature_key, filename in feature_sets.items():
            portfolio_path = model_dir / filename

            if not portfolio_path.exists():
                raise FileNotFoundError(
                    f"Backtest result not found: {portfolio_path}"
                )

            portfolio = pd.read_parquet(portfolio_path)

            portfolio_metrics, returns = GetMetrics(
                portfolio
            ).run_data()

            prediction_metrics, ic, hit = GetPredictionMetrics(
                portfolio
            ).run_data()

            results[model_key][feature_key] = {
                "portfolio_data": portfolio,
                "metrics": {
                    "prediction": prediction_metrics,
                    "portfolio": portfolio_metrics
                },
                "returns": returns,
                "ic": ic,
                "hit": hit
            }

    return results


def select_overall_configuration(results: dict) -> dict:
    """
    Select the best model-feature configuration.

    Sharpe ratio is the primary model-selection criterion. Sortino ratio,
    Calmar ratio, maximum drawdown and annual return are used as tie-breakers.
    """

    candidates = []

    for model_key, feature_results in results.items():
        for feature_key, result in feature_results.items():
            portfolio = result["metrics"]["portfolio"]
            prediction = result["metrics"]["prediction"]

            candidates.append({
                "model_key": model_key,
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "feature_key": feature_key,
                "feature_name": FEATURE_DISPLAY_NAMES[feature_key],
                "sharpe_ratio": portfolio["sharpe_ratio"],
                "sortino_ratio": portfolio["sortino_ratio"],
                "calmar_ratio": portfolio["calmar_ratio"],
                "max_drawdown": portfolio["max_drawdown"],
                "annual_return": portfolio["annual_return"],
                "annual_volatility": portfolio["annual_volatility"],
                "win_rate": portfolio["win_rate"],
                "mae": prediction["mae"],
                "mse": prediction.get(
                    "mse",
                    prediction["rmse"] ** 2
                ),
                "rmse": prediction["rmse"],
                "mean_ic": prediction["mean_ic"],
                "annualised_icir": prediction["annualised_icir"]
            })

    ranking = (
        pd.DataFrame(candidates)
        .sort_values(
            by=[
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "max_drawdown",
                "annual_return"
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return {
        "winner": ranking.iloc[0].to_dict(),
        "ranking": ranking
    }


def select_prediction_winner(results: dict) -> dict:
    """Select the configuration with the lowest RMSE, then lowest MAE."""

    candidates = []

    for model_key, feature_results in results.items():
        for feature_key, result in feature_results.items():
            prediction = result["metrics"]["prediction"]

            candidates.append({
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "feature_name": FEATURE_DISPLAY_NAMES[feature_key],
                "mae": prediction["mae"],
                "mse": prediction.get(
                    "mse",
                    prediction["rmse"] ** 2
                ),
                "rmse": prediction["rmse"]
            })

    ranking = (
        pd.DataFrame(candidates)
        .sort_values(
            by=["rmse", "mae"],
            ascending=True
        )
        .reset_index(drop=True)
    )

    return ranking.iloc[0].to_dict()



# ============================================================
# Data-driven executive-summary utilities
# ============================================================

SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash")


def _finite_float(value, default: float = 0.0) -> float:
    """Convert NumPy/pandas values into JSON-safe Python floats."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    return numeric if np.isfinite(numeric) else default


def build_summary_payload(
    selection: dict,
    hypothesis_p_values: dict,
    alpha: float = 0.05
) -> dict:
    """
    Build the factual payload used by both the deterministic fallback and GenAI.

    Python remains responsible for:
    - selecting the winning configuration;
    - identifying each metric winner;
    - deciding statistical significance.

    GenAI is only allowed to explain these pre-computed facts.
    """
    ranking = selection["ranking"].copy()
    winner = selection["winner"].copy()

    numeric_columns = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "annual_return",
        "annual_volatility",
        "win_rate",
        "mae",
        "mse",
        "rmse",
        "mean_ic",
        "annualised_icir"
    ]

    for column in numeric_columns:
        ranking[column] = pd.to_numeric(
            ranking[column],
            errors="coerce"
        )

    def highest(metric: str) -> dict:
        row = ranking.loc[ranking[metric].idxmax()]
        return {
            "model": str(row["model_name"]),
            "feature_set": str(row["feature_name"]),
            "value": _finite_float(row[metric])
        }

    def lowest(metric: str) -> dict:
        row = ranking.loc[ranking[metric].idxmin()]
        return {
            "model": str(row["model_name"]),
            "feature_set": str(row["feature_name"]),
            "value": _finite_float(row[metric])
        }

    # Drawdowns are normally negative. The highest value is the least negative
    # and therefore represents the smallest drawdown.
    metric_winners = {
        "annual_return": highest("annual_return"),
        "sharpe_ratio": highest("sharpe_ratio"),
        "sortino_ratio": highest("sortino_ratio"),
        "calmar_ratio": highest("calmar_ratio"),
        "smallest_max_drawdown": highest("max_drawdown"),
        "lowest_volatility": lowest("annual_volatility"),
        "weekly_win_rate": highest("win_rate"),
        "lowest_mae": lowest("mae"),
        "lowest_rmse": lowest("rmse"),
        "mean_ic": highest("mean_ic"),
        "annualised_icir": highest("annualised_icir")
    }

    tests = {}

    for test_name, p_value in hypothesis_p_values.items():
        clean_p_value = _finite_float(p_value, default=1.0)
        tests[test_name] = {
            "p_value": clean_p_value,
            "significant": clean_p_value < alpha
        }

    return {
        "selection_rule": (
            "Highest Sharpe ratio, followed by Sortino ratio, Calmar ratio, "
            "maximum drawdown and annual return as deterministic tie-breakers."
        ),
        "selected_configuration": {
            "model": str(winner["model_name"]),
            "feature_set": str(winner["feature_name"]),
            "annual_return": _finite_float(winner["annual_return"]),
            "sharpe_ratio": _finite_float(winner["sharpe_ratio"]),
            "sortino_ratio": _finite_float(winner["sortino_ratio"]),
            "calmar_ratio": _finite_float(winner["calmar_ratio"]),
            "max_drawdown": _finite_float(winner["max_drawdown"]),
            "annual_volatility": _finite_float(
                winner["annual_volatility"]
            ),
            "win_rate": _finite_float(winner["win_rate"]),
            "mae": _finite_float(winner["mae"]),
            "rmse": _finite_float(winner["rmse"]),
            "mean_ic": _finite_float(winner["mean_ic"]),
            "annualised_icir": _finite_float(
                winner["annualised_icir"]
            )
        },
        "metric_winners": metric_winners,
        "hypothesis_tests": tests,
        "alpha": alpha
    }


def build_deterministic_summary(payload: dict) -> dict:
    """Create a fully factual summary without calling an external model."""
    selected = payload["selected_configuration"]
    winners = payload["metric_winners"]
    tests = payload["hypothesis_tests"]

    selected_model = selected["model"]
    selected_features = selected["feature_set"]

    return_winner = winners["annual_return"]
    sharpe_winner = winners["sharpe_ratio"]
    ranking_winner = winners["mean_ic"]
    icir_winner = winners["annualised_icir"]
    rmse_winner = winners["lowest_rmse"]
    drawdown_winner = winners["smallest_max_drawdown"]

    all_not_significant = (
        bool(tests)
        and not any(test["significant"] for test in tests.values())
    )

    if all_not_significant:
        significance_sentence = (
            "None of the reported hypothesis tests was statistically "
            "significant at the 5% level, so the observed differences should "
            "not be treated as proof of model superiority."
        )
    else:
        significant_names = [
            name.replace("_", " ")
            for name, result in tests.items()
            if result["significant"]
        ]

        significance_sentence = (
            "Statistically significant evidence was detected for "
            + ", ".join(significant_names)
            + "; the remaining comparisons were not significant at the 5% "
              "level."
            if significant_names
            else (
                "The available hypothesis-test results do not establish "
                "statistically significant model superiority."
            )
        )

    findings = [
        (
            f"<strong>{return_winner['model']}</strong> using "
            f"<strong>{return_winner['feature_set']}</strong> achieved the "
            f"highest annual return at "
            f"<strong>{return_winner['value']:.1%}</strong>."
        ),
        (
            f"<strong>{sharpe_winner['model']}</strong> using "
            f"<strong>{sharpe_winner['feature_set']}</strong> recorded the "
            f"highest Sharpe ratio at "
            f"<strong>{sharpe_winner['value']:.2f}</strong>."
        ),
        (
            f"<strong>{ranking_winner['model']}</strong> produced the highest "
            f"mean IC ({ranking_winner['value']:.4f}), while "
            f"<strong>{icir_winner['model']}</strong> produced the highest "
            f"annualised ICIR ({icir_winner['value']:.4f})."
        ),
        significance_sentence
    ]

    reasons = [
        {
            "label": "Selection criterion",
            "text": (
                f"Ranked first under the pre-defined Sharpe-led selection "
                f"rule with a Sharpe ratio of "
                f"{selected['sharpe_ratio']:.2f}."
            )
        },
        {
            "label": "Realised portfolio",
            "text": (
                f"Generated an annual return of "
                f"{selected['annual_return']:.1%}, a Sortino ratio of "
                f"{selected['sortino_ratio']:.2f} and a Calmar ratio of "
                f"{selected['calmar_ratio']:.2f}."
            )
        },
        {
            "label": "Risk profile",
            "text": (
                f"Recorded a maximum drawdown of "
                f"{selected['max_drawdown']:.1%} and annual volatility of "
                f"{selected['annual_volatility']:.1%}."
            )
        }
    ]

    note = (
        f"The selected configuration is <strong>{selected_model}</strong> "
        f"with <strong>{selected_features}</strong>. "
        f"{rmse_winner['model']} achieved the lowest RMSE, while "
        f"{drawdown_winner['model']} recorded the smallest maximum drawdown. "
        "The final choice follows the stated portfolio-selection rule rather "
        "than allowing forecast error or a language model to override it."
    )

    return {
        "overall_results": findings,
        "selection_reasons": reasons,
        "selection_note": note
    }


def _normalise_genai_summary(
    raw_summary: dict,
    fallback: dict
) -> dict:
    """Validate and sanitise GenAI output before placing it in HTML."""
    if not isinstance(raw_summary, dict):
        return fallback

    findings = raw_summary.get("overall_results")
    reasons = raw_summary.get("selection_reasons")
    note = raw_summary.get("selection_note")

    if (
        not isinstance(findings, list)
        or not 3 <= len(findings) <= 4
        or not all(isinstance(item, str) for item in findings)
    ):
        return fallback

    if (
        not isinstance(reasons, list)
        or len(reasons) != 3
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("text"), str)
            for item in reasons
        )
    ):
        return fallback

    if not isinstance(note, str):
        return fallback

    # GenAI output is plain text. Escape it so it cannot inject HTML.
    return {
        "overall_results": [
            html.escape(item.strip())
            for item in findings
        ],
        "selection_reasons": [
            {
                "label": html.escape(item["label"].strip()),
                "text": html.escape(item["text"].strip())
            }
            for item in reasons
        ],
        "selection_note": html.escape(note.strip())
    }


@st.cache_data(show_spinner=False, ttl=3600)
def generate_genai_summary(payload_json: str) -> dict:
    """
    Ask Gemini to narrate pre-computed results.

    Failure is non-fatal: callers should fall back to the deterministic
    summary whenever an API key, network connection or valid response is
    unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    payload = json.loads(payload_json)
    fallback = build_deterministic_summary(payload)

    prompt = f"""
You are writing a concise executive summary for a quantitative model-
comparison dashboard.

The Python application has already selected the winning configuration.
You must NOT select a different model, alter the selection rule, infer
unreported results or claim statistical significance when p >= alpha.

Use only the JSON facts below:
{json.dumps(payload, indent=2)}

Return valid JSON only with exactly this structure:
{{
  "overall_results": [
    "Three or four concise factual findings.",
    "Each finding should mention the relevant model and metric.",
    "Distinguish prediction quality from realised portfolio performance."
  ],
  "selection_reasons": [
    {{"label": "Short label", "text": "Reason one"}},
    {{"label": "Short label", "text": "Reason two"}},
    {{"label": "Short label", "text": "Reason three"}}
  ],
  "selection_note": "One concise trade-off and statistical caveat."
}}

Rules:
- Australian English.
- No markdown, HTML, headings or bullet symbols.
- The selected model and feature set must exactly match
  selected_configuration.
- Do not describe one model as globally best merely because it wins one metric.
- Do not say results are statistically significant unless the supplied test
  says significant=true.
- Keep each sentence concise.
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=SUMMARY_MODEL,
        contents=prompt
    )

    response_text = (response.text or "").strip()
    response_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    parsed = json.loads(response_text)
    return _normalise_genai_summary(parsed, fallback)


def get_executive_summary(
    selection: dict,
    hypothesis_p_values: dict,
    alpha: float = 0.05
) -> tuple[dict, bool]:
    """
    Return the GenAI narrative when available, otherwise a deterministic one.

    The boolean indicates whether GenAI was successfully used.
    """
    payload = build_summary_payload(
        selection=selection,
        hypothesis_p_values=hypothesis_p_values,
        alpha=alpha
    )

    fallback = build_deterministic_summary(payload)
    payload_json = json.dumps(payload, sort_keys=True)

    try:
        return generate_genai_summary(payload_json), True
    except Exception:
        return fallback, False


def render_final_model_summary(
    selection: dict,
    hypothesis_p_values: dict,
    model_colour_map: dict,
    alpha: float = 0.05
) -> None:
    """Render the factual executive summary and deterministic selection."""
    selected = selection["winner"]
    selected_model = selected["model_name"]
    selected_features = selected["feature_name"]
    selected_colours = model_colour_map[selected_model]

    summary, used_genai = get_executive_summary(
        selection=selection,
        hypothesis_p_values=hypothesis_p_values,
        alpha=alpha
    )

    findings_html = "".join(
        f"<li>{finding}</li>"
        for finding in summary["overall_results"]
    )

    reasons_html = "".join(
        (
            '<div class="selection-reason">'
            '<span class="selection-check">✓</span>'
            f'<span><strong>{reason["label"]}:</strong> '
            f'{reason["text"]}</span>'
            '</div>'
        )
        for reason in summary["selection_reasons"]
    )

    source_label = (
        "AI-assisted narrative; model selection calculated in Python."
        if used_genai
        else "Deterministic narrative; model selection calculated in Python."
    )

    st.html(
        f"""
        <style>
        .conclusion-grid {{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:0.9rem;
            margin:1rem 0 1.4rem 0;
        }}

        .conclusion-card {{
            border:1px solid #E2E8F0;
            border-radius:10px;
            padding:1rem 1.15rem;
            min-height:270px;
            box-shadow:0 2px 7px rgba(15,23,42,0.04);
        }}

        .findings-card {{
            border-left:5px solid #2563EB;
            background:#F8FAFC;
        }}

        .selection-card {{
            border-left:5px solid {selected_colours["accent"]};
            background:{selected_colours["background"]};
        }}

        .conclusion-title {{
            display:flex;
            align-items:center;
            gap:0.45rem;
            margin-bottom:0.8rem;
            font-size:1rem;
            font-weight:800;
        }}

        .findings-title {{
            color:#1D4ED8;
        }}

        .selection-title {{
            color:{selected_colours["dark"]};
        }}

        .conclusion-list {{
            margin:0;
            padding-left:1.15rem;
            color:#334155;
            font-size:0.9rem;
            line-height:1.58;
        }}

        .conclusion-list li {{
            margin-bottom:0.58rem;
        }}

        .selected-model-label {{
            color:#64748B;
            font-size:0.78rem;
            font-weight:700;
            letter-spacing:0.05em;
            text-transform:uppercase;
            margin-bottom:0.2rem;
        }}

        .selected-model-name {{
            color:{selected_colours["accent"]};
            font-size:1.35rem;
            font-weight:800;
            margin-bottom:0.15rem;
        }}

        .selected-feature-name {{
            color:#64748B;
            font-size:0.8rem;
            font-weight:700;
            margin-bottom:0.8rem;
        }}

        .selection-reason {{
            display:flex;
            align-items:flex-start;
            gap:0.5rem;
            margin-bottom:0.58rem;
            color:#334155;
            font-size:0.9rem;
            line-height:1.48;
        }}

        .selection-check {{
            color:{selected_colours["accent"]};
            font-weight:900;
            line-height:1.45;
        }}

        .selection-note {{
            border-top:1px solid {selected_colours["border"]};
            margin-top:0.8rem;
            padding-top:0.7rem;
            color:#64748B;
            font-size:0.84rem;
            line-height:1.48;
        }}

        .summary-source {{
            margin-top:0.65rem;
            color:#94A3B8;
            font-size:0.68rem;
            font-weight:650;
        }}

        @media (max-width:900px) {{
            .conclusion-grid {{
                grid-template-columns:1fr;
            }}

            .conclusion-card {{
                min-height:auto;
            }}
        }}
        </style>

        <div class="conclusion-grid">
            <div class="conclusion-card findings-card">
                <div class="conclusion-title findings-title">
                    <span>◆</span>
                    <span>Overall Results</span>
                </div>

                <ul class="conclusion-list">
                    {findings_html}
                </ul>

                <div class="summary-source">{source_label}</div>
            </div>

            <div class="conclusion-card selection-card">
                <div class="conclusion-title selection-title">
                    <span>★</span>
                    <span>Final Model Selection</span>
                </div>

                <div class="selected-model-label">Selected configuration</div>
                <div class="selected-model-name">{selected_model}</div>
                <div class="selected-feature-name">{selected_features}</div>

                {reasons_html}

                <div class="selection-note">
                    {summary["selection_note"]}
                </div>
            </div>
        </div>
        """
    )



# ============================================================
# Main Streamlit page
# ============================================================

def render_model_comparison():
    st.set_page_config(
        page_title="ASX Alpha System - Model Comparison",
        page_icon="🧩",
        layout="wide"
    )

    results = load_model_comparison_results()

    selection = select_overall_configuration(results)
    overall_result = selection["winner"]
    prediction_winner = select_prediction_winner(results)

    overall_model = overall_result["model_name"]
    overall_feature_set = overall_result["feature_name"]

    latest_rebalance_date = max(
        pd.to_datetime(
            result["portfolio_data"]["Date"],
            errors="coerce"
        ).max()
        for feature_results in results.values()
        for result in feature_results.values()
    )

    if pd.isna(latest_rebalance_date):
        raise ValueError(
            "The model-comparison results do not contain a valid "
            "rebalance date."
        )

    st.html(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .model-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 88% 18%, rgba(124,58,237,0.16), transparent 28%),
                radial-gradient(circle at 72% 108%, rgba(37,99,235,0.14), transparent 36%),
                linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 48%, #F5F3FF 100%);
            border: 1px solid #DCE7F5;
            border-radius: 22px;
            padding: 1.75rem 1.9rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 34px rgba(37,99,235,0.08);
        }


        .model-hero-title {
            margin:0;
            color:#0F172A;
            font-size:2.25rem;
            font-weight:850;
            line-height:1.08;
        }

        .model-hero-description {
            margin-top:0.65rem;
            max-width:940px;
            color:#52647A;
            font-size:0.96rem;
            line-height:1.62;
        }

        .latest-rebalance {
            display:inline-flex;
            align-items:center;
            margin:0 0 0.25rem 0;
            padding:0.38rem 0.68rem;
            border-radius:999px;
            background:rgba(255,255,255,0.78);
            border:1px solid #DCE7F5;
            color:#64748B;
            font-size:0.73rem;
            font-weight:700;
        }


        .model-info-banner {
            display:flex;
            align-items:flex-start;
            gap:0.7rem;
            background:#F8FAFC;
            border:1px solid #CBD5E1;
            border-radius:14px;
            padding:0.9rem 1rem;
            color:#475569;
            font-size:0.84rem;
            line-height:1.5;
            margin-bottom:1rem;
        }

        .model-info-icon {
            display:flex;
            align-items:center;
            justify-content:center;
            flex:0 0 auto;
            width:1.8rem;
            height:1.8rem;
            border-radius:999px;
            background:#DBEAFE;
            color:#1D4ED8;
            font-weight:850;
        }

        
        .comparison-section-header {
            display:flex;
            align-items:flex-start;
            gap:0.7rem;
            margin-top:1.75rem;
            margin-bottom:0.8rem;
        }

        .comparison-section-icon {
            display:flex;
            align-items:center;
            justify-content:center;
            width:2.15rem;
            height:2.15rem;
            flex:0 0 auto;
            border-radius:12px;
            background:#EFF6FF;
            color:#2563EB;
            font-size:1rem;
            font-weight:850;
        }

        .comparison-section-title {
            color:#0F172A;
            font-size:1.25rem;
            font-weight:850;
            line-height:1.2;
        }

        .comparison-section-description {
            color:#64748B;
            font-size:0.83rem;
            line-height:1.45;
            margin-top:0.22rem;
        }

        .feature-set-banner {
            display:flex;
            align-items:center;
            gap:0.8rem;
            border-radius:17px;
            padding:1rem 1.15rem;
            margin:1rem 0 0.4rem 0;
            box-shadow:0 6px 18px rgba(15,23,42,0.04);
        }

        .feature-stock-banner {
            background:linear-gradient(135deg,#EFF6FF,#F8FAFC);
            border:1px solid #BFDBFE;
            border-left:5px solid #2563EB;
        }

        .feature-market-banner {
            background:linear-gradient(135deg,#FEF2F2,#FFF7ED);
            border:1px solid #FECACA;
            border-left:5px solid #EF4444;
        }

        .feature-set-icon {
            display:flex;
            align-items:center;
            justify-content:center;
            width:2.35rem;
            height:2.35rem;
            border-radius:999px;
            background:rgba(255,255,255,0.75);
            font-size:1.05rem;
        }

        .feature-set-title {
            color:#0F172A;
            font-size:1.02rem;
            font-weight:850;
        }

        .feature-set-description {
            color:#64748B;
            font-size:0.80rem;
            line-height:1.45;
            margin-top:0.2rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius:18px;
            box-shadow:0 7px 22px rgba(15,23,42,0.045);
        }

        div[data-testid="stPlotlyChart"] {
            border-radius:14px;
        }

        div[data-testid="stTabs"] {
            margin-top:1rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap:0.4rem;
            border-bottom:1px solid #E2E8F0;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            border:1px solid #E2E8F0;
            border-bottom:none;
            border-radius:12px 12px 0 0;
            padding:0.65rem 1rem;
            background:#F8FAFC;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background:#EFF6FF;
            border-color:#BFDBFE;
            color:#1D4ED8;
        }


        .model-candidate-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:0.9rem;
            margin:0.9rem 0 1.35rem 0;
        }

        .model-candidate-card {
            --candidate-accent:#2563EB;
            --candidate-soft:#DBEAFE;
            position:relative;
            overflow:hidden;
            min-height:165px;
            height:165px;
            box-sizing:border-box;
            border:1px solid color-mix(
                in srgb,
                var(--candidate-accent) 24%,
                #E2E8F0
            );
            border-radius:17px;
            background:linear-gradient(
                145deg,
                #FFFFFF 0%,
                var(--candidate-soft) 165%
            );
            padding:1rem 1.05rem;
            box-shadow:0 7px 22px rgba(15,23,42,0.045);
        }

        .model-candidate-card::before {
            content:"";
            position:absolute;
            inset:0 0 auto 0;
            height:5px;
            background:var(--candidate-accent);
        }

        .candidate-dt {
            --candidate-accent:#F59E0B;
            --candidate-soft:#FEF3C7;
        }

        .candidate-lgbm {
            --candidate-accent:#10B981;
            --candidate-soft:#D1FAE5;
        }

        .candidate-xgb {
            --candidate-accent:#2563EB;
            --candidate-soft:#DBEAFE;
        }

        .candidate-name {
            color:var(--candidate-accent);
            font-size:0.98rem;
            font-weight:850;
            margin-bottom:0.45rem;
        }

        .candidate-description {
            color:#475569;
            font-size:0.80rem;
            line-height:1.5;
        }

        div[data-testid="stTabs"] button {
            font-weight:750;
        }

        @media (max-width:900px) {
            .model-candidate-grid {
                grid-template-columns:1fr;
            }

            .model-hero-title {
                font-size:1.8rem;
            }
        }
        </style>

        <div class="model-hero">
            <h1 class="model-hero-title">Model Comparison</h1>

            <div class="model-hero-description">
                Compare Decision Tree, LightGBM and XGBoost across stock-only
                and stock-plus-market feature sets using identical walk-forward
                validation, portfolio construction and transaction-cost rules.
            </div>

        </div>

        <div class="latest-rebalance">
            Latest rebalance: __LATEST_REBALANCE_DATE__
        </div>

        <div class="comparison-section-header">
            <div class="comparison-section-icon">&#9733;</div>
            <div>
                <div class="comparison-section-title">
                    Executive Summary
                </div>
                <div class="comparison-section-description">
                    Automated recommendation based on out-of-sample
                    portfolio performance.
                </div>
            </div>
        </div>

        <div class="model-info-banner">
            <div class="model-info-icon">i</div>
            <div>
                All models were evaluated using identical expanding-window
                splits, prediction horizons, transaction-cost assumptions and
                portfolio construction rules.
            </div>
        </div>
        """.replace(
            "__LATEST_REBALANCE_DATE__",
            latest_rebalance_date.strftime("%d %B %Y")
        )
    )

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
            "Automated selection",
            (
                f"{overall_feature_set} produced the highest-ranked "
                "model-feature configuration under the portfolio-selection "
                "framework."
            )
        ),
        (
            "Risk-adjusted objective",
            (
                "Feature sets were compared using realised portfolio outcomes, "
                "with Sharpe ratio used as the primary selection criterion."
            )
        ),
        (
            "Consistent experiment",
            (
                "Every configuration used identical walk-forward splits, "
                "prediction horizons, transaction costs and portfolio rules."
            )
        )
    ]

    model_reasons = [
        (
            "Risk-adjusted performance",
            (
                f"Achieved the selected highest Sharpe ratio of "
                f"{overall_result['sharpe_ratio']:.2f}, alongside a Sortino "
                f"ratio of {overall_result['sortino_ratio']:.2f}."
            )
        ),
        (
            "Downside control",
            (
                f"Recorded a maximum drawdown of "
                f"{overall_result['max_drawdown']:.1%} and a Calmar ratio of "
                f"{overall_result['calmar_ratio']:.2f}."
            )
        ),
        (
            "Realised portfolio outcome",
            (
                f"Generated an annual return of "
                f"{overall_result['annual_return']:.1%} with annual volatility "
                f"of {overall_result['annual_volatility']:.1%}."
            )
        )
    ]

    same_prediction_and_portfolio_winner = (
        prediction_winner["model_name"] == overall_model
        and prediction_winner["feature_name"] == overall_feature_set
    )

    if same_prediction_and_portfolio_winner:
        prediction_takeaway = (
            f"{overall_model} also achieved the lowest forecast error. "
            "For this experiment, prediction accuracy and realised portfolio "
            "performance were aligned."
        )
    else:
        prediction_takeaway = (
            f"{prediction_winner['model_name']} using "
            f"{prediction_winner['feature_name']} achieved the lowest RMSE, "
            f"while {overall_model} using {overall_feature_set} generated the "
            "strongest risk-adjusted portfolio. Stronger point-forecast "
            "accuracy therefore did not translate directly into superior "
            "investment performance."
        )


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

            f"""
            <div class="overall-recommendation-card">
                <div class="overall-recommendation-title">
                    Overall recommended configuration
                </div>

                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:0.45rem;
                    margin:0.55rem 0 0.75rem 0;
                ">
                    <span style="
                        background:{model_colours["background"]};
                        border:1px solid {model_colours["border"]};
                        color:{model_colours["dark"]};
                        border-radius:999px;
                        padding:0.3rem 0.65rem;
                        font-size:0.74rem;
                        font-weight:800;
                    ">
                        {overall_model}
                    </span>

                    <span style="
                        background:{feature_colours["background"]};
                        border:1px solid {feature_colours["border"]};
                        color:{feature_colours["dark"]};
                        border-radius:999px;
                        padding:0.3rem 0.65rem;
                        font-size:0.74rem;
                        font-weight:800;
                    ">
                        {overall_feature_set}
                    </span>

                    <span style="
                        background:#EFF6FF;
                        border:1px solid #BFDBFE;
                        color:#1D4ED8;
                        border-radius:999px;
                        padding:0.3rem 0.65rem;
                        font-size:0.74rem;
                        font-weight:800;
                    ">
                        Sharpe {overall_result["sharpe_ratio"]:.2f}
                    </span>
                </div>

                <div>
                    <strong>{overall_model} using {overall_feature_set}</strong>
                    was selected automatically because it produced the strongest
                    risk-adjusted portfolio. Selection prioritises Sharpe ratio,
                    supported by Sortino ratio, drawdown, Calmar ratio and annual
                    return.
                </div>

                <div style="
                    margin-top:0.75rem;
                    padding-top:0.75rem;
                    border-top:1px solid #CBD5E1;
                ">
                    <strong>Key takeaway:</strong>
                    {prediction_takeaway}
                </div>
            </div>
            """

        '</div>'
    )

    st.html(overall_summary_html)

    st.html(
        """
        <div class="model-candidate-grid">
            <div class="model-candidate-card candidate-dt">
                <div class="candidate-name">Decision Tree</div>
                <div class="candidate-description">
                    Transparent nonlinear baseline with straightforward
                    interpretation and low model complexity.
                </div>
            </div>

            <div class="model-candidate-card candidate-lgbm">
                <div class="candidate-name">LightGBM</div>
                <div class="candidate-description">
                    Leaf-wise gradient boosting designed to capture nonlinear
                    feature interactions efficiently.
                </div>
            </div>

            <div class="model-candidate-card candidate-xgb">
                <div class="candidate-name">XGBoost</div>
                <div class="candidate-description">
                    Regularised gradient boosting with robust optimisation and
                    strong cross-sectional ranking capacity.
                </div>
            </div>
        </div>
        """
    )

    stock_tab, market_tab = st.tabs(
        [
            "📈 Stock Features",
            "🌐 Stock + Market"
        ]
    )

    hit_contingency_tables = {}

    for feature_name in results["dt"].keys():
        decision_tree_hit = results["dt"][feature_name]["hit"]
        lightgbm_hit = results["lightgbm"][feature_name]["hit"]

        hit_contingency_tables[feature_name] = (
            get_hit_contingency_table(
                decision_tree_hit,
                lightgbm_hit
            ).to_numpy()
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
        dt_ic=results["dt"]["market"]["ic"],
        xgboost_ic=results["xgboost"]["market"]["ic"],
        lgbm_ic=results["lightgbm"]["market"]["ic"], 
        hit_contingency_table=hit_contingency_tables["market"],
        dt_returns=results["dt"]["market"]["returns"]["portfolio_return"],
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
        
        
        render_section_header(
            icon="⚖",
            title="Statistical Hypothesis Tests",
            description=(
                "Test whether LightGBM significantly improves upon the Decision Tree "
                "baseline across ranking quality, directional accuracy, "
                "weekly returns and Sharpe ratio."
            )
        )

        
        render_hypothesis_card(
            number=1,
            question="Does LightGBM produce a higher mean weekly IC than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ \mu_{IC,\mathrm{LGBM}}"
                r"\leq"
                r"\mu_{IC,\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ \mu_{IC,\mathrm{LGBM}}"
                r">"
                r"\mu_{IC,\mathrm{DT}}"
            ),
            test_name="One-sided paired t-test on aligned weekly IC observations.",
            accent_colour="#2563EB",
            background_colour="#EFF6FF"
        )
        
        t_stat, p_value, statement = pipeline_stock.mean_weekly_ic()
        
        render_hypothesis_result(
            statistic_label="t-statistic",
            statistic_value=t_stat,
            p_value=p_value,
            statement=statement,
            alpha=pipeline_stock.alpha
        )

        render_hypothesis_card(
            number=2,
            question="Do Decision Tree and LightGBM have different directional hit rates?",
            null_hypothesis=(
                r"H_0:\ p_{\mathrm{hit,DT}}"
                r"="
                r"p_{\mathrm{hit,LGBM}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ p_{\mathrm{hit,DT}}"
                r"\neq "
                r"p_{\mathrm{hit,LGBM}}"
            ),
            test_name="McNemar test on paired correct and incorrect predictions.",
            accent_colour="#7C3AED",
            background_colour="#F5F3FF"
        )
        
        chi_squared_stat, p_value, statement = pipeline_stock.mcnemar_test()
        
        render_hypothesis_result(
            statistic_label="χ² statistic",
            statistic_value=chi_squared_stat,
            p_value=p_value,
            statement=statement,
            alpha=pipeline_stock.alpha
        )

        render_hypothesis_card(
            number=3,
            question="Does LightGBM produce a higher mean weekly portfolio return than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ \mu_{r,\mathrm{LGBM}}"
                r"\leq"
                r"\mu_{r,\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ \mu_{r,\mathrm{LGBM}}"
                r">"
                r"\mu_{r,\mathrm{DT}}"
            ),
            test_name="One-sided paired t-test on aligned weekly portfolio returns.",
            accent_colour="#10B981",
            background_colour="#F0FDF4"
        )
        
        t_stat, p_value, statement = pipeline_stock.portfolio_returns_test()
        
        render_hypothesis_result(
            statistic_label="t-statistic",
            statistic_value=t_stat,
            p_value=p_value,
            statement=statement,
            alpha=pipeline_stock.alpha
        )

        render_hypothesis_card(
            number=4,
            question="Does LightGBM achieve a higher Sharpe ratio than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ SR_{\mathrm{LGBM}}"
                r"\leq "
                r"SR_{\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ SR_{\mathrm{LGBM}}"
                r">"
                r"SR_{\mathrm{DT}}"
            ),
            test_name=(
                "One-sided Sharpe-ratio difference test using the "
                "Jobson–Korkie test with Memmel correction."
            ),
            accent_colour="#EA580C",
            background_colour="#FFF7ED"
        )
        
        results_dict = pipeline_stock.sharpe_ratio_test()

        if isinstance(results_dict, dict):
            z_statistic = results_dict.get(
                "z_statistic",
                results_dict.get(
                    "test_statistic",
                    results_dict.get("z", float("nan"))
                )
            )
            sharpe_p_value = results_dict.get("p_value", float("nan"))
            sharpe_statement = results_dict.get(
                "statement",
                "See the reported p-value for the Sharpe-ratio comparison."
            )

            render_hypothesis_result(
                statistic_label="z-statistic",
                statistic_value=z_statistic,
                p_value=sharpe_p_value,
                statement=sharpe_statement,
                alpha=pipeline_stock.alpha
            )
        
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
        
        render_section_header(
            icon="⚖",
            title="Statistical Hypothesis Tests",
            description=(
                "Test whether LightGBM significantly improves upon the Decision Tree "
                "baseline across ranking quality, directional accuracy, "
                "weekly returns and Sharpe ratio."
            )
        )
        
        render_hypothesis_card(
            number=1,
            question="Does LightGBM produce a higher mean weekly IC than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ \mu_{IC,\mathrm{LGBM}}"
                r"\leq "
                r"\mu_{IC,\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ \mu_{IC,\mathrm{LGBM}}"
                r">"
                r"\mu_{IC,\mathrm{DT}}"
            ),
            test_name="One-sided paired t-test on aligned weekly IC observations.",
            accent_colour="#2563EB",
            background_colour="#EFF6FF"
        )
        
        t_stat, ic_p_value, statement = pipeline_market.mean_weekly_ic()
        
        render_hypothesis_result(
            statistic_label="t-statistic",
            statistic_value=t_stat,
            p_value=ic_p_value,
            statement=statement,
            alpha=pipeline_market.alpha
        )

        render_hypothesis_card(
            number=2,
            question="Do Decision Tree and LightGBM have different directional hit rates?",
            null_hypothesis=(
                r"H_0:\ p_{\mathrm{hit,DT}}"
                r"="
                r"p_{\mathrm{hit,LGBM}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ p_{\mathrm{hit,DT}}"
                r"\neq "
                r"p_{\mathrm{hit,LGBM}}"
            ),
            test_name="McNemar test on paired correct and incorrect predictions.",
            accent_colour="#7C3AED",
            background_colour="#F5F3FF"
        )
        
        chi_squared_stat, hit_rate_p_value, statement = pipeline_market.mcnemar_test()
        
        render_hypothesis_result(
            statistic_label="χ² statistic",
            statistic_value=chi_squared_stat,
            p_value=hit_rate_p_value,
            statement=statement,
            alpha=pipeline_market.alpha
        )

        render_hypothesis_card(
            number=3,
            question="Does LightGBM produce a higher mean weekly portfolio return than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ \mu_{r,\mathrm{LGBM}}"
                r"\leq"
                r"\mu_{r,\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ \mu_{r,\mathrm{LGBM}}"
                r">"
                r"\mu_{r,\mathrm{DT}}"
            ),
            test_name="One-sided paired t-test on aligned weekly portfolio returns.",
            accent_colour="#10B981",
            background_colour="#F0FDF4"
        )
        
        t_stat, return_p_value, statement = pipeline_market.portfolio_returns_test()
        
        render_hypothesis_result(
            statistic_label="t-statistic",
            statistic_value=t_stat,
            p_value=return_p_value,
            statement=statement,
            alpha=pipeline_market.alpha
        )

        render_hypothesis_card(
            number=4,
            question="Does LightGBM achieve a higher Sharpe ratio than the Decision Tree?",
            null_hypothesis=(
                r"H_0:\ SR_{\mathrm{LGBM}}"
                r"\leq "
                r"SR_{\mathrm{DT}}"
            ),
            alternative_hypothesis=(
                r"H_1:\ SR_{\mathrm{LGBM}}"
                r">"
                r"SR_{\mathrm{DT}}"
            ),
            test_name=(
                "One-sided Sharpe-ratio difference test using the "
                "Jobson–Korkie test with Memmel correction."
            ),
            accent_colour="#EA580C",
            background_colour="#FFF7ED"
        )
        
        results_dict = pipeline_market.sharpe_ratio_test()

        if isinstance(results_dict, dict):
            z_statistic = results_dict.get(
                "z_statistic",
                results_dict.get(
                    "test_statistic",
                    results_dict.get("z", float("nan"))
                )
            )
            sharpe_p_value = results_dict.get("p_value", float("nan"))
            sharpe_statement = results_dict.get(
                "statement",
                "See the reported p-value for the Sharpe-ratio comparison."
            )

            render_hypothesis_result(
                statistic_label="z-statistic",
                statistic_value=z_statistic,
                p_value=sharpe_p_value,
                statement=sharpe_statement,
                alpha=pipeline_market.alpha
            )
        
        hypothesis_p_values = {
            "mean_weekly_ic": ic_p_value,
            "directional_hit_rate": hit_rate_p_value,
            "mean_weekly_return": return_p_value,
            "sharpe_ratio": sharpe_p_value
        }

        render_final_model_summary(
            selection=selection,
            hypothesis_p_values=hypothesis_p_values,
            model_colour_map=MODEL_COLOURS,
            alpha=pipeline_market.alpha
        )

