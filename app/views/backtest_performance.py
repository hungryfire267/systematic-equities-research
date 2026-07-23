import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from html import escape
import html
import os
import re
import pandas as pd
from pathlib import Path
import streamlit as st
import sys
from dotenv import load_dotenv
from google import genai



BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

load_dotenv()

api_key = st.secrets.get(
    "GOOGLE_API_KEY",
    os.getenv("GOOGLE_API_KEY")
)

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY was not found in Streamlit secrets or the environment."
    )

gemini_client = genai.Client(api_key=api_key)


from scripts.portfolio.metrics import GetMetrics
from scripts.dashboard.get_asx_metrics import ASXMetrics
from scripts.dashboard.alpha_metrics import AlphaMetrics


MODEL_DIRECTORIES = {
    "dt": BASE_DIR / "results" / "backtest" / "dt",
    "lightgbm": BASE_DIR / "results" / "backtest" / "lightgbm",
    "xgboost": BASE_DIR / "results" / "backtest" / "xgboost",
}

MODEL_NAMES = {
    "dt": "Decision Tree",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

FEATURE_NAMES = {
    "stock": "Stock Features",
    "market": "Stock + Market",
}

ASX_DIR = BASE_DIR / "data" / "raw" / "asx"


def _load_selected_backtest() -> tuple[
    dict,
    pd.DataFrame,
    pd.Series,
    str,
    str,
]:
    """
    Load the portfolio selected on the Model Comparison page.

    Falls back to LightGBM + Stock Features when session state has
    not yet been populated.
    """
    selected_configuration = st.session_state.get(
        "selected_configuration",
        {}
    )

    model_key = selected_configuration.get(
        "model_key",
        st.session_state.get(
            "selected_model_key",
            "lightgbm"
        )
    )

    feature_key = selected_configuration.get(
        "feature_key",
        st.session_state.get(
            "selected_feature_key",
            "stock"
        )
    )

    model_name = selected_configuration.get(
        "model_name",
        st.session_state.get(
            "selected_model",
            MODEL_NAMES.get(model_key, model_key)
        )
    )

    feature_name = selected_configuration.get(
        "feature_name",
        st.session_state.get(
            "selected_feature_set",
            FEATURE_NAMES.get(feature_key, feature_key)
        )
    )

    if model_key not in MODEL_DIRECTORIES:
        raise KeyError(
            f"Unknown model key: {model_key}. "
            f"Expected one of {sorted(MODEL_DIRECTORIES)}."
        )

    portfolio_path = (
        MODEL_DIRECTORIES[model_key]
        / f"final_portfolio_{feature_key}.parquet"
    )

    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Selected portfolio file was not found: {portfolio_path}"
        )

    final_portfolio_df = pd.read_parquet(portfolio_path)

    portfolio_metrics, portfolio_returns = (
        GetMetrics(final_portfolio_df).run_data()
    )

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

    strategy_name = f"{model_name} + {feature_name}"
    strategy_returns.name = strategy_name

    resolved_configuration = {
        **selected_configuration,
        "model_key": model_key,
        "model_name": model_name,
        "feature_key": feature_key,
        "feature_name": feature_name,
    }

    st.session_state["selected_configuration"] = resolved_configuration
    st.session_state["selected_model"] = model_name
    st.session_state["selected_feature_set"] = feature_name
    st.session_state["selected_model_key"] = model_key
    st.session_state["selected_feature_key"] = feature_key

    return (
        portfolio_metrics,
        portfolio_returns,
        strategy_returns,
        strategy_name,
        feature_name,
    )


(
    portfolio_metrics,
    portfolio_returns,
    strategy_returns,
    STRATEGY_NAME,
    SELECTED_FEATURE_NAME,
) = _load_selected_backtest()


LATEST_REBALANCE_DATE = pd.to_datetime(
    portfolio_returns["Date"],
    errors="coerce",
).max()

if pd.isna(LATEST_REBALANCE_DATE):
    raise ValueError(
        "The selected backtest does not contain a valid rebalance date."
    )


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



BACKTEST_PAGE_CSS = """
<style>
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.backtest-hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 88% 18%, rgba(124,58,237,0.18), transparent 28%),
        radial-gradient(circle at 72% 110%, rgba(37,99,235,0.15), transparent 36%),
        linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 48%, #F5F3FF 100%);
    border: 1px solid #DCE7F5;
    border-radius: 22px;
    padding: 1.75rem 1.9rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 34px rgba(37,99,235,0.08);
}

.backtest-title {
    margin: 0;
    color: #0F172A;
    font-size: 2.25rem;
    font-weight: 850;
    line-height: 1.08;
}

.backtest-description {
    margin-top: 0.65rem;
    max-width: 920px;
    color: #52647A;
    font-size: 0.96rem;
    line-height: 1.62;
}


.latest-rebalance {
    display: inline-flex;
    align-items: center;
    margin-top: 0.85rem;
    padding: 0.38rem 0.68rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.78);
    border: 1px solid #DCE7F5;
    color: #64748B;
    font-size: 0.73rem;
    font-weight: 700;
}

.section-header {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
}

.section-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.15rem;
    height: 2.15rem;
    border-radius: 12px;
    background: #EFF6FF;
    color: #2563EB;
    font-size: 1rem;
}

.section-title {
    color: #0F172A;
    font-size: 1.25rem;
    font-weight: 850;
    line-height: 1.2;
}

.section-caption {
    margin-top: 0.22rem;
    color: #64748B;
    font-size: 0.83rem;
    line-height: 1.45;
}

.backtest-takeaway {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.85rem;
    align-items: flex-start;
    margin-top: 0.9rem;
    padding: 1rem 1.15rem;
    border-radius: 16px;
    border: 1px solid #BFDBFE;
    border-left: 5px solid #6366F1;
    background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
}

.takeaway-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 999px;
    background: #E0E7FF;
    color: #4338CA;
    font-size: 1.05rem;
}

.takeaway-title {
    color: #312E81;
    font-size: 0.88rem;
    font-weight: 850;
    margin-bottom: 0.3rem;
}

.takeaway-text {
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.55;
}

.alpha-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.alpha-card {
    border-radius: 17px;
    padding: 1rem 1.05rem;
    min-height: 126px;
    border: 1px solid var(--alpha-border);
    background: linear-gradient(145deg, #FFFFFF 0%, var(--alpha-bg) 155%);
    box-shadow: 0 6px 18px rgba(15,23,42,0.045);
    text-align: center;
}

.alpha-label {
    color: #334155;
    font-size: 0.78rem;
    font-weight: 900;
    margin-bottom: 0.65rem;
}

.alpha-value {
    color: var(--alpha-accent);
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1.1;
    margin-bottom: 0.58rem;
    text-align: center;
}

.alpha-note {
    color: #64748B;
    font-size: 0.69rem;
    line-height: 1.4;
    text-align: center;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 16px;
}

.backtest-analysis-card {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.85rem;
    align-items: flex-start;
    width: 100%;
    box-sizing: border-box;
    margin-top: 1rem;
    margin-bottom: 1.25rem;
    padding: 1rem 1.15rem;
    border: 1px solid #BFDBFE;
    border-left: 5px solid #6366F1;
    border-radius: 16px;
    background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
}

.backtest-analysis-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 999px;
    background: #E0E7FF;
    color: #4338CA;
    font-size: 1.05rem;
}

.backtest-analysis-title {
    color: #312E81;
    font-size: 0.88rem;
    font-weight: 850;
    margin-bottom: 0.3rem;
}

.backtest-analysis-list {
    margin: 0;
    padding-left: 1.15rem;
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.55;
}

.backtest-analysis-list li {
    margin-bottom: 0.42rem;
    padding-left: 0.08rem;
}

.backtest-analysis-list li:last-child {
    margin-bottom: 0;
}

.backtest-analysis-list li::marker {
    color: #6366F1;
}

@media (max-width: 900px) {
    .alpha-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
</style>
"""


def _format_alpha_value(value, percentage=False):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}" if percentage else f"{value:.2f}"


def render_alpha_metric_cards(alpha_metrics: dict) -> None:
    definitions = [
        ("Annualised Alpha", "annualised_alpha", True, "#059669", "#D1FAE5", "#A7F3D0",
         "Estimated excess return after benchmark exposure."),
        ("Market Beta", "beta", False, "#2563EB", "#DBEAFE", "#BFDBFE",
         "Sensitivity of strategy returns to the ASX 200."),
        ("Market R²", "r_squared", True, "#7C3AED", "#EDE9FE", "#DDD6FE",
         "Share of strategy-return variation explained by the benchmark."),
        ("Information Ratio", "information_ratio", False, "#EA580C", "#FFEDD5", "#FED7AA",
         "Active return generated per unit of benchmark-relative risk."),
    ]

    cards = []
    for label, key, percentage, accent, bg, border, note in definitions:
        value = alpha_metrics.get(key)
        cards.append(
            f"""
            <div class="alpha-card" style="
                --alpha-accent:{accent};
                --alpha-bg:{bg};
                --alpha-border:{border};
            ">
                <div class="alpha-label">{label}</div>
                <div class="alpha-value">{_format_alpha_value(value, percentage)}</div>
                <div class="alpha-note">{note}</div>
            </div>
            """
        )

    st.html(f'<div class="alpha-grid">{"".join(cards)}</div>')


def _format_metric(
    value: float | None,
    as_percentage: bool
) -> str:
    if value is None or pd.isna(value):
        return "—"

    if as_percentage:
        return f"{value:.1%}"

    return f"{value:.2f}"


def _format_difference(
    difference: float,
    as_percentage: bool
) -> str:
    if as_percentage:
        return f"{difference:+.1%}"

    return f"{difference:+.2f}"


def render_backtest_metric_cards(
    strategy_metrics: dict,
    benchmark_metrics: dict,
    strategy_name: str,
    benchmark_name: str = "ASX 200"
) -> None:
    """
    Render headline strategy performance cards.
    """

    metrics = [
        {
            "label": "Annual Return",
            "key": "annual_return",
            "symbol": "↗",
            "percentage": True,
            "higher_is_better": True,
            "card_class": "annual-return-card"
        },
        {
            "label": "Sharpe Ratio",
            "key": "sharpe_ratio",
            "symbol": "◆",
            "percentage": False,
            "higher_is_better": True,
            "card_class": "sharpe-card"
        },
        {
            "label": "Sortino Ratio",
            "key": "sortino_ratio",
            "symbol": "▲",
            "percentage": False,
            "higher_is_better": True,
            "card_class": "sortino-card"
        },
        {
            "label": "Maximum Drawdown",
            "key": "max_drawdown",
            "symbol": "↓",
            "percentage": True,
            "higher_is_better": True,
            "card_class": "drawdown-card"
        },
        {
            "label": "Weekly Win Rate",
            "key": "win_rate",
            "symbol": "✓",
            "percentage": True,
            "higher_is_better": True,
            "card_class": "win-rate-card"
        }
    ]

    cards = []

    for metric in metrics:
        key = metric["key"]

        strategy_value = strategy_metrics.get(key)
        benchmark_value = benchmark_metrics.get(key)

        formatted_value = _format_metric(
            strategy_value,
            metric["percentage"]
        )

        if (
            strategy_value is None
            or benchmark_value is None
            or pd.isna(strategy_value)
            or pd.isna(benchmark_value)
        ):
            delta_class = "metric-card-delta-neutral"
            delta_text = "Benchmark unavailable"

        else:
            difference = strategy_value - benchmark_value

            is_better = (
                strategy_value > benchmark_value
                if metric["higher_is_better"]
                else strategy_value < benchmark_value
            )

            delta_class = (
                "metric-card-delta-positive"
                if is_better
                else "metric-card-delta-negative"
            )

            formatted_difference = _format_difference(
                difference,
                metric["percentage"]
            )

            delta_text = (
                f"{formatted_difference} "
                f"vs {escape(benchmark_name)}"
            )

        cards.append(
            f"""
            <div class="
                headline-metric-card
                {metric["card_class"]}
            ">
                <div class="headline-metric-label">
                    {escape(metric["label"])}
                </div>

                <div class="headline-metric-top">
                    <div class="headline-metric-symbol">
                        {metric["symbol"]}
                    </div>

                    <div class="headline-metric-value">
                        {formatted_value}
                    </div>
                </div>

                <div class="headline-metric-strategy">
                    {escape(strategy_name)}
                </div>

                <div class="
                    headline-metric-delta
                    {delta_class}
                ">
                    {delta_text}
                </div>
            </div>
            """
        )

    cards_html = "".join(cards)

    st.html(
        f"""
        <style>
            .headline-metrics-section {{
                width: 100%;
                margin-top: 18px;
            }}

            .headline-metrics-header {{
                margin-bottom: 12px;
            }}

            .headline-metrics-title {{
                color: #0F172A;
                font-size: 16px;
                font-weight: 700;
                line-height: 1.3;
            }}

            .headline-metrics-caption {{
                margin-top: 4px;
                color: #64748B;
                font-size: 13px;
                line-height: 1.4;
            }}

            .headline-metrics-grid {{
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 12px;
            }}

            .headline-section-header {{
                display: flex;
                align-items: flex-start;
                gap: 0.7rem;
                margin-bottom: 0.9rem;
            }}

            .headline-section-icon {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 2.15rem;
                height: 2.15rem;
                flex: 0 0 2.15rem;
                border-radius: 12px;
                background: #EFF6FF;
                color: #2563EB;
                font-size: 1rem;
                font-weight: 800;
            }}

            .headline-section-title {{
                color: #0F172A;
                font-size: 1.25rem;
                font-weight: 850;
                line-height: 1.2;
            }}

            .headline-section-caption {{
                margin-top: 0.22rem;
                color: #64748B;
                font-size: 0.83rem;
                line-height: 1.45;
            }}

            .headline-metric-card {{
                min-width: 0;
                padding: 16px;
                border-radius: 13px;
                box-sizing: border-box;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
                text-align: center;
            }}

            .annual-return-card {{
                border: 1px solid #A7F3D0;
                background: #ECFDF5;
            }}

            .sharpe-card {{
                border: 1px solid #BFDBFE;
                background: #EFF6FF;
            }}

            .sortino-card {{
                border: 1px solid #DDD6FE;
                background: #F5F3FF;
            }}

            .drawdown-card {{
                border: 1px solid #FECACA;
                background: #FEF2F2;
            }}

            .win-rate-card {{
                border: 1px solid #FDE68A;
                background: #FFFBEB;
            }}

            .headline-metric-top {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                min-width: 0;
                margin-top: 14px;
            }}

            .headline-metric-symbol {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 28px;
                height: 28px;
                flex: 0 0 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 800;
            }}

            .annual-return-card .headline-metric-symbol {{
                background: #D1FAE5;
                color: #047857;
            }}

            .sharpe-card .headline-metric-symbol {{
                background: #DBEAFE;
                color: #1D4ED8;
            }}

            .sortino-card .headline-metric-symbol {{
                background: #EDE9FE;
                color: #6D28D9;
            }}

            .drawdown-card .headline-metric-symbol {{
                background: #FEE2E2;
                color: #B91C1C;
            }}

            .win-rate-card .headline-metric-symbol {{
                background: #FEF3C7;
                color: #A16207;
            }}

            .headline-metric-label {{
                overflow: hidden;
                color: #334155;
                font-size: 12px;
                font-weight: 850;
                line-height: 1.3;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: center;
            }}

            .headline-metric-value {{
                color: #0F172A;
                font-size: 25px;
                font-weight: 800;
                line-height: 1;
                font-variant-numeric: tabular-nums;
                text-align: center;
            }}

            .headline-metric-strategy {{
                margin-top: 10px;
                color: #64748B;
                font-size: 11px;
                font-weight: 600;
                text-align: center;
            }}

            .headline-metric-delta {{
                margin-top: 11px;
                font-size: 11px;
                font-weight: 700;
                line-height: 1.3;
                text-align: center;
            }}

            .metric-card-delta-positive {{
                color: #059669;
            }}

            .metric-card-delta-negative {{
                color: #DC2626;
            }}

            .metric-card-delta-neutral {{
                color: #64748B;
            }}

            @media (max-width: 1100px) {{
                .headline-metrics-grid {{
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                }}
            }}

            @media (max-width: 700px) {{
                .headline-metrics-grid {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
            }}
        </style>

        <div class="headline-metrics-section">
            <div class="headline-section-header">
                <div class="headline-section-icon">✦</div>
                <div>
                    <div class="headline-section-title">Performance Highlights</div>
                    <div class="headline-section-caption">
                        Headline {escape(strategy_name)} results compared with
                        the {escape(benchmark_name)}.
                    </div>
                </div>
            </div>

            <div class="headline-metrics-grid">
                {cards_html}
            </div>
        </div>
        """
    )

PERFORMANCE_COLOURS = {
    "strategy": "#10B981",
    "benchmark": "#334155",
    "cash": "#94A3B8"
}


def _extract_return_series_cumulative(
    returns: pd.Series | pd.DataFrame,
    output_name: str,
    return_col: str | None = None
) -> pd.Series:
    """
    Convert a Series or DataFrame into a clean return Series.
    """

    if isinstance(returns, pd.Series):
        series = returns.copy()

    elif isinstance(returns, pd.DataFrame):
        returns_df = returns.copy()

        if return_col is not None:
            if return_col not in returns_df.columns:
                raise ValueError(
                    f"Column '{return_col}' was not found. "
                    f"Available columns: {list(return_col.columns)}"
                )

            series = returns_df[return_col].copy()

        else:
            numeric_columns = (
                returns_df
                .select_dtypes(include=np.number)
                .columns
                .tolist()
            )

            if len(numeric_columns) != 1:
                raise ValueError(
                    f"{output_name} returns contain multiple numeric columns. "
                    "Pass the required column using return_col. "
                    f"Available columns: {list(returns_df.columns)}"
                )

            series = returns_df[numeric_columns[0]].copy()

    else:
        raise TypeError(
            f"{output_name} returns must be a pandas Series or DataFrame, "
            f"not {type(returns).__name__}."
        )

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    if series.empty:
        raise ValueError(
            f"{output_name} returns contain no valid observations."
        )

    series.name = output_name

    return series


def _prepare_cumulative_returns(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    cash_returns: pd.Series | pd.DataFrame | None = None,
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None,
    cash_return_col: str | None = None
) -> pd.DataFrame:
    """
    Clean and align return series, then calculate cumulative returns.
    """

    strategy_series = _extract_return_series_cumulative(
        returns=strategy_returns,
        output_name="Strategy",
        return_col=strategy_return_col
    )

    benchmark_series = _extract_return_series_cumulative(
        returns=benchmark_returns,
        output_name="Benchmark",
        return_col=benchmark_return_col
    )

    series_to_align = [
        strategy_series,
        benchmark_series
    ]

    if cash_returns is not None:
        cash_series = _extract_return_series_cumulative(
            returns=cash_returns,
            output_name="Cash",
            return_col=cash_return_col
        )

        series_to_align.append(cash_series)

    returns_df = pd.concat(
        series_to_align,
        axis=1,
        join="inner"
    )

    returns_df = (
        returns_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if returns_df.empty:
        raise ValueError(
            "No aligned return observations were found."
        )

    cumulative_returns = (
        1 + returns_df
    ).cumprod() - 1

    return cumulative_returns


def _add_final_value_annotation(
    figure: go.Figure,
    series: pd.Series,
    colour: str
) -> None:
    """
    Add the latest cumulative return to the right side of the chart.
    """

    final_date = series.index[-1]
    final_value = float(series.iloc[-1])

    figure.add_annotation(
        x=final_date,
        y=final_value,
        text=f"<b>{final_value:.1%}</b>",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(
            color=colour,
            size=12
        ),
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=2
    )


def _create_cumulative_return_figure(
    cumulative_returns: pd.DataFrame,
    strategy_name: str,
    benchmark_name: str,
    cash_name: str
) -> go.Figure:
    """
    Create the cumulative portfolio-return chart.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns["Strategy"],
            name=strategy_name,
            mode="lines",
            line=dict(
                color=PERFORMANCE_COLOURS["strategy"],
                width=2.4
            ),
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Cumulative return: %{y:.1%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns["Benchmark"],
            name=benchmark_name,
            mode="lines",
            line=dict(
                color=PERFORMANCE_COLOURS["benchmark"],
                width=2.2
            ),
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Cumulative return: %{y:.1%}"
                "<extra></extra>"
            )
        )
    )

    if "Cash" in cumulative_returns.columns:
        figure.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns["Cash"],
                name=cash_name,
                mode="lines",
                line=dict(
                    color=PERFORMANCE_COLOURS["cash"],
                    width=2
                ),
                hovertemplate=(
                    "%{x|%d %b %Y}"
                    "<br>Cumulative return: %{y:.1%}"
                    "<extra></extra>"
                )
            )
        )

    all_values = cumulative_returns.to_numpy().flatten()
    all_values = all_values[np.isfinite(all_values)]

    y_min = min(float(all_values.min()), 0)
    y_max = max(float(all_values.max()), 0)

    y_range = y_max - y_min
    y_padding = max(y_range * 0.10, 0.03)

    figure.add_hline(
        y=0,
        line_width=1,
        line_color="#94A3B8"
    )

    figure.update_layout(
        height=TOP_PLOT_HEIGHT,
        margin=dict(
            l=15,
            r=78,
            t=45,
            b=15
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                size=11,
                color="#475569"
            )
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            tickfont=dict(
                color="#64748B"
            )
        ),
        yaxis=dict(
            title=None,
            tickformat=".0%",
            range=[
                y_min - y_padding,
                y_max + y_padding
            ],
            gridcolor="rgba(148, 163, 184, 0.20)",
            zeroline=False,
            fixedrange=True,
            tickfont=dict(
                color="#64748B"
            )
        )
    )

    return figure


def render_cumulative_returns(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    strategy_name: str,
    benchmark_name: str = "ASX 200",
    cash_returns: pd.Series | pd.DataFrame | None = None,
    cash_name: str = "Cash",
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None,
    cash_return_col: str | None = None
) -> None:
    """
    Render cumulative strategy, benchmark and optional cash returns.
    """

    cumulative_returns = _prepare_cumulative_returns(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        cash_returns=cash_returns,
        strategy_return_col=strategy_return_col,
        benchmark_return_col=benchmark_return_col,
        cash_return_col=cash_return_col
    )

    figure = _create_cumulative_return_figure(
        cumulative_returns=cumulative_returns,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
        cash_name=cash_name
    )

    with st.container(border=True, height=TOP_CHART_CARD_HEIGHT):
        st.markdown(
            """
            <div style="
                color: #0F172A;
                font-size: 16px;
                font-weight: 700;
                line-height: 1.3;
                margin-bottom: 2px;
            ">
                Cumulative Portfolio Return
            </div>

            <div style="
                color: #64748B;
                font-size: 13px;
                line-height: 1.4;
                margin-bottom: 0;
            ">
                Growth of the strategy relative to the benchmark and cash.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False
            },
            key="cumulative_portfolio_return"
        )

DRAWDOWN_COLOURS = {
    "strategy": "#10B981",
    "benchmark": "#334155"
}

TOP_CHART_CARD_HEIGHT = 455
TOP_PLOT_HEIGHT = 350
BOTTOM_CARD_HEIGHT = 610


def _extract_return_series_drawdown(
    returns: pd.Series | pd.DataFrame,
    output_name: str,
    return_col: str | None = None
) -> pd.Series:
    """
    Convert a Series or DataFrame into a clean return Series.
    """

    if isinstance(returns, pd.Series):
        series = returns.copy()

    elif isinstance(returns, pd.DataFrame):
        returns_df = returns.copy()

        if return_col is not None:
            if return_col not in returns_df.columns:
                raise ValueError(
                    f"Column '{return_col}' was not found. "
                    f"Available columns: {list(returns_df.columns)}"
                )

            series = returns_df[return_col].copy()

        else:
            numeric_columns = (
                returns_df
                .select_dtypes(include=np.number)
                .columns
                .tolist()
            )

            if len(numeric_columns) != 1:
                raise ValueError(
                    f"{output_name} returns contain multiple numeric columns. "
                    "Pass the required column using return_col. "
                    f"Available columns: {list(returns_df.columns)}"
                )

            series = returns_df[numeric_columns[0]].copy()

    else:
        raise TypeError(
            f"{output_name} returns must be a pandas Series or DataFrame, "
            f"not {type(returns).__name__}."
        )

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    if series.empty:
        raise ValueError(
            f"{output_name} returns contain no valid observations."
        )

    series.name = output_name

    return series


def _prepare_drawdowns(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None
) -> pd.DataFrame:
    """
    Clean and align returns, then calculate drawdown series.
    """

    strategy_series = _extract_return_series_drawdown(
        returns=strategy_returns,
        output_name="Strategy",
        return_col=strategy_return_col
    )

    benchmark_series = _extract_return_series_drawdown(
        returns=benchmark_returns,
        output_name="Benchmark",
        return_col=benchmark_return_col
    )

    returns_df = pd.concat(
        [
            strategy_series,
            benchmark_series
        ],
        axis=1,
        join="inner"
    )

    returns_df = (
        returns_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if returns_df.empty:
        raise ValueError(
            "No aligned strategy and benchmark return observations "
            "were found."
        )

    wealth_index = (
        1 + returns_df
    ).cumprod()

    running_peak = wealth_index.cummax()

    drawdowns = (
        wealth_index / running_peak
    ) - 1

    return drawdowns


def _add_final_drawdown_annotation(
    figure: go.Figure,
    series: pd.Series,
    colour: str
) -> None:
    """
    Add the latest drawdown value to the right side of the chart.
    """

    final_date = series.index[-1]
    final_value = float(series.iloc[-1])

    figure.add_annotation(
        x=final_date,
        y=final_value,
        text=f"<b>{final_value:.1%}</b>",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(
            color=colour,
            size=12
        ),
        bgcolor="rgba(255,255,255,0.78)",
        borderpad=2
    )


def _create_drawdown_figure(
    drawdowns: pd.DataFrame,
    strategy_name: str,
    benchmark_name: str
) -> go.Figure:
    """
    Create strategy and benchmark drawdown chart.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns["Strategy"],
            name=strategy_name,
            mode="lines",
            line=dict(
                color=DRAWDOWN_COLOURS["strategy"],
                width=2.2
            ),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Drawdown: %{y:.1%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns["Benchmark"],
            name=benchmark_name,
            mode="lines",
            line=dict(
                color=DRAWDOWN_COLOURS["benchmark"],
                width=2
            ),
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Drawdown: %{y:.1%}"
                "<extra></extra>"
            )
        )
    )

    minimum_drawdown = float(
        drawdowns.min().min()
    )

    lower_bound = min(
        minimum_drawdown * 1.12,
        -0.05
    )

    figure.add_hline(
        y=0,
        line_width=1.2,
        line_color="#64748B"
    )

    figure.update_layout(
        height=TOP_PLOT_HEIGHT,
        margin=dict(
            l=15,
            r=78,
            t=45,
            b=15
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                size=11,
                color="#475569"
            )
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            tickfont=dict(
                color="#64748B"
            )
        ),
        yaxis=dict(
            title=None,
            tickformat=".0%",
            range=[
                lower_bound,
                0.01
            ],
            gridcolor="rgba(148, 163, 184, 0.20)",
            zeroline=False,
            fixedrange=True,
            tickfont=dict(
                color="#64748B"
            )
        )
    )

    return figure


def render_drawdown(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    strategy_name: str,
    benchmark_name: str = "ASX 200",
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None
) -> None:
    """
    Render strategy and benchmark drawdowns over time.
    """

    drawdowns = _prepare_drawdowns(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        strategy_return_col=strategy_return_col,
        benchmark_return_col=benchmark_return_col
    )

    figure = _create_drawdown_figure(
        drawdowns=drawdowns,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name
    )

    with st.container(border=True, height=TOP_CHART_CARD_HEIGHT):
        st.markdown(
            """
            <div style="
                color: #0F172A;
                font-size: 16px;
                font-weight: 700;
                line-height: 1.3;
                margin-bottom: 2px;
            ">
                Drawdown Over Time
            </div>

            <div style="
                color: #64748B;
                font-size: 13px;
                line-height: 1.4;
                margin-bottom: 0;
            ">
                Peak-to-trough portfolio declines relative to the benchmark.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False
            },
            key="portfolio_drawdown"
        )

RETURN_SERIES_COLOURS = {
    "strategy": "#10B981",
    "benchmark": "#334155"
}

# Use this same height for the summary-table container.
BACKTEST_CARD_HEIGHT = BOTTOM_CARD_HEIGHT


def _extract_return_series_distribution(
    returns: pd.Series | pd.DataFrame,
    output_name: str,
    return_col: str | None = None
) -> pd.Series:
    """
    Convert a Series or DataFrame into a clean return Series.

    For a DataFrame:
    - Uses return_col when supplied.
    - Otherwise uses the only numeric column when exactly one exists.
    """

    if isinstance(returns, pd.Series):
        series = returns.copy()

    elif isinstance(returns, pd.DataFrame):
        returns_df = returns.copy()

        if return_col is not None:
            if return_col not in returns_df.columns:
                raise ValueError(
                    f"Column '{return_col}' was not found. "
                    f"Available columns: {list(returns_df.columns)}"
                )

            series = returns_df[return_col].copy()

        else:
            numeric_columns = (
                returns_df
                .select_dtypes(include=np.number)
                .columns
                .tolist()
            )

            if len(numeric_columns) == 1:
                series = returns_df[numeric_columns[0]].copy()

            else:
                raise ValueError(
                    f"{output_name} returns were provided as a DataFrame "
                    "with multiple numeric columns. Pass the appropriate "
                    "return column using return_col. "
                    f"Available columns: {list(returns_df.columns)}"
                )

    else:
        raise TypeError(
            f"{output_name} returns must be a pandas Series "
            f"or DataFrame, not {type(returns).__name__}."
        )

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    series.name = output_name

    if series.empty:
        raise ValueError(
            f"{output_name} returns contain no valid observations."
        )

    return series


def _prepare_returns(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None
) -> pd.DataFrame:
    """
    Extract, clean and align strategy and benchmark returns by index.
    """

    strategy_series = _extract_return_series_distribution(
        returns=strategy_returns,
        output_name="Strategy",
        return_col=strategy_return_col
    )

    benchmark_series = _extract_return_series_distribution(
        returns=benchmark_returns,
        output_name="Benchmark",
        return_col=benchmark_return_col
    )

    returns_df = pd.concat(
        [
            strategy_series,
            benchmark_series
        ],
        axis=1,
        join="inner"
    )

    returns_df = (
        returns_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if returns_df.empty:
        raise ValueError(
            "No aligned strategy and benchmark return observations "
            "were found."
        )

    return returns_df


def _calculate_plot_range(
    returns_df: pd.DataFrame
) -> tuple[float, float]:
    """
    Create a shared x-axis range for the histogram and box plot.
    """

    combined_returns = pd.concat(
        [
            returns_df["Strategy"],
            returns_df["Benchmark"]
        ],
        ignore_index=True
    ).dropna()

    x_min = float(combined_returns.min())
    x_max = float(combined_returns.max())

    data_range = x_max - x_min

    if data_range == 0:
        padding = 0.01
    else:
        padding = data_range * 0.08

    return x_min - padding, x_max + padding


def _add_kde_trace(
    figure: go.Figure,
    returns: pd.Series,
    name: str,
    colour: str,
    x_min: float,
    x_max: float
) -> None:
    """
    Add a kernel-density-estimate line to the histogram.

    KDE is skipped when there are too few observations or no variation.
    """

    clean_returns = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(clean_returns) < 2:
        return

    if clean_returns.nunique() < 2:
        return

    try:
        kde = gaussian_kde(
            clean_returns.to_numpy()
        )

        x_grid = np.linspace(
            x_min,
            x_max,
            400
        )

        density = kde(x_grid)

        figure.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name=f"{name} KDE",
                line=dict(
                    color=colour,
                    width=2.5
                ),
                hovertemplate=(
                    "Weekly return: %{x:.2%}"
                    "<br>Estimated density: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

    except (np.linalg.LinAlgError, ValueError):
        # KDE may fail for singular or effectively constant data.
        return


def _create_histogram(
    returns_df: pd.DataFrame,
    strategy_name: str,
    benchmark_name: str,
    x_min: float,
    x_max: float
) -> go.Figure:
    """
    Create overlapping probability-density histograms with KDE lines.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=returns_df["Strategy"],
            name=strategy_name,
            histnorm="probability density",
            opacity=0.48,
            marker=dict(
                color=RETURN_SERIES_COLOURS["strategy"],
                line=dict(
                    color=RETURN_SERIES_COLOURS["strategy"],
                    width=0.5
                )
            ),
            hovertemplate=(
                "Weekly return: %{x:.2%}"
                "<br>Density: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Histogram(
            x=returns_df["Benchmark"],
            name=benchmark_name,
            histnorm="probability density",
            opacity=0.48,
            marker=dict(
                color=RETURN_SERIES_COLOURS["benchmark"],
                line=dict(
                    color=RETURN_SERIES_COLOURS["benchmark"],
                    width=0.5
                )
            ),
            hovertemplate=(
                "Weekly return: %{x:.2%}"
                "<br>Density: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    _add_kde_trace(
        figure=figure,
        returns=returns_df["Strategy"],
        name=strategy_name,
        colour=RETURN_SERIES_COLOURS["strategy"],
        x_min=x_min,
        x_max=x_max
    )

    _add_kde_trace(
        figure=figure,
        returns=returns_df["Benchmark"],
        name=benchmark_name,
        colour=RETURN_SERIES_COLOURS["benchmark"],
        x_min=x_min,
        x_max=x_max
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_color="#64748B"
    )

    figure.update_layout(
        barmode="overlay",
        bargap=0.03,
        height=285,
        margin=dict(
            l=12,
            r=12,
            t=36,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                size=11,
                color="#475569"
            )
        ),
        xaxis=dict(
            title="Weekly return",
            range=[x_min, x_max],
            tickformat=".0%",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="Density",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            fixedrange=True
        )
    )

    return figure


def _create_boxplot(
    returns_df: pd.DataFrame,
    strategy_name: str,
    benchmark_name: str,
    x_min: float,
    x_max: float
) -> go.Figure:
    """
    Create horizontal box plots for strategy and benchmark returns.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Box(
            x=returns_df["Strategy"],
            name=strategy_name,
            orientation="h",
            boxmean=True,
            boxpoints="outliers",
            fillcolor="rgba(16, 185, 129, 0.45)",
            marker=dict(
                color=RETURN_SERIES_COLOURS["strategy"],
                size=5
            ),
            line=dict(
                color=RETURN_SERIES_COLOURS["strategy"],
                width=1.5
            ),
            hovertemplate=(
                "Weekly return: %{x:.2%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Box(
            x=returns_df["Benchmark"],
            name=benchmark_name,
            orientation="h",
            boxmean=True,
            boxpoints="outliers",
            fillcolor="rgba(51, 65, 85, 0.28)",
            marker=dict(
                color=RETURN_SERIES_COLOURS["benchmark"],
                size=5
            ),
            line=dict(
                color=RETURN_SERIES_COLOURS["benchmark"],
                width=1.5
            ),
            hovertemplate=(
                "Weekly return: %{x:.2%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_color="#64748B"
    )

    figure.update_layout(
        height=205,
        margin=dict(
            l=10,
            r=12,
            t=0,
            b=10
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Weekly return",
            range=[x_min, x_max],
            tickformat=".0%",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            fixedrange=True,
            categoryorder="array",
            categoryarray=[
                strategy_name,
                benchmark_name
            ]
        )
    )

    return figure


def render_return_distribution(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    strategy_name: str,
    benchmark_name: str = "ASX 200",
    strategy_return_col: str | None = None,
    benchmark_return_col: str | None = None,
    container_height: int = BACKTEST_CARD_HEIGHT
) -> None:
    """
    Render weekly strategy and benchmark return distributions.

    Set container_height to the same value used by the adjacent
    risk-and-return summary container so both cards align.
    """

    returns_df = _prepare_returns(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        strategy_return_col=strategy_return_col,
        benchmark_return_col=benchmark_return_col
    )

    x_min, x_max = _calculate_plot_range(
        returns_df=returns_df
    )

    histogram = _create_histogram(
        returns_df=returns_df,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
        x_min=x_min,
        x_max=x_max
    )

    boxplot = _create_boxplot(
        returns_df=returns_df,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
        x_min=x_min,
        x_max=x_max
    )

    with st.container(
        border=True,
        height=container_height
    ):
        st.markdown(
            """
<div style="
    color: #0F172A;
    font-size: 16px;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 3px 0;
">
    Return Distribution
</div>
<div style="
    color: #64748B;
    font-size: 13px;
    line-height: 1.4;
    margin: 0 0 2px 0;
">
    Weekly return dispersion, central tendency and outliers.
</div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            histogram,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False
            },
            key="return_distribution_histogram"
        )

        st.plotly_chart(
            boxplot,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False
            },
            key="return_distribution_boxplot"
        )

def _format_value(
    value: float | None,
    as_percentage: bool = False
) -> str:
    if value is None or pd.isna(value):
        return "—"

    if as_percentage:
        return f"{value:.1%}"

    return f"{value:.2f}"


def render_return_summary(
    strategy_metrics: dict,
    benchmark_metrics: dict,
    strategy_name: str,
    benchmark_name: str = "ASX 200",
    container_height: int = BACKTEST_CARD_HEIGHT,
) -> None:
    """
    Render a risk and return comparison table.
    """

    metrics = [
        ("Annual Return", "annual_return", True, True),
        ("Total Return", "total_return", True, True),
        ("Annual Volatility", "annual_volatility", True, False),
        ("Sharpe Ratio", "sharpe_ratio", False, True),
        ("Sortino Ratio", "sortino_ratio", False, True),
        ("Maximum Drawdown", "max_drawdown", True, True),
        ("Calmar Ratio", "calmar_ratio", False, True),
        ("Weekly Win Rate", "win_rate", True, True),
        ("Worst Week", "worst_week", True, True)
    ]

    rows = []

    for label, key, as_percentage, higher_is_better in metrics:
        strategy_value = strategy_metrics.get(key)
        benchmark_value = benchmark_metrics.get(key)

        strategy_display = _format_value(
            strategy_value,
            as_percentage
        )

        benchmark_display = _format_value(
            benchmark_value,
            as_percentage
        )

        if (
            strategy_value is None
            or benchmark_value is None
            or pd.isna(strategy_value)
            or pd.isna(benchmark_value)
        ):
            winner = "—"
            winner_class = "winner-neutral"
        elif strategy_value == benchmark_value:
            winner = "Tie"
            winner_class = "winner-neutral"
        else:
            strategy_wins = (
                strategy_value > benchmark_value
                if higher_is_better
                else strategy_value < benchmark_value
            )
            strategy_winner_label = strategy_name.replace(" Features", "")
            winner = strategy_winner_label if strategy_wins else benchmark_name
            winner_class = (
                "winner-strategy" if strategy_wins else "winner-benchmark"
            )

        rows.append(
            f"""
            <tr>
                <td class="metric-name">{escape(label)}</td>
                <td class="metric-value">{strategy_display}</td>
                <td class="metric-value">{benchmark_display}</td>
                <td class="winner-cell"><span class="winner-pill {winner_class}">{escape(winner)}</span></td>
            </tr>
            """
        )

    rows_html = "".join(rows)

    html = f"""
        <style>
            .return-summary-card {{
                width: 100%;
                height: {container_height}px;
                padding: 20px;
                border: 1px solid #DCE3EC;
                border-radius: 14px;
                background: #FFFFFF;
                box-sizing: border-box;

                display: flex;                 /* <-- Added */
                flex-direction: column;        /* <-- Added */
            }}

            .return-summary-title {{
                margin: 0;
                color: #0F172A;
                font-size: 16px;
                font-weight: 700;
                line-height: 1.3;
            }}

            .return-summary-caption {{
                margin-top: 5px;
                margin-bottom: 16px;
                color: #64748B;
                font-size: 13px;
                line-height: 1.4;
            }}

            .return-summary-table-wrapper {{
                width: 100%;
                flex: 1;
                min-height: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}

            .return-summary-table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}

            .return-summary-table th {{
                padding: 11px 10px;
                border-bottom: 1px solid #CBD5E1;
                background: #F8FAFC;
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                text-align: right;
            }}

            .return-summary-table th:first-child {{
                width: 27%;
                text-align: left;
                border-top-left-radius: 8px;
            }}

            .return-summary-table th:last-child {{
                border-top-right-radius: 8px;
            }}

            .return-summary-table td {{
                padding: 10px 10px;
                border-bottom: 1px solid #E2E8F0;
                color: #0F172A;
                font-size: 13px;
            }}

            .return-summary-table tbody tr:last-child td {{
                border-bottom: none;
            }}

            .metric-name {{
                font-weight: 600;
                text-align: left;
            }}

            .metric-value {{
                font-variant-numeric: tabular-nums;
                text-align: right;
            }}

            .winner-cell {{
                text-align: center;
                font-size: 11px;
                font-weight: 750;
                line-height: 1.25;
            }}

            .winner-pill {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                max-width: 100%;
                padding: 4px 8px;
                border-radius: 999px;
                white-space: normal;
            }}

            .winner-strategy {{
                color: #047857;
                background: #ECFDF5;
                border: 1px solid #A7F3D0;
            }}

            .winner-benchmark {{
                color: #334155;
                background: #F1F5F9;
                border: 1px solid #CBD5E1;
            }}

            .winner-neutral {{
                color: #64748B;
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
            }}
        </style>

        <div class="return-summary-card">

            <div class="return-summary-title">
                Risk &amp; Return Summary
            </div>

            <div class="return-summary-caption">
                Comparison of portfolio performance against the benchmark.
            </div>

            <div class="return-summary-table-wrapper">

                <table class="return-summary-table">

                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>{escape(strategy_name)}</th>
                            <th>{escape(benchmark_name)}</th>
                            <th>Winner</th>
                        </tr>
                    </thead>

                    <tbody>
                        {rows_html}
                    </tbody>

                </table>

            </div>

        </div>
        """

    st.html(html)



def _metric_value(metrics: dict, key: str) -> float:
    """Return a metric as a float, falling back to NaN when unavailable."""
    value = metrics.get(key, np.nan)
    return float(value) if value is not None else float("nan")


@st.cache_data(show_spinner=False)
def generate_backtest_analysis(
    strategy_name: str,
    feature_set: str,
    portfolio_metrics: dict,
    benchmark_metrics: dict,
    alpha_metrics: dict,
) -> str:
    """Generate three concise observations about the backtest using Gemini."""

    prompt = f"""
You are a systematic equities analyst reviewing a quantitative long-short
ASX equity strategy backtest.

Strategy configuration:
- Model: {strategy_name}
- Feature set: {feature_set}
- Forecast horizon: 5 trading days
- Rebalancing frequency: weekly
- Portfolio construction: long-short and dollar-neutral
- Validation: expanding-window walk-forward validation
- Return frequency: weekly
- Annualisation factor: 52
- Risk-free rate: 0.0
- Benchmark: ASX 200
- Transaction costs: excluded; all results are gross and pre-cost

Strategy performance:
- Annual return: {_metric_value(portfolio_metrics, "annual_return"):.1%}
- Total return: {_metric_value(portfolio_metrics, "total_return"):.1%}
- Annual volatility: {_metric_value(portfolio_metrics, "annual_volatility"):.1%}
- Sharpe ratio: {_metric_value(portfolio_metrics, "sharpe_ratio"):.2f}
- Sortino ratio: {_metric_value(portfolio_metrics, "sortino_ratio"):.2f}
- Maximum drawdown: {_metric_value(portfolio_metrics, "max_drawdown"):.1%}
- Calmar ratio: {_metric_value(portfolio_metrics, "calmar_ratio"):.2f}
- Weekly win rate: {_metric_value(portfolio_metrics, "win_rate"):.1%}
- Worst week: {_metric_value(portfolio_metrics, "worst_week"):.1%}

ASX 200 benchmark performance:
- Annual return: {_metric_value(benchmark_metrics, "annual_return"):.1%}
- Total return: {_metric_value(benchmark_metrics, "total_return"):.1%}
- Annual volatility: {_metric_value(benchmark_metrics, "annual_volatility"):.1%}
- Sharpe ratio: {_metric_value(benchmark_metrics, "sharpe_ratio"):.2f}
- Sortino ratio: {_metric_value(benchmark_metrics, "sortino_ratio"):.2f}
- Maximum drawdown: {_metric_value(benchmark_metrics, "max_drawdown"):.1%}
- Calmar ratio: {_metric_value(benchmark_metrics, "calmar_ratio"):.2f}
- Weekly win rate: {_metric_value(benchmark_metrics, "win_rate"):.1%}
- Worst week: {_metric_value(benchmark_metrics, "worst_week"):.1%}

Alpha analysis:
- Annualised alpha: {_metric_value(alpha_metrics, "annualised_alpha"):.1%}
- Market beta: {_metric_value(alpha_metrics, "beta"):.2f}
- Market R-squared: {_metric_value(alpha_metrics, "r_squared"):.1%}
- Information ratio: {_metric_value(alpha_metrics, "information_ratio"):.2f}

Write an executive summary of the strategy using exactly three concise bullet points. Emphasise the overall performance, investment behaviour, and key conclusion of the backtest, drawing on all available metrics rather than discussing them individually. Avoid generic statements and unnecessary repetition.

Every line must begin with "- ". Do not add a heading, introduction,
conclusion, numbering, blank lines or nested bullets. Keep the complete
response below 140 words. Use plain text only, avoid Markdown bolding, and
remain objective and appropriately critical.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


def _extract_analysis_bullets(response_text: str) -> list[str]:
    """Parse bullets, numbered points or sentence-style Gemini output."""
    bullets: list[str] = []

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(
            r"^(?:[-*•]|\d+[.)])\s*(.+)$",
            line
        )

        if match:
            bullet = match.group(1).replace("**", "").strip()
            if bullet:
                bullets.append(bullet)

    if len(bullets) >= 3:
        return bullets[:3]

    plain_text = re.sub(
        r"^(?:[-*•]|\d+[.)])\s*",
        "",
        response_text,
        flags=re.MULTILINE,
    )
    plain_text = plain_text.replace("**", "").strip()

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            plain_text
        )
        if sentence.strip()
    ]

    return sentences[:3]


def _fallback_analysis_bullets(
    portfolio_metrics: dict,
    benchmark_metrics: dict,
    alpha_metrics: dict,
) -> list[str]:
    """Create three reliable observations when Gemini formatting is invalid."""
    return [
        (
            "Performance: The strategy achieved a Sharpe ratio of "
            f"{_metric_value(portfolio_metrics, 'sharpe_ratio'):.2f} "
            "versus "
            f"{_metric_value(benchmark_metrics, 'sharpe_ratio'):.2f} "
            "for the ASX 200, indicating stronger historical return per "
            "unit of volatility."
        ),
        (
            "Market relationship: A beta of "
            f"{_metric_value(alpha_metrics, 'beta'):.2f} and market "
            f"R-squared of {_metric_value(alpha_metrics, 'r_squared'):.1%} "
            "suggest that broad-market movements explained only a limited "
            "share of strategy performance."
        ),
        (
            "Limitation: Results exclude transaction costs, short-borrow "
            "fees and market impact, so realised performance would likely "
            "be lower than the reported gross backtest."
        ),
    ]


def render_backtest_analysis(
    strategy_name: str,
    feature_set: str,
    portfolio_metrics: dict,
    benchmark_metrics: dict,
    alpha_metrics: dict,
) -> None:
    """Render Gemini's three-point interpretation in one analysis box."""
    try:
        with st.spinner("Generating backtest analysis..."):
            response_text = generate_backtest_analysis(
                strategy_name=strategy_name,
                feature_set=feature_set,
                portfolio_metrics=portfolio_metrics,
                benchmark_metrics=benchmark_metrics,
                alpha_metrics=alpha_metrics,
            )

        bullets = _extract_analysis_bullets(response_text)

        if len(bullets) < 3:
            bullets = _fallback_analysis_bullets(
                portfolio_metrics=portfolio_metrics,
                benchmark_metrics=benchmark_metrics,
                alpha_metrics=alpha_metrics,
            )

        bullet_html = "".join(
            f"<li>{html.escape(bullet)}</li>"
            for bullet in bullets
        )

        st.html(
            f"""
            <div class="backtest-analysis-card">
                <div class="backtest-analysis-icon">★</div>
                <div>
                    <div class="backtest-analysis-title">
                        Analysis of Backtest Performance
                    </div>
                    <ul class="backtest-analysis-list">
                        {bullet_html}
                    </ul>
                </div>
            </div>
            """
        )

    except Exception as exc:
        st.warning(
            "The AI backtest analysis could not be generated. "
            f"Details: {exc}"
        )

def render_backtesting():
    st.set_page_config(
        page_title="ASX Alpha System - Methodology",
        page_icon="📈",
        layout="wide"
    )
    st.html(BACKTEST_PAGE_CSS)

    st.html(
        f"""
        <div class="backtest-hero">
            <h1 class="backtest-title">Backtest Performance</h1>
            <div class="backtest-description">
                Compare the selected <b>{STRATEGY_NAME}</b> strategy with the
                ASX 200 benchmark across return, downside risk, consistency
                and alpha-generation measures using aligned weekly observations.
            </div>
        </div>
        """
    )

    st.html(
        f"""
        <div class="latest-rebalance">
            Latest rebalance: {LATEST_REBALANCE_DATE:%d %B %Y}
            &nbsp;&nbsp; {STRATEGY_NAME}
        </div>
        """
    )

    render_backtest_metric_cards(
        strategy_metrics=portfolio_metrics,
        benchmark_metrics=asx_metrics_full,
        strategy_name=STRATEGY_NAME,
        benchmark_name="ASX 200"
    )

    annual_return = portfolio_metrics.get("annual_return")
    sharpe = portfolio_metrics.get("sharpe_ratio")
    max_drawdown = portfolio_metrics.get("max_drawdown")
    benchmark_return = asx_metrics_full.get("annual_return")

    st.html(
        f"""
        <div class="backtest-takeaway">
            <div class="takeaway-icon">★</div>
            <div>
                <div class="takeaway-title">Backtest Takeaway</div>
                <div class="takeaway-text">
                    The <b>{STRATEGY_NAME}</b> strategy generated an annualised
                    return of <b>{annual_return:.1%}</b> versus
                    <b>{benchmark_return:.1%}</b> for the ASX 200, with a
                    Sharpe ratio of <b>{sharpe:.2f}</b>. Maximum drawdown was
                    <b>{max_drawdown:.1%}</b>, indicating stronger return
                    generation alongside meaningful but controlled downside
                    variation. Results are pre-cost and exclude transaction
                    costs.
                </div>
            </div>
        </div>
        """
    )

    st.html(
        """
        <div class="section-header">
            <div class="section-icon">↗</div>
            <div>
                <div class="section-title">Portfolio Performance</div>
                <div class="section-caption">
                    Growth, drawdowns and weekly-return characteristics relative to the benchmark.
                </div>
            </div>
        </div>
        """
    )

    performance_col, drawdown_col = st.columns(2, gap="large")

    with performance_col:
        render_cumulative_returns(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name=STRATEGY_NAME,
            benchmark_name="ASX 200"
        )

    with drawdown_col:
        render_drawdown(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name=STRATEGY_NAME,
            benchmark_name="ASX 200"
        )

    left_column, right_column = st.columns(2, gap="large")

    with left_column:
        render_return_summary(
            strategy_metrics=portfolio_metrics,
            benchmark_metrics=asx_metrics_full,
            strategy_name=STRATEGY_NAME,
            benchmark_name="ASX 200"
        )

    with right_column:
        render_return_distribution(
            strategy_returns=strategy_returns,
            benchmark_returns=asx_returns,
            strategy_name=STRATEGY_NAME,
            benchmark_name="ASX 200"
        )

    alpha_analysis = AlphaMetrics(
        strategy_returns=strategy_returns,
        benchmark_returns=asx_returns,
        periods_per_year=52,
        risk_free_rate=0.0
    )

    alpha_metrics = alpha_analysis.get_metrics()

    st.html(
        """
        <div class="section-header">
            <div class="section-icon">α</div>
            <div>
                <div class="section-title">Alpha Analysis</div>
                <div class="section-caption">
                    Benchmark-adjusted return, market sensitivity and active-risk efficiency.
                </div>
            </div>
        </div>
        """
    )

    render_alpha_metric_cards(alpha_metrics)

    render_backtest_analysis(
        strategy_name=st.session_state.get("selected_model", STRATEGY_NAME),
        feature_set=SELECTED_FEATURE_NAME,
        portfolio_metrics=portfolio_metrics,
        benchmark_metrics=asx_metrics_full,
        alpha_metrics=alpha_metrics,
    )

