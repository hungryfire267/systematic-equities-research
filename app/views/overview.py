import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.portfolio.metrics import GetMetrics


MODEL_DIRECTORIES = {
    "dt": BASE_DIR / "results" / "backtest" / "dt",
    "lightgbm": BASE_DIR / "results" / "backtest" / "lightgbm",
    "xgboost": BASE_DIR / "results" / "backtest" / "xgboost",
}


def _load_selected_strategy() -> tuple[dict, pd.DataFrame, dict]:
    """
    Load the portfolio corresponding to the model-feature configuration
    selected on the Model Comparison page.

    Falls back to LightGBM + Stock Features when no selection has yet
    been stored in Streamlit session state.
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
            {
                "dt": "Decision Tree",
                "lightgbm": "LightGBM",
                "xgboost": "XGBoost",
            }.get(model_key, model_key)
        )
    )

    feature_name = selected_configuration.get(
        "feature_name",
        st.session_state.get(
            "selected_feature_set",
            {
                "stock": "Stock Features",
                "market": "Stock + Market",
            }.get(feature_key, feature_key)
        )
    )

    if model_key not in MODEL_DIRECTORIES:
        raise KeyError(
            f"Unknown selected model key: {model_key}. "
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

    portfolio_df = pd.read_parquet(portfolio_path)

    portfolio_metrics, portfolio_returns = GetMetrics(
        portfolio_df
    ).run_data()

    resolved_configuration = {
        **selected_configuration,
        "model_key": model_key,
        "model_name": model_name,
        "feature_key": feature_key,
        "feature_name": feature_name,
        "portfolio_path": str(portfolio_path),
    }

    # Keep every page synchronised with the resolved active strategy.
    st.session_state["selected_configuration"] = resolved_configuration
    st.session_state["selected_model"] = model_name
    st.session_state["selected_feature_set"] = feature_name
    st.session_state["selected_model_key"] = model_key
    st.session_state["selected_feature_key"] = feature_key

    return portfolio_metrics, portfolio_returns, resolved_configuration



def _fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}%}"


def _fmt_num(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _metric_card(
    title: str,
    value: str,
    subtitle: str,
    icon: str,
    accent: str,
    soft_background: str,
) -> None:
    st.markdown(
        f"""
        <div class="overview-metric-card" style="
            --metric-accent:{accent};
            --metric-soft:{soft_background};
        ">
            <div class="overview-metric-topline"></div>
            <div class="overview-metric-label">{title}</div>
            <div class="overview-metric-main">
                <div class="overview-metric-icon">{icon}</div>
                <div class="overview-metric-value">{value}</div>
            </div>
            <div class="overview-metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _supporting_card(
    title: str,
    value: str,
    note: str,
    accent: str,
    background: str,
) -> None:
    st.markdown(
        f"""
        <div class="supporting-stat-card" style="
            --support-accent:{accent};
            --support-bg:{background};
        ">
            <div class="supporting-stat-label">{title}</div>
            <div class="supporting-stat-value">{value}</div>
            <div class="supporting-stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _create_equity_curve(portfolio_returns: pd.DataFrame) -> go.Figure:
    df = portfolio_returns.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "equity_curve" not in df.columns:
        df["equity_curve"] = (1 + df["portfolio_return"].fillna(0)).cumprod()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["equity_curve"],
            mode="lines",
            name="Portfolio",
            line=dict(color="#2563EB", width=3),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate=(
                "<b>Portfolio</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Equity: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Portfolio value",
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#64748B"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color="#64748B"),
    )

    return fig


def _create_drawdown_chart(portfolio_returns: pd.DataFrame) -> go.Figure:
    df = portfolio_returns.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "equity_curve" not in df.columns:
        df["equity_curve"] = (1 + df["portfolio_return"].fillna(0)).cumprod()

    rolling_peak = df["equity_curve"].cummax()
    df["drawdown"] = df["equity_curve"] / rolling_peak - 1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["drawdown"],
            mode="lines",
            line=dict(color="#EF4444", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.12)",
            hovertemplate=(
                "<b>Drawdown</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Drawdown: %{y:.2%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#64748B"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color="#64748B"),
    )

    return fig


def _create_weekly_return_chart(portfolio_returns: pd.DataFrame) -> go.Figure:
    df = portfolio_returns.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    bar_colours = np.where(
        df["portfolio_return"] >= 0,
        "#10B981",
        "#EF4444",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["portfolio_return"],
            marker_color=bar_colours,
            hovertemplate=(
                "<b>Weekly Return</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Return: %{y:.2%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Weekly return",
        yaxis_tickformat=".0%",
        bargap=0.22,
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#64748B"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.35)",
        tickfont=dict(color="#64748B"),
    )

    return fig


def render_overview(
    strategy_name: str = "Systematic ASX Equities Alpha Generation Platform",
) -> None:
    """
    Render a colourful portfolio overview.

    Expected portfolio_metrics keys:
        sharpe_ratio
        annual_return
        max_drawdown
        sortino_ratio
        win_rate
        annual_volatility
        calmar_ratio
        worst_week

    Expected portfolio_returns columns:
        Date
        portfolio_return

    Optional:
        equity_curve
    """

    # Always reload the portfolio selected on the Model Comparison page.
    # This prevents old Decision Tree metrics passed by the calling page
    # from overriding the current LightGBM strategy.
    (
        portfolio_metrics,
        portfolio_returns,
        selected_configuration,
    ) = _load_selected_strategy()

    selected_model = selected_configuration.get(
        "model_name",
        st.session_state.get("selected_model", "LightGBM")
    )

    selected_feature_set = selected_configuration.get(
        "feature_name",
        st.session_state.get(
            "selected_feature_set",
            "Stock Features"
        )
    )

    selected_model_key = selected_configuration.get(
        "model_key",
        st.session_state.get("selected_model_key", "lightgbm")
    )

    selected_feature_key = selected_configuration.get(
        "feature_key",
        st.session_state.get("selected_feature_key", "stock")
    )

    feature_descriptions = {
        "stock": "stock-specific and industry-level features",
        "market": "stock-specific, industry and broader market features",
    }

    active_feature_description = feature_descriptions.get(
        selected_feature_key,
        selected_feature_set.lower()
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .overview-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 86% 18%, rgba(139,92,246,0.18), transparent 27%),
                radial-gradient(circle at 70% 100%, rgba(14,165,233,0.16), transparent 33%),
                linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 42%, #F5F3FF 100%);
            border: 1px solid #DCE7F5;
            border-radius: 22px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 12px 34px rgba(37, 99, 235, 0.08);
        }


        .overview-title {
            margin: 0;
            color: #0F172A;
            font-size: 2.25rem;
            font-weight: 850;
            line-height: 1.08;
            max-width: 980px;
        }

        .overview-description {
            color: #52647A;
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 940px;
            margin-top: 0.72rem;
            margin-bottom: 1rem;
        }




        .executive-summary-card {
            margin-top: 0.15rem;
            margin-bottom: 1.45rem;
            padding: 1rem 1.2rem;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #DCE7F5;
            border-left: 4px solid #2563EB;
            border-radius: 14px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
        }

        .executive-summary-title {
            color: #0F172A;
            font-size: 0.95rem;
            font-weight: 850;
            margin-bottom: 0.35rem;
        }

        .executive-summary-text {
            color: #52647A;
            font-size: 0.84rem;
            line-height: 1.6;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
        }

        .section-title {
            color: #0F172A;
            font-size: 1.3rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .section-caption {
            color: #64748B;
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }

        .overview-metric-card {
            position: relative;
            overflow: hidden;
            min-height: 150px;
            background:
                linear-gradient(145deg, #FFFFFF 0%, var(--metric-soft) 155%);
            border: 1px solid color-mix(in srgb, var(--metric-accent) 25%, #E2E8F0);
            border-radius: 18px;
            padding: 1.05rem 1rem 0.95rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .overview-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
        }

        .overview-metric-topline {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: var(--metric-accent);
        }

        .overview-metric-label {
            color: #334155;
            font-size: 0.78rem;
            font-weight: 800;
            text-align: center;
            margin-top: 0.15rem;
            margin-bottom: 0.8rem;
        }

        .overview-metric-main {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.75rem;
        }

        .overview-metric-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 2.45rem;
            height: 2.45rem;
            border-radius: 999px;
            background: var(--metric-soft);
            color: var(--metric-accent);
            font-size: 1.12rem;
            font-weight: 800;
        }

        .overview-metric-value {
            color: var(--metric-accent);
            font-size: 1.58rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: -0.02em;
        }

        .overview-metric-subtitle {
            color: #64748B;
            font-size: 0.72rem;
            line-height: 1.4;
            text-align: center;
        }

        .performance-shell {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 1rem 1.1rem 0.5rem;
            box-shadow: 0 7px 24px rgba(15, 23, 42, 0.05);
        }

        .takeaway-card {
            background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
            border: 1px solid #BFDBFE;
            border-left: 5px solid #6366F1;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin-top: 0.85rem;
        }

        .takeaway-title {
            color: #1E3A8A;
            font-size: 0.9rem;
            font-weight: 850;
            margin-bottom: 0.35rem;
        }

        .takeaway-text {
            color: #334155;
            font-size: 0.86rem;
            line-height: 1.62;
        }

        .supporting-stat-card {
            min-height: 126px;
            background: linear-gradient(145deg, #FFFFFF 0%, var(--support-bg) 150%);
            border: 1px solid color-mix(in srgb, var(--support-accent) 22%, #E2E8F0);
            border-radius: 17px;
            padding: 1rem 1.05rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
        }

        .supporting-stat-label {
            color: #64748B;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .supporting-stat-value {
            color: var(--support-accent);
            font-size: 1.42rem;
            font-weight: 850;
            line-height: 1.1;
            margin-bottom: 0.42rem;
        }

        .supporting-stat-note {
            color: #64748B;
            font-size: 0.7rem;
            line-height: 1.4;
        }

        .snapshot-card {
            min-height: 138px;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 5px 18px rgba(15,23,42,0.045);
        }

        .snapshot-blue {
            background: linear-gradient(145deg, #EFF6FF 0%, #FFFFFF 115%);
            border-color: #BFDBFE;
        }

        .snapshot-purple {
            background: linear-gradient(145deg, #F5F3FF 0%, #FFFFFF 115%);
            border-color: #DDD6FE;
        }

        .snapshot-green {
            background: linear-gradient(145deg, #ECFDF5 0%, #FFFFFF 115%);
            border-color: #A7F3D0;
        }

        .snapshot-orange {
            background: linear-gradient(145deg, #FFF7ED 0%, #FFFFFF 115%);
            border-color: #FED7AA;
        }

        .snapshot-icon {
            font-size: 1.2rem;
            margin-bottom: 0.45rem;
        }

        .snapshot-title {
            color: #0F172A;
            font-size: 0.88rem;
            font-weight: 850;
            margin-bottom: 0.3rem;
        }

        .snapshot-text {
            color: #64748B;
            font-size: 0.74rem;
            line-height: 1.48;
        }

        div[data-testid="stSegmentedControl"] {
            margin-bottom: 0.65rem;
        }

        @media (max-width: 900px) {
            .overview-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sharpe = float(portfolio_metrics["sharpe_ratio"])
    annual_return = float(portfolio_metrics["annual_return"])
    max_drawdown = float(portfolio_metrics["max_drawdown"])
    sortino = float(portfolio_metrics["sortino_ratio"])
    win_rate = float(portfolio_metrics["win_rate"])
    annual_volatility = float(portfolio_metrics["annual_volatility"])
    calmar = float(portfolio_metrics["calmar_ratio"])
    worst_week = float(portfolio_metrics["worst_week"])

    st.markdown(
        f"""
        <div class="overview-hero">
            <h1 class="overview-title">{strategy_name}</h1>
            <div class="overview-description">
                A systematic research platform for testing machine-learning
                stock-selection signals and translating forecasts into an
                out-of-sample long–short equity strategy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="executive-summary-card">
            <div class="executive-summary-title">Executive Summary</div>
            <div class="executive-summary-text">
                This dashboard presents the automatically selected
                <b>{selected_model} + {selected_feature_set}</b> configuration
                for the systematic long–short ASX 200 strategy. All portfolio
                metrics and charts below are loaded from this selected
                model-feature backtest. The <b>{selected_model}</b>
                cross-sectional regression model uses
                {active_feature_description} to predict each stock’s five-day
                forward return. Stocks are ranked by their forecasts
                and used to form a weekly rebalanced, dollar-neutral portfolio
                under expanding walk-forward validation. Reported results are
                pre-cost and currently exclude transaction costs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Portfolio Performance</div>
                <div class="section-caption">
                    Core return, risk and consistency measures from the selected backtest.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)

    cards = [
        (
            "Sharpe Ratio",
            _fmt_num(sharpe),
            "Risk-adjusted return",
            "↗",
            "#2563EB",
            "#DBEAFE",
        ),
        (
            "Annual Return",
            _fmt_pct(annual_return),
            "Annualised performance",
            "◆",
            "#7C3AED",
            "#EDE9FE",
        ),
        (
            "Max Drawdown",
            _fmt_pct(max_drawdown),
            "Largest portfolio decline",
            "↘",
            "#EF4444",
            "#FEE2E2",
        ),
        (
            "Sortino Ratio",
            _fmt_num(sortino),
            "Downside-adjusted return",
            "◎",
            "#EA580C",
            "#FFEDD5",
        ),
        (
            "Winning Weeks",
            _fmt_pct(win_rate, 0),
            "Percentage of positive weeks",
            "★",
            "#0891B2",
            "#CFFAFE",
        ),
    ]

    for column, card in zip(metric_columns, cards):
        with column:
            _metric_card(*card)

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Strategy Performance</div>
                <div class="section-caption">
                    Explore cumulative growth, portfolio drawdowns and weekly return behaviour.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_view = st.segmented_control(
        "Performance view",
        options=["Equity Curve", "Drawdown", "Weekly Returns"],
        default="Equity Curve",
        label_visibility="collapsed",
    )

    with st.container(border=False):
        st.markdown('<div class="performance-shell">', unsafe_allow_html=True)

        if selected_view == "Drawdown":
            chart = _create_drawdown_chart(portfolio_returns)
        elif selected_view == "Weekly Returns":
            chart = _create_weekly_return_chart(portfolio_returns)
        else:
            chart = _create_equity_curve(portfolio_returns)

        st.plotly_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return_quality = (
        "strong"
        if sharpe >= 1.5
        else "solid"
        if sharpe >= 1
        else "positive but moderate"
    )

    st.markdown(
        f"""
        <div class="takeaway-card">
            <div class="takeaway-title">★ Performance Takeaway</div>
            <div class="takeaway-text">
                The strategy delivered a {return_quality} risk-adjusted result,
                with a Sharpe ratio of <b>{sharpe:.2f}</b>, annualised return
                of <b>{annual_return:.1%}</b> and winning-week rate of
                <b>{win_rate:.0%}</b>. The return profile is encouraging,
                although annual volatility of <b>{annual_volatility:.1%}</b>
                and maximum drawdown of <b>{max_drawdown:.1%}</b> show that
                the strategy still experiences meaningful variation through
                time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Additional Statistics</div>
                <div class="section-caption">
                    Supporting measures describing volatility, downside risk and return efficiency.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    supporting_columns = st.columns(4)

    supporting_cards = [
        (
            "Annual Volatility",
            _fmt_pct(annual_volatility),
            "Annualised dispersion of weekly returns",
            "#2563EB",
            "#EFF6FF",
        ),
        (
            "Sortino Ratio",
            _fmt_num(sortino),
            "Return relative to downside volatility",
            "#EA580C",
            "#FFF7ED",
        ),
        (
            "Calmar Ratio",
            _fmt_num(calmar),
            "Annual return relative to maximum drawdown",
            "#7C3AED",
            "#F5F3FF",
        ),
        (
            "Worst Week",
            _fmt_pct(worst_week),
            "Largest single-week portfolio loss",
            "#DC2626",
            "#FEF2F2",
        ),
    ]

    for column, card in zip(supporting_columns, supporting_cards):
        with column:
            _supporting_card(*card)

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Strategy Snapshot</div>
                <div class="section-caption">
                    A concise overview of how predictions become portfolio decisions.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot_columns = st.columns(4)

    snapshot_cards = [
        (
            "snapshot-blue",
            "📈",
            "Cross-Sectional Forecasting",
            "Models estimate each stock's five-day return and rank the investable universe.",
        ),
        (
            "snapshot-purple",
            "🕒",
            "Walk-Forward Validation",
            "Each forecast uses only information available before its rebalance date.",
        ),
        (
            "snapshot-green",
            "⚖️",
            "Dollar-Neutral Portfolio",
            "Long and short books are formed from the strongest and weakest predictions.",
        ),
        (
            "snapshot-orange",
            "🔁",
            "Weekly Rebalancing",
            "Positions are refreshed each week and assessed using realised forward returns.",
        ),
    ]

    for column, (class_name, icon, title, text) in zip(
        snapshot_columns,
        snapshot_cards,
    ):
        with column:
            st.markdown(
                f"""
                <div class="snapshot-card {class_name}">
                    <div class="snapshot-icon">{icon}</div>
                    <div class="snapshot-title">{title}</div>
                    <div class="snapshot-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
