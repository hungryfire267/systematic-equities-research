import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from components.overview_bar import render_metric_card
from scripts.portfolio.metrics import GetMetrics


BACKTEST_RESULTS_DIR = BASE_DIR / "results" / "backtest"


OVERVIEW_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.overview-header {
    margin-bottom: 1.25rem;
}

.overview-title {
    margin: 0;
    color: #0F172A;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.1;
}

.overview-description {
    margin-top: 0.55rem;
    max-width: 760px;
    color: #64748B;
    font-size: 0.95rem;
    line-height: 1.55;
}

.section-title {
    margin-top: 1.7rem;
    margin-bottom: 0.3rem;
    color: #0F172A;
    font-size: 1.25rem;
    font-weight: 800;
}

.section-description {
    margin-bottom: 0.9rem;
    color: #64748B;
    font-size: 0.88rem;
    line-height: 1.5;
}

.metric-card {
    min-height: 142px;
    padding: 1rem 0.8rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    box-shadow:
        0 1px 2px rgba(15, 23, 42, 0.04),
        0 5px 15px rgba(15, 23, 42, 0.035);
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
}

.metric-title {
    color: #172033;
    font-size: 0.78rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.85rem;
}

.metric-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
}

.metric-icon {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1.3rem;
    font-weight: 700;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    white-space: nowrap;
}

.metric-subtitle {
    margin-top: 0.8rem;
    color: #64748B;
    font-size: 0.72rem;
    text-align: center;
}

div[data-testid="stSegmentedControl"] {
    margin-bottom: 0.8rem;
}

@media (max-width: 1200px) {
    .metric-value {
        font-size: 1.45rem;
    }

    .metric-icon {
        width: 35px;
        height: 35px;
        font-size: 1rem;
    }
}
</style>
"""


def render_overview() -> None:
    st.html(OVERVIEW_CSS)

    st.html(
        """
        <div class="overview-header">
            <h1 class="overview-title">Overview</h1>

            <div class="overview-description">
                Systematic long/short equity strategy across the ASX 200 universe,
                driven by machine learning predictions, walk-forward validation
                and portfolio construction.
            </div>
        </div>
        """
    )

    portfolio_path = BACKTEST_RESULTS_DIR / "final_portfolio.parquet"

    if not portfolio_path.exists():
        st.error(
            f"Backtest results could not be found at: {portfolio_path}"
        )
        return

    try:
        final_portfolio_df = pd.read_parquet(portfolio_path)

        backtest_metrics, strategy_returns = GetMetrics(
            final_portfolio_df
        ).run_data()

    except Exception as error:
        st.error(f"Unable to load the overview data: {error}")
        return

    annual_return = backtest_metrics["annual_return"]
    total_return = backtest_metrics["total_return"]
    sharpe = backtest_metrics["sharpe_ratio"]
    sortino_ratio = backtest_metrics["sortino_ratio"]
    annual_volatility = backtest_metrics["annual_volatility"]
    max_drawdown = backtest_metrics["max_drawdown"]
    calmar_ratio = backtest_metrics["calmar_ratio"]
    win_rate = backtest_metrics["win_rate"]
    worst_week = backtest_metrics["worst_week"]

    metric_columns = st.columns(6, gap="medium")

    with metric_columns[0]:
        render_metric_card(
            title="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            subtitle="Risk-adjusted return",
            icon="〽",
            value_colour="#2F80ED",
            icon_colour="#2F80ED",
            icon_background="#E8F1FF",
        )

    with metric_columns[1]:
        render_metric_card(
            title="Cumulative Return",
            value=f"{total_return:+.1%}",
            subtitle="Since inception",
            icon="↗",
            value_colour="#0EAD79",
            icon_colour="#0EAD79",
            icon_background="#DDF8EE",
        )

    with metric_columns[2]:
        render_metric_card(
            title="Annual Return",
            value=f"{annual_return:.1%}",
            subtitle="Annualised performance",
            icon="↑",
            value_colour="#8B5CD6",
            icon_colour="#8B5CD6",
            icon_background="#F0E7FF",
        )

    with metric_columns[3]:
        render_metric_card(
            title="Max Drawdown",
            value=f"{max_drawdown:.1%}",
            subtitle="Largest portfolio decline",
            icon="↘",
            value_colour="#FF554D",
            icon_colour="#FF554D",
            icon_background="#FFE7E5",
        )

    with metric_columns[4]:
        render_metric_card(
            title="Sortino Ratio",
            value=f"{sortino_ratio:.2f}",
            subtitle="Downside-adjusted return",
            icon="◎",
            value_colour="#F79009",
            icon_colour="#F79009",
            icon_background="#FFF0D8",
        )

    with metric_columns[5]:
        render_metric_card(
            title="Winning Weeks",
            value=f"{win_rate:.0%}",
            subtitle="Percentage of positive weeks",
            icon="🏆",
            value_colour="#11AEB4",
            icon_colour="#11AEB4",
            icon_background="#DDF7F6",
        )

    st.html(
        """
        <div class="section-title">Strategy Performance</div>
        <div class="section-description">
            View cumulative growth, portfolio drawdown or weekly returns
            across the backtest period.
        </div>
        """
    )

    plot_choice = st.segmented_control(
        "Performance View",
        options=[
            "Equity Curve",
            "Drawdown",
            "Weekly Returns",
        ],
        default="Equity Curve",
        label_visibility="collapsed",
    )

    chart_df = strategy_returns.copy()

    if "Date" not in chart_df.columns:
        chart_df = chart_df.reset_index()

    if "Date" not in chart_df.columns:
        st.error("The strategy return data must contain a Date column.")
        return

    if "portfolio_return" not in chart_df.columns:
        st.error(
            "The strategy return data must contain a "
            "'portfolio_return' column."
        )
        return

    chart_df["Date"] = pd.to_datetime(chart_df["Date"])

    chart_df = (
        chart_df
        .sort_values("Date")
        .dropna(subset=["portfolio_return"])
    )

    chart_df["equity_curve"] = (
        1 + chart_df["portfolio_return"]
    ).cumprod()

    chart_df["drawdown"] = (
        chart_df["equity_curve"]
        / chart_df["equity_curve"].cummax()
        - 1
    )

    if plot_choice == "Equity Curve":
        st.line_chart(
            chart_df,
            x="Date",
            y="equity_curve",
            height=420,
            use_container_width=True,
        )

    elif plot_choice == "Drawdown":
        st.line_chart(
            chart_df,
            x="Date",
            y="drawdown",
            height=420,
            use_container_width=True,
        )

    else:
        st.bar_chart(
            chart_df,
            x="Date",
            y="portfolio_return",
            height=420,
            use_container_width=True,
        )

    st.html(
        """
        <div class="section-title">Additional Statistics</div>
        <div class="section-description">
            Supporting measures for portfolio volatility and downside behaviour.
        </div>
        """
    )

    additional_columns = st.columns(3)

    additional_columns[0].metric(
        label="Annual Volatility",
        value=f"{annual_volatility:.2%}",
    )

    additional_columns[1].metric(
        label="Calmar Ratio",
        value=f"{calmar_ratio:.2f}",
    )

    additional_columns[2].metric(
        label="Worst Week",
        value=f"{worst_week:.2%}",
    )