from html import escape

import pandas as pd
import streamlit as st


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
                <div class="headline-metric-top">
                    <div class="headline-metric-symbol">
                        {metric["symbol"]}
                    </div>

                    <div class="headline-metric-label">
                        {escape(metric["label"])}
                    </div>
                </div>

                <div class="headline-metric-value">
                    {formatted_value}
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

            .headline-metric-card {{
                min-width: 0;
                padding: 16px;
                border-radius: 13px;
                box-sizing: border-box;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
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
                gap: 8px;
                min-width: 0;
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
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                line-height: 1.3;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}

            .headline-metric-value {{
                margin-top: 14px;
                color: #0F172A;
                font-size: 25px;
                font-weight: 800;
                line-height: 1;
                font-variant-numeric: tabular-nums;
            }}

            .headline-metric-strategy {{
                margin-top: 6px;
                color: #64748B;
                font-size: 11px;
                font-weight: 600;
            }}

            .headline-metric-delta {{
                margin-top: 11px;
                font-size: 11px;
                font-weight: 700;
                line-height: 1.3;
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
            <div class="headline-metrics-header">
                <div class="headline-metrics-title">
                    Performance Highlights
                </div>

                <div class="headline-metrics-caption">
                    Headline {escape(strategy_name)} results compared with
                    the {escape(benchmark_name)}.
                </div>
            </div>

            <div class="headline-metrics-grid">
                {cards_html}
            </div>
        </div>
        """
    )