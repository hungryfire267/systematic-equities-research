from html import escape

import pandas as pd
import streamlit as st


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
    benchmark_name: str = "ASX 200"
) -> None:
    """
    Render a risk and return comparison table.
    """

    metrics = [
        ("Annual Return", "annual_return", True),
        ("Total Return", "total_return", True),
        ("Annual Volatility", "annual_volatility", True),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Sortino Ratio", "sortino_ratio", False),
        ("Maximum Drawdown", "max_drawdown", True),
        ("Calmar Ratio", "calmar_ratio", False),
        ("Weekly Win Rate", "win_rate", True),
        ("Worst Week", "worst_week", True)
    ]

    rows = []

    for label, key, as_percentage in metrics:
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

        rows.append(
            f"""
            <tr>
                <td class="metric-name">{escape(label)}</td>
                <td class="metric-value">{strategy_display}</td>
                <td class="metric-value">{benchmark_display}</td>
            </tr>
            """
        )

    rows_html = "".join(rows)

    html = f"""
        <style>
            .return-summary-card {{
                width: 100%;
                height: 520px;                 /* <-- Added */
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
                flex: 1;                       /* <-- Added */
                display: flex;                 /* <-- Added */
                flex-direction: column;        /* <-- Added */
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
                width: 40%;
                text-align: left;
                border-top-left-radius: 8px;
            }}

            .return-summary-table th:last-child {{
                border-top-right-radius: 8px;
            }}

            .return-summary-table td {{
                padding: 11px 10px;
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