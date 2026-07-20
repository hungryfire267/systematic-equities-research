import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PERFORMANCE_COLOURS = {
    "strategy": "#2563EB",
    "benchmark": "#F97316",
    "cash": "#64748B"
}


def _extract_return_series(
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

    strategy_series = _extract_return_series(
        returns=strategy_returns,
        output_name="Strategy",
        return_col=strategy_return_col
    )

    benchmark_series = _extract_return_series(
        returns=benchmark_returns,
        output_name="Benchmark",
        return_col=benchmark_return_col
    )

    series_to_align = [
        strategy_series,
        benchmark_series
    ]

    if cash_returns is not None:
        cash_series = _extract_return_series(
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

    _add_final_value_annotation(
        figure=figure,
        series=cumulative_returns["Strategy"],
        colour=PERFORMANCE_COLOURS["strategy"]
    )

    _add_final_value_annotation(
        figure=figure,
        series=cumulative_returns["Benchmark"],
        colour=PERFORMANCE_COLOURS["benchmark"]
    )

    if "Cash" in cumulative_returns.columns:
        _add_final_value_annotation(
            figure=figure,
            series=cumulative_returns["Cash"],
            colour=PERFORMANCE_COLOURS["cash"]
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
        height=390,
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

    st.markdown(
        """
        <style>
            div[data-testid="stPlotlyChart"] {
                margin-top: -0.45rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
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