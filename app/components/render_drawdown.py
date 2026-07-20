import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DRAWDOWN_COLOURS = {
    "strategy": "#EF4444",
    "benchmark": "#F97316"
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
            fillcolor="rgba(239, 68, 68, 0.12)",
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

    _add_final_drawdown_annotation(
        figure=figure,
        series=drawdowns["Strategy"],
        colour=DRAWDOWN_COLOURS["strategy"]
    )

    _add_final_drawdown_annotation(
        figure=figure,
        series=drawdowns["Benchmark"],
        colour=DRAWDOWN_COLOURS["benchmark"]
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