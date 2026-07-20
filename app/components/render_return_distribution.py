import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde


MODEL_COLOURS = {
    "strategy": "#10B981",
    "benchmark": "#2563EB"
}

# Use this same height for the summary-table container.
BACKTEST_CARD_HEIGHT = 520


def _extract_return_series(
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
                color=MODEL_COLOURS["strategy"],
                line=dict(
                    color=MODEL_COLOURS["strategy"],
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
                color=MODEL_COLOURS["benchmark"],
                line=dict(
                    color=MODEL_COLOURS["benchmark"],
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
        colour=MODEL_COLOURS["strategy"],
        x_min=x_min,
        x_max=x_max
    )

    _add_kde_trace(
        figure=figure,
        returns=returns_df["Benchmark"],
        name=benchmark_name,
        colour=MODEL_COLOURS["benchmark"],
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
        height=245,
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
                color=MODEL_COLOURS["strategy"],
                size=5
            ),
            line=dict(
                color=MODEL_COLOURS["strategy"],
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
            fillcolor="rgba(37, 99, 235, 0.40)",
            marker=dict(
                color=MODEL_COLOURS["benchmark"],
                size=5
            ),
            line=dict(
                color=MODEL_COLOURS["benchmark"],
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
        height=175,
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