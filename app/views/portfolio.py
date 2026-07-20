import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

BACKTEST_RESULTS_DIR = BASE_DIR / "results" / "backtest"
PORTFOLIO_PATH = BACKTEST_RESULTS_DIR / "final_portfolio.parquet"
UNIVERSE_PATH = BASE_DIR / "data" / "asx_companies.csv"


PORTFOLIO_CSS = """
<style>
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-left: 1.6rem;
    padding-right: 1.6rem;
    padding-bottom: 3rem;
}

/* Hero */
.portfolio-hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 88% 15%, rgba(16,185,129,0.18), transparent 28%),
        radial-gradient(circle at 72% 105%, rgba(37,99,235,0.15), transparent 35%),
        linear-gradient(135deg, #ECFDF5 0%, #F8FAFC 48%, #EFF6FF 100%);
    border: 1px solid #CFE8E0;
    border-radius: 22px;
    padding: 1.75rem 1.9rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 34px rgba(15, 118, 110, 0.08);
}

.portfolio-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 999px;
    padding: 0.38rem 0.72rem;
    color: #047857;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.portfolio-title {
    margin: 0;
    color: #0F172A;
    font-size: 2.25rem;
    font-weight: 850;
    line-height: 1.08;
}

.portfolio-description {
    margin-top: 0.65rem;
    max-width: 920px;
    color: #52647A;
    font-size: 0.96rem;
    line-height: 1.62;
}

.portfolio-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.95rem;
}

.portfolio-tag {
    background: rgba(255,255,255,0.84);
    border: 1px solid #D6E5E1;
    border-radius: 999px;
    padding: 0.4rem 0.72rem;
    color: #334155;
    font-size: 0.77rem;
    font-weight: 750;
}

.latest-date {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.85rem;
    padding: 0.38rem 0.68rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.72);
    border: 1px solid #DCE7E4;
    color: #64748B;
    font-size: 0.73rem;
    font-weight: 700;
}

/* Section headings */
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
    flex: 0 0 auto;
    width: 2.15rem;
    height: 2.15rem;
    border-radius: 12px;
    background: #EFF6FF;
    color: #2563EB;
    font-size: 1rem;
}

.section-title {
    margin: 0;
    color: #0F172A;
    font-size: 1.25rem;
    font-weight: 850;
    line-height: 1.2;
}

.section-description {
    margin-top: 0.22rem;
    margin-bottom: 0;
    color: #64748B;
    font-size: 0.83rem;
    line-height: 1.45;
}

/* Headline metric cards */
.portfolio-card {
    --card-accent: #2563EB;
    --card-soft: #DBEAFE;
    position: relative;
    overflow: hidden;
    min-height: 145px;
    padding: 1rem 0.8rem 0.9rem;
    border-radius: 18px;
    border: 1px solid color-mix(in srgb, var(--card-accent) 24%, #E2E8F0);
    background: linear-gradient(145deg, #FFFFFF 0%, var(--card-soft) 160%);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
    display: flex;
    flex-direction: column;
    align-items: center;
    box-sizing: border-box;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.portfolio-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
}

.portfolio-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 5px;
    background: var(--card-accent);
}

.portfolio-card-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    margin-top: 0.15rem;
    margin-bottom: 0.55rem;
    border-radius: 999px;
    background: var(--card-soft);
    color: var(--card-accent);
    font-size: 1rem;
    font-weight: 850;
}

.portfolio-card-title {
    color: #334155;
    font-size: 0.72rem;
    font-weight: 800;
    text-align: center;
}

.portfolio-card-value {
    margin-top: 0.38rem;
    color: var(--card-accent);
    font-size: 1.62rem;
    font-weight: 850;
    line-height: 1;
    text-align: center;
    letter-spacing: -0.02em;
}

.portfolio-card-subtitle {
    margin-top: 0.5rem;
    color: #64748B;
    font-size: 0.67rem;
    line-height: 1.35;
    text-align: center;
}

.card-green { --card-accent: #059669; --card-soft: #D1FAE5; }
.card-red { --card-accent: #DC2626; --card-soft: #FEE2E2; }
.card-blue { --card-accent: #2563EB; --card-soft: #DBEAFE; }
.card-teal { --card-accent: #0D9488; --card-soft: #CCFBF1; }
.card-purple { --card-accent: #7C3AED; --card-soft: #EDE9FE; }
.card-orange { --card-accent: #EA580C; --card-soft: #FFEDD5; }

/* Takeaway */
.portfolio-takeaway {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.85rem;
    align-items: flex-start;
    margin-top: 0.9rem;
    padding: 1rem 1.15rem;
    border-radius: 16px;
    border: 1px solid #A7F3D0;
    border-left: 5px solid #10B981;
    background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
}

.takeaway-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 999px;
    background: #D1FAE5;
    color: #047857;
    font-size: 1.05rem;
}

.takeaway-title {
    color: #065F46;
    font-size: 0.88rem;
    font-weight: 850;
    margin-bottom: 0.3rem;
}

.takeaway-text {
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.55;
}

/* Chart panels */
.panel-label {
    margin-bottom: 0.2rem;
    color: #0F172A;
    font-size: 0.95rem;
    font-weight: 850;
}

.panel-caption {
    margin-bottom: 0.45rem;
    color: #64748B;
    font-size: 0.74rem;
}

div[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 0.55rem;
    box-shadow: 0 7px 22px rgba(15, 23, 42, 0.045);
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
}

div[data-testid="stSegmentedControl"] {
    margin-bottom: 0.55rem;
}

.long-heading,
.short-heading {
    padding: 0.65rem 0.85rem;
    border-radius: 12px 12px 0 0;
    font-size: 0.88rem;
    font-weight: 850;
    margin-bottom: 0.35rem;
}

.long-heading {
    color: #047857;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
}

.short-heading {
    color: #B91C1C;
    background: #FEF2F2;
    border: 1px solid #FECACA;
}

@media (max-width: 1200px) {
    .portfolio-card-value {
        font-size: 1.38rem;
    }

    .portfolio-card-subtitle {
        font-size: 0.62rem;
    }
}

@media (max-width: 900px) {
    .portfolio-title {
        font-size: 1.8rem;
    }
}
</style>
"""


def render_portfolio_card(
    title: str,
    value: str,
    subtitle: str,
    card_class: str,
) -> None:
    icons = {
        "Long Positions": "↗",
        "Short Positions": "↘",
        "Gross Exposure": "◎",
        "Net Exposure": "⚖",
        "Largest Long": "▲",
        "Largest Short": "▼",
    }

    icon = icons.get(title, "•")

    st.html(
        f"""
        <div class="portfolio-card {card_class}">
            <div class="portfolio-card-icon">{icon}</div>
            <div class="portfolio-card-title">{title}</div>
            <div class="portfolio-card-value">{value}</div>
            <div class="portfolio-card-subtitle">{subtitle}</div>
        </div>
        """
    )


def find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str | None:
    for column in possible_names:
        if column in dataframe.columns:
            return column

    return None


def get_latest_portfolio(
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_df = portfolio_df.copy()

    if "Date" not in portfolio_df.columns:
        portfolio_df = portfolio_df.reset_index()

    if "Date" not in portfolio_df.columns:
        raise ValueError(
            "The portfolio data must contain a Date column."
        )

    portfolio_df["Date"] = pd.to_datetime(
        portfolio_df["Date"]
    )

    latest_date = portfolio_df["Date"].max()

    return (
        portfolio_df.loc[
            portfolio_df["Date"] == latest_date
        ]
        .copy()
        .reset_index(drop=True)
    )


def attach_sector_data(
    portfolio_df: pd.DataFrame,
    ticker_col: str,
) -> pd.DataFrame:
    portfolio_df = portfolio_df.copy()

    if not UNIVERSE_PATH.exists():
        portfolio_df["industry"] = "Unknown"
        return portfolio_df

    universe_df = pd.read_csv(UNIVERSE_PATH)

    if (
        "asxCode" not in universe_df.columns
        or "industry" not in universe_df.columns
    ):
        portfolio_df["industry"] = "Unknown"
        return portfolio_df

    universe_df["asxCode"] = (
        universe_df["asxCode"]
        .astype(str)
        .str.strip()
        .str.replace(".AX", "", regex=False)
        + ".AX"
    )

    portfolio_df[ticker_col] = (
        portfolio_df[ticker_col]
        .astype(str)
        .str.strip()
    )

    portfolio_df = portfolio_df.merge(
        universe_df[
            [
                "asxCode",
                "industry",
            ]
        ],
        left_on=ticker_col,
        right_on="asxCode",
        how="left",
    )

    portfolio_df["industry"] = (
        portfolio_df["industry"]
        .fillna("Unknown")
    )

    return portfolio_df


def build_sector_composition(
    portfolio_df: pd.DataFrame,
    weight_col: str,
) -> pd.DataFrame:
    sector_df = portfolio_df.copy()

    sector_df["Absolute Weight"] = (
        sector_df[weight_col].abs()
    )

    sector_df = (
        sector_df
        .groupby(
            "industry",
            as_index=False,
        )["Absolute Weight"]
        .sum()
        .query("`Absolute Weight` > 0")
        .sort_values(
            "Absolute Weight",
            ascending=False,
        )
    )

    total_weight = sector_df["Absolute Weight"].sum()

    if total_weight == 0:
        return sector_df

    sector_df["Gross Exposure (%)"] = (
        sector_df["Absolute Weight"]
        / total_weight
        * 100
    )

    if len(sector_df) > 7:
        top_sectors = sector_df.head(6).copy()

        other_exposure = (
            sector_df
            .iloc[6:]["Gross Exposure (%)"]
            .sum()
        )

        other_weight = (
            sector_df
            .iloc[6:]["Absolute Weight"]
            .sum()
        )

        other_row = pd.DataFrame(
            {
                "industry": ["Other"],
                "Absolute Weight": [other_weight],
                "Gross Exposure (%)": [other_exposure],
            }
        )

        sector_df = pd.concat(
            [
                top_sectors,
                other_row,
            ],
            ignore_index=True,
        )

    return sector_df


def render_portfolio() -> None:
    st.html(PORTFOLIO_CSS)

    st.html(
        """
        <div class="portfolio-hero">
            <div class="portfolio-kicker">● Current portfolio allocation</div>
            <h1 class="portfolio-title">Portfolio</h1>

            <div class="portfolio-description">
                Explore the latest long and short positions, portfolio weights,
                sector composition and realised return contributions generated
                by the systematic ASX equity strategy.
            </div>

            <div class="portfolio-tags">
                <span class="portfolio-tag">Dollar Neutral</span>
                <span class="portfolio-tag">Weekly Rebalancing</span>
                <span class="portfolio-tag">Long / Short</span>
                <span class="portfolio-tag">Prediction Ranked</span>
                <span class="portfolio-tag">Sector Monitored</span>
            </div>
        </div>
        """
    )

    if not PORTFOLIO_PATH.exists():
        st.error(
            f"Portfolio results could not be found at: "
            f"{PORTFOLIO_PATH}"
        )
        return

    try:
        portfolio_df = pd.read_parquet(PORTFOLIO_PATH)
        latest_portfolio = get_latest_portfolio(
            portfolio_df
        )

    except Exception as error:
        st.error(
            f"Unable to load portfolio data: {error}"
        )
        return

    ticker_col = find_column(
        latest_portfolio,
        [
            "Ticker",
            "ticker",
            "Symbol",
            "symbol",
        ],
    )

    weight_col = find_column(
        latest_portfolio,
        [
            "weight",
            "portfolio_weight",
            "final_weight",
            "Weight",
        ],
    )

    prediction_col = find_column(
        latest_portfolio,
        [
            "prediction",
            "predicted_return",
            "future_return_prediction",
            "Predicted Return",
        ],
    )

    realised_return_col = find_column(
        latest_portfolio,
        [
            "future_return_5d",
            "realised_return",
            "portfolio_return",
            "return",
        ],
    )

    if ticker_col is None:
        st.error(
            "The portfolio data must contain a ticker column."
        )
        return

    if weight_col is None:
        st.error(
            "The portfolio data must contain a portfolio "
            "weight column."
        )
        return

    latest_portfolio[weight_col] = pd.to_numeric(
        latest_portfolio[weight_col],
        errors="coerce",
    ).fillna(0)

    if prediction_col is not None:
        latest_portfolio[prediction_col] = pd.to_numeric(
            latest_portfolio[prediction_col],
            errors="coerce",
        )

    if realised_return_col is not None:
        latest_portfolio[realised_return_col] = (
            pd.to_numeric(
                latest_portfolio[realised_return_col],
                errors="coerce",
            )
        )

    try:
        latest_portfolio = attach_sector_data(
            latest_portfolio,
            ticker_col,
        )

    except Exception as error:
        st.warning(
            f"Unable to attach sector data: {error}"
        )
        latest_portfolio["industry"] = "Unknown"

    latest_date = latest_portfolio["Date"].max()

    active_positions = latest_portfolio.loc[
        latest_portfolio[weight_col] != 0
    ].copy()

    long_positions = active_positions.loc[
        active_positions[weight_col] > 0
    ].copy()

    short_positions = active_positions.loc[
        active_positions[weight_col] < 0
    ].copy()

    gross_exposure = (
        active_positions[weight_col]
        .abs()
        .sum()
    )

    net_exposure = (
        active_positions[weight_col]
        .sum()
    )

    largest_long = (
        long_positions[weight_col].max()
        if not long_positions.empty
        else 0
    )

    largest_short = (
        short_positions[weight_col].min()
        if not short_positions.empty
        else 0
    )

    st.html(
        f"""
        <div class="latest-date">
            🗓 Latest rebalance: {latest_date:%d %B %Y}
        </div>
        """
    )

    metric_columns = st.columns(
        6,
        gap="small",
    )

    with metric_columns[0]:
        render_portfolio_card(
            title="Long Positions",
            value=f"{len(long_positions)}",
            subtitle="Positive portfolio weights",
            card_class="card-green",
        )

    with metric_columns[1]:
        render_portfolio_card(
            title="Short Positions",
            value=f"{len(short_positions)}",
            subtitle="Negative portfolio weights",
            card_class="card-red",
        )

    with metric_columns[2]:
        render_portfolio_card(
            title="Gross Exposure",
            value=f"{gross_exposure:.1%}",
            subtitle="Absolute portfolio exposure",
            card_class="card-blue",
        )

    with metric_columns[3]:
        render_portfolio_card(
            title="Net Exposure",
            value=f"{net_exposure:.1%}",
            subtitle="Long exposure less short",
            card_class="card-teal",
        )

    with metric_columns[4]:
        render_portfolio_card(
            title="Largest Long",
            value=f"{largest_long:.1%}",
            subtitle="Largest positive position",
            card_class="card-purple",
        )

    with metric_columns[5]:
        render_portfolio_card(
            title="Largest Short",
            value=f"{largest_short:.1%}",
            subtitle="Largest negative position",
            card_class="card-orange",
        )

    exposure_status = (
        "market neutral"
        if abs(net_exposure) < 0.01
        else "net long"
        if net_exposure > 0
        else "net short"
    )

    st.html(
        f"""
        <div class="portfolio-takeaway">
            <div class="takeaway-icon">★</div>
            <div>
                <div class="takeaway-title">Portfolio Snapshot</div>
                <div class="takeaway-text">
                    The latest portfolio contains
                    <b>{len(long_positions)} long</b> and
                    <b>{len(short_positions)} short</b> positions with
                    <b>{gross_exposure:.1%} gross exposure</b>. Net exposure is
                    <b>{net_exposure:.1%}</b>, leaving the strategy broadly
                    <b>{exposure_status}</b>. The largest individual long and
                    short allocations are <b>{largest_long:.1%}</b> and
                    <b>{largest_short:.1%}</b>, respectively.
                </div>
            </div>
        </div>
        """
    )

    st.html(
        """
        <div class="section-header">
            <div class="section-icon">◫</div>
            <div>
                <div class="section-title">Portfolio Composition</div>
                <div class="section-description">
                    Review the largest individual positions and the distribution
                    of gross exposure across industries.
                </div>
            </div>
        </div>
        """
    )

    chart_col, sector_col = st.columns(
        [
            1.35,
            1,
        ],
        gap="medium",
    )

    with chart_col:
        st.html(
            """
            <div class="panel-label">
                Largest Portfolio Positions
            </div>

            <div class="panel-caption">
                Largest long and short positions by portfolio weight.
            </div>
            """
        )

        top_long_chart = (
            long_positions
            .nlargest(
                8,
                weight_col,
            )[
                [
                    ticker_col,
                    weight_col,
                ]
            ]
        )

        top_short_chart = (
            short_positions
            .nsmallest(
                8,
                weight_col,
            )[
                [
                    ticker_col,
                    weight_col,
                ]
            ]
        )

        position_chart_df = (
            pd.concat(
                [
                    top_short_chart,
                    top_long_chart,
                ],
                ignore_index=True,
            )
            .sort_values(weight_col)
        )

        position_chart_df["Position"] = (
            position_chart_df[weight_col]
            .apply(
                lambda weight: (
                    "Long"
                    if weight > 0
                    else "Short"
                )
            )
        )

        position_chart_df["Weight (%)"] = (
            position_chart_df[weight_col]
            * 100
        )

        position_figure = px.bar(
            position_chart_df,
            x="Weight (%)",
            y=ticker_col,
            color="Position",
            orientation="h",
            color_discrete_map={
                "Long": "#10B981",
                "Short": "#EF4444",
            },
            labels={
                ticker_col: "",
                "Weight (%)": "Portfolio weight (%)",
            },
        )

        position_figure.update_layout(
            height=390,
            margin=dict(
                l=5,
                r=10,
                t=10,
                b=10,
            ),
            legend_title_text="",
            legend_orientation="h",
            legend_y=1.08,
            legend_x=0,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="#E2E8F0",
                zeroline=True,
                zerolinecolor="#94A3B8",
            ),
            yaxis=dict(
                categoryorder="total ascending",
            ),
        )

        position_figure.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weight: %{x:.2f}%"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            position_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    with sector_col:
        st.html(
            """
            <div class="panel-label">
                Sector Composition
            </div>

            <div class="panel-caption">
                Share of gross portfolio exposure allocated to each industry.
            </div>
            """
        )

        sector_composition_df = build_sector_composition(
            active_positions,
            weight_col,
        )

        if sector_composition_df.empty:
            st.info(
                "Sector composition is unavailable for "
                "the current portfolio."
            )

        else:
            sector_figure = px.pie(
                sector_composition_df,
                names="industry",
                values="Gross Exposure (%)",
                hole=0.55,
            )

            sector_figure.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Share of gross exposure: %{value:.1f}%"
                    "<extra></extra>"
                ),
                marker=dict(
                    line=dict(
                        color="#FFFFFF",
                        width=2,
                    )
                ),
            )

            sector_figure.update_layout(
                height=390,
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=10,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.0,
                    font=dict(
                        size=10,
                    ),
                ),
                annotations=[
                    dict(
                        text=(
                            f"<b>{gross_exposure:.0%}</b>"
                            "<br>"
                            "<span style='font-size:11px'>"
                            "Gross exposure"
                            "</span>"
                        ),
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        align="center",
                    )
                ],
            )

            st.plotly_chart(
                sector_figure,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                },
            )

    st.html(
        """
        <div class="section-header">
            <div class="section-icon">≡</div>
            <div>
                <div class="section-title">Current Positions</div>
                <div class="section-description">
                    Filter active holdings by portfolio side and inspect their
                    weights, predictions and realised returns.
                </div>
            </div>
        </div>
        """
    )

    portfolio_side = st.segmented_control(
        "Portfolio Side",
        options=[
            "All Positions",
            "Long Positions",
            "Short Positions",
        ],
        default="All Positions",
        label_visibility="collapsed",
    )

    if portfolio_side == "Long Positions":
        display_df = long_positions.copy()

    elif portfolio_side == "Short Positions":
        display_df = short_positions.copy()

    else:
        display_df = active_positions.copy()

    display_columns = [
        "Date",
        ticker_col,
        "industry",
        weight_col,
    ]

    if prediction_col is not None:
        display_columns.append(
            prediction_col
        )

    if realised_return_col is not None:
        display_columns.append(
            realised_return_col
        )

    display_df = (
        display_df[
            display_columns
        ]
        .sort_values(
            weight_col,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    table_df = display_df.copy()
    table_df[weight_col] *= 100

    column_config = {
        "Date": st.column_config.DateColumn(
            "Date",
            format="DD MMM YYYY",
        ),
        ticker_col: st.column_config.TextColumn(
            "Ticker",
        ),
        "industry": st.column_config.TextColumn(
            "Industry",
        ),
        weight_col: st.column_config.NumberColumn(
            "Portfolio Weight",
            format="%.2f%%",
        ),
    }

    if prediction_col is not None:
        table_df[prediction_col] *= 100

        column_config[prediction_col] = (
            st.column_config.NumberColumn(
                "Predicted Return",
                format="%.2f%%",
            )
        )

    if realised_return_col is not None:
        table_df[realised_return_col] *= 100

        column_config[realised_return_col] = (
            st.column_config.NumberColumn(
                "Realised Return",
                format="%.2f%%",
            )
        )

    st.dataframe(
        table_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=390,
    )

    st.html(
        """
        <div class="section-header">
            <div class="section-icon">↕</div>
            <div>
                <div class="section-title">Portfolio Contribution</div>
                <div class="section-description">
                    Positions with the largest positive and negative realised
                    contribution to the latest portfolio return.
                </div>
            </div>
        </div>
        """
    )

    if realised_return_col is None:
        st.info(
            "A realised-return column is required to calculate "
            "portfolio contribution."
        )
        return

    contribution_df = active_positions.copy()

    contribution_df["return_contribution"] = (
        contribution_df[weight_col]
        * contribution_df[realised_return_col]
    )

    positive_contributors = (
        contribution_df
        .nlargest(
            5,
            "return_contribution",
        )[
            [
                ticker_col,
                weight_col,
                realised_return_col,
                "return_contribution",
            ]
        ]
        .copy()
    )

    negative_contributors = (
        contribution_df
        .nsmallest(
            5,
            "return_contribution",
        )[
            [
                ticker_col,
                weight_col,
                realised_return_col,
                "return_contribution",
            ]
        ]
        .copy()
    )

    for dataframe in [
        positive_contributors,
        negative_contributors,
    ]:
        dataframe[weight_col] *= 100
        dataframe[realised_return_col] *= 100
        dataframe["return_contribution"] *= 100

    contribution_config = {
        ticker_col: st.column_config.TextColumn(
            "Ticker",
        ),
        weight_col: st.column_config.NumberColumn(
            "Weight",
            format="%.2f%%",
        ),
        realised_return_col: (
            st.column_config.NumberColumn(
                "Realised Return",
                format="%.2f%%",
            )
        ),
        "return_contribution": (
            st.column_config.NumberColumn(
                "Contribution",
                format="%.2f%%",
            )
        ),
    }

    positive_col, negative_col = st.columns(
        2,
        gap="medium",
    )

    with positive_col:
        st.html(
            """
            <div class="long-heading">
                ▲ Top Positive Contributors
            </div>
            """
        )

        st.dataframe(
            positive_contributors,
            column_config=contribution_config,
            hide_index=True,
            use_container_width=True,
            height=225,
        )

    with negative_col:
        st.html(
            """
            <div class="short-heading">
                ▼ Top Negative Contributors
            </div>
            """
        )

        st.dataframe(
            negative_contributors,
            column_config=contribution_config,
            hide_index=True,
            use_container_width=True,
            height=225,
        )
