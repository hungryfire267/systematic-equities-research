import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from textwrap import dedent


MODEL_COLOURS = {
    "Decision Tree": "#F59E0B",
    "LightGBM": "#10B981",
    "XGBoost": "#2563EB"
}


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

    st.markdown("## Prediction Analytics")
    st.caption(
        "Measures how effectively each model ranks future returns, predicts "
        "direction and limits forecast error before portfolio construction."
    )

    st.markdown(f"### {feature_title}")
    st.caption(feature_description)

    top_left, top_right = st.columns(2)

    with top_left:
        with st.container(border=True):
            st.plotly_chart(
                create_prediction_metric_chart(
                    dt_pred,
                    lgbm_pred,
                    xgb_pred
                ),
                width="stretch"
            )

    with top_right:
        with st.container(border=True):
            st.plotly_chart(
                create_ic_chart(
                    dt_ic,
                    lightgbm_ic,
                    xgboost_ic
                ),
                width="stretch"
            )

    st.markdown(
        "<div style='height:8px;'></div>",
        unsafe_allow_html=True
    )

    with st.container(border=True):
        render_hit_rate_cards(
            dt_pred,
            lgbm_pred,
            xgb_pred
        )
        # Adds extra breathing room beneath the three hit-rate cards so the
        # lower edge of the section does not feel cramped.
        st.markdown(
            "<div style='height:12px;'></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div style='height:10px;'></div>",
        unsafe_allow_html=True
    )

    render_forecast_error_section(
        dt_pred,
        lgbm_pred,
        xgb_pred
    )

    st.markdown(
        "<div style='height:10px;'></div>",
        unsafe_allow_html=True
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

    st.markdown("## Performance Analytics")
    st.caption(
        "Translates the model forecasts into realised portfolio outcomes, "
        "including cumulative return, drawdown and risk-adjusted performance."
    )

    top_left, top_right = st.columns(2)

    with top_left:
        with st.container(border=True):
            st.plotly_chart(
                create_equity_curve(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns
                ),
                width="stretch"
            )

    with top_right:
        with st.container(border=True):
            st.plotly_chart(
                create_drawdown_curve(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns
                ),
                width="stretch"
            )

    # Equal column weights ensure that the performance table and rolling
    # Sharpe chart receive exactly half of the available row width.
    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        with st.container(border=True):
            render_portfolio_performance_table(
                lgbm_port,
                xgb_port,
                dt_port
            )

    with bottom_right:
        with st.container(border=True):
            st.plotly_chart(
                create_rolling_sharpe_chart(
                    dt_returns,
                    lightgbm_returns,
                    xgboost_returns,
                    window=13
                ),
                width="stretch"
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
            f'<td style="text-align:left;">{label}</td>'
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
        "Decision Tree": dt_prediction,
        "LightGBM": lightgbm_prediction,
        "XGBoost": xgboost_prediction
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

    mae_winner = min(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["mae"]
    )

    rmse_winner = min(
        prediction_metrics,
        key=lambda model: prediction_metrics[model]["rmse"]
    )

    def badge(model: str) -> str:
        style = model_styles[model]

        return (
            f'<span style="background:{style["header_background"]};'
            f'color:{style["colour"]};'
            'padding:0.18rem 0.55rem;'
            'border-radius:999px;'
            'font-size:0.68rem;'
            'font-weight:700;'
            'white-space:nowrap;'
            '">'
            f'{model}'
            '</span>'
        )

    if mae_winner == rmse_winner:
        takeaway = (
            f"<strong>{mae_winner}</strong> records the lowest MAE and RMSE, "
            "indicating the strongest overall point-forecast accuracy."
        )
        takeaway_style = model_styles[mae_winner]
        takeaway_background = takeaway_style["background"]
        takeaway_header_background = takeaway_style["header_background"]
        takeaway_border = takeaway_style["border"]
        takeaway_colour = takeaway_style["colour"]
    else:
        takeaway = (
            f"<strong>{mae_winner}</strong> records the lowest MAE, while "
            f"<strong>{rmse_winner}</strong> records the lowest RMSE. "
            "The models therefore differ in average forecast error and "
            "sensitivity to larger misses."
        )
        mae_style = model_styles[mae_winner]
        rmse_style = model_styles[rmse_winner]
        takeaway_background = (
            f'linear-gradient(135deg,'
            f'{mae_style["background"]},'
            f'{rmse_style["background"]})'
        )
        takeaway_header_background = (
            f'linear-gradient(135deg,'
            f'{mae_style["header_background"]},'
            f'{rmse_style["header_background"]})'
        )
        takeaway_border = mae_style["border"]
        takeaway_colour = mae_style["colour"]

    table_html = (
        '<div style="'
        'border:1px solid #E2E8F0;'
        'border-radius:12px;'
        'overflow:hidden;'
        'background:#FFFFFF;'
        'height:172px;'
        'box-sizing:border-box;'
        '">'
        '<div style="'
        'padding:0.72rem 0.85rem;'
        'font-size:0.88rem;'
        'font-weight:750;'
        'color:#0F172A;'
        'border-bottom:1px solid #E2E8F0;'
        'background:#FFFFFF;'
        '">'
        'Forecast Error '
        '<span style="'
        'color:#64748B;'
        'font-size:0.75rem;'
        'font-weight:500;'
        '">'
        '(Lower is Better)'
        '</span>'
        '</div>'
        '<table style="'
        'width:100%;'
        'border-collapse:collapse;'
        'font-size:0.76rem;'
        'table-layout:fixed;'
        '">'
        '<thead>'
        '<tr style="background:#F8FAFC;">'
        '<th style="padding:0.52rem;text-align:left;width:15%;">Metric</th>'
        '<th style="padding:0.52rem;text-align:center;width:17%;">DT</th>'
        '<th style="padding:0.52rem;text-align:center;width:21%;">LightGBM</th>'
        '<th style="padding:0.52rem;text-align:center;width:20%;">XGBoost</th>'
        '<th style="padding:0.52rem;text-align:center;width:27%;">Better</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        '<tr>'
        '<td style="padding:0.58rem;border-top:1px solid #E2E8F0;text-align:left;">MAE</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{dt_prediction["mae"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{lightgbm_prediction["mae"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{xgboost_prediction["mae"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{badge(mae_winner)}</td>'
        '</tr>'
        '<tr>'
        '<td style="padding:0.58rem;border-top:1px solid #E2E8F0;text-align:left;">RMSE</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{dt_prediction["rmse"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{lightgbm_prediction["rmse"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{xgboost_prediction["rmse"]:.4f}</td>'
        f'<td style="padding:0.58rem;text-align:center;border-top:1px solid #E2E8F0;">{badge(rmse_winner)}</td>'
        '</tr>'
        '</tbody>'
        '</table>'
        '</div>'
    )

    takeaway_html = (
        '<div style="'
        f'border:1px solid {takeaway_border};'
        'border-radius:12px;'
        'overflow:hidden;'
        f'background:{takeaway_background};'
        'height:172px;'
        'box-sizing:border-box;'
        '">'
        '<div style="'
        'padding:0.72rem 0.85rem;'
        'font-size:0.88rem;'
        'font-weight:750;'
        f'color:{takeaway_colour};'
        f'border-bottom:1px solid {takeaway_border};'
        f'background:{takeaway_header_background};'
        '">'
        '★ Key Takeaway'
        '</div>'
        '<div style="'
        'padding:0.9rem 1rem;'
        'font-size:0.8rem;'
        'line-height:1.6;'
        'color:#334155;'
        f'background:{takeaway_background};'
        '">'
        f'{takeaway}'
        '</div>'
        '</div>'
    )

    table_col, takeaway_col = st.columns(
        [1, 1],
        gap="small"
    )

    with table_col:
        st.markdown(
            table_html,
            unsafe_allow_html=True
        )

    with takeaway_col:
        st.markdown(
            takeaway_html,
            unsafe_allow_html=True
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
    """
    Render a styled hypothesis-testing card.

    Parameters
    ----------
    number:
        Question number displayed at the top of the card.
    question:
        Research question being tested.
    null_hypothesis:
        LaTeX expression for H_0.
    alternative_hypothesis:
        LaTeX expression for H_1.
    test_name:
        Name and brief description of the statistical test.
    accent_colour:
        Colour used for the left border and question label.
    background_colour:
        Background colour of the question header.
    """

    st.markdown(
        (
            f'<div style="'
            f'background:{background_colour};'
            f'border:1px solid #E2E8F0;'
            f'border-left:6px solid {accent_colour};'
            f'border-radius:14px;'
            f'padding:1.1rem 1.25rem;'
            f'margin-top:0.8rem;'
            f'margin-bottom:0.9rem;'
            f'box-shadow:0 4px 14px rgba(15,23,42,0.05);'
            f'">'
            f'<div style="'
            f'font-size:0.72rem;'
            f'font-weight:800;'
            f'letter-spacing:0.10em;'
            f'color:{accent_colour};'
            f'margin-bottom:0.35rem;'
            f'">'
            f'QUESTION {number}'
            f'</div>'
            f'<div style="'
            f'font-size:1.08rem;'
            f'font-weight:750;'
            f'color:#0F172A;'
            f'line-height:1.4;'
            f'">'
            f'{question}'
            f'</div>'
            f'</div>'
        ),
        unsafe_allow_html=True
    )

    null_col, alternative_col = st.columns(2)

    with null_col:
        st.markdown(
            """
            <p style="
                margin:0 0 0.2rem 0;
                font-size:0.72rem;
                font-weight:800;
                color:#64748B;
                letter-spacing:0.08em;
            ">
                NULL HYPOTHESIS
            </p>
            """,
            unsafe_allow_html=True
        )

        st.latex(null_hypothesis)

    with alternative_col:
        st.markdown(
            """
            <p style="
                margin:0 0 0.2rem 0;
                font-size:0.72rem;
                font-weight:800;
                color:#64748B;
                letter-spacing:0.08em;
            ">
                ALTERNATIVE HYPOTHESIS
            </p>
            """,
            unsafe_allow_html=True
        )

        st.latex(alternative_hypothesis)

    st.markdown(
        (
            '<div style="'
            'background:#F8FAFC;'
            'border:1px solid #E2E8F0;'
            'border-radius:9px;'
            'padding:0.75rem 1rem;'
            'margin-top:0.35rem;'
            'margin-bottom:1.4rem;'
            'font-size:0.88rem;'
            'line-height:1.5;'
            'color:#475569;'
            '">'
            '<strong style="color:#0F172A;">Recommended test:</strong> '
            f'{test_name}'
            '</div>'
        ),
        unsafe_allow_html=True
    )