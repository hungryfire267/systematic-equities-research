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
    xgboost_ic: pd.Series,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    dt_pred = dt_results["prediction"]
    lgbm_pred = lightgbm_results["prediction"]
    xgb_pred = xgboost_results["prediction"]

    st.markdown(f"### {feature_title}")
    st.caption(feature_description)

    top_left, top_right = st.columns(2)

    with top_left:
        st.plotly_chart(
            create_prediction_metric_chart(
                dt_pred,
                lgbm_pred,
                xgb_pred
            ),
            width="stretch"
        )

        render_hit_rate_cards(
            lgbm_pred,
            xgb_pred
        )

    with top_right:
        st.plotly_chart(
            create_ic_chart(
                dt_ic,
                lightgbm_ic,
                xgboost_ic
            ),
            width="stretch"
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        
def render_performance_comparison(
    lightgbm_results: dict,
    xgboost_results: dict,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
) -> None:
    lgbm_port = lightgbm_results["portfolio"]
    xgb_port = xgboost_results["portfolio"]

    top_left, top_right = st.columns(2)

    with top_left:
        st.plotly_chart(
            create_equity_curve(
                lightgbm_returns,
                xgboost_returns
            ),
            width="stretch"
        )

    with top_right:
        st.plotly_chart(
            create_drawdown_curve(
                lightgbm_returns,
                xgboost_returns
            ),
            width="stretch"
        )

    bottom_left, bottom_right = st.columns([1, 1.05])

    with bottom_left:
        render_portfolio_performance_table(
            lgbm_port,
            xgb_port
        )

    with bottom_right:
        st.plotly_chart(
            create_rolling_sharpe_chart(
                lightgbm_returns,
                xgboost_returns,
                window=13
            ),
            width="stretch"
        )
        
def render_portfolio_performance_table(
    lightgbm_portfolio: dict,
    xgboost_portfolio: dict
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

    rows = ""

    for label, key, percentage, higher_is_better in metrics:
        lgbm_value = lightgbm_portfolio[key]
        xgb_value = xgboost_portfolio[key]

        if higher_is_better:
            winner = (
                "LightGBM"
                if lgbm_value > xgb_value
                else "XGBoost"
            )
        else:
            winner = (
                "LightGBM"
                if lgbm_value < xgb_value
                else "XGBoost"
            )

        if percentage:
            lgbm_display = f"{lgbm_value:.1%}"
            xgb_display = f"{xgb_value:.1%}"
        else:
            lgbm_display = f"{lgbm_value:.2f}"
            xgb_display = f"{xgb_value:.2f}"

        badge_background = (
            "#DCFCE7" if winner == "LightGBM" else "#DBEAFE"
        )
        badge_colour = (
            "#059669" if winner == "LightGBM" else "#2563EB"
        )

        rows += (
            "<tr>"
            f'<td style="text-align:left;">{label}</td>'
            f"<td>{lgbm_display}</td>"
            f"<td>{xgb_display}</td>"
            "<td>"
            f'<span style="background:{badge_background};'
            f'color:{badge_colour};padding:0.2rem 0.6rem;'
            'border-radius:999px;font-size:0.72rem;'
            f'font-weight:700;">{winner}</span>'
            "</td>"
            "</tr>"
        )

    table_html = (
        '<div style="border:1px solid #E2E8F0;'
        'border-radius:12px;overflow:hidden;background:#FFFFFF;">'
        '<div style="padding:0.8rem 0.9rem;'
        'font-size:0.9rem;font-weight:750;color:#0F172A;">'
        "Portfolio Performance Summary"
        "</div>"
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.8rem;">'
        "<thead>"
        '<tr style="background:#F8FAFC;">'
        '<th style="text-align:left;">Metric</th>'
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
    lightgbm_portfolio: dict,
    xgboost_portfolio: dict
) -> None:
    lgbm_sharpe = lightgbm_portfolio["sharpe_ratio"]
    xgb_sharpe = xgboost_portfolio["sharpe_ratio"]

    winner = "LightGBM" if lgbm_sharpe > xgb_sharpe else "XGBoost"
    difference = abs(lgbm_sharpe - xgb_sharpe)

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
        'grid-template-columns:1fr 1fr;'
        'gap:0.8rem;'
        '">'
        '<div style="'
        'background:#FFFFFF;'
        'border:1px solid #A7F3D0;'
        'border-radius:10px;'
        'padding:0.85rem;'
        '">'
        '<p style="'
        'margin:0;'
        'color:#64748B;'
        'font-size:0.72rem;'
        'font-weight:800;'
        '">'
        'LIGHTGBM'
        '</p>'
        '<p style="'
        'margin:0.2rem 0 0 0;'
        'font-size:1.7rem;'
        'font-weight:800;'
        'color:#10B981;'
        '">'
        f'{lgbm_sharpe:.2f}'
        '</p>'
        '</div>'
        '<div style="'
        'background:#FFFFFF;'
        'border:1px solid #BFDBFE;'
        'border-radius:10px;'
        'padding:0.85rem;'
        '">'
        '<p style="'
        'margin:0;'
        'color:#64748B;'
        'font-size:0.72rem;'
        'font-weight:800;'
        '">'
        'XGBOOST'
        '</p>'
        '<p style="'
        'margin:0.2rem 0 0 0;'
        'font-size:1.7rem;'
        'font-weight:800;'
        'color:#2563EB;'
        '">'
        f'{xgb_sharpe:.2f}'
        '</p>'
        '</div>'
        '</div>'
        '<p style="'
        'margin:0.9rem 0 0 0;'
        'color:#334155;'
        'font-size:0.84rem;'
        'line-height:1.5;'
        '">'
        f'<strong>{winner}</strong> achieved the higher Sharpe ratio by '
        f'{difference:.2f}, indicating stronger risk-adjusted portfolio performance.'
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
        height=300,
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
    lightgbm_prediction: dict,
    xgboost_prediction: dict
):
    st.markdown(
        """
        <p style="
            font-size:0.95rem;
            font-weight:700;
            color:#0F172A;
            margin:0 0 0.6rem 0;
        ">
            Directional Hit Rate
        </p>
        """,
        unsafe_allow_html=True
    )

    lgbm_col, xgb_col = st.columns(2)

    lgbm_card = (
        f'<div style="border:1px solid #A7F3D0;'
        f'border-left:6px solid {MODEL_COLOURS["LightGBM"]};'
        f'border-radius:12px;padding:14px 16px;'
        f'background:#F0FDF4;min-height:92px;">'
        f'<p style="color:#64748B;font-size:0.72rem;font-weight:800;'
        f'letter-spacing:0.08em;margin:0;">LIGHTGBM</p>'
        f'<p style="color:#0F172A;font-size:1.65rem;font-weight:800;'
        f'margin:4px 0 0 0;">'
        f'{lightgbm_prediction["hit_rate"]:.2%}'
        f'</p>'
        f'</div>'
    )

    xgb_card = (
        f'<div style="border:1px solid #BFDBFE;'
        f'border-left:6px solid {MODEL_COLOURS["XGBoost"]};'
        f'border-radius:12px;padding:14px 16px;'
        f'background:#EFF6FF;min-height:92px;">'
        f'<p style="color:#64748B;font-size:0.72rem;font-weight:800;'
        f'letter-spacing:0.08em;margin:0;">XGBOOST</p>'
        f'<p style="color:#0F172A;font-size:1.65rem;font-weight:800;'
        f'margin:4px 0 0 0;">'
        f'{xgboost_prediction["hit_rate"]:.2%}'
        f'</p>'
        f'</div>'
    )

    with lgbm_col:
        st.markdown(lgbm_card, unsafe_allow_html=True)

    with xgb_col:
        st.markdown(xgb_card, unsafe_allow_html=True)


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

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.2,
        line_color="#64748B"
    )

    fig.update_layout(
        title="IC Through Time",
        height=420,
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
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    lgbm = lightgbm_returns.copy()
    xgb = xgboost_returns.copy()

    lgbm["Date"] = pd.to_datetime(lgbm["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])

    lgbm = lgbm.sort_values("Date")
    xgb = xgb.sort_values("Date")

    lgbm["Equity"] = (
        1 + lgbm["portfolio_return"]
    ).cumprod() - 1

    xgb["Equity"] = (
        1 + xgb["portfolio_return"]
    ).cumprod() - 1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lgbm["Date"],
            y=lgbm["Equity"],
            mode="lines",
            name="LightGBM",
            line=dict(
                color=MODEL_COLOURS["LightGBM"],
                width=2.8
            ),
            hovertemplate=(
                "<b>LightGBM</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Cumulative return: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xgb["Date"],
            y=xgb["Equity"],
            mode="lines",
            name="XGBoost",
            line=dict(
                color=MODEL_COLOURS["XGBoost"],
                width=2.8
            ),
            hovertemplate=(
                "<b>XGBoost</b><br>"
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
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    lgbm = lightgbm_returns.copy()
    xgb = xgboost_returns.copy()

    lgbm["Date"] = pd.to_datetime(lgbm["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])

    lgbm = lgbm.sort_values("Date")
    xgb = xgb.sort_values("Date")

    def calculate_drawdown(df: pd.DataFrame) -> pd.Series:
        equity = (1 + df["portfolio_return"]).cumprod()
        return equity / equity.cummax() - 1

    lgbm["Drawdown"] = calculate_drawdown(lgbm)
    xgb["Drawdown"] = calculate_drawdown(xgb)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lgbm["Date"],
            y=lgbm["Drawdown"],
            mode="lines",
            name="LightGBM",
            line=dict(
                color=MODEL_COLOURS["LightGBM"],
                width=2.8
            ),
            hovertemplate=(
                "<b>LightGBM</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Drawdown: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xgb["Date"],
            y=xgb["Drawdown"],
            mode="lines",
            name="XGBoost",
            line=dict(
                color=MODEL_COLOURS["XGBoost"],
                width=2.8
            ),
            hovertemplate=(
                "<b>XGBoost</b><br>"
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
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame,
    window: int = 26
):
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

    lgbm["Rolling Sharpe"] = rolling_sharpe(
        lgbm["portfolio_return"]
    )

    xgb["Rolling Sharpe"] = rolling_sharpe(
        xgb["portfolio_return"]
    )

    fig = go.Figure()

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
        title="Rolling Sharpe Ratio",
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

def render_forecast_error_section(lightgbm_data: dict, xgboost_data: dict) -> None:
    lgbm_pred = lightgbm_data["metrics"]["prediction"]
    xgb_pred = xgboost_data["metrics"]["prediction"]

    table_col, takeaway_col = st.columns([1, 1.35])

    with table_col:
        mae_winner = (
            "LightGBM"
            if lgbm_pred["mae"] < xgb_pred["mae"]
            else "XGBoost"
        )

        rmse_winner = (
            "LightGBM"
            if lgbm_pred["rmse"] < xgb_pred["rmse"]
            else "XGBoost"
        )

        def badge(model: str) -> str:
            if model == "LightGBM":
                return (
                    '<span style="background:#DCFCE7;color:#059669;'
                    'padding:0.18rem 0.55rem;border-radius:999px;'
                    'font-size:0.72rem;font-weight:700;">LightGBM</span>'
                )

            return (
                '<span style="background:#DBEAFE;color:#2563EB;'
                'padding:0.18rem 0.55rem;border-radius:999px;'
                'font-size:0.72rem;font-weight:700;">XGBoost</span>'
            )

        table_html = (
            '<div style="border:1px solid #E2E8F0;border-radius:10px;'
            'overflow:hidden;background:#FFFFFF;">'
            '<div style="padding:0.65rem 0.8rem;font-size:0.86rem;'
            'font-weight:750;color:#0F172A;">'
            'Forecast Error <span style="color:#64748B;">'
            '(Lower is Better)</span></div>'
            '<table style="width:100%;border-collapse:collapse;'
            'font-size:0.78rem;">'
            '<thead><tr style="background:#F8FAFC;">'
            '<th style="padding:0.5rem;text-align:left;">Metric</th>'
            '<th style="padding:0.5rem;text-align:center;">LightGBM</th>'
            '<th style="padding:0.5rem;text-align:center;">XGBoost</th>'
            '<th style="padding:0.5rem;text-align:center;">Better</th>'
            '</tr></thead>'
            '<tbody>'
            '<tr>'
            '<td style="padding:0.5rem;border-top:1px solid #E2E8F0;">MAE</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{lgbm_pred["mae"]:.4f}</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{xgb_pred["mae"]:.4f}</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{badge(mae_winner)}</td>'
            '</tr>'
            '<tr>'
            '<td style="padding:0.5rem;border-top:1px solid #E2E8F0;">RMSE</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{lgbm_pred["rmse"]:.4f}</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{xgb_pred["rmse"]:.4f}</td>'
            f'<td style="padding:0.5rem;text-align:center;border-top:1px solid #E2E8F0;">'
            f'{badge(rmse_winner)}</td>'
            '</tr>'
            '</tbody></table></div>'
        )

        st.markdown(table_html, unsafe_allow_html=True)

    with takeaway_col:
        if mae_winner == rmse_winner:
            takeaway = (
                f"{mae_winner} delivers lower MAE and RMSE, indicating "
                "stronger forecast accuracy across both error measures."
            )
        else:
            takeaway = (
                f"{mae_winner} records the lower MAE, while "
                f"{rmse_winner} records the lower RMSE."
            )

        takeaway_html = (
            '<div style="border:1px solid #BFDBFE;border-radius:10px;'
            'background:linear-gradient(135deg,#F8FBFF,#EFF6FF);'
            'padding:0.9rem 1rem;min-height:118px;">'
            '<div style="font-size:0.86rem;font-weight:750;'
            'color:#2563EB;margin-bottom:0.45rem;">'
            '☆ Key Takeaway</div>'
            f'<div style="font-size:0.8rem;line-height:1.5;color:#334155;">'
            f'{takeaway}</div>'
            '</div>'
        )

        st.markdown(takeaway_html, unsafe_allow_html=True)

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
    
    