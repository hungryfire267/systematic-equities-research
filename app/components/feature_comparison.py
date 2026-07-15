import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from textwrap import dedent


MODEL_COLOURS = {
    "LightGBM": "#10B981",
    "XGBoost": "#2563EB"
}


def render_feature_comparison(
    feature_title: str,
    feature_description: str,
    lightgbm_results: dict,
    xgboost_results: dict,
    lightgbm_ic: pd.Series,
    xgboost_ic: pd.Series,
    lightgbm_returns: pd.DataFrame,
    xgboost_returns: pd.DataFrame
):
    lgbm_pred = lightgbm_results["prediction"]
    xgb_pred = xgboost_results["prediction"]

    st.markdown(f"### {feature_title}")
    st.caption(feature_description)

    top_left, top_right = st.columns(2)

    with top_left:
        st.plotly_chart(
            create_prediction_metric_chart(
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
): 
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
    


def create_prediction_metric_chart(
    lightgbm_prediction: dict,
    xgboost_prediction: dict
):
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean IC",
                "Annualised ICIR"
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
    lightgbm_ic: pd.Series,
    xgboost_ic: pd.Series
):
    ic_df = pd.concat(
        [
            lightgbm_ic.rename("LightGBM"),
            xgboost_ic.rename("XGBoost")
        ],
        axis=1
    ).reset_index()

    fig = go.Figure()

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
    
    