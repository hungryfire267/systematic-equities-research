import html

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    subtitle: str,
    icon: str,
    value_colour: str,
    icon_colour: str,
    icon_background: str,
) -> None:
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">
            {html.escape(title)}
        </div>

        <div class="metric-main">
            <div
                class="metric-icon"
                style="
                    color: {icon_colour};
                    background: {icon_background};
                "
            >
                {html.escape(icon)}
            </div>

            <div
                class="metric-value"
                style="color: {value_colour};"
            >
                {html.escape(value)}
            </div>
        </div>

        <div class="metric-subtitle">
            {html.escape(subtitle)}
        </div>
    </div>
    """

    st.html(card_html)