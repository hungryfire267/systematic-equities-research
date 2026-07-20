import streamlit as st


def render_alpha_metric_cards(alpha_metrics: dict) -> None:
    st.markdown(
        """
        <style>
        .alpha-card {
            border: 1px solid;
            border-radius: 14px;
            padding: 1.15rem 1.2rem;
            min-height: 170px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }

        .alpha-card-title {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .alpha-card-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.85rem;
        }

        .alpha-card-description {
            color: #64748B;
            font-size: 0.85rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = [
        {
            "title": "Annualised Alpha",
            "value": f"{alpha_metrics['annualised_alpha']:.1%}",
            "description": (
                "Estimated annual excess return after accounting "
                "for ASX 200 exposure."
            ),
            "background": "#ECFDF5",
            "border": "#A7F3D0",
            "accent": "#047857"
        },
        {
            "title": "Market Beta",
            "value": f"{alpha_metrics['beta']:.2f}",
            "description": (
                "Sensitivity of strategy returns to movements "
                "in the ASX 200."
            ),
            "background": "#EFF6FF",
            "border": "#BFDBFE",
            "accent": "#1D4ED8"
        },
        {
            "title": "Market R²",
            "value": f"{alpha_metrics['r_squared']:.1%}",
            "description": (
                "Proportion of strategy return variation explained "
                "by the ASX 200."
            ),
            "background": "#F5F3FF",
            "border": "#DDD6FE",
            "accent": "#6D28D9"
        },
        {
            "title": "Information Ratio",
            "value": f"{alpha_metrics['information_ratio']:.2f}",
            "description": (
                "Active return generated per unit of "
                "benchmark-relative risk."
            ),
            "background": "#FFF7ED",
            "border": "#FED7AA",
            "accent": "#C2410C"
        }
    ]

    columns = st.columns(4, gap="medium")

    for column, card in zip(columns, cards):
        with column:
            card_html = (
                f'<div class="alpha-card" '
                f'style="background:{card["background"]};'
                f'border-color:{card["border"]};">'
                f'<div class="alpha-card-title" '
                f'style="color:{card["accent"]};">'
                f'{card["title"]}'
                f'</div>'
                f'<div class="alpha-card-value" '
                f'style="color:{card["accent"]};">'
                f'{card["value"]}'
                f'</div>'
                f'<div class="alpha-card-description">'
                f'{card["description"]}'
                f'</div>'
                f'</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )