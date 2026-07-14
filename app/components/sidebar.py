import base64
import streamlit as st
from pathlib import Path


def render_sidebar():
    logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: #F4F7FB;
        }

        /* Hide radio circles */
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        /* Navigation buttons */
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            width: 100%;
            padding: 0.72rem 0.85rem;
            margin-bottom: 0.35rem;
            border-radius: 10px;
            cursor: pointer;
            transition: 0.2s ease;
        }

        /* Hover */
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #E6EEF8;
            transform: translateX(3px);
        }

        /* Selected item */
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(
                90deg,
                #1FB7A6 0%,
                #2F80ED 100%
            );
            box-shadow: 0 5px 15px rgba(47, 128, 237, 0.22);
        }

        /* Selected text */
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: white !important;
            font-weight: 700;
        }

        /* Normal text */
        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 0.92rem;
            font-weight: 600;
            color: #223047;
        }
        
        [data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stSidebarCollapseButton"] {
            display: none;
        }


        .nav-label {
            margin-top: 1.1rem;
            margin-bottom: 0.6rem;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            color: #8A99AD;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    
    with open(logo_path, "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()

    with st.sidebar:
        logo_col, title_col = st.columns(
            [1.15, 2.85],
            vertical_alignment="center"
        )

        with logo_col:
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    height:72px;
                ">
                    <img
                        src="data:image/png;base64,{logo_base64}"
                        style="
                            width:68px;
                            height:68px;
                            object-fit:contain;
                            display:block;
                        "
                    >
                </div>
                """,
                unsafe_allow_html=True
            )

        with title_col:
            st.markdown(
                """
                <div style="
                    display:flex;
                    align-items:center;
                    height:72px;
                    font-size:1.35rem;
                    font-weight:800;
                    line-height:1.1;
                    color:#0F172A;
                ">
                    ASX Equities<br>Platform
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(
            '<div class="nav-label">NAVIGATION</div>',
            unsafe_allow_html=True
        )
    
        page = st.radio(
            "Navigation",
            options=[
                "🏠  Overview",
                "📊  Portfolio",
                "🎯  Model Prediction",
                "📈  Backtest Performance",
                "🛡️  Risk Analysis",
                "⚖️  Model Comparison",
                "✨  AI Insights"
            ],
            label_visibility="collapsed"
        )
    return page