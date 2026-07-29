import base64
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


def _format_sidebar_date(value) -> str:
    """Format a date-like value for the strategy overview."""
    if value is None:
        return "Not available"

    parsed_date = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed_date):
        return "Not available"

    return parsed_date.strftime("%d %b %Y")


def _get_strategy_overview() -> dict[str, str]:
    """Read the active strategy details stored by the dashboard pages."""
    selected_configuration = st.session_state.get(
        "selected_configuration",
        {}
    )

    model_key = selected_configuration.get(
        "model_key",
        st.session_state.get(
            "selected_model_key",
            "lightgbm"
        )
    )

    feature_key = selected_configuration.get(
        "feature_key",
        st.session_state.get(
            "selected_feature_key",
            "stock"
        )
    )

    model_names = {
        "dt": "Decision Tree",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
    }

    feature_names = {
        "stock": "Stock Features",
        "market": "Stock + Market",
    }

    selected_model = model_names.get(
        str(model_key).strip().lower(),
        "LightGBM"
    )

    selected_feature_set = feature_names.get(
        str(feature_key).strip().lower(),
        "Stock Features"
    )

    backtest_start_date = selected_configuration.get(
        "backtest_start_date",
        st.session_state.get("backtest_start_date")
    )

    latest_rebalance_date = selected_configuration.get(
        "latest_rebalance_date",
        st.session_state.get("latest_rebalance_date")
    )

    dataset_name = selected_configuration.get(
        "dataset_name",
        st.session_state.get(
            "dataset_name",
            "ASX equities"
        )
    )

    if backtest_start_date is None or latest_rebalance_date is None:
        base_dir = Path(__file__).resolve().parents[2]
        portfolio_path = (
            base_dir
            / "results"
            / "backtest"
            / str(model_key)
            / f"final_portfolio_{feature_key}.parquet"
        )

        if portfolio_path.exists():
            portfolio_dates = pd.read_parquet(
                portfolio_path,
                columns=["Date"]
            )
            parsed_dates = pd.to_datetime(
                portfolio_dates["Date"],
                errors="coerce"
            ).dropna()

            if not parsed_dates.empty:
                if backtest_start_date is None:
                    backtest_start_date = parsed_dates.min()

                if latest_rebalance_date is None:
                    latest_rebalance_date = parsed_dates.max()

                st.session_state["backtest_start_date"] = (
                    backtest_start_date
                )
                st.session_state["latest_rebalance_date"] = (
                    latest_rebalance_date
                )

    return {
        "start_date": _format_sidebar_date(backtest_start_date),
        "latest_date": _format_sidebar_date(latest_rebalance_date),
        "dataset": str(dataset_name),
        "model": str(selected_model),
        "feature_set": str(selected_feature_set),
    }


def render_sidebar():
    logo_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "logo.png"
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: #F4F7FB;
        }

        [data-testid="stSidebarContent"] {
            padding-bottom: 12.5rem;
        }

        /* Hide radio circles */
        [data-testid="stSidebar"]
        div[role="radiogroup"] label > div:first-child {
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
        [data-testid="stSidebar"]
        div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(
                90deg,
                #1FB7A6 0%,
                #2F80ED 100%
            );
            box-shadow: 0 5px 15px rgba(47, 128, 237, 0.22);
        }

        /* Selected text */
        [data-testid="stSidebar"]
        div[role="radiogroup"] label:has(input:checked) p {
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



        .author-card {
            position: fixed;
            left: 1rem;
            top: 55%;
            transform: translateY(-50%);
            width: 13rem;
            z-index: 20;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.7rem;
            padding: 0.72rem 0.82rem;
            border: 1px solid #D8E0EA;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
        }

        .author-card-text {
            color: #334155;
            font-size: 0.72rem;
            font-weight: 750;
            line-height: 1.25;
        }

        .author-github-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 50%;
            color: #0F172A;
            text-decoration: none;
            transition: 0.2s ease;
        }

        .author-github-link:hover {
            background: #EAF1F8;
            transform: translateY(-1px);
        }

        .author-github-icon {
            width: 1.38rem;
            height: 1.38rem;
            display: block;
            object-fit: contain;
        }

        .strategy-overview-card {
            position: fixed;
            left: 1rem;
            bottom: 0.85rem;
            width: 13rem;
            z-index: 20;
            box-sizing: border-box;
            padding: 0.78rem 0.82rem;
            border: 1px solid #D8E0EA;
            border-left: 3px solid #64748B;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
        }

        .strategy-overview-title {
            display: flex;
            align-items: center;
            gap: 0.42rem;
            margin-bottom: 0.6rem;
            color: #0F172A;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .strategy-overview-title-symbol {
            color: #64748B;
            font-size: 0.82rem;
        }

        .strategy-overview-row {
            display: grid;
            grid-template-columns: 3.5rem minmax(0, 1fr);
            gap: 0.45rem;
            align-items: start;
            margin-top: 0.38rem;
            font-size: 0.66rem;
            line-height: 1.4;
        }

        .strategy-overview-label {
            color: #8291A6;
            font-weight: 750;
        }

        .strategy-overview-value {
            color: #334155;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .strategy-overview-period {
            margin-top: 0.55rem;
            padding-top: 0.55rem;
            border-top: 1px solid #E2E8F0;
            color: #64748B;
            font-size: 0.62rem;
            line-height: 1.45;
        }

        .strategy-overview-period strong {
            color: #334155;
            font-weight: 750;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if not logo_path.exists():
        raise FileNotFoundError(
            f"Sidebar logo was not found: {logo_path}"
        )

    with open(logo_path, "rb") as image_file:
        logo_base64 = base64.b64encode(
            image_file.read()
        ).decode()

    strategy_overview = _get_strategy_overview()

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

        NAV_OPTIONS = [
            "🪐  Overview",
            "💠  Portfolio",
            "🌊  Backtest Performance",
            "🧩  Model Comparison",
            "✨  Methodology"
        ]

        SLUG_TO_LABEL = {
            "overview": "🪐  Overview",
            "portfolio": "💠  Portfolio",
            "backtest_performance": "🌊  Backtest Performance",
            "model_comparison": "🧩  Model Comparison",
            "methodology": "✨  Methodology",
        }
        LABEL_TO_SLUG = {v: k for k, v in SLUG_TO_LABEL.items()}

        # Seed initial selection from the URL, once per session
        if "nav_page" not in st.session_state:
            url_slug = st.query_params.get("page", "overview")
            st.session_state.nav_page = SLUG_TO_LABEL.get(url_slug, NAV_OPTIONS[0])

        page = st.radio(
            "Navigation",
            options=NAV_OPTIONS,
            label_visibility="collapsed",
            key="nav_page"
        )

        # Keep the URL in sync with whatever is currently selected
        st.query_params["page"] = LABEL_TO_SLUG.get(page, "overview")



        github_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="#0F172A" d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58 0-.29-.01-1.04-.02-2.04-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23A11.5 11.5 0 0 1 12 6.32c1.02 0 2.04.14 3 .4 2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.62-5.49 5.92.43.37.81 1.1.81 2.22 0 1.61-.01 2.9-.01 3.29 0 .32.22.69.82.57A12.01 12.01 0 0 0 24 12c0-6.63-5.37-12-12-12Z"/>
        </svg>
        """.strip()

        github_icon_base64 = base64.b64encode(
            github_svg.encode("utf-8")
        ).decode("utf-8")

        st.markdown(
            f"""
            <div class="author-card">
                <div class="author-card-text">By Gordon Li</div>
                <a
                    class="author-github-link"
                    href="https://github.com/hungryfire267/systematic-equities-research"
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="View Gordon Li's GitHub repository"
                    title="View GitHub repository"
                >
                    <img
                        class="author-github-icon"
                        src="data:image/svg+xml;base64,{github_icon_base64}"
                        alt="GitHub"
                    >
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.html(
            f"""
            <div class="strategy-overview-card">
                <div class="strategy-overview-title">
                    <span class="strategy-overview-title-symbol">✦</span>
                    <span>Strategy Overview</span>
                </div>

                <div class="strategy-overview-row">
                    <div class="strategy-overview-label">Dataset</div>
                    <div class="strategy-overview-value">
                        {escape(strategy_overview["dataset"])}
                    </div>
                </div>

                <div class="strategy-overview-row">
                    <div class="strategy-overview-label">Model</div>
                    <div class="strategy-overview-value">
                        {escape(strategy_overview["model"])}
                    </div>
                </div>

                <div class="strategy-overview-row">
                    <div class="strategy-overview-label">Features</div>
                    <div class="strategy-overview-value">
                        {escape(strategy_overview["feature_set"])}
                    </div>
                </div>

                <div class="strategy-overview-period">
                    Evaluation period:
                    <strong>{escape(strategy_overview["start_date"])}</strong>
                    to
                    <strong>{escape(strategy_overview["latest_date"])}</strong>
                </div>
            </div>
            """
        )

    return page
