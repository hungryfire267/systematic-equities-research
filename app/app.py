import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="ASX Alpha System",
    page_icon="📈",
    layout="wide"
)

render_sidebar()

pages = { 
    "Navigation": [
        st.Page(
            "views/5_Model_Comparison.py",
            title="Model Comparison",
            icon=":material/compare_arrows:"
        )
    ]
}

selected_page = st.navigation(pages)
selected_page.run()