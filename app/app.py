import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ASX Systematic Equities Dashboard",
    layout="wide"
)

st.title("ASX Systematic Equities Dashboard")

st.markdown("""
Use the sidebar to explore:

- Data Overview
- Signals
- Backtest
- Portfolio
- Model Diagnostics
""")

col1, col2 = st.columns(2)

with col1: 
    st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis")
    
with col2: 
    st.page_link("pages/2_Economic_Analysis.py", label="Economic Analysis")