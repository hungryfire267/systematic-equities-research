import streamlit as st
from components.sidebar import render_sidebar
from views.overview import render_overview
from views.portfolio import render_portfolio
from views.backtest_performance import render_backtesting
from views.model_comparison import render_model_comparison
from views.methodology_lessons import render_methodology_lessons

st.set_page_config(
    page_title="ASX Alpha System",
    page_icon="📈",
    layout="wide"
)

selected_page = render_sidebar()

if selected_page == "🏠  Overview":
    render_overview()
elif selected_page == "📊  Portfolio":
    render_portfolio()
elif selected_page == "📈  Backtest Performance": 
    render_backtesting()
elif selected_page == "⚖️  Model Comparison":
    render_model_comparison()
elif selected_page == "✨  Methodology":
    render_methodology_lessons()
    
