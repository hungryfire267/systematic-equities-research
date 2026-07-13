import os
import pandas as pd
from pathlib import Path
import streamlit as st
import sys



BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from scripts.portfolio.metrics import GetMetrics, GetPredictionMetrics
BACKTEST_RESULTS_LIGHTGBM_DIR = BASE_DIR / "results" /  "backtest" / "lightgbm"
BACKTEST_RESULTS_XGBOOST_DIR = BASE_DIR / "results" /  "backtest" / "xgboost"

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.markdown("""
<h2 style='color:#4F8BF9;'>
📊 Model Comparison
</h2>
""", unsafe_allow_html=True)



#### Get Data for each model

final_portfolio_lightgbm = pd.read_parquet(os.path.join(BACKTEST_RESULTS_LIGHTGBM_DIR, "final_portfolio_stock.parquet"))
final_portfolio_xgboost = pd.read_parquet(os.path.join(BACKTEST_RESULTS_XGBOOST_DIR, "final_portfolio_stock.parquet"))

test_preds_lightgbm = pd.read_parquet(os.path.join(BACKTEST_RESULTS_LIGHTGBM_DIR, "test_preds_stock.parquet"))





metrics_lightgbm, _ = GetMetrics(final_portfolio_lightgbm).run_data()
metrics_xgboost, _ = GetMetrics(final_portfolio_xgboost).run_data()


prediction_metrics_lightgbm_dict = GetPredictionMetrics(final_portfolio_lightgbm).run_data()
prediction_metrics_xgboost_dict = GetPredictionMetrics(final_portfolio_xgboost).run_data()

prediction_metrics_df = pd.DataFrame({
    "Metric": list(prediction_metrics_lightgbm_dict.keys()),
    "LightGBM": list(prediction_metrics_lightgbm_dict.values()),
    "XGBoost": list(prediction_metrics_xgboost_dict.values())
})


st.table(prediction_metrics_df)
st.table(final_portfolio_lightgbm.head(10))
st.table(final_portfolio_xgboost.head(10))



st.write(metrics_lightgbm)
st.write(metrics_xgboost)

