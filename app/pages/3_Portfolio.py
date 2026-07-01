import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Portfolio Analysis",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]
BACKTEST_RESULTS_DIR = BASE_DIR / "results" /  "backtest"
UNIVERSE_PATH = os.path.join(BASE_DIR, "data/asx_companies.csv")

portfolio_dict = { 
    "portfolio": os.path.join(BACKTEST_RESULTS_DIR, "final_portfolio.parquet"),
    "rank": os.path.join(BACKTEST_RESULTS_DIR, "test_preds_rank.parquet")   
}

company_df = pd.read_csv(UNIVERSE_PATH)
print(company_df)

st.title("Portfolio Analysis")
st.header("Overview")

final_portfolio_df = pd.read_parquet(portfolio_dict["portfolio"])
test_pred_rank_df = pd.read_parquet(portfolio_dict["rank"])
test_pred_rank_df = test_pred_rank_df.set_index("Date")

rebalance_date = final_portfolio_df["Date"].iloc[-1]
latest_date = rebalance_date.strftime("%Y-%m-%d")

portfolio_df = final_portfolio_df[final_portfolio_df["Date"] == rebalance_date].copy()
industry_list, rank_list = [], []
for ticker in portfolio_df["Ticker"]:
    ticker_code = ticker.split(".")[0]
    industry_row = company_df[company_df["asxCode"] == ticker_code]    
    industry = industry_row["industry"].values[0]
    industry_list.append(industry)
    
    rank = test_pred_rank_df.loc[rebalance_date, ticker]
    rank_list.append(rank)
portfolio_df["industry"] = industry_list
portfolio_df["rank"] = rank_list
    

portfolio_df = portfolio_df[portfolio_df["weight"] != 0]
portfolio_df = portfolio_df[["Ticker", "side", "weight", "prediction", "rank", "industry"]].rename(
    columns = {
        "side": "Side", 
        "weight": "Weight", 
        "prediction": "Predicted 5D Return", 
        "rank": "Rank", 
        "industry": "Industry"
    }
)
long_portfolio_df = portfolio_df[portfolio_df["Side"] == "long"].copy()
short_portfolio_df = portfolio_df[portfolio_df["Side"] == "short"].copy()

n_positions = portfolio_df.shape[0]
n_long = long_portfolio_df.shape[0]
n_short = short_portfolio_df.shape[0]

net_exposure = portfolio_df["Weight"].sum()
if abs(net_exposure) < 1e-10:
    net_exposure = 0
gross_exposure = portfolio_df["Weight"].abs().sum()
long_exposure = portfolio_df.loc[
    portfolio_df["Weight"] > 0,
    "Weight"
].sum()
short_exposure = portfolio_df.loc[
    portfolio_df["Weight"] < 0,
    "Weight"
].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Rebalance Date", latest_date)
col2.metric("Number of Positions", n_positions)
col3.metric("Number of Long Postiions", n_long)
col4.metric("Number of Short Positions", n_short)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Net Exposure", f"{net_exposure:.2%}")
col6.metric("Gross Exposure", f"{gross_exposure:.2f}x")
col7.metric("Long Exposure", f"{long_exposure:.2f}")
col8.metric("Short Exposure", f"{short_exposure:.2f}")

st.header("Current Holdings")
st.markdown("""
<div style="
    background-color:#E8F5E9;
    padding:15px;
    border-radius:8px;
    border-left:5px solid #4CAF50;
    font-size:20px;
    line-height:1.6;
">
<strong>Note:</strong> Rankings are calculated over the full cross-sectional universe at each
rebalance date. The holdings table only includes securities with non-zero portfolio
weights; consequently, displayed ranks are not necessarily consecutive.
</div>
""", unsafe_allow_html=True)
st.subheader("Long Position")
st.dataframe(long_portfolio_df, hide_index=True)
st.subheader("Short Position")
st.dataframe(short_portfolio_df, hide_index=True)

weight_df = portfolio_df.copy() 
weight_df["Weight (%)"] = weight_df["Weight"] * 100
weight_df = weight_df.sort_values("Weight")

st.header("Current Portfolio Weight Distribution")
fig = px.bar(
    weight_df,
    x="Weight (%)",
    y="Ticker",
    color="Side",
    title="Plot",
    orientation="h",
    text="Weight (%)",
    hover_data=["Weight", "Predicted 5D Return", "Rank", "Industry"]
)

fig.update_traces(
    texttemplate="%{x:.2f}%",
    textposition="inside",
    insidetextanchor="middle",
    textfont_size=16
)

fig.update_layout(
    height=900,
    font=dict(size=18),
    title=dict(font=dict(size=26)),
    xaxis_title="Portfolio Weight (%)",
    yaxis_title="Ticker",
    uniformtext_minsize=12,
    uniformtext_mode="show"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Header")
industry_alloc = (
    weight_df
    .groupby(["Industry", "Side"])["Weight"]
    .sum()
    .reset_index()
)

fig = px.bar(
    industry_alloc,
    x="Weight",
    y="Industry",
    color="Side",
    orientation="h",
    barmode="group",
    title="Portfolio Allocation by Industry",
    text="Weight"
)

fig.update_traces(
    texttemplate="%{y:.2%}",
    textposition="outside"
)

fig.update_layout(
    yaxis_tickformat=".0%",
    xaxis_title="Industry",
    yaxis_title="Portfolio Weight",
    font=dict(size=18),
    title_font_size=26
)

st.plotly_chart(fig, use_container_width=True)

