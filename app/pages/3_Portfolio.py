from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import streamlit as st

load_dotenv()

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
st.info(
    """
    Rankings are calculated over the full cross-sectional universe at each
    rebalance date. The holdings table only includes securities with non-zero portfolio
    weights; consequently, displayed ranks are not necessarily consecutive.
    """
)
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

st.header("Portfolio Allocation by Industry")
industry_alloc = (
    weight_df
    .groupby(["Industry", "Side"])["Weight"]
    .sum()
    .reset_index()
)

industry_alloc["Weight (%)"] = industry_alloc["Weight"] * 100

fig = px.bar(
    industry_alloc,
    x="Weight",
    y="Industry",
    color="Side",
    orientation="h",
    barmode="group",
    title=None,
    text="Weight"
)

fig.update_traces(
    texttemplate="%{x:.2f}%",
    textposition="inside",
    insidetextanchor="middle",
    textfont_size=16
)

fig.update_layout(
    height=900,
    yaxis_tickformat=".0%",
    xaxis_title="Industry",
    yaxis_title="Portfolio Weight",
    font=dict(size=18),
    title_font_size=26,
    uniformtext_minsize=12,
    uniformtext_mode="show",
    title_text=""
)

st.plotly_chart(fig, use_container_width=True)

weight_df["Predicted Contribution"] = (
    weight_df["Weight"] * weight_df["Predicted 5D Return"]
)

weight_df["Predicted Contribution (%)"] = weight_df["Predicted Contribution"] * 100

st.header("Predicted Contribution Distribution")
st.markdown("""
This chart shows the **expected contribution of each holding to the portfolio's 5-day return forecast**.
For each stock, predicted contribution is calculated as:
""")

st.latex(r"\text{Predicted Contribution}_i = w_i \hat{r}_i")

st.markdown(r"""
where:

- $w_i$ is the portfolio weight of stock $i$
- st.latex(r"\hat{r}_i") is the model-predicted 5-day return for stock \(i\)

For **long positions**, a positive predicted return increases expected portfolio contribution.  
For **short positions**, a negative predicted return can also contribute positively because the portfolio benefits if the stock falls.
""")

fig = px.bar(
    weight_df.sort_values("Predicted Contribution (%)"),
    x="Predicted Contribution (%)",
    y="Ticker",
    color="Side",
    orientation="h",
    title="Predicted Contribution to Portfolio Return",
    text="Predicted Contribution"
)

fig.update_traces(
    texttemplate="%{x:.3f}%",
    textposition="inside",
    insidetextanchor="middle",
    textfont_size=16
)

fig.update_xaxes(tickformat=".1%")

fig.update_layout(
    height=900,
    yaxis_tickformat=".0%",
    xaxis_title="Portfolio Contribution (%)",
    yaxis_title="Ticker",
    font=dict(size=18),
    title_font_size=26,
    uniformtext_minsize=12,
    uniformtext_mode="show",
    title_text=""
)

st.plotly_chart(fig, use_container_width=True)

def predicted_contribution_summary(df: pd.DataFrame) -> str:
    df = df.copy()
    
    df["predicted_contribution"] = df["Weight"] * df["Predicted 5D Return"]

    n_positions = len(df)
    n_positive = (df["predicted_contribution"] > 0).sum()
    n_negative = (df["predicted_contribution"] < 0).sum()

    long_positive = ((df["Side"] == "long") & (df["predicted_contribution"] > 0)).sum()
    long_negative = ((df["Side"] == "long") & (df["predicted_contribution"] < 0)).sum()

    short_positive = ((df["Side"] == "short") & (df["predicted_contribution"] > 0)).sum()
    short_negative = ((df["Side"] == "short") & (df["predicted_contribution"] < 0)).sum()

    top_pos_row = df.loc[df["predicted_contribution"].idxmax()]
    top_neg_row = df.loc[df["predicted_contribution"].idxmin()]

    top_pos_ticker = top_pos_row["Ticker"]
    top_pos_side = top_pos_row["Side"]
    top_pos_value = top_pos_row["predicted_contribution"]

    top_neg_ticker = top_neg_row["Ticker"]
    top_neg_side = top_neg_row["Side"]
    top_neg_value = top_neg_row["predicted_contribution"]

    summary = (
        f"On the selected rebalance date, the portfolio contains {n_positions} active positions. "
        f"Of these, {n_positive} positions have positive predicted contributions and {n_negative} have negative predicted contributions. "
        f"The long book contains {long_positive} positive and {long_negative} negative contributors, "
        f"while the short book contains {short_positive} positive and {short_negative} negative contributors. "
        f"The largest positive contribution comes from the {top_pos_side.lower()} position in {top_pos_ticker} "
        f"({top_pos_value:.3%}), while the most negative contribution comes from the {top_neg_side.lower()} position in "
        f"{top_neg_ticker} ({top_neg_value:.3%})."
    )

    return summary

st.write(predicted_contribution_summary(weight_df))

st.header("AI Summary")
ai_client = genai.Client()
ai_prompt = f"""
You are a quantitative systematic portfolio analyst at an asset manager.

Write a concise portfolio commentary for the current rebalance of an ASX market-neutral top-20 long / top-20 short equity strategy.

Strategy context:
- The model ranks stocks by predicted 5-day return.
- The strategy goes long the highest-ranked stocks and short the lowest-ranked stocks.
- The portfolio targets approximately 1.00 long exposure, -1.00 short exposure, 0.00 net exposure, and 2.00x gross exposure.
- The displayed active portfolio may contain fewer than 20 names on either side if some names receive zero weight after portfolio construction.

Use the portfolio tables below to summarise:
1. portfolio structure
2. long and short positioning
3. industry exposure and sector tilts
4. expected return drivers from predicted contribution
5. key concentration or portfolio
6. Talk about expected impact to portfolio and any reasons why this has happened over {rebalance_date} in backtesting (e.g. economy and what has happened to this stock recently)

Interpretation rules:
- Long positions benefit from positive predicted returns.
- Short positions benefit from negative predicted stock returns.
- Do not restate every row in the tables.
- Focus on the largest positions, largest industry tilts, top contributors, and concentration risks.
- If the portfolio is concentrated in a small number of names or industries, say so.
- Write in a professional buy-side tone.

Keep the response under 250 words. No Headings.

=== LONG HOLDINGS TABLE ===
{long_portfolio_df}

=== SHORT HOLDINGS TABLE ===
{short_portfolio_df}

=== INDUSTRY EXPOSURE TABLE ===
{weight_df}

=== PREDICTED CONTRIBUTION TABLE ===
{weight_df}
"""

try: 
    response = ai_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=ai_prompt,
        config=types.GenerateContentConfig(
            temperature=0.5
        )
    )
    
    st.write(response.text)
except Exception as e: 
    print(e)