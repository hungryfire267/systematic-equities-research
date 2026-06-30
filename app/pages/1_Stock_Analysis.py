from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
from scripts.dashboard.get_stock_metrics import GetStockMetrics
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data"/ "raw" / "companies"

companies_path_dict = { 
    "Prices": os.path.join(COMPANIES_DIR, "prices.parquet"), 
    "Market Cap": os.path.join(COMPANIES_DIR, "market_cap.parquet"),
    "Volume": os.path.join(COMPANIES_DIR, "volume.parquet"),
    "Returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}








load_dotenv()

ai_prompt_instructions = {
    "overview": """
        You are writing a concise company overview for an investment dashboard.

        Using the information below, write a factual summary in 3–4 sentences.

        Include:
        - What the company does.
        - Its primary products or services.
        - The industries or sectors it operates in.
        - Any notable geographic presence or market position.

        Do not discuss recent share price performance, future outlook, investment recommendations or financial advice.
        Keep the tone professional and objective.
        
    """
}


companies_path_dict = { 
    "Prices": os.path.join(COMPANIES_DIR, "prices.parquet"), 
    "Market Cap": os.path.join(COMPANIES_DIR, "market_cap.parquet"),
    "Volume": os.path.join(COMPANIES_DIR, "volume.parquet"),
    "Returns": os.path.join(COMPANIES_DIR, "returns.parquet")
}




company_df_path = os.path.join(BASE_DIR, "data/asx_companies.csv")
company_df = pd.read_csv(company_df_path)
asx_codes = company_df["asxCode"].tolist()



st.title("Stock Analysis")

company_code = st.text_input(
    "Enter company code",
    placeholder="e.g. WBC, CBA, BHP"
).upper().strip()

if "success" not in st.session_state: 
    st.session_state.success = False
if "company_code" not in st.session_state:
    st.session_state.company_code = ""

if st.button("Generate"):
    if not company_code:
        st.warning("Please enter a company code.")
    elif company_code not in asx_codes:
        st.error("Invalid ASX code.")
        st.session_state.success = False   
    else:
        st.session_state.success = True
        st.session_state.company_code = company_code
        st.success(f"Generating analysis for {company_code}")
        

if st.session_state.success:
    company_code = st.session_state.company_code
    final_company_code = company_code + ".AX"
    
    company_name = company_df.loc[company_df["asxCode"] == company_code, "companyName"].iloc[0]
    company_name = company_name.rstrip(".")
    
    st.markdown(f"""
        <p style="font-size:20px; line-height:1.6;">
        This page provides a stock-level overview of 
        <strong>{company_name} ({final_company_code})</strong>, showing how its price,
        returns and risk profile have changed over the selected period.
        </p>

        <p style="font-size:20px; line-height:1.6;">
        The visualisations allow users to inspect historical price movements, compare different metrics,
        and assess the stock’s behaviour through quantitative indicators such as returns, volatility,
        drawdowns and model-based signals.
        </p>

        <p style="font-size:20px; line-height:1.6;">
        This overview is intended to support exploratory equity analysis rather than provide financial advice.
        </p>
        """, unsafe_allow_html=True
    )
    
    st.subheader("Summary")
    ai_client = genai.Client()
    full_prompt = ai_prompt_instructions["overview"] + f"\n The company name is {company_name}"
    print(full_prompt)
    try: 
        response = ai_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.5
            )
        )
        
        st.write(response.text)
    except Exception as e: 
        print(e)
    
    prices_df = pd.read_parquet(companies_path_dict["Prices"])
    market_cap_df = pd.read_parquet(companies_path_dict["Market Cap"])
    volume_df = pd.read_parquet(companies_path_dict["Volume"])
    returns_df = pd.read_parquet(companies_path_dict["Returns"])
    
    st.subheader(f"Price Statistics")
    price_statistics_dict = GetStockMetrics(prices_df, final_company_code).get_price_statistics()
    
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", price_statistics_dict["start_date"].strftime("%Y-%m-%d"))
    col2.metric("Latest Date", price_statistics_dict["end_date"].strftime("%Y-%m-%d"))
    col3.metric("Current Price", "$" + str(round(price_statistics_dict["latest_price"], 2)))
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Start Price", "$" + str(round(price_statistics_dict["start_price"], 2)))
    col5.metric("Lowest Price", "$" + str(round(price_statistics_dict["lowest_price"], 2)))
    col6.metric("Highest Price", "$" + str(round(price_statistics_dict["highest_price"], 2)))

    col7, col8, col9 = st.columns(3)
    col7.metric("Total Return", str(round(100 * price_statistics_dict["total_return"], 2)) + "%")
    col8.metric("Latest 21D Return", str(round(100 * price_statistics_dict["Latest 21d Return"][0], 2)) + "%")
    col9.metric("Latest 63D Return", str(round(100 * price_statistics_dict["Latest_63d Return"][0], 2)) + "%")
    
    st.subheader("Risk Statistics")
    
    st.subheader()
    
    prediction_df = { 
    
    
    
    
    
    
    
    
    
    }
    
    
    
    
    
    st.subheader(f"Visualisations for {company_name}")
    
    df_map_dict = dict() 
    for metric in companies_path_dict.keys(): 
        df = pd.read_parquet(companies_path_dict[metric])[["Date", final_company_code]]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.set_index("Date")
        df_map_dict[metric] = df
        

    left_col, right_col = st.columns([3, 1])
    with right_col: 
        st.subheader("Filters")
        metric = st.selectbox(
            "Select Metric to Plot", 
            ["Prices", "Market Cap", "Volume", "Returns"]
        )
        
        df = df_map_dict[metric]
        
        min_date = df.index.min() 
        max_date = df.index.max()
        
        date_range = st.slider(
            "Date Range", 
            min_value = min_date, 
            max_value = max_date, 
            value = (min_date, max_date),
            format="DD/MM/YYYY"
        )
        
        start_date, end_date = date_range
    
    start_condition = (df.index >= start_date)
    end_condition = (df.index <= end_date)
    
    filtered_df = df.loc[start_condition & end_condition]

    with left_col:

        chart_df = filtered_df.reset_index()

        chart_df = chart_df.melt(
            id_vars="Date",
            var_name="Series",
            value_name="Value"
        )

        fig = px.line(
            chart_df,
            x="Date",
            y="Value",
            color="Series",
            title=f"{metric} Over Time",
            labels={
                "date": "Date",
                "Value": "Rate / Yield (%)",
                "Series": "Series"
            }
        )

        fig.update_xaxes(tickformat="%b %Y")
        fig.update_layout(
            height=500,
            legend_title_text="Series"
        )

        st.plotly_chart(fig, width="stretch")

        
        
        
    

    
        
        

    



