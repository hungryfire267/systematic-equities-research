from dotenv import load_dotenv

from google import genai
from google.genai import types
import os
import pandas as pd 
from pathlib import Path 
import streamlit as st

import matplotlib.pyplot as plt

import plotly.express as px


BASE_DIR = Path(__file__).resolve().parents[2]  

DATA_DIR = BASE_DIR/"data"/ "raw"/ "macro"

load_dotenv()

macro_path_dict = {
    "Currency Rates": os.path.join(DATA_DIR, "currency_rates.parquet"),
    "Interest Rates": os.path.join(DATA_DIR, "interest_rates.parquet"),
    "Yield Curves": os.path.join(DATA_DIR, "yield_curves.parquet")
}

ai_prompt_instructions = {
    "Currency Rates": """
        You are a systematic equities quantitative analyst analysing currency exchange rate data. You are currently reviewing several macroeconomic indicators, one of which is currency rates. It is important not to make unsupported assumptions. Focus only on currency movements and avoid detailed discussion of interest rates, CPI, unemployment, GDP, or other macroeconomic indicators, as they are analysed separately.

    Your task is to analyse the following:

    1. Summarise how each currency pair changed over the selected period.
    2. Describe the latest exchange rate levels.
    3. Highlight any major trend, turning point, period of stability, or unusual movement.
    4. Compare which currencies strengthened or weakened the most.
    5. Provide a high-level explanation of likely drivers behind the observed movements, referencing major economic, monetary policy, commodity, or geopolitical events where relevant.
    6. Briefly discuss what currency movements may imply for equity returns, sector performance, export/import exposure, risk appetite, or systematic investment signals. Do not predict future market movements.

    Keep the response concise, clear, and suitable for an investment dashboard. Maximum 20 sentences (no headings).
    """,
    "Interest Rates": """
        You are a systematic equities investment quantitative analyst. You are currently looking currently 
        looking at several macroeconomic indicators. One of which is interest rates. It is important to not assume things. 
        The only focus is on interest rates, try to avoid CPI, Unemployment, Currency as
        much as possible as this will be discussed in their respective sections.
        
        Your task is to analyse the following: 
        1. Summarise how interest rates changed over the full selected period.
        2. Describe the latest interest rate levels.
        3. Highlight any major trend, seasonality, turning point, or unusual movement.
        4. Explain likely causes of the movements, but only at a high level (like any events that caused it like higher inflation and why or unemployment or 
        geopolitical and refer to specific events e.g. Iran War).
        5. Briefly explain what this may mean for equity returns or portfolio signals for this dataset (don't predict the future though).
    
        Keep it concise, simple to the reader in a dashboard and keep it maximum 10 sentences long (no headings)
    """,
    "Yield Curves": """
        You are a systematic equities quantitative analyst analysing Australian government bond yield curve data. You are currently reviewing several macroeconomic indicators, one of which is the yield curve. It is important not to make unsupported assumptions. Focus only on the yield curve and interest-rate expectations. Avoid detailed discussion of inflation, unemployment, currency markets, or other macroeconomic indicators, as they are analysed separately.

        Your task is to analyse the following:

        1. Summarise how yields across the curve changed over the selected period.
        2. Describe the latest yield levels across short-, medium-, and long-term maturities.
        3. Highlight any major trend, seasonality, turning point, or unusual movement.
        4. Describe whether the yield curve became steeper, flatter, or more inverted over time.
        5. Provide a high-level explanation of likely drivers behind the observed movements, referencing major economic, monetary policy, or geopolitical events where relevant.
        6. Briefly discuss what the yield curve behaviour may imply for equity market conditions, sector performance, risk appetite, or systematic investment signals. Do not predict future market movements.

        Keep the response concise, clear, and suitable for an investment dashboard. Maximum 10 sentences. (no headings)
    """
}









st.title("Macroeconomic Analysis")


macro_data_list = list(macro_path_dict.keys())
selected_indicator = st.selectbox("Select a Macroeconomic indicator:", macro_data_list)
macro_df = pd.read_parquet(macro_path_dict[selected_indicator])
macro_df = macro_df.set_index("Date")
st.subheader(f"Past 5 day history of the {selected_indicator}")
st.write(macro_df.tail())

numeric_cols = macro_df.select_dtypes(include="number").columns.tolist()


left_col, right_col = st.columns([3, 1])

with right_col: 
    st.subheader("Filters")

    min_date = macro_df.index.min()
    max_date = macro_df.index.max() 
    
    date_range = st.slider(
        "Date Range", 
        min_value = min_date, 
        max_value = max_date, 
        value = (min_date, max_date),
        format="DD/MM/YYYY"
    )
    
    start_date, end_date = date_range
    
    st.write("Select series to plot")
    
    numeric_cols = macro_df.select_dtypes(include="number").columns.tolist()

    
    if selected_indicator == "Currency Rates":
        selected_col = st.selectbox("Currency pair", numeric_cols)
        selected_cols = [selected_col]
    else:
        selected_cols = []
        for col in numeric_cols:
            if st.checkbox(col, value=True):
                selected_cols.append(col)
    
filtered_df = macro_df.loc[
    (macro_df.index >= start_date) &
    (macro_df.index <= end_date)
]

with left_col:
    st.subheader(f"{selected_indicator} Over Time")

    if selected_cols:
        chart_df = filtered_df[selected_cols].reset_index()

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
            title=f"{selected_indicator} Over Time",
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
    else:
        st.info("Select at least one series to plot.")
        
st.subheader("AI Analysis")
ai_client = genai.Client()
full_prompt = ai_prompt_instructions[selected_indicator] + f"\n The dataset is given as follows: {filtered_df}"
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




