import os
import pandas as pd 
from pathlib import Path 
import streamlit as st

import matplotlib.pyplot as plt

import plotly.express as px


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR/"data"/ "raw"/ "macro"


macro_path_dict = {
    "Interest Rates": os.path.join(DATA_DIR, "interest_rates.parquet"),
    "Yield Curves": os.path.join(DATA_DIR, "yield_curves.parquet")
}

ai_prompt_instructions = {
    "Interest Rates": """
        You are a systematic equities investment quantitative analyst. You are currently looking currently 
        looking at several macroeconomic indicators. One of which is interest rates. It is important to not assume things. 
        The only focus is on interest rates, try to avoid CPI, Unemployment, Currency as
        much as possible as this will be discussed in their respective sections.
        
        Your task is to analyse the following: 
        1. Summarise how interest rates changed over the full selected period.
        2. Describe the latest interest rate levels.
        3. Highlight any major trend, turning point, or unusual movement.
        4. Explain likely causes of the movements, but only at a high level.
        5. Briefly explain what this may mean for equity returns or portfolio signals.
    
        Keep it concise, simple to the reader in a dashboard and keep it maximum 10 sentences long. 
    """
}









st.title("Macroeconomic Analysis")


macro_data_list = list(macro_path_dict.keys())
selected_indicator = st.selectbox("Select a Macroeconomic indicator:", macro_data_list)
macro_df = pd.read_parquet(macro_path_dict[selected_indicator])
macro_df = macro_df.set_index("Date")
st.subheader(f"Past 5 day history of the {selected_indicator}")
st.write(macro_df.tail())

st.write("Select series to plot")

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
    
    st.write("Series")
    
    numeric_cols = macro_df.select_dtypes(include="number").columns.tolist()

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

try: 
    response = ai_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    
except Exception as e: 
    print(e)




