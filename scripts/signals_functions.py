import numpy as np 
import pandas as pd


def date_parser(df:pd.DataFrame) -> pd.DataFrame: 
    df.index = pd.to_datetime(df["Date"])
    df = df.drop(columns=["Date"])
    return df