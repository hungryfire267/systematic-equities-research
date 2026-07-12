import pandas as pd

def date_parser(df:pd.DataFrame) -> pd.DataFrame: 
    df.index = pd.to_datetime(df["Date"])
    df = df.drop(columns=["Date"])
    return df

def cross_sectional_ranking(df: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame: 
    mean = df.mean(axis=1)
    std = df.std(axis=1, skipna=True)
    new_df = df.sub(mean, axis=0).div(std, axis=0)
    rank = new_df.rank(axis=1, pct=True, ascending=higher_is_better)
    return rank