import pandas as pd

def get_hit_contingency_table(lightgbm_hit_df, xgboost_hit_df):
    hit_comparison = (
        lightgbm_hit_df[
            ["Date", "Ticker", "hit"]
        ]
        .rename(columns={"hit": "lightgbm_hit"})
        .merge(
            xgboost_hit_df[
                ["Date", "Ticker", "hit"]
            ].rename(columns={"hit": "xgboost_hit"}),
            on=["Date", "Ticker"],
            how="inner",
            validate="one_to_one",
        )
        .dropna(subset=["lightgbm_hit", "xgboost_hit"])
    )

    hit_contingency_table = pd.crosstab(
        hit_comparison["lightgbm_hit"],
        hit_comparison["xgboost_hit"],
    ).reindex(
        index=[True, False],
        columns=[True, False],
        fill_value=0,
    )

    hit_contingency_table.index = [
        "LightGBM hit",
        "LightGBM miss",
    ]

    hit_contingency_table.columns = [
        "XGBoost hit",
        "XGBoost miss",
    ]
    
    return hit_contingency_table