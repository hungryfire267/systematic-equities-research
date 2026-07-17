from functools import reduce
import os
from pathlib import Path

import pandas as pd

from scripts.preprocessing.build_feature_matrix import FeatureMatrixBuilder
from scripts.preprocessing.build_macromarket_matrix import BuildMacroMarketMatrix
from scripts.preprocessing.build_targets import ForwardReturns
from scripts.preprocessing.get_feature_signals import GetFeatureSignals
from scripts.signals.market import MarketSignals


BASE_DIR = Path(__file__).resolve().parents[1]

MACRO_DIR = BASE_DIR / "data" / "raw" / "macro"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_COMPANIES_DIR = PROCESSED_DIR / "companies"
PROCESSED_MARKETS_DIR = PROCESSED_DIR / "markets"
PROCESSED_FEATURE_DIR = PROCESSED_DIR / "features"

for directory in [
    PROCESSED_DIR,
    PROCESSED_COMPANIES_DIR,
    PROCESSED_MARKETS_DIR,
    PROCESSED_FEATURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ================================================================
# STOCK SIGNAL PATHS
# ================================================================

processed_paths_dict = {
    "autocorr": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "autocorr.parquet",
    ),
    "beta": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "beta.parquet",
    ),
    "mean_volatility": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "mean_volatility.parquet",
    ),
    "microstructure": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "microstructure.parquet",
    ),
    "momentum": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "momentum.parquet",
    ),
    "momentum_liquidity": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "momentum_liquidity.parquet",
    ),
    "pvo": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "pvo.parquet",
    ),
    "reversal": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "reversal.parquet",
    ),
    "reversal_illiquidity": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "reversal_illiquidity.parquet",
    ),
    "trend": os.path.join(
        PROCESSED_COMPANIES_DIR,
        "trend.parquet",
    ),
}


predictive_factors_dict = {
    "autocorr": [
        "autocorr_21",
        "autocorr_63",
    ],
    "beta": [
        "market_beta_63",
        "industry_beta_63",
        "market_resid_vol_126",
        "industry_resid_vol_126",
    ],
    "mean_volatility": [
        "mean_volatility_10",
        "mean_volatility_21",
        "mean_volatility_63",
    ],
    "microstructure": [
        "amihud_21",
    ],
    "momentum": [
        "momentum_252_21",
    ],
    "momentum_liquidity": [
        "momentum_liquidity_21",
    ],
    "reversal": [
        "reversal_5",
        "rsr_21",
    ],
    "trend": [
        "trend_21",
        "trend_63",
        "trend_126",
        "r2_21",
        "r2_63",
        "r2_126",
    ],
}


# ================================================================
# MARKET SIGNAL PATHS
# ================================================================

market_paths_dict = {
    "beta_market_return": os.path.join(
        PROCESSED_MARKETS_DIR,
        "beta_market_return.parquet",
    ),
    "beta_market_volatility": os.path.join(
        PROCESSED_MARKETS_DIR,
        "beta_market_volatility.parquet",
    ),
    "beta_market_drawdown": os.path.join(
        PROCESSED_MARKETS_DIR,
        "beta_market_drawdown.parquet",
    ),
}


# ================================================================
# MACRO SIGNAL PATHS
# ================================================================

macro_paths_dict = {
    "currency_rates": os.path.join(
        MACRO_DIR,
        "currency_rates.parquet",
    ),
    "interest_rates": os.path.join(
        MACRO_DIR,
        "interest_rates.parquet",
    ),
    "vix": os.path.join(
        MACRO_DIR,
        "vix.parquet",
    ),
}


# ================================================================
# OUTPUT PATHS
# ================================================================

feature_matrix_pipeline_dict = {
    "feature_matrix_stock": os.path.join(
        PROCESSED_FEATURE_DIR,
        "feature_matrix_stock.parquet",
    ),
    "feature_matrix_market": os.path.join(
        PROCESSED_FEATURE_DIR,
        "feature_matrix_market.parquet",
    ),
    "feature_matrix_macro_market": os.path.join(
        PROCESSED_FEATURE_DIR,
        "feature_matrix_macro_market.parquet",
    ),
}


# ================================================================
# HELPERS
# ================================================================

def ensure_date_column(
    dataframe: pd.DataFrame,
    dataframe_name: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "Date" not in dataframe.columns:
        dataframe = dataframe.reset_index()

    if "Date" not in dataframe.columns:
        raise ValueError(
            f"{dataframe_name} does not contain a Date column."
        )

    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"]
    )

    return dataframe


def load_stock_feature_dataframes() -> dict[str, pd.DataFrame]:
    """
    Load each stock-level feature table.

    Every dataframe should contain:
        Date
        Ticker
        selected feature columns
    """
    feature_dfs_dict = {}

    for feature_name, feature_columns in predictive_factors_dict.items():
        feature_path = processed_paths_dict[feature_name]

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Missing stock feature file: {feature_path}"
            )

        feature_df = pd.read_parquet(
            feature_path
        )

        feature_df = ensure_date_column(
            feature_df,
            feature_name,
        )

        required_columns = [
            "Date",
            "Ticker",
            *feature_columns,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in feature_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{feature_name} is missing columns: "
                f"{missing_columns}"
            )

        feature_df = (
            feature_df[required_columns]
            .sort_values(["Date", "Ticker"])
            .drop_duplicates(
                subset=["Date", "Ticker"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        feature_dfs_dict[feature_name] = feature_df

    return feature_dfs_dict


def collapse_market_feature(
    feature_name: str,
    feature_path: str,
) -> pd.DataFrame:
    """
    Convert a wide market feature parquet into one market value per date.

    Input example:
        Date | CBA.AX | BHP.AX | ZIP.AX | ...

    Output:
        Date | beta_market_return

    The current implementation uses the cross-sectional mean.
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Missing market feature file: {feature_path}"
        )

    feature_df = pd.read_parquet(
        feature_path
    )

    feature_df = ensure_date_column(
        feature_df,
        feature_name,
    )

    value_columns = [
        column
        for column in feature_df.columns
        if column != "Date"
    ]

    if not value_columns:
        raise ValueError(
            f"{feature_name} contains no feature-value columns."
        )

    feature_df[value_columns] = feature_df[
        value_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    feature_df[feature_name] = (
        feature_df[value_columns]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    return (
        feature_df[
            [
                "Date",
                feature_name,
            ]
        ]
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def build_market_matrix(
    paths_dict: dict[str, str],
) -> pd.DataFrame:
    """
    Build a date-level market matrix containing exactly one column
    for each key in market_paths_dict.
    """
    market_dataframes = []

    for feature_name, feature_path in paths_dict.items():
        market_feature_df = collapse_market_feature(
            feature_name=feature_name,
            feature_path=feature_path,
        )

        market_dataframes.append(
            market_feature_df
        )

    if not market_dataframes:
        raise ValueError(
            "No market feature dataframes were created."
        )

    market_df = reduce(
        lambda left, right: left.merge(
            right,
            on="Date",
            how="outer",
            validate="one_to_one",
        ),
        market_dataframes,
    )

    market_df = (
        market_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    expected_columns = [
        "Date",
        *paths_dict.keys(),
    ]

    market_df = market_df[
        expected_columns
    ]

    return market_df


def prepare_macro_matrix(
    paths_dict: dict[str, str],
) -> pd.DataFrame:
    macro_df = BuildMacroMarketMatrix(
        paths_dict
    ).run_data()

    macro_df = ensure_date_column(
        macro_df,
        "macro matrix",
    )

    return (
        macro_df
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def validate_matrix_columns(
    stock_matrix: pd.DataFrame,
    market_matrix: pd.DataFrame,
    combined_matrix: pd.DataFrame,
) -> None:
    expected_market_columns = set(
        market_paths_dict.keys()
    )

    actual_new_columns = (
        set(combined_matrix.columns)
        - set(stock_matrix.columns)
    )

    if actual_new_columns != expected_market_columns:
        raise ValueError(
            "Unexpected market columns were added.\n"
            f"Expected: {sorted(expected_market_columns)}\n"
            f"Actual: {sorted(actual_new_columns)}"
        )

    if list(market_matrix.columns) != [
        "Date",
        *market_paths_dict.keys(),
    ]:
        raise ValueError(
            "Market matrix columns are not in the expected format."
        )


# ================================================================
# PIPELINE
# ================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1. Generate market signal parquet files
    # ------------------------------------------------------------

    print("1. Generating market signals")

    MarketSignals(
        market_paths_dict
    ).run_data()


    # ------------------------------------------------------------
    # 2. Build three-column market matrix
    # ------------------------------------------------------------

    print("2. Building market matrix")

    market_df = build_market_matrix(
        market_paths_dict
    )

    print(
        "Market matrix columns:",
        market_df.columns.tolist(),
    )

    print(
        "Market matrix shape:",
        market_df.shape,
    )


    # ------------------------------------------------------------
    # 3. Build macro matrix
    # ------------------------------------------------------------

    print("3. Building macro matrix")

    macro_df = prepare_macro_matrix(
        macro_paths_dict
    )

    print(
        "Macro matrix columns:",
        macro_df.columns.tolist(),
    )

    print(
        "Macro matrix shape:",
        macro_df.shape,
    )


    # ------------------------------------------------------------
    # 4. Generate stock-level signals
    # ------------------------------------------------------------

    print("4. Generating stock signals")

    GetFeatureSignals(
        processed_paths_dict
    ).run_data()


    # ------------------------------------------------------------
    # 5. Load stock features
    # ------------------------------------------------------------

    print("5. Loading stock features")

    feature_dfs_dict = (
        load_stock_feature_dataframes()
    )


    # ------------------------------------------------------------
    # 6. Build target data
    # ------------------------------------------------------------

    print("6. Building forward-return target")

    target_df = (
        ForwardReturns()
        .run_data()[
            [
                "Date",
                "Ticker",
                "future_return_5d",
            ]
        ]
        .copy()
    )

    target_df = ensure_date_column(
        target_df,
        "forward-return target",
    )

    target_df = (
        target_df
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(
            subset=["Date", "Ticker"],
            keep="last",
        )
        .reset_index(drop=True)
    )


    # ------------------------------------------------------------
    # 7. Build stock-only feature matrix
    # ------------------------------------------------------------

    print("7. Building stock-only feature matrix")

    feature_matrix_stock = FeatureMatrixBuilder(
        feature_dfs_dict,
        target_df,
    ).run_data()

    feature_matrix_stock = ensure_date_column(
        feature_matrix_stock,
        "stock feature matrix",
    )

    feature_matrix_stock = (
        feature_matrix_stock
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )


    # ------------------------------------------------------------
    # 8. Merge stock and market features
    # ------------------------------------------------------------

    print("8. Building stock + market feature matrix")

    feature_matrix_market = (
        feature_matrix_stock
        .merge(
            market_df,
            on="Date",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    validate_matrix_columns(
        stock_matrix=feature_matrix_stock,
        market_matrix=market_df,
        combined_matrix=feature_matrix_market,
    )


    # ------------------------------------------------------------
    # 9. Merge stock, market and macro features
    # ------------------------------------------------------------

    print("9. Building stock + market + macro feature matrix")

    overlapping_macro_columns = [
        column
        for column in macro_df.columns
        if column in feature_matrix_market.columns
        and column != "Date"
    ]

    if overlapping_macro_columns:
        raise ValueError(
            "Macro matrix contains columns already present in the "
            "stock + market matrix: "
            f"{overlapping_macro_columns}"
        )

    feature_matrix_macro_market = (
        feature_matrix_market
        .merge(
            macro_df,
            on="Date",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )


    # ------------------------------------------------------------
    # 10. Save feature matrices
    # ------------------------------------------------------------

    print("10. Saving feature matrices")

    feature_matrix_stock.to_parquet(
        feature_matrix_pipeline_dict[
            "feature_matrix_stock"
        ],
        index=False,
        engine="pyarrow",
    )

    feature_matrix_market.to_parquet(
        feature_matrix_pipeline_dict[
            "feature_matrix_market"
        ],
        index=False,
        engine="pyarrow",
    )

    feature_matrix_macro_market.to_parquet(
        feature_matrix_pipeline_dict[
            "feature_matrix_macro_market"
        ],
        index=False,
        engine="pyarrow",
    )


    # ------------------------------------------------------------
    # 11. Final checks
    # ------------------------------------------------------------

    print("\nFeature matrices saved successfully.")

    print(
        "Stock-only shape:",
        feature_matrix_stock.shape,
    )

    print(
        "Stock + market shape:",
        feature_matrix_market.shape,
    )

    print(
        "Stock + market + macro shape:",
        feature_matrix_macro_market.shape,
    )

    print(
        "\nMarket features added:",
        [
            column
            for column in feature_matrix_market.columns
            if column not in feature_matrix_stock.columns
        ],
    )

    expected_market_column_count = (
        feature_matrix_stock.shape[1]
        + len(market_paths_dict)
    )

    if (
        feature_matrix_market.shape[1]
        != expected_market_column_count
    ):
        raise ValueError(
            "The stock + market matrix contains an unexpected "
            "number of columns.\n"
            f"Expected: {expected_market_column_count}\n"
            f"Actual: {feature_matrix_market.shape[1]}"
        )