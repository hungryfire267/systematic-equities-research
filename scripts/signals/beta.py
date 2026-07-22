import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.signals.utils import (
    cross_sectional_ranking,
    date_parser,
)


BASE_DIR = Path(__file__).resolve().parents[2]

UNIVERSE_PATH = BASE_DIR / "data" / "asx_companies.csv"
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"
ASX_DIR = BASE_DIR / "data" / "raw" / "asx"
INDUSTRY_DIR = BASE_DIR / "data" / "raw" / "industry"

returns_paths_dict = {
    "company_returns": os.path.join(
        COMPANIES_DIR,
        "returns.parquet",
    ),
    "asx_returns": os.path.join(
        ASX_DIR,
        "asx_returns.parquet",
    ),
    "industry_returns": os.path.join(
        INDUSTRY_DIR,
        "industry_returns.parquet",
    ),
}


class BetaFeatures:
    def __init__(self, window_list: list[int]):
        self.companies_df = pd.read_csv(UNIVERSE_PATH)

        self.returns_df = date_parser(
            pd.read_parquet(
                returns_paths_dict["company_returns"]
            )
        )

        self.asx_returns_df = date_parser(
            pd.read_parquet(
                returns_paths_dict["asx_returns"]
            )
        )

        self.industry_returns_df = date_parser(
            pd.read_parquet(
                returns_paths_dict["industry_returns"]
            )
        )

        self.companies_df["asxCode"] = (
            self.companies_df["asxCode"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        self.companies_df["industry"] = (
            self.companies_df["industry"]
            .astype("string")
            .str.strip()
        )

        self.industry_returns_df.columns = [
            str(column).strip()
            for column in self.industry_returns_df.columns
        ]

        self.window_list = window_list

    @property
    def company_columns(self) -> list[str]:
        return [
            column
            for column in self.returns_df.columns
            if column != "Date"
        ]

    @staticmethod
    def beta_calculation(
        combined_df: pd.DataFrame,
        beta_type: str,
        window: int,
    ) -> pd.Series:
        covariance = (
            combined_df["company"]
            .rolling(window=window)
            .cov(combined_df[beta_type])
        )

        variance = (
            combined_df[beta_type]
            .rolling(window=window)
            .var()
        )

        variance = variance.replace(0, np.nan)

        return covariance / variance

    @staticmethod
    def vol_calculation(
        combined_df: pd.DataFrame,
        beta_type: str,
        window: int,
        beta: pd.Series,
    ) -> pd.Series:
        residuals = (
            combined_df["company"]
            - beta * combined_df[beta_type]
        )

        return residuals.rolling(window=window).std()

    def get_market_return_column(self) -> str:
        possible_columns = [
            "^AXJO",
            "ASX200",
            "Close",
        ]

        for column in possible_columns:
            if column in self.asx_returns_df.columns:
                return column

        non_date_columns = [
            column
            for column in self.asx_returns_df.columns
            if column != "Date"
        ]

        if len(non_date_columns) == 1:
            return non_date_columns[0]

        raise KeyError(
            "Could not identify the ASX market-return column. "
            f"Available columns: "
            f"{list(self.asx_returns_df.columns)}"
        )

    def get_market_rolling_beta_vol(
        self,
        window: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        market_beta_df_dict = {}
        market_vol_df_dict = {}

        market_column = self.get_market_return_column()
        market_returns = self.asx_returns_df[
            market_column
        ].copy()

        for company in self.company_columns:
            company_returns = self.returns_df[company]

            combined_df = pd.concat(
                [market_returns, company_returns],
                axis=1,
                join="inner",
            )

            combined_df.columns = [
                "market",
                "company",
            ]

            beta = self.beta_calculation(
                combined_df=combined_df,
                beta_type="market",
                window=window,
            )

            vol = self.vol_calculation(
                combined_df=combined_df,
                beta_type="market",
                window=window,
                beta=beta,
            )

            market_beta_df_dict[company] = beta
            market_vol_df_dict[company] = vol

        market_beta_df = pd.DataFrame(
            market_beta_df_dict
        )

        market_vol_df = pd.DataFrame(
            market_vol_df_dict
        )

        return market_beta_df, market_vol_df

    def get_industry_company_return(
        self,
        company: str,
    ) -> pd.Series | None:
        company_code = (
            company
            .split(".")[0]
            .strip()
            .upper()
        )

        condition = (
            self.companies_df["asxCode"]
            == company_code
        )

        if not condition.any():
            print(
                f"Skipping {company}: "
                "company not found in universe"
            )
            return None

        company_industry = self.companies_df.loc[
            condition,
            "industry",
        ].iloc[0]

        if pd.isna(company_industry):
            print(
                f"Skipping {company}: "
                "industry is missing"
            )
            return None

        company_industry = str(
            company_industry
        ).strip()

        invalid_industries = {
            "",
            "Class Pend",
            "Classification Pending",
            "Not Applic",
            "Unknown",
            "N/A",
        }

        if company_industry in invalid_industries:
            print(
                f"Skipping {company}: invalid industry "
                f"'{company_industry}'"
            )
            return None

        if (
            company_industry
            not in self.industry_returns_df.columns
        ):
            print(
                f"Skipping {company}: industry "
                f"'{company_industry}' is not present "
                "in industry_returns.parquet"
            )
            return None

        return self.industry_returns_df[
            company_industry
        ]

    def get_industry_rolling_beta_vol(
        self,
        window: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        industry_beta_df_dict = {}
        industry_vol_df_dict = {}

        for company in self.company_columns:
            company_returns = self.returns_df[
                company
            ]

            industry_returns = (
                self.get_industry_company_return(
                    company
                )
            )

            if industry_returns is None:
                empty_series = pd.Series(
                    np.nan,
                    index=company_returns.index,
                    dtype=float,
                )

                industry_beta_df_dict[
                    company
                ] = empty_series.copy()

                industry_vol_df_dict[
                    company
                ] = empty_series.copy()

                continue

            combined_df = pd.concat(
                [industry_returns, company_returns],
                axis=1,
                join="inner",
            )

            combined_df.columns = [
                "industry",
                "company",
            ]

            beta = self.beta_calculation(
                combined_df=combined_df,
                beta_type="industry",
                window=window,
            )

            vol = self.vol_calculation(
                combined_df=combined_df,
                beta_type="industry",
                window=window,
                beta=beta,
            )

            industry_beta_df_dict[company] = beta
            industry_vol_df_dict[company] = vol

        industry_beta_df = pd.DataFrame(
            industry_beta_df_dict
        )

        industry_vol_df = pd.DataFrame(
            industry_vol_df_dict
        )

        return industry_beta_df, industry_vol_df

    def run_data(self) -> dict:
        market_beta_df_dict = {}
        market_vol_df_dict = {}
        industry_beta_df_dict = {}
        industry_vol_df_dict = {}

        for window in self.window_list:
            market_beta_df, market_vol_df = (
                self.get_market_rolling_beta_vol(
                    window
                )
            )

            industry_beta_df, industry_vol_df = (
                self.get_industry_rolling_beta_vol(
                    window
                )
            )

            market_beta_df_dict[window] = (
                cross_sectional_ranking(
                    market_beta_df,
                    higher_is_better=True,
                )
                .reset_index()
            )

            industry_beta_df_dict[window] = (
                cross_sectional_ranking(
                    industry_beta_df,
                    higher_is_better=True,
                )
                .reset_index()
            )

            market_vol_df_dict[window] = (
                cross_sectional_ranking(
                    market_vol_df,
                    higher_is_better=True,
                )
                .reset_index()
            )

            industry_vol_df_dict[window] = (
                cross_sectional_ranking(
                    industry_vol_df,
                    higher_is_better=True,
                )
                .reset_index()
            )

        return {
            "market_beta": market_beta_df_dict,
            "industry_beta": industry_beta_df_dict,
            "market_resid_vol": market_vol_df_dict,
            "industry_resid_vol": industry_vol_df_dict,
        }