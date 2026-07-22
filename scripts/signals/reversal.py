import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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


class Reversal:
    def __init__(self, windows_list: list[int]):
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

        self.companies_df = pd.read_csv(UNIVERSE_PATH)

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

        self.companies_df["code"] = (
            self.companies_df["asxCode"] + ".AX"
        )

        self.industry_returns_df.columns = [
            str(column).strip()
            for column in self.industry_returns_df.columns
        ]

        self.industry_dict = (
            self.companies_df
            .set_index("code")["industry"]
            .to_dict()
        )

        self.windows_list = windows_list

    @property
    def company_columns(self) -> list[str]:
        return [
            column
            for column in self.returns_df.columns
            if column != "Date"
        ]

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
            "Could not determine the ASX market-return column. "
            f"Available columns: "
            f"{list(self.asx_returns_df.columns)}"
        )

    def get_reversal(
        self,
        window: int,
    ) -> pd.DataFrame:
        company_returns = self.returns_df[
            self.company_columns
        ]

        cumulative_returns = (
            (1 + company_returns)
            .rolling(window=window)
            .apply(np.prod, raw=True)
            - 1
        )

        return -cumulative_returns

    def get_company_industry(
        self,
        company: str,
    ) -> str | None:
        industry = self.industry_dict.get(company)

        if industry is None or pd.isna(industry):
            print(
                f"Skipping RSR for {company}: "
                "industry is missing"
            )
            return None

        industry = str(industry).strip()

        invalid_industries = {
            "",
            "Class Pend",
            "Classification Pending",
            "Not Applic",
            "Unknown",
            "N/A",
        }

        if industry in invalid_industries:
            print(
                f"Skipping RSR for {company}: "
                f"invalid industry '{industry}'"
            )
            return None

        if industry not in self.industry_returns_df.columns:
            print(
                f"Skipping RSR for {company}: "
                f"industry '{industry}' is not present "
                "in industry_returns.parquet"
            )
            return None

        return industry

    def get_rsr(
        self,
        window: int,
    ) -> pd.DataFrame:
        market_column = self.get_market_return_column()
        market_returns = self.asx_returns_df[
            market_column
        ]

        rsr_dict = {}

        for company in self.company_columns:
            company_returns = self.returns_df[
                company
            ]

            industry = self.get_company_industry(
                company
            )

            if industry is None:
                rsr_dict[company] = pd.Series(
                    np.nan,
                    index=company_returns.index,
                    dtype=float,
                )
                continue

            industry_returns = (
                self.industry_returns_df[industry]
            )

            total_returns = pd.concat(
                [
                    industry_returns.rename("industry"),
                    market_returns.rename("market"),
                    company_returns.rename("company"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            if total_returns.shape[0] < window:
                print(
                    f"Skipping RSR for {company}: "
                    "insufficient observations"
                )

                rsr_dict[company] = pd.Series(
                    np.nan,
                    index=company_returns.index,
                    dtype=float,
                )
                continue

            y_returns = total_returns["company"]

            X_returns = total_returns[
                ["industry", "market"]
            ]

            linear_model = LinearRegression()

            linear_model.fit(
                X_returns,
                y_returns,
            )

            residuals = pd.Series(
                y_returns
                - linear_model.predict(X_returns),
                index=total_returns.index,
                name=company,
            )

            cumulative_residual_return = (
                (1 + residuals)
                .rolling(window=window)
                .apply(np.prod, raw=True)
                - 1
            )

            rsr_dict[company] = (
                -cumulative_residual_return
            )

        return pd.DataFrame(rsr_dict)

    def get_reversal_ranks(
        self,
        window: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        reversal_score = self.get_reversal(window)
        rsr_score = self.get_rsr(window)

        reversal_rank_df = (
            cross_sectional_ranking(
                reversal_score,
                higher_is_better=True,
            )
            .reset_index()
        )

        rsr_rank_df = (
            cross_sectional_ranking(
                rsr_score,
                higher_is_better=True,
            )
            .reset_index()
        )

        return reversal_rank_df, rsr_rank_df

    def run_data(self) -> dict:
        reversal_df_dict = {}
        rsr_df_dict = {}

        for window in self.windows_list:
            (
                reversal_df_dict[window],
                rsr_df_dict[window],
            ) = self.get_reversal_ranks(window)

        return {
            "reversal": reversal_df_dict,
            "rsr": rsr_df_dict,
        }