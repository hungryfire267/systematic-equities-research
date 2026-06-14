import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Quintile: 
    def __init__(self,  return_df: pd.DataFrame, signal_df: pd.DataFrame, n_quantiles: int = 5):
        self.return_df = return_df.copy()
        self.signal_df = signal_df.copy()
        self.n_quantiles = n_quantiles
        
        self.return_df["Date"] = pd.to_datetime(self.return_df["Date"])
        self.signal_df["Date"] = pd.to_datetime(self.signal_df["Date"])
        
        self.factor_list = [
            col for col in self.signal_df.columns
            if col not in {"Date", "Ticker"}
        ]
        
    def merge_dfs(self) -> pd.DataFrame: 
        self.df = pd.merge(
            left=self.return_df,
            right=self.signal_df,
            how="inner",
            on=["Date", "Ticker"]
        )
        return self.df
        
    def assign_quantiles(self, group: pd.DataFrame, factor_col: str) -> pd.DataFrame:
        group = group.copy()
        valid_factor = group[factor_col].replace([np.inf, -np.inf], np.nan).dropna()

        if valid_factor.nunique() < self.n_quantiles or len(valid_factor) < self.n_quantiles:
            group["quintile"] = np.nan
            return group

        ranks = group[factor_col].rank(method="first")
        group["quintile"] = pd.qcut(
            ranks,
            q=self.n_quantiles,
            labels=range(1, self.n_quantiles + 1)
        )
        return group

    def calculate(
        self,
        factor_col: str,
        target_col: str = "future_return_5d"
    ) -> tuple[pd.DataFrame, pd.DataFrame]: 
        if not hasattr(self, "df"):
            self.merge_dfs()

        required_cols = ["Date", "Ticker", factor_col, target_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = self.df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
        df = (
            df.groupby("Date", group_keys=False)
              .apply(lambda group: self.assign_quantiles(group, factor_col))
              .dropna(subset=["quintile"])
        )
        df["quintile"] = df["quintile"].astype(int)

        quintile_returns = (
            df.groupby(["Date", "quintile"])[target_col]
              .mean()
              .reset_index()
        )

        average_returns = (
            quintile_returns.groupby("quintile")[target_col]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(
                  columns={
                      "mean": "mean_forward_return",
                      "std": "std_forward_return",
                      "count": "n_periods"
                  }
              )
        )

        wide_returns = quintile_returns.pivot(
            index="Date",
            columns="quintile",
            values=target_col
        )

        if 1 in wide_returns.columns and self.n_quantiles in wide_returns.columns:
            spread = wide_returns[self.n_quantiles] - wide_returns[1]
            spread_row = pd.DataFrame([{
                "quintile": f"Q{self.n_quantiles}-Q1",
                "mean_forward_return": spread.mean(),
                "std_forward_return": spread.std(),
                "n_periods": spread.count()
            }])
            average_returns = pd.concat([average_returns, spread_row], ignore_index=True)

        return quintile_returns, average_returns
        
    def run_data(
        self,
        factor_col: str | None = None,
        target_col: str = "future_return_5d"
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]: 
        self.merge_dfs()
        
        factors = [factor_col] if factor_col is not None else self.factor_list
        quintile_dict = {}
        summary_rows = []

        for factor in factors:
            quintile_returns, average_returns = self.calculate(factor, target_col)
            quintile_dict[factor] = quintile_returns

            summary = average_returns.copy()
            summary.insert(0, "factor", factor)
            summary.insert(1, "target", target_col)
            summary_rows.append(summary)

        summary_df = pd.concat(summary_rows, ignore_index=True)
        return quintile_dict, summary_df

    def plot_quintiles(
        self,
        average_returns: pd.DataFrame,
        title: str | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        plot_df = average_returns[
            average_returns["quintile"].apply(lambda x: isinstance(x, (int, np.integer)))
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            plot_df["quintile"].astype(str),
            plot_df["mean_forward_return"]
        )
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Quintile")
        ax.set_ylabel("Mean forward return")
        ax.set_title(title or "Mean Forward Return by Signal Quintile")
        plt.tight_layout()

        return fig, ax
