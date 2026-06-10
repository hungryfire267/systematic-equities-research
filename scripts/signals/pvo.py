import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Sequence
from scripts.signals.utils import date_parser, cross_sectional_ranking

BASE_DIR = Path(__file__).resolve().parents[2]
COMPANIES_DIR = BASE_DIR / "data" / "raw" / "companies"

companies_paths_dict = { 
    "volume": os.path.join(COMPANIES_DIR, "volume.parquet")
}


class PVO: 
    def __init__(self, extreme_list: Sequence[tuple[float, float]], span_list: Sequence[tuple[int, int]]): 
        self.volume_df = date_parser(pd.read_parquet(companies_paths_dict["volume"]))
        self.extreme_list = extreme_list
        self.span_list = span_list
        
    def compute_ema(self, span: int) -> pd.DataFrame:
        return self.volume_df.ewm(span=span, adjust=False).mean()
    
    def calculate_pvo(self, extremes: tuple[float, float], spans: tuple[int, int]) -> pd.DataFrame: 
        lower_extreme, upper_extreme = extremes
        slow_span, fast_span = spans
        
        ema_slow = self.compute_ema(span=slow_span)
        ema_fast = self.compute_ema(span=fast_span)
        pvo_df = (ema_fast - ema_slow)/ema_slow.replace(0, np.nan)
        
        pvo_df_score = pvo_df.clip(lower=pvo_df.quantile(lower_extreme), upper=pvo_df.quantile(upper_extreme), axis=1) # Capping the extremes
        pvo_df_rank = cross_sectional_ranking(pvo_df_score, higher_is_better=True).reset_index()
        return pvo_df_rank
    
    
    def run_data(self) -> dict: 
        pvo_df_dict = dict()
        for extremes, spans in zip(self.extreme_list, self.span_list): 
            keys = (extremes, spans)
            pvo_df_dict[keys] = self.calculate_pvo(extremes, spans)
        return {"pvo": pvo_df_dict}
