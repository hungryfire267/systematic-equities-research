import pandas as pd
from microstructure import Microstructure
from reversal import Reversal
from utils import cross_sectional_ranking


class ReversalIlliquidity:
    def __init__(self, reversal_window_list: list, illiquidity_window_list: list):
        self.reversal_window_list = reversal_window_list
        self.illiquidity_window_list = illiquidity_window_list
        
        
    def build_reversal_illiquidity_rank(self, reversal_rank: pd.DataFrame, amihud_rank: pd.DataFrame) -> pd.DataFrame: 
        reversal_amihud_score = reversal_rank * amihud_rank
        reversal_amihud_rank = cross_sectional_ranking(reversal_amihud_score, higher_is_better=True)
        
        return reversal_amihud_rank
                 
                 
    def run_data(self) ->  dict[tuple[int, int], pd.DataFrame]:
        reversal_dict = Reversal(self.reversal_window_list).run_data()
        _, amihud_dict = Microstructure(self.illiquidity_window_list).run_data()
        
        reversal_illiquidity_dict = dict()
        for reversal_window in self.reversal_window_list: 
            reversal_rank = reversal_dict[reversal_window]
            
            for illiquidity_window in self.illiquidity_window_list: 
                amihud_rank = amihud_dict[illiquidity_window]
                
                key = (reversal_window, illiquidity_window)
                
                reversal_illiquidity_dict[key] = self.build_reversal_illiquidity_rank(reversal_rank, amihud_rank)
        
        return reversal_illiquidity_dict