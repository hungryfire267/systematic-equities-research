import pandas as pd
from scripts.signals.microstructure import Microstructure
from scripts.signals.reversal import Reversal
from scripts.signals.utils import cross_sectional_ranking, date_parser


class ReversalIlliquidity:
    def __init__(self, reversal_window_list: list, illiquidity_window_list: list):
        self.reversal_window_list = reversal_window_list
        self.illiquidity_window_list = illiquidity_window_list
        
        
    def build_reversal_illiquidity_rank(self, reversal_rank: pd.DataFrame, amihud_rank: pd.DataFrame) -> pd.DataFrame: 
        reversal_amihud_score = reversal_rank * amihud_rank
        reversal_amihud_rank = cross_sectional_ranking(reversal_amihud_score, higher_is_better=True).reset_index()
        
        return reversal_amihud_rank
                 
                 
    def run_data(self) ->  dict[tuple[int, int], pd.DataFrame]:
        final_reversal_dict = Reversal(self.reversal_window_list).run_data()
        final_microstructure_dict = Microstructure(self.illiquidity_window_list).run_data()
        
        reversal_dict = final_reversal_dict["reversal"]
        amihud_dict = final_microstructure_dict["amihud"]
        
        
        reversal_illiquidity_dict = dict()
        for reversal_window in self.reversal_window_list: 
            reversal_rank = reversal_dict[reversal_window].set_index("Date")
            
            
            for illiquidity_window in self.illiquidity_window_list: 
                amihud_rank = amihud_dict[illiquidity_window].set_index("Date")
                
                key = (reversal_window, illiquidity_window)
                
                reversal_illiquidity_dict[key] = self.build_reversal_illiquidity_rank(reversal_rank, amihud_rank)
        
        final_reversal_illiquidity_dict = {
            "reversal_illiquidity": reversal_illiquidity_dict
        }
        return final_reversal_illiquidity_dict