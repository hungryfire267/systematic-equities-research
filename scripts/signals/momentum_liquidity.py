import pandas as pd
from scripts.signals.microstructure import Microstructure
from scripts.signals.momentum import Momentum
from scripts.signals.utils import cross_sectional_ranking

class MomentumLiquidity: 
    def __init__(self, liquidity_window_list: list): 
        self.liquidity_window_list = liquidity_window_list
    
    def build_momentum_liquidity_rank(self, momentum_rank, dv_rank) -> pd.DataFrame: 
        momentum_dv_score = momentum_rank * dv_rank
        momentum_dv_rank = cross_sectional_ranking(momentum_dv_score, higher_is_better=True).reset_index()
    
        return momentum_dv_rank
    
    def run_data(self) -> dict: 
        final_momentum_dict = Momentum().run_data()
        final_microstructure_dict = Microstructure(self.liquidity_window_list).run_data()
        
        momentum_liquidity_dict = dict()
        
        momentum_rank = final_momentum_dict["momentum"]["252_12"].set_index("Date")
        for window in self.liquidity_window_list: 
            dv_liquidity_rank = final_microstructure_dict["amihud"][window].set_index("Date")
            
            momentum_liquidity_dict[window] = self.build_momentum_liquidity_rank(
                momentum_rank, dv_liquidity_rank
            )
            
        final_momentum_liquidity_dict = {
            "momentum_liquidity": momentum_liquidity_dict
        }
        
        return final_momentum_liquidity_dict