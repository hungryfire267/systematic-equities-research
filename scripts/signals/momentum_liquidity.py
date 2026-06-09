import pandas as pd
from microstructure import Microstructure
from momentum import Momentum
from utils import cross_sectional_ranking

class MomentumLiquidity: 
    def __init__(self, momentum_weights: list, liquidity_window_list: list): 
        self.momentum_weights = momentum_weights 
        self.liquidity_window_list = liquidity_window_list
    
    def build_momentum_liquidity_rank(self, momentum_rank, dv_rank) -> pd.DataFrame: 
        momentum_dv_score = momentum_rank * dv_rank
        momentum_dv_rank = cross_sectional_ranking(momentum_dv_score, higher_is_better=True)
    
        return momentum_dv_rank
    
    def run_data(self) -> dict: 
        momentum_rank = Momentum(self.momentum_weights, self.momentum_n).run_data()
        dv_liquidity_dict, _ = Microstructure(self.liquidity_window_list)
        
        momentum_liquidity_dict = dict()
        for window in self.liquidity_window_list: 
            momentum_liquidity_dict[window] = self.build_momentum_liquidity_rank(
                window, momentum_rank, dv_liquidity_dict[window]
            )
        return momentum_liquidity_dict