import numpy as np 
import pandas as pd

from scripts.models.lightgbm.lightgbm_model import LightGBMRegressionModel
from scripts.portfolio.optimiser import MeanVarianceOptimiser
from scripts.portfolio.selection import TopBottom20Selector

print("hello")
model_class = LightGBMRegressionModel
print("hello")
topbottom20 = TopBottom20Selector(LightGBMRegressionModel).run_data()
returns_df 
MeanVarianceOptimiser(topbottom20, )
print("hello")
print(topbottom20)