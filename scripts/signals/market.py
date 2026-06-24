import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ASX_DIR = BASE_DIR / "data"

class MarketSignals: 
    def __init__(self): 
         