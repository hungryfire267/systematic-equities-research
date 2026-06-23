
import numpy as np 
import os 
import pandas as pd
from pathlib import Path

from scripts.models.walk_forward import WalkForwardValidator

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)




feature_matrix_pipeline_dict = {
    "feature_matrix_first": os.path.join(PROCESSED_DIR, "feature_matrix_first.parquet")
}

if __name__ == "__main__": 
    feature_matrix_df = pd.read_parquet(feature_matrix_pipeline_dict["feature_matrix_first"])
    WalkForwardValidator(feature_matrix_df, 2).run_data()
