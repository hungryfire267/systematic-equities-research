
import numpy as np 
import os 
import pandas as pd
from pathlib import Path

from scripts.models.walk_forward import WalkForwardValidator

from lightgbm import LGBMRegressor





BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)




feature_matrix_pipeline_dict = {
    "feature_matrix_first": os.path.join(PROCESSED_DIR, "feature_matrix_first.parquet")
}

if __name__ == "__main__": 
    model = LGBMRegressor(random_state=42)
    
    feature_matrix_df = pd.read_parquet(feature_matrix_pipeline_dict["feature_matrix_first"])
    output = WalkForwardValidator(feature_matrix_df, model, 2).run_data()
    
    
    ic_results = output.groupby("Date").apply(
        lambda x: x["prediction"].corr(
            x["future_return_5d"],
            method="spearman"
        )
    ).describe()
    print(ic_results)
