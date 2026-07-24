import os
import joblib
import pandas as pd
import numpy as np

class ForecastInferencePipeline:
    """
    Production inference engine for serving demand forecasts from serialized artifacts.
    """
    
    def __init__(self, model_path: str = "models/lightgbm_model.joblib", config_path: str = "models/model_config.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = joblib.load(model_path)
        self.config = joblib.load(config_path) if os.path.exists(config_path) else {}
        self.feature_cols = self.config.get('feature_cols', [])

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates demand forecasts for a batch feature matrix.
        """
        df = input_df.copy()
        
        # Ensure categorical encoding aligns with training
        if 'store_id' in df.columns and 'store_id_cat' not in df.columns:
            df['store_id_cat'] = df['store_id'].astype('category').cat.codes
        if 'product_id' in df.columns and 'product_id_cat' not in df.columns:
            df['product_id_cat'] = df['product_id'].astype('category').cat.codes
            
        X = df[self.feature_cols]
        predictions = self.model.predict(X)
        
        # Floor negative predictions at 0 (demand cannot be negative)
        df['forecast_demand'] = np.maximum(0, np.round(predictions))
        return df
