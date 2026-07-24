import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet
import joblib

class RetailForecaster:
    """
    Orchestrates baseline, statistical (Prophet), and GBDT (LightGBM, XGBoost) models.
    """
    
    def __init__(self, target_col: str = 'sales_quantity'):
        self.target_col = target_col
        self.models = {}

    def predict_naive(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Naive Baseline: Forecasts future as the last observed value."""
        last_val = train_df.groupby(['store_id', 'product_id'])[self.target_col].last()
        preds = test_df.set_index(['store_id', 'product_id']).index.map(last_val)
        return preds.fillna(0).values

    def predict_moving_average(self, train_df: pd.DataFrame, test_df: pd.DataFrame, window: int = 7) -> np.ndarray:
        """Moving Average Baseline: Forecasts future as the 7-day trailing average."""
        ma_val = train_df.groupby(['store_id', 'product_id'])[self.target_col].apply(lambda x: x.tail(window).mean())
        preds = test_df.set_index(['store_id', 'product_id']).index.map(ma_val)
        return preds.fillna(0).values

    def train_lightgbm(self, train_df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
        """Trains a LightGBM Regressor on engineered features."""
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]
        
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model

    def train_xgboost(self, train_df: pd.DataFrame, feature_cols: list) -> xgb.XGBRegressor:
        """Trains an XGBoost Regressor on engineered features."""
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]
        
        model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model

    def train_prophet_sku(self, df: pd.DataFrame, store_id: str, product_id: str, forecast_horizon: int = 30) -> pd.DataFrame:
        """Trains Facebook Prophet on a single Store-SKU series."""
        sku_df = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)].copy()
        sku_df['ds'] = pd.to_datetime(sku_df['date'])
        sku_df['y'] = sku_df[self.target_col]
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(sku_df[['ds', 'y']])
        
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(forecast_horizon)
