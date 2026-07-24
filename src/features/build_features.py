import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class RetailFeatureEngineer:
    """
    Transforms raw retail demand time series into a rich feature matrix with
    calendar signals, Fourier terms, lag variables, and rolling statistics.
    """

    def __init__(self, target_col: str = 'sales_quantity'):
        self.target_col = target_col

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts date-based signals and cyclical Fourier terms."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Standard Calendar Signals
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Day of year for Fourier terms
        day_of_year = df['date'].dt.dayofyear
        
        # Annual Fourier Seasonality (K=2 harmonics)
        for k in [1, 2]:
            df[f'sin_annual_k{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
            df[f'cos_annual_k{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
            
        # Weekly Fourier Seasonality (K=1 harmonic)
        df['sin_weekly'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['cos_weekly'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        return df

    def add_lag_features(self, df: pd.DataFrame, lags: list = [7, 14, 21, 28]) -> pd.DataFrame:
        """Generates historical lag features grouped by Store and Product."""
        df = df.copy()
        df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
        
        for lag in lags:
            df[f'lag_{lag}'] = (
                df.groupby(['store_id', 'product_id'])[self.target_col]
                .shift(lag)
            )
        return df

    def add_rolling_features(self, df: pd.DataFrame, windows: list = [7, 14, 28]) -> pd.DataFrame:
        """Generates rolling statistics using shifted targets to prevent target leakage."""
        df = df.copy()
        df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
        
        for w in windows:
            # Shift by 1 first so rolling calculations DO NOT include current day target
            grouped = df.groupby(['store_id', 'product_id'])[self.target_col].shift(1)
            
            df[f'rolling_mean_{w}'] = (
                grouped.groupby([df['store_id'], df['product_id']])
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )
            df[f'rolling_std_{w}'] = (
                grouped.groupby([df['store_id'], df['product_id']])
                .transform(lambda x: x.rolling(window=w, min_periods=1).std())
            )
        return df

    def test_stationarity(self, series: pd.Series) -> dict:
        """Performs the Augmented Dickey-Fuller test to evaluate stationarity."""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'used_lags': result[2],
            'is_stationary': result[1] < 0.05
        }


 
