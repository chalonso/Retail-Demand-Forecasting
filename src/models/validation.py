import numpy as np
import pandas as pd
from typing import Generator, Tuple

class TimeSeriesWalkForwardCV:
    """
    Implements Expanding Window Walk-Forward Cross-Validation for Time Series.
    Ensures zero temporal data leakage across folds.
    """
    def __init__(self, initial_train_days: int = 365, forecast_horizon_days: int = 30, step_days: int = 30):
        self.initial_train_days = initial_train_days
        self.forecast_horizon_days = forecast_horizon_days
        self.step_days = step_days

    def split(self, df: pd.DataFrame, date_col: str = 'date') -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields train and test dataframes for each expanding window fold.
        """
        dates = pd.to_datetime(df[date_col]).sort_values().unique()
        min_date = dates[0]
        max_date = dates[-1]
        
        current_train_end = min_date + pd.Timedelta(days=self.initial_train_days)
        fold = 1
        
        while current_train_end + pd.Timedelta(days=self.forecast_horizon_days) <= max_date:
            test_end = current_train_end + pd.Timedelta(days=self.forecast_horizon_days)
            
            train_mask = (pd.to_datetime(df[date_col]) <= current_train_end)
            test_mask = (pd.to_datetime(df[date_col]) > current_train_end) & (pd.to_datetime(df[date_col]) <= test_end)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            print(f"Fold {fold}: Train ({train_df[date_col].min()} to {train_df[date_col].max()}) | Test ({test_df[date_col].min()} to {test_df[date_col].max()})")
            
            yield train_df, test_df
            
            current_train_end += pd.Timedelta(days=self.step_days)
            fold += 1
