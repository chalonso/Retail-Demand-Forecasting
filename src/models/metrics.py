import numpy as np
import pandas as pd

class ForecastEvaluator:
    """
    Computes statistical and financial metrics for demand forecasting.
    """
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, holding_cost_per_unit: float = 2.0, lost_margin_per_unit: float = 5.0) -> dict:
        """
        Calculates MAE, RMSE, MAPE, and Financial Inventory Costs.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero for MAPE
        epsilon = 1e-8
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # Financial Impact Logic
        # Overstock: y_pred > y_true -> Holding Cost incurred
        # Stockout: y_pred < y_true -> Lost Gross Margin
        errors = y_pred - y_true
        overstock_units = np.sum(np.maximum(0, errors))
        stockout_units = np.sum(np.maximum(0, -errors))
        
        total_holding_cost = overstock_units * holding_cost_per_unit
        total_lost_margin = stockout_units * lost_margin_per_unit
        total_financial_impact = total_holding_cost + total_lost_margin
        
        return {
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'MAPE (%)': round(float(mape), 2),
            'Overstock Units': int(overstock_units),
            'Stockout Units': int(stockout_units),
            'Holding Cost ($)': round(float(total_holding_cost), 2),
            'Lost Margin ($)': round(float(total_lost_margin), 2),
            'Total Financial Loss ($)': round(float(total_financial_impact), 2)
        }
