import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ExecutiveDashboardPlotter:
    """
    Generates interactive Plotly figures for executive leadership and supply chain managers.
    """
    
    @staticmethod
    def plot_forecast_vs_actual(test_df: pd.DataFrame, store_id: str, product_id: str) -> go.Figure:
        """Generates actual vs. forecast demand timeline with stockout alert markers."""
        subset = test_df[(test_df['store_id'] == store_id) & (test_df['product_id'] == product_id)].copy()
        subset['date'] = pd.to_datetime(subset['date'])
        subset = subset.sort_values('date')
        
        fig = go.Figure()
        
        # Actual Sales
        fig.add_trace(go.Scatter(
            x=subset['date'], y=subset['sales_quantity'],
            mode='lines+markers', name='Actual Sales',
            line=dict(color='#2b5c8f', width=2)
        ))
        
        # Forecasted Demand
        fig.add_trace(go.Scatter(
            x=subset['date'], y=subset['forecast_demand'],
            mode='lines', name='LightGBM Forecast',
            line=dict(color='#e05d06', width=2, dash='dash')
        ))
        
        # Highlight Stockout events
        stockouts = subset[subset['is_stockout'] == 1]
        if not stockouts.empty:
            fig.add_trace(go.Scatter(
                x=stockouts['date'], y=stockouts['sales_quantity'],
                mode='markers', name='Stockout Event',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
        fig.update_layout(
            title=f"Demand Forecast vs Actual Sales | Store: {store_id}, SKU: {product_id}",
            xaxis_title="Date",
            yaxis_title="Units",
            template="plotly_white",
            hovermode="x unified"
        )
        return fig

    @staticmethod
    def plot_financial_impact_comparison(metrics_df: pd.DataFrame) -> go.Figure:
        """Generates a bar chart comparing total financial loss across models."""
        fig = px.bar(
            metrics_df,
            x='Model',
            y='Total Financial Loss ($)',
            color='Model',
            text='Total Financial Loss ($)',
            title="Total Inventory Financial Loss by Model ($ Overstock Holding + $ Stockout Margin Lost)",
            template="plotly_white"
        )
        fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Financial Impact ($)")
        return fig
