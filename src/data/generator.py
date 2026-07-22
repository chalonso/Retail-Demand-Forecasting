import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class EnterpriseDataGenerator:
    """
    Generates realistic enterprise-level retail demand data with multi-level seasonality,
    price elasticity, promotional spikes, holidays, and stockout truncation.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_stores(self, num_stores: int = 10) -> pd.DataFrame:
        regions = ['North', 'South', 'East', 'West', 'Central']
        stores = []
        for store_id in range(1, num_stores + 1):
            stores.append({
                'store_id': f'STR_{store_id:02d}',
                'region': np.random.choice(regions),
                'store_size_sqft': np.random.randint(15000, 80000),
                'location_tier': np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.5, 0.3, 0.2])
            })
        return pd.DataFrame(stores)

    def generate_products(self, num_products: int = 50) -> pd.DataFrame:
        categories = {
            'Grocery': (2.5, 15.0),
            'Electronics': (50.0, 500.0),
            'Apparel': (10.0, 80.0),
            'Home & Kitchen': (15.0, 150.0)
        }
        products = []
        cat_names = list(categories.keys())
        
        for prod_id in range(1, num_products + 1):
            cat = np.random.choice(cat_names)
            min_cost, max_cost = categories[cat]
            base_cost = round(np.random.uniform(min_cost, max_cost), 2)
            # Apply a 30% to 70% markup
            markup = np.random.uniform(1.3, 1.7)
            base_price = round(base_cost * markup, 2)
            
            products.append({
                'product_id': f'SKU_{prod_id:03d}',
                'category_name': cat,
                'base_cost': base_cost,
                'base_price': base_price,
                'price_elasticity': round(np.random.uniform(-2.5, -0.8), 2)  # Elasticity coefficient
            })
        return pd.DataFrame(products)

    def generate_sales_demand(
        self, 
        stores_df: pd.DataFrame, 
        products_df: pd.DataFrame, 
        start_date: str = '2024-01-01', 
        days: int = 730
    ) -> pd.DataFrame:
        
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        records = []
        
        for date in dates:
            day_of_week = date.weekday()  # 0=Mon, 6=Sun
            month = date.month
            
            # Base seasonality multipliers
            dow_mult = 1.35 if day_of_week in [4, 5, 6] else 0.90  # Weekend surge
            month_mult = 1.40 if month in [11, 12] else (0.85 if month in [1, 2] else 1.0) # Q4 holiday surge
            
            for _, store in stores_df.iterrows():
                # Store size multiplier
                store_mult = store['store_size_sqft'] / 40000.0
                
                for _, prod in products_df.iterrows():
                    # Base latent daily demand
                    base_demand = np.random.poisson(lam=15)
                    
                    # Promotion logic (10% chance of promo)
                    is_promo = np.random.binomial(1, 0.10)
                    discount = np.random.choice([0.10, 0.20, 0.30]) if is_promo else 0.0
                    price_actual = round(prod['base_price'] * (1 - discount), 2)
                    
                    # Calculate elasticity effect: Demand Lift = % Price Change * Elasticity
                    pct_price_change = (price_actual - prod['base_price']) / prod['base_price']
                    elasticity_mult = 1.0 + (pct_price_change * prod['price_elasticity'])
                    
                    # Total Latent Demand calculation
                    latent_demand = int(
                        base_demand * dow_mult * month_mult * store_mult * elasticity_mult
                    )
                    latent_demand = max(0, latent_demand + np.random.randint(-3, 4))
                    
                    # Stockout simulation (5% chance of severe inventory shortage)
                    stock_on_hand = np.random.randint(0, latent_demand + 20) if np.random.rand() > 0.05 else 0
                    
                    # Truncated Sales: You can only sell what you have in stock
                    sales_quantity = min(latent_demand, stock_on_hand)
                    is_stockout = 1 if stock_on_hand < latent_demand else 0
                    
                    records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'store_id': store['store_id'],
                        'product_id': prod['product_id'],
                        'base_price': prod['base_price'],
                        'price_actual': price_actual,
                        'is_promotion': is_promo,
                        'stock_on_hand': stock_on_hand,
                        'latent_demand': latent_demand,
                        'sales_quantity': sales_quantity,
                        'is_stockout': is_stockout,
                        'revenue': round(sales_quantity * price_actual, 2)
                    })
                    
        return pd.DataFrame(records)
  
