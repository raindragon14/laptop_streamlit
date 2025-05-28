import pandas as pd
import numpy as np
import os

def load_sample_data():
    """Load or generate sample laptop dataset"""
    data_path = os.path.join('data', 'laptop_data.csv')
    
    # Check if dataset exists, if not create a sample one
    if not os.path.exists(data_path):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        np.random.seed(42)
        n_samples = 1000
        
        brands = ['HP', 'Dell', 'Lenovo', 'ASUS', 'Acer', 'Apple', 'MSI', 'Samsung']
        processors = ['Intel i3', 'Intel i5', 'Intel i7', 'Intel i9', 'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9']
        ram_options = [4, 8, 16, 32, 64]
        storage_options = [128, 256, 512, 1024, 2048]
        gpu_types = ['Integrated', 'GTX 1650', 'GTX 1660', 'RTX 3050', 'RTX 3060', 'RTX 3070', 'RTX 4060', 'RTX 4070']
        screen_sizes = [13.3, 14.0, 15.6, 16.0, 17.3]
        
        data = {
            'Brand': np.random.choice(brands, n_samples),
            'Processor': np.random.choice(processors, n_samples),
            'RAM_GB': np.random.choice(ram_options, n_samples),
            'Storage_GB': np.random.choice(storage_options, n_samples),
            'GPU': np.random.choice(gpu_types, n_samples),
            'Screen_Size': np.random.choice(screen_sizes, n_samples),
            'Weight_kg': np.round(np.random.uniform(1.2, 3.5, n_samples), 2),
            'Battery_Hours': np.round(np.random.uniform(4, 12, n_samples), 1)
        }
        
        # Generate prices based on specifications
        prices = []
        for i in range(n_samples):
            base_price = 500
            
            # Brand premium
            brand_multiplier = {'HP': 1.0, 'Dell': 1.1, 'Lenovo': 1.05, 'ASUS': 1.15, 
                              'Acer': 0.95, 'Apple': 1.8, 'MSI': 1.3, 'Samsung': 1.2}
            base_price *= brand_multiplier[data['Brand'][i]]
            
            # Processor premium
            proc_multiplier = {'Intel i3': 1.0, 'Intel i5': 1.3, 'Intel i7': 1.6, 'Intel i9': 2.2,
                             'AMD Ryzen 3': 0.95, 'AMD Ryzen 5': 1.25, 'AMD Ryzen 7': 1.55, 'AMD Ryzen 9': 2.1}
            base_price *= proc_multiplier[data['Processor'][i]]
            
            # RAM premium
            base_price += data['RAM_GB'][i] * 15
            
            # Storage premium
            base_price += data['Storage_GB'][i] * 0.8
            
            # GPU premium
            gpu_premium = {'Integrated': 0, 'GTX 1650': 200, 'GTX 1660': 300, 'RTX 3050': 400,
                          'RTX 3060': 600, 'RTX 3070': 900, 'RTX 4060': 700, 'RTX 4070': 1100}
            base_price += gpu_premium[data['GPU'][i]]
            
            # Screen size premium
            base_price += (data['Screen_Size'][i] - 13) * 50
            
            # Add some noise
            base_price *= np.random.uniform(0.9, 1.1)
            
            prices.append(round(base_price, 2))
        
        data['Price_USD'] = prices
        
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return df
    else:
        return pd.read_csv(data_path)

def get_categorical_columns():
    """Return list of categorical columns"""
    return ['Brand', 'Processor', 'GPU']

def get_numerical_columns():
    """Return list of numerical columns (excluding price)"""
    return ['RAM_GB', 'Storage_GB', 'Screen_Size', 'Weight_kg', 'Battery_Hours']
