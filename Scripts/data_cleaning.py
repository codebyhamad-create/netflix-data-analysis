import pandas as pd
import numpy as np

def load_dataset(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_data(df):
    """Performs data cleaning and feature engineering."""
    # 1. Handle missing values
    df['country'] = df['country'].fillna('Unknown')
    df.dropna(subset=['date_added', 'rating'], inplace=True)
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')

    # 2. Standardize data types
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    
    # 3. Extract useful features
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month_name()
    
    # Extract numeric duration
    df['duration_val'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0)
    
    # Primary country (first one listed)
    df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0])
    
    return df
