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
    # Replace missing country with 'Unknown'
    df['country'] = df['country'].fillna('Unknown')
    
    # Drop rows where date_added or rating is missing (small percentage usually)
    df.dropna(subset=['date_added', 'rating'], inplace=True)
    
    # Fill missing director and cast with 'Unknown' for completeness
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')

    # 2. Standardize data types
    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    
    # 3. Extract useful features
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month_name()
    
    # Extract numeric duration
    # Movies have 'min', TV Shows have 'Season' or 'Seasons'
    # We extract the integer part. For TV shows, this represents number of seasons.
    df['duration_val'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0)
    
    # Primary country (first one listed) for simplified analysis
    df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0])
    
    return df