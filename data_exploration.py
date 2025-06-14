import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading dengue data...")
try:
    df = pd.read_excel('Dengue_Data (2010-2020).xlsx')
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Try to identify key columns
    for col in df.columns:
        if 'district' in col.lower():
            print(f"\nUnique values in {col}:")
            print(df[col].unique())
        elif 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
            print(f"\nDate/Time column {col}:")
            print(df[col].unique()[:10])  # Show first 10 unique values
            
except Exception as e:
    print(f"Error loading data: {e}") 