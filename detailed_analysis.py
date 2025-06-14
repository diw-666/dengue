import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load and analyze the data in detail
df = pd.read_excel('Dengue_Data (2010-2020).xlsx')

print("=== DETAILED DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Number of unique cities/districts: {df['City'].nunique()}")
print(f"\nCities/Districts in the dataset:")
cities = df['City'].unique()
for i, city in enumerate(cities, 1):
    city_data = df[df['City'] == city]
    print(f"{i:2d}. {city:15s} - {len(city_data):3d} records, Cases range: {city_data['Value'].min():4d} to {city_data['Value'].max():4d}")

# Check data completeness
print(f"\n=== DATA COMPLETENESS ===")
date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='MS')
expected_records = len(cities) * len(date_range)
actual_records = len(df)
print(f"Expected records (all cities × all months): {expected_records}")
print(f"Actual records: {actual_records}")
print(f"Data completeness: {actual_records/expected_records*100:.1f}%")

# Time series analysis
print(f"\n=== TIME SERIES PATTERNS ===")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Monthly patterns
monthly_avg = df.groupby('Month')['Value'].mean()
print("Average cases by month:")
for month, avg_cases in monthly_avg.items():
    print(f"Month {month:2d}: {avg_cases:6.1f} cases")

# Yearly patterns
yearly_total = df.groupby('Year')['Value'].sum()
print("\nTotal cases by year:")
for year, total_cases in yearly_total.items():
    print(f"{year}: {total_cases:6d} cases")

# Top districts by total cases
district_totals = df.groupby('City')['Value'].sum().sort_values(ascending=False)
print(f"\nTop 10 districts by total cases (2010-2020):")
for i, (district, total) in enumerate(district_totals.head(10).items(), 1):
    print(f"{i:2d}. {district:15s}: {total:6d} cases")

print(f"\n=== DATA READY FOR NEURAL NETWORK MODELING ===") 