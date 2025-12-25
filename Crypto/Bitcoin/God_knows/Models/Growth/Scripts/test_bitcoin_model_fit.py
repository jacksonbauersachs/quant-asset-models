"""
TEST BITCOIN MODEL FIT
======================

This script tests the current Bitcoin growth model to see if it's fitted correctly
and identifies any issues with the coefficients or data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Current Bitcoin growth model coefficients (updated 2025-07-19)
BITCOIN_GROWTH_COEFFICIENTS = {
    'slope': 1.8277429956323488,
    'intercept': -10.880943376278237,
    'r2': 0.9402752052678194,
    'start_day': 365,  # Skip first year of Bitcoin
    'data_range': '2011-07-17 to 2025-07-19'
}

# Bitcoin genesis date (January 3, 2009)
BITCOIN_GENESIS = datetime(2009, 1, 3)

def calculate_bitcoin_days(date):
    """Calculate days since Bitcoin genesis for a given date"""
    return (date - BITCOIN_GENESIS).days

def calculate_model_price(days):
    """Calculate Bitcoin price using the growth model formula"""
    if days < BITCOIN_GROWTH_COEFFICIENTS['start_day']:
        return None
    
    slope = BITCOIN_GROWTH_COEFFICIENTS['slope']
    intercept = BITCOIN_GROWTH_COEFFICIENTS['intercept']
    
    # Formula: log10(price) = slope * ln(day) + intercept
    log_price = slope * np.log(days) + intercept
    price = 10**log_price
    
    return price

def main():
    """Test the Bitcoin growth model fit"""
    
    print("=" * 60)
    print("TESTING BITCOIN GROWTH MODEL FIT")
    print("=" * 60)
    
    # Load Bitcoin data
    df = pd.read_csv("Portfolio/Data/Bitcoin_all_time_price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle Price column
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    else:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    
    # Calculate days since genesis
    df['Days'] = df['Date'].apply(calculate_bitcoin_days)
    
    # Filter to only include days >= start_day (as the model was fitted)
    df_filtered = df[df['Days'] >= BITCOIN_GROWTH_COEFFICIENTS['start_day']].copy()
    
    print(f"\nFiltered to {len(df_filtered):,} days starting from day {BITCOIN_GROWTH_COEFFICIENTS['start_day']}")
    print(f"Filtered date range: {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Filtered price range: ${df_filtered['Price'].min():.2f} to ${df_filtered['Price'].max():,.2f}")
    
    # Test the current model on this data
    print(f"\nTESTING CURRENT MODEL:")
    print(f"Formula: log10(price) = {BITCOIN_GROWTH_COEFFICIENTS['slope']:.6f} * ln(day) + {BITCOIN_GROWTH_COEFFICIENTS['intercept']:.6f}")
    
    # Calculate model predictions
    df_filtered['Model_Price'] = df_filtered['Days'].apply(calculate_model_price)
    
    # Check early predictions
    early_data = df_filtered.head(10)
    print(f"\nEARLY PREDICTIONS (first 10 days after day {BITCOIN_GROWTH_COEFFICIENTS['start_day']}):")
    for _, row in early_data.iterrows():
        print(f"Day {row['Days']}: Actual ${row['Price']:.2f}, Model ${row['Model_Price']:.2f}")
    
    # Check current predictions
    current_data = df_filtered.tail(5)
    print(f"\nCURRENT PREDICTIONS (last 5 days):")
    for _, row in current_data.iterrows():
        print(f"Day {row['Days']}: Actual ${row['Price']:.2f}, Model ${row['Model_Price']:.2f}")
    
    # Calculate R² for the current model on this data
    log_actual = np.log10(df_filtered['Price'])
    log_model = np.log10(df_filtered['Model_Price'])
    
    # Calculate R²
    ss_res = np.sum((log_actual - log_model) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nMODEL PERFORMANCE ON CURRENT DATA:")
    print(f"R² = {r2:.6f}")
    print(f"Expected R² = {BITCOIN_GROWTH_COEFFICIENTS['r2']:.6f}")
    
    # Now let's refit the model on this exact data to see what we get
    print(f"\nREFITTING MODEL ON CURRENT DATA:")
    
    X = np.log(df_filtered['Days'])
    Y = np.log10(df_filtered['Price'])
    
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    R2 = r_value ** 2
    
    print(f"New Formula: log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}")
    print(f"New R² = {R2:.6f}")
    
    # Test the new model
    df_filtered['New_Model_Price'] = 10**(slope * np.log(df_filtered['Days']) + intercept)
    
    print(f"\nCOMPARING MODELS:")
    print(f"Current Model R²: {r2:.6f}")
    print(f"Refitted Model R²: {R2:.6f}")
    
    # Check if the models are significantly different
    if abs(slope - BITCOIN_GROWTH_COEFFICIENTS['slope']) > 0.01:
        print(f"⚠️  WARNING: Slope differs significantly!")
        print(f"   Current: {BITCOIN_GROWTH_COEFFICIENTS['slope']:.6f}")
        print(f"   Refitted: {slope:.6f}")
        print(f"   Difference: {abs(slope - BITCOIN_GROWTH_COEFFICIENTS['slope']):.6f}")
    
    if abs(intercept - BITCOIN_GROWTH_COEFFICIENTS['intercept']) > 0.1:
        print(f"⚠️  WARNING: Intercept differs significantly!")
        print(f"   Current: {BITCOIN_GROWTH_COEFFICIENTS['intercept']:.6f}")
        print(f"   Refitted: {intercept:.6f}")
        print(f"   Difference: {abs(intercept - BITCOIN_GROWTH_COEFFICIENTS['intercept']):.6f}")
    
    # Show what the refitted model predicts for early days
    print(f"\nREFITTED MODEL EARLY PREDICTIONS:")
    for _, row in early_data.iterrows():
        new_pred = 10**(slope * np.log(row['Days']) + intercept)
        print(f"Day {row['Days']}: Actual ${row['Price']:.2f}, New Model ${new_pred:.2f}")
    
    print(f"\nCONCLUSION:")
    if abs(slope - BITCOIN_GROWTH_COEFFICIENTS['slope']) < 0.01 and abs(intercept - BITCOIN_GROWTH_COEFFICIENTS['intercept']) < 0.1:
        print("✅ Current model coefficients are accurate for this dataset")
    else:
        print("❌ Current model coefficients need updating for this dataset")
        print("   The model was fitted on different data or a different time period")

if __name__ == "__main__":
    main() 