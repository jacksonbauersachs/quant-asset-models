"""
Optimize slope for a fixed power value using analytical linear regression.
This is Step 2 of the two-step approach:
1. Pick power using interactive_rainbow_v2.py
2. Run this script to get the optimal slope for that power
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import gc
import subprocess
import platform

# Import configuration
try:
    from config import (
        CURRENCY_NAME, CURRENCY_NAME_LOWER, DATA_FOLDER, HISTORICAL_DATA_FILE,
        OUTPUT_FOLDER, START_DATE, get_output_files, validate_config
    )
    try:
        from config import GENESIS_DATE
    except ImportError:
        GENESIS_DATE = None
    try:
        from config import POWER  # Power value from interactive chart
    except ImportError:
        POWER = 1.0  # Default to linear
except ImportError:
    print("Error: config.py not found!")
    sys.exit(1)

BASE = 10.0  # Fixed base

def load_data():
    """Load and prepare historical price data."""
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    
    # Get data file path - handle relative paths (go up one level if running from log_model folder)
    if os.path.basename(os.getcwd()) == 'log_model':
        data_file = os.path.join('..', DATA_FOLDER, HISTORICAL_DATA_FILE)
    else:
        data_file = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILE)
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found - {data_file}")
        print(f"Current working directory: {os.getcwd()}")
        return None
    
    try:
        df = pd.read_csv(data_file)
        
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            print("Error: Could not find date column in CSV")
            return None
        
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        price_col = None
        for col in df.columns:
            if 'price' in col.lower() and 'close' not in col.lower():
                price_col = col
                break
        if price_col is None:
            for col in df.columns:
                if 'close' in col.lower():
                    price_col = col
                    break
        
        if price_col is None:
            print("Error: Could not find price column in CSV")
            return None
        
        if df[price_col].dtype == 'object':
            df['Price'] = df[price_col].str.replace(',', '').astype(float)
        else:
            df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        df = df.dropna(subset=['Price', 'Date'])
        df = df[df['Price'] > 0]
        
        if GENESIS_DATE is None:
            genesis_date = df['Date'].min()
            print(f"Auto-detected genesis date from data: {genesis_date.date()}")
        else:
            genesis_date = pd.to_datetime(GENESIS_DATE)
        
        if START_DATE is None or START_DATE == "":
            start_date = df['Date'].min()
            print(f"Auto-detected start date from data: {start_date.date()}")
        else:
            start_date = pd.to_datetime(START_DATE)
        
        df = df[df['Date'] >= start_date].copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        df['Days_Since_Genesis'] = (df['Date'] - genesis_date).dt.days
        df['Days_Since_Genesis'] = df['Days_Since_Genesis'].clip(lower=1)
        
        print(f"{CURRENCY_NAME} data loaded: {len(df)} records")
        return df[['Date', 'Price', 'Days_Since_Genesis']]
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def rainbow_model(x, slope, intercept, base=10, power=1.0):
    """Growth model: price = base^(slope * (days^power) + intercept)"""
    x = np.maximum(x, 1)
    exponent = slope * (x**power) + intercept
    exponent = np.clip(exponent, -30, 30)
    result = base**exponent
    result = np.where(np.isfinite(result), result, np.nan)
    return result

def calculate_r2(y_actual, y_pred, base=10):
    """Calculate R² score in log space."""
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    if np.sum(valid_mask) < 2:
        return 0.0
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    log_base = np.log(base)
    log_actual = np.log(y_actual_valid) / log_base
    log_pred = np.log(y_pred_valid) / log_base
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def optimize_slope_analytically(df, power, base=10.0):
    """Optimize slope and intercept analytically using linear regression."""
    days = df['Days_Since_Genesis'].values
    prices = df['Price'].values
    first_price = prices[0]
    first_days = days[0]
    
    z = days**power
    log_base = np.log(base)
    log_prices = np.log(prices) / log_base
    
    z_centered = z - (first_days**power)
    log_prices_centered = log_prices - (np.log(first_price) / log_base)
    
    denominator = np.sum(z_centered**2)
    if denominator == 0:
        slope = 0.0
    else:
        slope = np.sum(z_centered * log_prices_centered) / denominator
    
    intercept = np.log(first_price) / log_base - slope * (first_days**power)
    
    y_pred = rainbow_model(days, slope, intercept, base, power)
    r2 = calculate_r2(prices, y_pred, base)
    
    return slope, intercept, r2

def main():
    """Main optimization function."""
    print(f"{CURRENCY_NAME} Slope Optimization (Fixed Power)")
    print("=" * 40)
    print(f"Using power: {POWER:.2f} (from config.py)")
    print(f"Base: {BASE:.0f} (fixed)")
    
    df = load_data()
    if df is None:
        return
    
    print(f"\nOptimizing slope analytically for power={POWER:.2f}...")
    slope, intercept, r2 = optimize_slope_analytically(df, POWER, BASE)
    
    print(f"\nOptimization Results:")
    print(f"  Power: {POWER:.2f}")
    print(f"  Optimal Slope: {slope:.6f}")
    print(f"  Optimal Intercept: {intercept:.6f}")
    print(f"  Maximum R²: {r2:.6f}")
    
    # Calculate model predictions
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, slope, intercept, BASE, POWER)
    )
    
    # Cap predictions
    min_price = df['Price'].min()
    max_price = df['Price'].max()
    df['Model_Price'] = df['Model_Price'].clip(lower=min_price * 0.5, upper=max_price * 2.0)
    
    current_price = df['Price'].iloc[-1]
    model_fair_value = df['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    print(f"\nCurrent Status:")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  Model Fair Value: ${model_fair_value:,.2f}")
    print(f"  Deviation: {deviation:+.1f}%")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"To use these parameters, update config.py:")
    print(f"  POWER = {POWER:.2f}")
    print(f"  FIXED_GROWTH_RATE = {slope:.6f}  # This is the slope")

if __name__ == "__main__":
    main()

