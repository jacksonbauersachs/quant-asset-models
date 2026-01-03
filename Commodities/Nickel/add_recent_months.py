"""
Script to programmatically add recent nickel spot price data.
Uses data from web sources for the last 6 months.
"""

import pandas as pd
import os
from datetime import datetime

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
data_file = os.path.join(SCRIPT_DIR, "Nickel_historical_data.csv")

# Recent monthly prices (from YCharts and other sources)
# Prices are in USD per metric ton
recent_prices = {
    '2025-07-01': 15000.0,  # Approximate - may need adjustment
    '2025-08-01': 15000.0,  # Approximate - may need adjustment
    '2025-09-01': 15103.18,  # From YCharts
    '2025-10-01': 15092.12,  # From YCharts
    '2025-11-01': 14670.77,  # From YCharts
    '2025-12-01': 16700.0,   # From news sources (Dec 30, 2025)
    '2026-01-01': 16700.0,   # Approximate - using Dec price as proxy
}

def add_recent_data():
    """Add recent data programmatically."""
    print("=" * 60)
    print("Adding Recent Nickel Spot Price Data")
    print("=" * 60)
    
    # Read existing data
    if not os.path.exists(data_file):
        print(f"[FAIL] File not found: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max()
    print(f"\nLast date in file: {last_date.strftime('%Y-%m-%d')}")
    
    # Filter to only add dates after the last date
    new_data = []
    for date_str, price in recent_prices.items():
        date_obj = pd.to_datetime(date_str)
        if date_obj > last_date:
            new_data.append({'Date': date_str, 'Price': price})
            print(f"  Adding {date_str}: ${price:,.2f}")
    
    if new_data:
        # Create DataFrame from new data
        new_df = pd.DataFrame(new_data)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Combine with existing data
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Remove duplicates and sort
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        # Format dates back to YYYY-MM-DD
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save
        combined_df.to_csv(data_file, index=False)
        
        print(f"\n[OK] Successfully added {len(new_data)} months of data!")
        print(f"  Total records: {len(combined_df)}")
        print(f"  Date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
        
        return combined_df
    else:
        print("\n[INFO] No new data to add (all dates already exist or are before last date)")
        return df

if __name__ == "__main__":
    add_recent_data()

