"""
Fix the format of the crude oil historical data CSV file.
- Add proper headers
- Remove dollar signs from prices
- Sort by date (oldest first)
- Ensure proper CSV format
"""

import pandas as pd
import os

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
data_file = os.path.join(SCRIPT_DIR, "Crude_oil_historical_data.csv")

def fix_format():
    """Fix the format of the CSV file."""
    print("=" * 60)
    print("Fixing Crude Oil Historical Data Format")
    print("=" * 60)
    
    if not os.path.exists(data_file):
        print(f"[FAIL] File not found: {data_file}")
        return None
    
    # Read the file (tab-separated, no header)
    print("\nReading file...")
    df = pd.read_csv(data_file, sep='\t', header=None, names=['Date', 'Price'])
    
    print(f"  Found {len(df)} records")
    
    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean prices (remove $, commas)
    if df['Price'].dtype == 'object':
        df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Remove any rows with invalid data
    df = df.dropna(subset=['Date', 'Price'])
    
    # Sort by date (oldest first)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Format dates back to YYYY-MM-DD
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Save as proper CSV
    df.to_csv(data_file, index=False)
    
    print(f"\n[OK] Successfully fixed format!")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"  Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

if __name__ == "__main__":
    fix_format()

