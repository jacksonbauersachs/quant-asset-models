"""
Script to manually add missing nickel spot price data.
Use this to add the last 6 months that Macrotrends is missing.
"""

import pandas as pd
import os
from datetime import datetime

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
data_file = os.path.join(SCRIPT_DIR, "Nickel_historical_data.csv")

def add_manual_data():
    """Add data manually entered by user."""
    print("=" * 60)
    print("Add Missing Nickel Spot Price Data")
    print("=" * 60)
    
    # Read existing data
    if not os.path.exists(data_file):
        print(f"[FAIL] File not found: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max()
    print(f"\nLast date in file: {last_date.strftime('%Y-%m-%d')}")
    
    # Calculate missing months
    today = datetime.now()
    months_needed = []
    current = last_date + pd.DateOffset(months=1)
    
    while current <= today:
        months_needed.append((current.strftime('%Y-%m-01'), current.strftime('%B %Y')))
        current += pd.DateOffset(months=1)
    
    if not months_needed:
        print("\n[OK] Data appears to be up to date!")
        return None
    
    print(f"\nYou need data for {len(months_needed)} month(s):")
    print("\nGet spot prices from:")
    print("  - YCharts: https://ycharts.com/indicators/nickel_spot_price")
    print("  - Trading Economics: https://tradingeconomics.com/commodity/nickel")
    print("  - LME: https://www.lme.com/en/Market-data")
    print("\n" + "-" * 60)
    
    new_data = []
    for date_str, month_name in months_needed:
        while True:
            try:
                price_input = input(f"{month_name} ({date_str}): $").strip()
                if not price_input:
                    print(f"  Skipped {month_name}")
                    break
                price = float(price_input.replace('$', '').replace(',', ''))
                new_data.append({'Date': date_str, 'Price': price})
                print(f"  [OK] Added {month_name}: ${price:.2f}")
                break
            except ValueError:
                print("  Invalid price. Enter a number (e.g., 25000.50)")
            except KeyboardInterrupt:
                print("\n\nCancelled.")
                return None
    
    if not new_data:
        print("\nNo new data added.")
        return None
    
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

if __name__ == "__main__":
    add_manual_data()

