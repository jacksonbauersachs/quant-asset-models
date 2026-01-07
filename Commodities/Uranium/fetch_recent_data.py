"""
Simple script to fetch recent Uranium data using yfinance.
This is a fallback if the cookie/crumb method doesn't work.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# Try SPUT tickers (most likely to have recent data)
tickers_to_try = ["SRUUF", "U.UN", "U-UN.TO"]

print("Attempting to fetch recent Uranium data...")
print("=" * 50)

for ticker in tickers_to_try:
    try:
        print(f"\nTrying ticker: {ticker}")
        ticker_obj = yf.Ticker(ticker)
        
        # Fetch data from June 2025 onwards
        df = ticker_obj.history(start="2025-06-02", end=None)
        
        if not df.empty:
            print(f"✓ Successfully fetched {len(df)} days of data for {ticker}")
            print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            
            # Convert to the format we need
            result = pd.DataFrame({
                'Date': df.index.strftime('%Y-%m-%d'),
                'Price': df['Close'].values
            })
            
            # Save to a temporary file
            output_file = f"recent_uranium_data_{ticker.replace('.', '_').replace('-', '_')}.csv"
            result.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")
            print(f"\n✓ SUCCESS! Use data from {output_file}")
            break
        else:
            print(f"  ✗ No data returned")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue
else:
    print("\n✗ Could not fetch data from any ticker.")
    print("\nAlternative options:")
    print("1. Try manually getting data from YCharts (https://ycharts.com/indicators/uranium_spot_price)")
    print("2. Check EIA reports for recent data")
    print("3. Use a different data source")


