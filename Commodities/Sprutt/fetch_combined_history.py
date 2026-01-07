"""
Fetch and combine SPUT (Sprott Physical Uranium Trust) historical price data.
Combines:
- Pre-reorg: Uranium Participation Corporation (UPC) - U.TO (inception to July 16, 2021)
- Post-reorg: SPUT - SRUUF or U-U.TO (July 19, 2021 to today)
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reorganization date
REORG_DATE = "2021-07-19"  # SPUT started trading
PRE_REORG_END = "2021-07-16"  # Last trading day before reorg

# Output file
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "SPUT_historical_data.csv")

# Optional: Manual UPC data file (if you have it)
MANUAL_UPC_FILE = os.path.join(SCRIPT_DIR, "UPC_historical_data.csv")

def fetch_upc_data():
    """Fetch UPC (U.TO) data from inception to July 16, 2021."""
    print("=" * 60)
    print("Fetching UPC (Pre-Reorg) Data")
    print("=" * 60)
    print(f"Ticker: U.TO (Uranium Participation Corporation)")
    print(f"Date range: Inception to {PRE_REORG_END}")
    
    # Try different approaches to get UPC data
    tickers_to_try = [
        "U.TO",
        "U-TO", 
        "UPC.TO",
        "UPC-TO"
    ]
    
    for ticker_symbol in tickers_to_try:
        print(f"\nTrying {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Try with period='max' first
            df = ticker.history(period='max', end=REORG_DATE)
            
            if df.empty:
                # Try with explicit start date (UPC started around 2005)
                df = ticker.history(start="2005-01-01", end=REORG_DATE)
            
            if not df.empty:
                # Filter to only dates before reorg
                df = df[df.index < pd.Timestamp(REORG_DATE)]
                
                if not df.empty:
                    # Convert to standard format
                    result = pd.DataFrame({
                        'Date': df.index.strftime('%Y-%m-%d'),
                        'Price': df['Close'].values
                    })
                    
                    print(f"[OK] Fetched {len(result)} records using {ticker_symbol}")
                    print(f"  Date range: {result['Date'].iloc[0]} to {result['Date'].iloc[-1]}")
                    print(f"  Price range: ${result['Price'].min():.2f} to ${result['Price'].max():.2f}")
                    
                    return result
            
        except Exception as e:
            print(f"  [WARN] Error with {ticker_symbol}: {e}")
            continue
    
    # Try loading from manual file if it exists
    if os.path.exists(MANUAL_UPC_FILE):
        print(f"\n[INFO] Found manual UPC data file: {MANUAL_UPC_FILE}")
        try:
            manual_df = pd.read_csv(MANUAL_UPC_FILE)
            
            # Handle different column names
            if 'Date' not in manual_df.columns:
                for col in manual_df.columns:
                    if 'date' in col.lower():
                        manual_df = manual_df.rename(columns={col: 'Date'})
                        break
            
            if 'Price' not in manual_df.columns:
                for col in manual_df.columns:
                    if 'price' in col.lower() or 'close' in col.lower():
                        manual_df = manual_df.rename(columns={col: 'Price'})
                        break
            
            # Convert dates and filter to pre-reorg
            manual_df['Date'] = pd.to_datetime(manual_df['Date'])
            manual_df = manual_df[manual_df['Date'] < pd.Timestamp(REORG_DATE)]
            
            if not manual_df.empty:
                # Format dates
                result = pd.DataFrame({
                    'Date': manual_df['Date'].dt.strftime('%Y-%m-%d'),
                    'Price': manual_df['Price'].values
                })
                
                print(f"[OK] Loaded {len(result)} UPC records from manual file")
                print(f"  Date range: {result['Date'].iloc[0]} to {result['Date'].iloc[-1]}")
                print(f"  Price range: ${result['Price'].min():.2f} to ${result['Price'].max():.2f}")
                
                return result
        except Exception as e:
            print(f"[WARN] Error loading manual UPC file: {e}")
    
    print("[FAIL] Could not fetch UPC data from any ticker")
    print("[INFO] UPC (U.TO) may be delisted and data unavailable via Yahoo Finance")
    print(f"[INFO] To add UPC data manually:")
    print(f"  1. Download UPC historical data from another source")
    print(f"  2. Save as CSV with columns: Date, Price")
    print(f"  3. Place it in this folder as: {os.path.basename(MANUAL_UPC_FILE)}")
    print(f"  4. Run this script again to combine with SPUT data")
    print("[INFO] Continuing with SPUT data only...")
    return None

def fetch_sput_data():
    """Fetch SPUT data from July 19, 2021 to today."""
    print("\n" + "=" * 60)
    print("Fetching SPUT (Post-Reorg) Data")
    print("=" * 60)
    
    # Try SRUUF first (US OTC), then U-U.TO (TSX)
    tickers_to_try = [
        ("SRUUF", "US OTC"),
        ("U-U.TO", "TSX")
    ]
    
    for ticker_symbol, exchange in tickers_to_try:
        print(f"\nTrying {ticker_symbol} ({exchange})...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Fetch from reorg date to today
            df = ticker.history(start=REORG_DATE, end=None)
            
            if not df.empty:
                # Convert to standard format
                result = pd.DataFrame({
                    'Date': df.index.strftime('%Y-%m-%d'),
                    'Price': df['Close'].values
                })
                
                print(f"[OK] Fetched {len(result)} records using {ticker_symbol}")
                print(f"  Date range: {result['Date'].iloc[0]} to {result['Date'].iloc[-1]}")
                print(f"  Price range: ${result['Price'].min():.2f} to ${result['Price'].max():.2f}")
                
                return result
            else:
                print(f"  [WARN] No data returned for {ticker_symbol}")
        except Exception as e:
            print(f"  [WARN] Error with {ticker_symbol}: {e}")
            continue
    
    print("[FAIL] Could not fetch SPUT data from any ticker")
    return None

def combine_and_save(upc_df, sput_df):
    """Combine UPC and SPUT data, remove duplicates, and save."""
    print("\n" + "=" * 60)
    print("Combining Data")
    print("=" * 60)
    
    if upc_df is None and sput_df is None:
        print("[FAIL] No data to combine")
        return None
    
    all_data = []
    
    if upc_df is not None:
        print(f"  Adding {len(upc_df)} UPC records (pre-reorg)")
        all_data.append(upc_df)
    
    if sput_df is not None:
        print(f"  Adding {len(sput_df)} SPUT records (post-reorg)")
        all_data.append(sput_df)
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert Date to datetime for sorting
    combined['Date'] = pd.to_datetime(combined['Date'])
    
    # Remove duplicates (keep last occurrence if any overlap)
    combined = combined.drop_duplicates(subset=['Date'], keep='last')
    
    # Sort by date
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Format dates back to YYYY-MM-DD
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    
    # Save
    combined.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[OK] Successfully saved {len(combined)} records to {OUTPUT_FILE}")
    print(f"  Date range: {combined['Date'].iloc[0]} to {combined['Date'].iloc[-1]}")
    print(f"  Price range: ${combined['Price'].min():.2f} to ${combined['Price'].max():.2f}")
    
    # Check for gap around reorg date
    upc_last = upc_df['Date'].iloc[-1] if upc_df is not None else None
    sput_first = sput_df['Date'].iloc[0] if sput_df is not None else None
    
    if upc_last and sput_first:
        upc_last_date = pd.to_datetime(upc_last)
        sput_first_date = pd.to_datetime(sput_first)
        gap_days = (sput_first_date - upc_last_date).days
        
        if gap_days > 3:  # More than a weekend gap
            print(f"\n[INFO] Gap of {gap_days} days between UPC ({upc_last}) and SPUT ({sput_first})")
            print(f"  This is expected due to the reorganization weekend")
    
    return combined

def main():
    """Main function."""
    print("=" * 60)
    print("SPUT Combined Historical Data Fetcher")
    print("Combines UPC (pre-reorg) + SPUT (post-reorg)")
    print("=" * 60)
    
    # Fetch UPC data (pre-reorg)
    upc_df = fetch_upc_data()
    
    # Fetch SPUT data (post-reorg)
    sput_df = fetch_sput_data()
    
    # Combine and save
    combined_df = combine_and_save(upc_df, sput_df)
    
    if combined_df is not None:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Combined dataset saved to: {OUTPUT_FILE}")
        print(f"Total records: {len(combined_df)}")
        print(f"Ready to use for analysis!")
    else:
        print("\n[FAIL] Could not create combined dataset")

if __name__ == "__main__":
    main()

