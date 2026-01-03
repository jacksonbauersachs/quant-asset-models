"""
Combine Macrotrends historical nickel spot price data with recent data
from other sources to fill in the missing last 6 months.
"""

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import os
import re

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_macrotrends_data(file_path=None):
    """Load Macrotrends data from CSV file."""
    print("=" * 60)
    print("Loading Macrotrends Historical Data")
    print("=" * 60)
    
    if file_path is None:
        file_path = os.path.join(SCRIPT_DIR, "Nickel_historical_data.csv")
    
    if not os.path.exists(file_path):
        print(f"\n[FAIL] File not found: {file_path}")
        print("\nTo get Macrotrends data:")
        print("1. Visit: https://www.macrotrends.net/1477/nickel-prices-historical-chart-data")
        print("2. Click 'Download Data' or copy the table")
        print("3. Save as CSV with columns: Date, Price")
        print("4. Place it in this folder as 'Nickel_historical_data.csv'")
        return None
    
    try:
        # Try reading with header first (most common case)
        try:
            df = pd.read_csv(file_path)
        except:
            # If that fails, try without header
            try:
                df = pd.read_csv(file_path, sep='\t', header=None, names=['Date', 'Price'])
            except:
                df = pd.read_csv(file_path, sep=',', header=None, names=['Date', 'Price'])
        
        # Handle different column names if header exists
        if 'Date' not in df.columns:
            # Try to find date column
            for col in df.columns:
                if 'date' in col.lower() or 'year' in col.lower():
                    df = df.rename(columns={col: 'Date'})
                    break
            # If still not found, assume first column is Date
            if 'Date' not in df.columns:
                df.columns = ['Date', 'Price'] + list(df.columns[2:])
        
        if 'Price' not in df.columns:
            # Try to find price column
            for col in df.columns:
                if 'price' in col.lower() or 'value' in col.lower() or 'close' in col.lower():
                    df = df.rename(columns={col: 'Price'})
                    break
            # If still not found, assume second column is Price
            if 'Price' not in df.columns and len(df.columns) >= 2:
                df = df.iloc[:, :2]
                df.columns = ['Date', 'Price']
        
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        
        # Clean price (remove $, commas)
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"\n[OK] Loaded {len(df)} records from Macrotrends")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"  Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"\n[FAIL] Error loading Macrotrends data: {e}")
        return None

def fetch_recent_spot_prices():
    """Fetch recent nickel spot prices from various sources."""
    print("\n" + "=" * 60)
    print("Fetching Recent Spot Prices (Last 6 Months)")
    print("=" * 60)
    
    # Calculate date 6 months ago
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    start_date = six_months_ago.strftime('%Y-%m-%d')
    
    all_data = []
    
    # Method 1: Try YCharts scraping (spot price)
    print("\n[1] Trying YCharts for spot prices...")
    try:
        url = "https://ycharts.com/indicators/nickel_spot_price"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # YCharts uses JavaScript, so direct scraping is limited
            # But we can try to find data in the page
            page_text = soup.get_text()
            # Look for recent price data patterns
            # This is a simplified approach - YCharts may require Selenium
            print("  [WARN] YCharts requires JavaScript rendering (would need Selenium)")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    
    # Method 2: Try Trading Economics
    print("\n[2] Trying Trading Economics...")
    try:
        url = "https://tradingeconomics.com/commodity/nickel"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Trading Economics also uses JavaScript
            print("  [WARN] Trading Economics requires JavaScript rendering")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    
    # Method 3: Use Yahoo Finance futures as proxy (not ideal, but available)
    print("\n[3] Trying Yahoo Finance (futures - will note as proxy)...")
    try:
        # Try different nickel futures tickers
        tickers_to_try = ["NICKEL=F", "NI=F", "NIL=F"]  # Common nickel futures tickers
        for ticker in tickers_to_try:
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start_date, end=None)
                
                if not df.empty:
                    print(f"  [OK] Found {len(df)} days of futures data using {ticker}")
                    print(f"  [WARN] Note: This is futures price, not spot, but may be close")
                    
                    result = pd.DataFrame({
                        'Date': df.index,
                        'Price': df['Close'].values,
                        'Source': f'Yahoo_Futures_{ticker}'  # Mark as futures
                    })
                    result['Date'] = pd.to_datetime(result['Date'])
                    return result
            except Exception as e:
                continue
        
        print("  [WARN] Could not find nickel futures data on Yahoo Finance")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
    
    # Method 4: Try LME (London Metal Exchange) - may need API or manual download
    print("\n[4] Note: LME data available at https://www.lme.com/en/Market-data")
    print("  [INFO] LME provides official nickel prices but requires manual download")
    
    return None

def combine_data(macrotrends_df, recent_df):
    """Combine Macrotrends historical data with recent data."""
    print("\n" + "=" * 60)
    print("Combining Data Sources")
    print("=" * 60)
    
    if macrotrends_df is None:
        print("[FAIL] No Macrotrends data to combine")
        return None
    
    # Get the last date from Macrotrends (make timezone-naive for comparison)
    last_macrotrends_date = macrotrends_df['Date'].max()
    if hasattr(last_macrotrends_date, 'tz') and last_macrotrends_date.tz is not None:
        last_macrotrends_date = last_macrotrends_date.tz_localize(None)
    print(f"\nLast Macrotrends date: {last_macrotrends_date.date()}")
    
    if recent_df is None or recent_df.empty:
        print("[WARN] No recent data found - using Macrotrends only")
        print("       You may need to manually add the last 6 months")
        combined = macrotrends_df.copy()
    else:
        # Make recent_df dates timezone-naive for comparison
        if recent_df['Date'].dt.tz is not None:
            recent_df = recent_df.copy()
            recent_df['Date'] = recent_df['Date'].dt.tz_localize(None)
        
        # Filter recent data to only dates after Macrotrends
        recent_df = recent_df[recent_df['Date'] > last_macrotrends_date]
        
        if recent_df.empty:
            print("[WARN] Recent data doesn't extend beyond Macrotrends")
            combined = macrotrends_df.copy()
        else:
            print(f"[OK] Adding {len(recent_df)} new records from recent source")
            
            # Prepare recent data (remove Source column if exists)
            recent_clean = pd.DataFrame({
                'Date': recent_df['Date'],
                'Price': recent_df['Price']
            })
            
            # Combine
            combined = pd.concat([macrotrends_df[['Date', 'Price']], recent_clean], ignore_index=True)
    
    # Remove duplicates (keep last)
    combined = combined.drop_duplicates(subset=['Date'], keep='last')
    
    # Sort by date
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Format dates
    output_df = pd.DataFrame({
        'Date': combined['Date'].dt.strftime('%Y-%m-%d'),
        'Price': combined['Price']
    })
    
    return output_df

def save_data(df, output_file=None):
    """Save combined data to CSV."""
    if df is None or df.empty:
        print("\n[FAIL] No data to save")
        return False
    
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "Nickel_historical_data.csv")
    else:
        # If relative path, make it relative to script directory
        if not os.path.isabs(output_file):
            output_file = os.path.join(SCRIPT_DIR, output_file)
    
    df.to_csv(output_file, index=False)
    
    print(f"\n[OK] Successfully saved {len(df)} records to {output_file}")
    print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"  Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("Nickel Price Data Combiner")
    print("Combines Macrotrends (historical) + Recent Sources (last 6 months)")
    print("=" * 60)
    
    # Load Macrotrends data
    macrotrends_df = load_macrotrends_data()
    
    if macrotrends_df is None:
        print("\n[FAIL] Cannot proceed without Macrotrends data")
        return
    
    # Fetch recent data
    recent_df = fetch_recent_spot_prices()
    
    # Combine
    combined_df = combine_data(macrotrends_df, recent_df)
    
    # Save
    if combined_df is not None:
        save_data(combined_df)
        
        # Check if we still need recent months
        last_date = pd.to_datetime(combined_df['Date'].max())
        days_old = (datetime.now(timezone.utc) - last_date.replace(tzinfo=timezone.utc)).days
        
        if days_old > 30:
            print(f"\n[WARN] Data is {days_old} days old")
            print("       Consider manually adding recent months using add_missing_data.py")
            print("       Sources for spot prices:")
            print("       - YCharts: https://ycharts.com/indicators/nickel_spot_price")
            print("       - Trading Economics: https://tradingeconomics.com/commodity/nickel")
            print("       - LME: https://www.lme.com/en/Market-data")

if __name__ == "__main__":
    main()

