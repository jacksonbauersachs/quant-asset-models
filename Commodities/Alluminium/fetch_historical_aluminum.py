"""
Fetch comprehensive aluminum price data from multiple sources.
Combines historical data from FRED, LME, and Yahoo Finance.
"""

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timezone
import os
import time

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def search_fred_series():
    """Search FRED for aluminum-related series."""
    print("\n[1a] Searching FRED for aluminum price series...")
    
    try:
        # Search FRED for aluminum series
        url = "https://api.stlouisfed.org/fred/series/search"
        params = {
            'search_text': 'aluminum price',
            'api_key': 'fred',
            'file_type': 'json',
            'limit': 20,
        }
        
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'seriess' in data:
                print(f"  [INFO] Found {len(data['seriess'])} potential series")
                for series in data['seriess'][:5]:  # Show first 5
                    print(f"    - {series.get('id', 'N/A')}: {series.get('title', 'N/A')}")
                return [s['id'] for s in data['seriess']]
    except Exception as e:
        print(f"  [WARN] Could not search FRED: {e}")
    
    return []

def fetch_from_fred():
    """Fetch aluminum price data from FRED (Federal Reserve Economic Data)."""
    print("\n[1] Trying FRED (Federal Reserve Economic Data)...")
    
    # First, search for aluminum series
    series_ids = search_fred_series()
    
    # Also try known series IDs
    known_series = [
        "APU0000717311",  # Producer Price Index
        "PALUMUSDM",      # Aluminum price USD per metric ton
    ]
    series_ids.extend(known_series)
    
    all_data = []
    
    for series_id in series_ids[:5]:  # Try first 5 series
        try:
            # FRED API endpoint
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': 'fred',
                'file_type': 'json',
                'observation_start': '1990-01-01',
                'observation_end': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and len(data['observations']) > 0:
                    series_data = []
                    for obs in data['observations']:
                        if obs.get('value') != '.' and obs.get('value') is not None:
                            try:
                                date = obs['date']
                                price = float(obs['value'])
                                series_data.append({'Date': date, 'Price': price})
                            except (ValueError, KeyError):
                                continue
                    
                    if series_data:
                        print(f"  [OK] Found {len(series_data)} observations from {series_id}")
                        all_data.extend(series_data)
                        break  # Use first successful series
        except Exception as e:
            continue
    
    if all_data:
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')
        return df
    else:
        print("  [FAIL] No data found from FRED")
        print("  [INFO] FRED may not have direct aluminum price series")
        print("  [INFO] Try USGS or LME data instead")
        return None

def fetch_from_yahoo():
    """Fetch recent data from Yahoo Finance (complements FRED)."""
    print("\n[2] Trying Yahoo Finance for recent data...")
    
    tickers = ["ALI=F", "ALF24.NYM"]  # Aluminum futures
    
    for ticker in tickers:
        try:
            print(f"  Trying ticker: {ticker}")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period='max')
            
            if not df.empty:
                print(f"  [OK] Found {len(df)} days from Yahoo Finance")
                result = pd.DataFrame({
                    'Date': df.index,
                    'Price': df['Close'].values
                })
                result['Date'] = pd.to_datetime(result['Date'])
                return result
        except Exception as e:
            print(f"  [WARN] Error with {ticker}: {e}")
            continue
    
    print("  [FAIL] No data from Yahoo Finance")
    return None

def fetch_from_lme_scrape():
    """Try to scrape LME data (limited success expected)."""
    print("\n[3] Trying LME website (may require manual download)...")
    print("  [INFO] LME data typically requires subscription or manual download")
    print("  [INFO] Visit: https://www.lme.com/en/Market-data/Historical-data")
    return None

def combine_and_save(fred_data, yahoo_data, output_file):
    """Combine data from all sources and save."""
    print("\n" + "=" * 60)
    print("Combining data from all sources...")
    print("=" * 60)
    
    all_dfs = []
    
    if fred_data is not None and not fred_data.empty:
        print(f"  FRED: {len(fred_data)} records from {fred_data['Date'].min().date()} to {fred_data['Date'].max().date()}")
        all_dfs.append(fred_data)
    
    if yahoo_data is not None and not yahoo_data.empty:
        print(f"  Yahoo: {len(yahoo_data)} records from {yahoo_data['Date'].min().date()} to {yahoo_data['Date'].max().date()}")
        all_dfs.append(yahoo_data)
    
    if not all_dfs:
        print("\n[FAIL] No data to combine!")
        return None
    
    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (keep last occurrence)
    combined = combined.drop_duplicates(subset=['Date'], keep='last')
    
    # Sort by date
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Format for output
    output_df = pd.DataFrame({
        'Date': combined['Date'].dt.strftime('%Y-%m-%d'),
        'Price': combined['Price']
    })
    
    # Save
    output_df.to_csv(output_file, index=False)
    
    print(f"\n[OK] Successfully saved {len(output_df)} records to {output_file}")
    print(f"  Date range: {output_df['Date'].iloc[0]} to {output_df['Date'].iloc[-1]}")
    print(f"  Price range: ${output_df['Price'].min():.2f} to ${output_df['Price'].max():.2f}")
    
    return output_df

def main():
    """Main function."""
    print("=" * 60)
    print("Fetching Historical Aluminum Price Data")
    print("=" * 60)
    print("\nAttempting multiple data sources...")
    
    # Fetch from different sources
    fred_data = fetch_from_fred()
    yahoo_data = fetch_from_yahoo()
    lme_data = fetch_from_lme_scrape()
    
    # Output file
    output_file = os.path.join(SCRIPT_DIR, "Alluminium_historical_data.csv")
    
    # Combine and save
    result = combine_and_save(fred_data, yahoo_data, output_file)
    
    if result is None:
        print("\n" + "=" * 60)
        print("Alternative Options:")
        print("=" * 60)
        print("1. USGS Mineral Industry Surveys:")
        print("   https://apps.usgs.gov/minerals-information-archives/aluminum/")
        print("   - Download monthly PDFs and extract prices")
        print("\n2. LME Historical Data:")
        print("   https://www.lme.com/en/Market-data/Historical-data")
        print("   - May require account/subscription")
        print("\n3. YCharts (manual):")
        print("   https://ycharts.com/indicators/aluminum_spot_price")
        print("   - Export data manually")
        print("\n4. Use add_missing_data.py to manually add data")

if __name__ == "__main__":
    main()

