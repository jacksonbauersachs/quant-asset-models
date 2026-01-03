"""
Uranium price data update script using Yahoo Finance.
This script fetches data from both UPC (old) and SPUT (new) and stitches them together.

Usage:
    python update_prices.py
"""

import requests
import pandas as pd
import os
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from io import StringIO
import sys

# Import configuration
try:
    from config import (
        CURRENCY_NAME, DATA_FOLDER, HISTORICAL_DATA_FILE
    )
except ImportError:
    print("Error: config.py not found! Make sure you're running from the clone folder.")
    sys.exit(1)

# Yahoo Finance endpoints
YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
YAHOO_QUOTE_URL = "https://finance.yahoo.com/quote/{}"
YAHOO_DOWNLOAD_URL = "https://query1.finance.yahoo.com/v7/finance/download/{}"

# Uranium tickers
UPC_TICKER = "U.TO"  # Uranium Participation Corp (old, delisted ~2021-07-16)
SPUT_TICKER_PRIMARY = "U.UN"  # Sprott Physical Uranium Trust (TSX CAD) - try U.UN format
SPUT_TICKER_FALLBACK1 = "U-UN.TO"  # Alternative format
SPUT_TICKER_FALLBACK2 = "SRUUF"  # Sprott Physical Uranium Trust (OTC, fallback)

# Transition date: UPC delisted around 2021-07-16, SPUT started ~2021-07-19
UPC_END_DATE = "2021-07-16"
SPUT_START_DATE = "2021-07-19"

def _get_yahoo_cookie_and_crumb(session: requests.Session, ticker: str):
    """
    1) Visit a quote page to set cookies.
    2) Extract crumb either from getcrumb endpoint or HTML.
    """
    # Try visiting a generic page first (SPY) to get cookies, then use the ticker
    try:
        # First, try visiting SPY to get cookies (more reliable)
        session.get("https://finance.yahoo.com/quote/SPY", timeout=30)
    except:
        pass
    
    # Step 1: visit quote page to set cookies (try the actual ticker)
    try:
        r = session.get(YAHOO_QUOTE_URL.format(ticker), timeout=30, allow_redirects=True)
        # Don't raise on 404, we'll handle it
        if r.status_code == 404:
            # Try with URL encoding
            encoded_ticker = urllib.parse.quote(ticker)
            r = session.get(YAHOO_QUOTE_URL.format(encoded_ticker), timeout=30, allow_redirects=True)
    except Exception as e:
        # If ticker page fails, try SPY history page
        r = session.get("https://finance.yahoo.com/quote/SPY/history", timeout=30)

    # Step 2: try crumb endpoint (works if cookie is valid)
    try:
        crumb_resp = session.get(YAHOO_CRUMB_URL, timeout=30)
        if crumb_resp.status_code == 200 and crumb_resp.text and "forbidden" not in crumb_resp.text.lower():
            crumb = crumb_resp.text.strip()
            if crumb and len(crumb) > 5:  # Valid crumbs are usually longer
                return crumb
    except:
        pass

    # Step 3: fallback: parse crumb from HTML (multiple patterns)
    patterns = [
        r'"crumb":"([^"]+)"',  # Standard pattern
        r'CrumbStore":{"crumb":"([^"]+)"}',  # Alternative pattern
        r'"crumb":"(.*?)"',  # More flexible
    ]
    
    for pattern in patterns:
        m = re.search(pattern, r.text)
        if m:
            crumb = m.group(1)
            # Handle unicode escape sequences
            try:
                crumb = crumb.encode("utf-8").decode("unicode_escape")
            except:
                pass
            if crumb and len(crumb) > 5:
                return crumb
    
    # Last resort: try to get crumb from any page
    try:
        r2 = session.get("https://finance.yahoo.com/quote/SPY/history", timeout=30)
        for pattern in patterns:
            m = re.search(pattern, r2.text)
            if m:
                crumb = m.group(1)
                try:
                    crumb = crumb.encode("utf-8").decode("unicode_escape")
                except:
                    pass
                if crumb and len(crumb) > 5:
                    return crumb
    except:
        pass
    
    raise RuntimeError("Could not extract crumb from Yahoo HTML. Yahoo may be blocking this IP/session.")

def yahoo_history_csv(ticker: str,
                      start: str = "2000-01-01",
                      end: str = None,
                      interval: str = "1d",
                      include_adjclose: bool = True,
                      session: requests.Session = None) -> pd.DataFrame:
    """
    Downloads historical OHLCV data from Yahoo Finance via CSV download endpoint.
    Handles cookie + crumb automatically.
    """
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Convert to unix epoch seconds
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int(pd.Timestamp(end).timestamp())

    if session is None:
        session = requests.Session()

    # User-Agent helps avoid blocks
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    })

    crumb = _get_yahoo_cookie_and_crumb(session, ticker)

    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true" if include_adjclose else "false",
        "crumb": crumb,
    }

    # Download CSV
    url = YAHOO_DOWNLOAD_URL.format(ticker)
    resp = session.get(url, params=params, timeout=30)

    # If forbidden, give a helpful error
    if resp.status_code == 403 or "User is unable to access this feature" in resp.text:
        raise RuntimeError(
            f"Yahoo blocked your request (403 Forbidden) for {ticker}. "
            "This often happens on cloud/VPS/corporate IPs. "
            "Try running from a home/residential network or use a VPN/residential proxy."
        )

    resp.raise_for_status()

    # Parse CSV into DataFrame
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        return df
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df

def fetch_uranium_data(start_date=None, end_date=None):
    """
    Fetch Uranium data by combining UPC (old) and SPUT (new).
    If start_date is None, fetches all available historical data.
    """
    session = requests.Session()
    
    # Set User-Agent
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    })
    
    all_dataframes = []
    
    # Determine date ranges
    if start_date is None:
        upc_start = "2005-01-01"
        sput_start = SPUT_START_DATE
    else:
        upc_start = start_date if start_date < SPUT_START_DATE else None
        sput_start = start_date if start_date >= SPUT_START_DATE else SPUT_START_DATE
    
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Fetch UPC data (if needed)
    if upc_start and upc_start <= UPC_END_DATE:
        upc_end = min(UPC_END_DATE, end_date) if end_date <= UPC_END_DATE else UPC_END_DATE
        if upc_start <= upc_end:
            try:
                print(f"Fetching UPC ({UPC_TICKER}) data from {upc_start} to {upc_end}...")
                df_upc = yahoo_history_csv(UPC_TICKER, start=upc_start, end=upc_end, session=session)
                if not df_upc.empty:
                    print(f"  ✓ UPC: {len(df_upc)} rows from {df_upc.index.min().date()} to {df_upc.index.max().date()}")
                    all_dataframes.append(df_upc)
                else:
                    print(f"  ⚠ No UPC data found")
            except Exception as e:
                print(f"  ⚠ Error fetching UPC data: {e}")
            time.sleep(1)  # Be nice to Yahoo
    
    # Fetch SPUT data (if needed)
    if sput_start and sput_start <= end_date:
        sput_tickers = [SPUT_TICKER_PRIMARY, SPUT_TICKER_FALLBACK1, SPUT_TICKER_FALLBACK2]
        sput_fetched = False
        
        for sput_ticker in sput_tickers:
            if sput_fetched:
                break
            try:
                if sput_ticker == SPUT_TICKER_PRIMARY:
                    print(f"Fetching SPUT ({sput_ticker}) data from {sput_start} to {end_date}...")
                else:
                    print(f"  Trying fallback ticker {sput_ticker}...")
                
                df_sput = yahoo_history_csv(sput_ticker, start=sput_start, end=end_date, session=session)
                if not df_sput.empty:
                    print(f"  ✓ SPUT ({sput_ticker}): {len(df_sput)} rows from {df_sput.index.min().date()} to {df_sput.index.max().date()}")
                    all_dataframes.append(df_sput)
                    sput_fetched = True
                else:
                    print(f"  ⚠ No data found for {sput_ticker}")
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "Not Found" in error_msg:
                    print(f"  ⚠ Ticker {sput_ticker} not found (404)")
                elif "403" in error_msg or "Forbidden" in error_msg:
                    print(f"  ⚠ Access forbidden for {sput_ticker} (403)")
                else:
                    print(f"  ⚠ Error fetching SPUT data ({sput_ticker}): {e}")
            
            if not sput_fetched and sput_ticker != sput_tickers[-1]:
                time.sleep(1)  # Be nice to Yahoo between attempts
        
        if not sput_fetched:
            print(f"  ✗ Failed to fetch SPUT data from all ticker options")
        time.sleep(1)  # Be nice to Yahoo
    
    if not all_dataframes:
        print("No data fetched from any source.")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes)
    
    # Remove duplicates (keep last occurrence)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
    
    # Filter by date range if specified
    if start_date:
        combined_df = combined_df[combined_df.index >= pd.Timestamp(start_date)]
    if end_date:
        combined_df = combined_df[combined_df.index <= pd.Timestamp(end_date)]
    
    return combined_df

def format_volume(volume):
    """Format volume to match CSV format (with K suffix)"""
    try:
        vol_num = float(volume)
        if pd.isna(vol_num) or vol_num == 0:
            return "0.00K"
        if vol_num >= 1000:
            return f"{vol_num/1000:.2f}K"
        else:
            return f"{vol_num:.2f}K"
    except:
        return "0.00K"

def calculate_change_percent(open_price, close_price):
    """Calculate percentage change from open to close"""
    try:
        if pd.isna(open_price) or open_price == 0:
            return "0.00%"
        change = ((close_price - open_price) / open_price) * 100
        return f"{change:.2f}%"
    except:
        return "0.00%"

def convert_data_to_dataframe(df):
    """
    Convert fetched DataFrame to expected format with proper columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    records = []
    for date, row in df.iterrows():
        # Use Adjusted Close if available, otherwise Close
        close_price = row.get('Adj Close', row.get('Close', 0))
        open_price = row.get('Open', close_price)
        high_price = row.get('High', close_price)
        low_price = row.get('Low', close_price)
        volume = row.get('Volume', 0)
        
        record = {
            'Date': date.strftime('%m/%d/%Y'),  # Convert to M/D/YYYY format
            'Price': float(close_price),  # Use adjusted close as Price
            'Open': float(open_price),
            'High': float(high_price),
            'Low': float(low_price),
            'Vol.': format_volume(volume),
            'Change %': calculate_change_percent(open_price, close_price)
        }
        records.append(record)
    
    result_df = pd.DataFrame(records)
    
    if result_df.empty:
        return result_df
    
    # Sort by date
    result_df['Date'] = pd.to_datetime(result_df['Date'], format='%m/%d/%Y')
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    
    return result_df

def get_last_date_from_file(file_path):
    """Get the last date from the existing data file."""
    try:
        df = pd.read_csv(file_path)
        # Try to parse date - handle both M/D/YYYY and YYYY-MM-DD formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        except ValueError:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            except ValueError:
                df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def update_prices():
    """
    Update price data from the last available date to today.
    If the file doesn't exist, fetches all available historical data.
    """
    # Build file path
    data_file = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILE)
    
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Data file not found. Fetching all available historical data for {CURRENCY_NAME}...")
        
        # Fetch all available historical data
        all_data = fetch_uranium_data()
        
        if all_data is None or all_data.empty:
            print("Failed to fetch historical data.")
            return None
        
        # Convert to DataFrame
        df = convert_data_to_dataframe(all_data)
        
        if df.empty:
            print("No data to save.")
            return None
        
        # Convert Date back to M/D/YYYY string format for CSV
        df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
        
        # Save the data
        df.to_csv(data_file, index=False)
        
        # Get price range (need to convert back to numeric since Date is now string)
        df_temp = df.copy()
        df_temp['Price'] = pd.to_numeric(df_temp['Price'], errors='coerce')
        
        print(f"\n✅ Successfully created {CURRENCY_NAME} data file!")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"Price range: ${df_temp['Price'].min():.2f} to ${df_temp['Price'].max():.2f}")
        
        return df
    
    # File exists - update from last date
    # Get the last date from the existing file
    last_date = get_last_date_from_file(data_file)
    if not last_date:
        print("Could not determine last date from file.")
        return None
    
    # Calculate today's date
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    print(f"Last date in file: {last_date}")
    print(f"Today's date: {today}")
    
    # Check if we need to update
    if last_date >= today:
        print("Data is already up to date!")
        return None
    
    # Calculate the start date for fetching (next day after last date)
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    start_date = (last_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # For Yahoo Finance, we need to fetch from start_date to tomorrow to include today
    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching new data from {start_date} to {today}...")
    
    # Fetch new data
    new_data = fetch_uranium_data(start_date=start_date, end_date=end_date)
    
    if new_data is None or new_data.empty:
        print("No new data was fetched.")
        return None
    
    # Convert new data to DataFrame with proper formatting
    # Filter to only include dates after last_date
    last_date_dt = pd.Timestamp(last_date)
    new_data = new_data[new_data.index > last_date_dt]
    
    if new_data.empty:
        print("No new records to add.")
        return None
    
    new_df = convert_data_to_dataframe(new_data)
    
    if new_df.empty:
        print("No new records to add.")
        return None
    
    print(f"Fetched {len(new_df)} new days of data")
    print(f"Date range: {new_df['Date'].min().strftime('%m/%d/%Y')} to {new_df['Date'].max().strftime('%m/%d/%Y')}")
    print(f"Price range: ${new_df['Price'].min():.2f} to ${new_df['Price'].max():.2f}")
    
    # Read existing data
    existing_df = pd.read_csv(data_file)
    # Try to parse date - handle both M/D/YYYY and YYYY-MM-DD formats
    try:
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%m/%d/%Y')
    except ValueError:
        try:
            existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d')
        except ValueError:
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Convert Date back to M/D/YYYY string format for consistency with original format
    combined_df['Date'] = combined_df['Date'].dt.strftime('%m/%d/%Y')
    
    # Save updated data
    combined_df.to_csv(data_file, index=False)
    
    print(f"\n✅ Successfully updated {CURRENCY_NAME} data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_df)} new records")
    
    return combined_df

def main():
    """Main function to run the price update."""
    
    print(f"{CURRENCY_NAME} Price Data Update")
    print("=" * 30)
    print("Fetching data from:")
    print(f"  - UPC (Uranium Participation Corp): {UPC_TICKER} (until {UPC_END_DATE})")
    print(f"  - SPUT (Sprott Physical Uranium Trust): {SPUT_TICKER_PRIMARY} (from {SPUT_START_DATE})")
    print(f"  - Fallbacks: {SPUT_TICKER_FALLBACK1}, {SPUT_TICKER_FALLBACK2}")
    print("=" * 30)
    
    try:
        updated_df = update_prices()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
