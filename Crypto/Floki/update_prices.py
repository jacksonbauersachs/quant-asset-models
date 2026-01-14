"""
Template script for updating currency price data from Yahoo Finance.
This script reads configuration from config.py.

Usage:
    1. Edit config.py with your currency information
    2. Run: python update_prices.py
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

# Import configuration
try:
    from config import (
        CURRENCY_NAME, YAHOO_TICKER, DATA_FOLDER, 
        HISTORICAL_DATA_FILE
    )
except ImportError:
    print("Error: config.py not found! Make sure you're running from the clone folder.")
    sys.exit(1)

def get_last_date_from_file(file_path):
    """Get the last date from the existing data file."""
    try:
        df = pd.read_csv(file_path)
        # Try to parse date - handle both M/D/YYYY and YYYY-MM-DD formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        except ValueError:
            # If that fails, try YYYY-MM-DD format
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            except ValueError:
                # Last resort: let pandas auto-detect
                df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def fetch_data_for_period(start_date=None, end_date=None):
    """
    Fetch currency data using Yahoo Finance.
    If start_date is None, fetches all available historical data.
    """
    try:
        print(f"Using Yahoo Finance ticker: {YAHOO_TICKER}")
        
        # Download data from Yahoo Finance
        ticker = yf.Ticker(YAHOO_TICKER)
        
        if start_date is None:
            # Fetch all available historical data
            print(f"Fetching ALL available historical data for {CURRENCY_NAME}...")
            df = ticker.history(period='max', interval='1d')
        else:
            # Fetch data for specific date range
            print(f"Fetching {CURRENCY_NAME} data from {start_date} to {end_date}...")
            df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"No data found for {YAHOO_TICKER}")
            return None
        
        # Convert to dictionary format matching expected structure
        data_dict = {}
        for date, row in df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            data_dict[date_str] = {
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume']
            }
        
        print(f"Successfully fetched {len(data_dict)} days of data")
        if start_date is None and len(data_dict) > 0:
            dates = sorted(data_dict.keys())
            print(f"Date range: {dates[0]} to {dates[-1]}")
        
        return data_dict
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print(f"Make sure {YAHOO_TICKER} is a valid Yahoo Finance ticker (e.g., 'DOGE-USD', 'BTC-USD', 'ETH-USD')")
        return None

def format_volume(volume):
    """Format volume to match CSV format (with K suffix)"""
    try:
        vol_num = float(volume)
        if vol_num >= 1000:
            return f"{vol_num/1000:.2f}K"
        else:
            return f"{vol_num:.2f}K"
    except:
        return "0.00K"

def calculate_change_percent(open_price, close_price):
    """Calculate percentage change from open to close"""
    try:
        change = ((close_price - open_price) / open_price) * 100
        return f"{change:.2f}%"
    except:
        return "0.00%"

def convert_data_to_dataframe(data_dict):
    """
    Convert fetched data dictionary to DataFrame with proper formatting
    """
    records = []
    for date, values in data_dict.items():
        date_dt = datetime.strptime(date, '%Y-%m-%d')
        open_price = float(values['Open'])
        close_price = float(values['Close'])
        
        record = {
            'Date': date_dt.strftime('%m/%d/%Y'),  # Convert to M/D/YYYY format
            'Price': close_price,  # Use close price as Price
            'Open': open_price,
            'High': float(values['High']),
            'Low': float(values['Low']),
            'Vol.': format_volume(values['Volume']),  # Format volume with K suffix
            'Change %': calculate_change_percent(open_price, close_price)  # Calculate percentage change
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    if df.empty:
        return df
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

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
        all_data = fetch_data_for_period()
        
        if not all_data:
            print("Failed to fetch historical data.")
            return
        
        # Convert to DataFrame
        df = convert_data_to_dataframe(all_data)
        
        if df.empty:
            print("No data to save.")
            return
        
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
        return
    
    # Calculate today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Last date in file: {last_date}")
    print(f"Today's date: {today}")
    
    # Check if we need to update
    if last_date >= today:
        print("Data is already up to date!")
        return
    
    # Calculate the start date for fetching (next day after last date)
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    start_date = (last_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # For Yahoo Finance, we need to fetch from start_date to tomorrow to include today
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching new data from {start_date} to {today}...")
    
    # Fetch new data
    new_data = fetch_data_for_period(start_date, end_date)
    
    if not new_data:
        print("No new data was fetched.")
        return
    
    # Convert new data to DataFrame with proper formatting
    # Filter to only include dates after last_date
    filtered_data = {}
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    for date, values in new_data.items():
        date_dt = datetime.strptime(date, '%Y-%m-%d')
        if date_dt > last_date_dt:  # Only include dates after last_date
            filtered_data[date] = values
    
    if not filtered_data:
        print("No new records to add.")
        return
    
    new_df = convert_data_to_dataframe(filtered_data)
    
    if new_df.empty:
        print("No new records to add.")
        return
    
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
    
    print(f"\nSuccessfully updated {CURRENCY_NAME} data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_df)} new records")
    
    return combined_df

def main():
    """Main function to run the price update."""
    
    print(f"{CURRENCY_NAME} Price Data Update")
    print("=" * 30)
    
    try:
        updated_df = update_prices()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")

if __name__ == "__main__":
    main()

