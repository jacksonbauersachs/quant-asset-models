import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_bitcoin_data_for_period(start_date, end_date, api_key):
    """
    Fetch Bitcoin data for a specific date range
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            print(f"API Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            print(f"API Note: {data['Note']}")
            return None
        
        # Extract time series data
        time_series_key = 'Time Series (Digital Currency Daily)'
        if time_series_key not in data:
            print("No time series data found")
            return None
        
        time_series = data[time_series_key]
        
        # Filter data for the specified date range
        filtered_data = {}
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for date_str, values in time_series.items():
            date_dt = datetime.strptime(date_str, '%Y-%m-%d')
            if start_dt <= date_dt <= end_dt:
                filtered_data[date_str] = values
        
        return filtered_data
        
    except Exception as e:
        print(f"Error fetching data for {start_date} to {end_date}: {e}")
        return None

def fetch_complete_bitcoin_history():
    """
    Fetch complete historical Bitcoin data by making multiple API calls
    """
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    print("Fetching complete historical Bitcoin data...")
    print("This will make multiple API calls to get all historical data.")
    
    # Define date ranges to fetch (Bitcoin started around 2010)
    # We'll fetch in 1-year chunks to work within API limits
    date_ranges = [
        ('2010-07-01', '2011-06-30'),
        ('2011-07-01', '2012-06-30'),
        ('2012-07-01', '2013-06-30'),
        ('2013-07-01', '2014-06-30'),
        ('2014-07-01', '2015-06-30'),
        ('2015-07-01', '2016-06-30'),
        ('2016-07-01', '2017-06-30'),
        ('2017-07-01', '2018-06-30'),
        ('2018-07-01', '2019-06-30'),
        ('2019-07-01', '2020-06-30'),
        ('2020-07-01', '2021-06-30'),
        ('2021-07-01', '2022-06-30'),
        ('2022-07-01', '2023-06-30'),
        ('2023-07-01', '2024-06-30'),
        ('2024-07-01', '2025-07-19')  # Current data
    ]
    
    all_data = {}
    
    for i, (start_date, end_date) in enumerate(date_ranges):
        print(f"Fetching data for {start_date} to {end_date}...")
        
        data = fetch_bitcoin_data_for_period(start_date, end_date, api_key)
        
        if data:
            all_data.update(data)
            print(f"  ✓ Got {len(data)} days of data")
        else:
            print(f"  ✗ No data for this period")
        
        # Rate limiting - wait between API calls
        if i < len(date_ranges) - 1:  # Don't wait after the last call
            print("  Waiting 12 seconds for rate limiting...")
            time.sleep(12)
    
    if not all_data:
        print("No data was fetched. Please check your API key and try again.")
        return None
    
    # Convert to DataFrame
    records = []
    for date, values in all_data.items():
        record = {
            'Date': date,
            'Open': float(values['1. open']),
            'High': float(values['2. high']),
            'Low': float(values['3. low']),
            'Close': float(values['4. close']),
            'Volume': float(values['5. volume'])
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (oldest first)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add Price column (using Close price)
    df['Price'] = df['Close']
    
    # Remove duplicates (in case of overlapping date ranges)
    df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    print(f"\nSuccessfully fetched {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    
    return df

def save_complete_bitcoin_data(df, filename=None):
    """
    Save complete Bitcoin data to CSV file
    """
    if filename is None:
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Data Sets/Bitcoin Data/Bitcoin_Complete_Historical_Data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Complete data saved to: {filename}")
    
    return filename

def main():
    """
    Main function to fetch and save complete Bitcoin historical data
    """
    print("="*70)
    print("COMPLETE BITCOIN HISTORICAL DATA FETCHER")
    print("="*70)
    print("This script will make multiple API calls to get all historical data.")
    print("It may take several minutes due to rate limiting.")
    print("="*70)
    
    # Fetch complete data
    df = fetch_complete_bitcoin_history()
    
    if df is not None:
        # Save data
        filename = save_complete_bitcoin_data(df)
        
        # Display summary statistics
        print("\n" + "="*50)
        print("COMPLETE DATA SUMMARY")
        print("="*50)
        print(f"Total days: {len(df):,}")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Days of data: {(df['Date'].max() - df['Date'].min()).days:,}")
        print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
        print(f"All-time high: ${df['Price'].max():,.2f}")
        print(f"All-time low: ${df['Price'].min():,.2f}")
        
        # Check for any missing data
        expected_days = (df['Date'].max() - df['Date'].min()).days + 1
        missing_days = expected_days - len(df)
        if missing_days > 0:
            print(f"Missing days: {missing_days}")
        else:
            print("No missing days detected")
        
        print(f"\nComplete data file ready: {filename}")
        print("You can now use this file for your rainbow chart analysis!")
        
    else:
        print("Failed to fetch complete Bitcoin data. Please check your API key and try again.")

if __name__ == "__main__":
    main() 