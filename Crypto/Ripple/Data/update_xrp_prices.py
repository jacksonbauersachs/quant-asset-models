import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_last_date_from_file(file_path):
    """Get the last date from the existing XRP data file."""
    try:
        df = pd.read_csv(file_path)
        # Convert the M/D/YYYY format to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        last_date = df['Date'].max()
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def fetch_xrp_data_for_period(start_date, end_date, api_key):
    """
    Fetch XRP data for a specific date range using Alpha Vantage API
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'XRP',  # XRP/Ripple symbol
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    try:
        print(f"Fetching XRP data from {start_date} to {end_date}...")
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

def format_volume(volume):
    """Format volume to match XRP CSV format (with K/M suffix, no quotes)"""
    try:
        vol_num = float(volume)
        if vol_num >= 1000000:
            return f"{vol_num/1000000:.2f}M"
        elif vol_num >= 1000:
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

def download_complete_xrp_history(api_key):
    """
    Download complete XRP history from Alpha Vantage API
    """
    print("Downloading complete XRP history...")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'XRP',
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
        
        # Convert to DataFrame with XRP-specific formatting
        records = []
        for date, values in time_series.items():
            open_price = float(values['1. open'])
            close_price = float(values['4. close'])
            
            record = {
                'Date': datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y'),  # Convert to M/D/YYYY format
                'Price': close_price,  # Use close price as Price
                'Open': open_price,
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Vol.': format_volume(values['5. volume']),  # Format volume with K/M suffix
                'Change %': calculate_change_percent(open_price, close_price)  # Calculate percentage change
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if df.empty:
            print("No records to save.")
            return None
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Downloaded {len(df)} days of XRP data")
        print(f"Date range: {df['Date'].min().strftime('%m/%d/%Y')} to {df['Date'].max().strftime('%m/%d/%Y')}")
        print(f"Price range: ${df['Price'].min():.4f} to ${df['Price'].max():.4f}")
        
        # Convert Date back to M/D/YYYY string format for consistency
        df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
        
        return df
        
    except Exception as e:
        print(f"Error downloading XRP data: {e}")
        return None

def update_xrp_prices():
    """
    Update XRP price data from the last available date to today
    If no file exists, download complete history
    """
    # Get API key
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    # Define file paths
    xrp_data_file = "Crypto/Ripple/Data/XRP Historical Data.csv"
    
    # Check if the file exists
    if not os.path.exists(xrp_data_file):
        print(f"XRP data file not found at {xrp_data_file}")
        print("Downloading complete XRP history...")
        
        # Download complete history
        df = download_complete_xrp_history(api_key)
        
        if df is None:
            print("Failed to download XRP data.")
            return
        
        # Save the complete data
        df.to_csv(xrp_data_file, index=False)
        
        print(f"\nSuccessfully downloaded complete XRP history!")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"Data saved to: {xrp_data_file}")
        
        return df
    
    # File exists, check for updates
    print(f"XRP data file found at {xrp_data_file}")
    
    # Get the last date from the existing file
    last_date = get_last_date_from_file(xrp_data_file)
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
    
    print(f"Fetching new data from {start_date} to {today}...")
    
    # Fetch new data
    new_data = fetch_xrp_data_for_period(start_date, today, api_key)
    
    if not new_data:
        print("No new data was fetched.")
        return
    
    # Convert new data to DataFrame with XRP-specific formatting
    records = []
    for date, values in new_data.items():
        open_price = float(values['1. open'])
        close_price = float(values['4. close'])
        
        record = {
            'Date': datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y'),  # Convert to M/D/YYYY format
            'Price': close_price,  # Use close price as Price
            'Open': open_price,
            'High': float(values['2. high']),
            'Low': float(values['3. low']),
            'Vol.': format_volume(values['5. volume']),  # Format volume with K/M suffix
            'Change %': calculate_change_percent(open_price, close_price)  # Calculate percentage change
        }
        records.append(record)
    
    new_df = pd.DataFrame(records)
    
    if new_df.empty:
        print("No new records to add.")
        return
    
    # Sort by date
    new_df['Date'] = pd.to_datetime(new_df['Date'], format='%m/%d/%Y')
    new_df = new_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Fetched {len(new_df)} new days of data")
    print(f"Date range: {new_df['Date'].min().strftime('%m/%d/%Y')} to {new_df['Date'].max().strftime('%m/%d/%Y')}")
    print(f"Price range: ${new_df['Price'].min():.4f} to ${new_df['Price'].max():.4f}")
    
    # Read existing data
    existing_df = pd.read_csv(xrp_data_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%m/%d/%Y')
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Convert Date back to M/D/YYYY string format for consistency with original format
    combined_df['Date'] = combined_df['Date'].dt.strftime('%m/%d/%Y')
    
    # Save updated data
    combined_df.to_csv(xrp_data_file, index=False)
    
    print(f"\nSuccessfully updated XRP data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_df)} new records")
    
    return combined_df

def main():
    """Main function to run the XRP price update."""
    
    print("XRP Price Data Update")
    print("=" * 30)
    
    try:
        updated_df = update_xrp_prices()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")

if __name__ == "__main__":
    main()


