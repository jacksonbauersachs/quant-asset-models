import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import yfinance as yf

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

def fetch_complete_xrp_history_yahoo():
    """
    Download complete XRP history from Yahoo Finance (completely free, no API key required)
    """
    print("Downloading complete XRP history from Yahoo Finance...")
    
    try:
        # XRP-USD ticker symbol for Yahoo Finance
        ticker = "XRP-USD"
        print(f"Fetching data for ticker: {ticker}")
        
        # Create yfinance object
        xrp = yf.Ticker(ticker)
        
        # Get historical data (max available)
        print("Downloading historical data...")
        hist = xrp.history(period="max")
        
        if hist.empty:
            print("❌ No data received from Yahoo Finance")
            return None
        
        print(f"✅ Received {len(hist)} data points from Yahoo Finance")
        print(f"Date range: {hist.index.min().strftime('%Y-%m-%d')} to {hist.index.max().strftime('%Y-%m-%d')}")
        
        # Convert to our format
        records = []
        for date, row in hist.iterrows():
            # Calculate daily change percentage
            open_price = row['Open']
            close_price = row['Close']
            change_pct = ((close_price - open_price) / open_price) * 100
            
            record = {
                'Date': date.strftime('%m/%d/%Y'),
                'Price': round(close_price, 4),
                'Open': round(open_price, 4),
                'High': round(row['High'], 4),
                'Low': round(row['Low'], 4),
                'Vol.': format_volume(row['Volume']),
                'Change %': f"{change_pct:.2f}%"
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if df.empty:
            print("No records to save.")
            return None
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Processed {len(df)} days of XRP data")
        print(f"Date range: {df['Date'].min().strftime('%m/%d/%Y')} to {df['Date'].max().strftime('%m/%d/%Y')}")
        print(f"Price range: ${df['Price'].min():.4f} to ${df['Price'].max():.4f}")
        
        # Convert Date back to M/D/YYYY string format for consistency
        df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
        
        return df
        
    except Exception as e:
        print(f"Error downloading XRP data from Yahoo Finance: {e}")
        return None

def fetch_recent_xrp_data_yahoo(days=30):
    """
    Fetch recent XRP data from Yahoo Finance for updates
    """
    print(f"Fetching last {days} days of XRP data from Yahoo Finance...")
    
    try:
        ticker = "XRP-USD"
        xrp = yf.Ticker(ticker)
        
        # Get recent data
        hist = xrp.history(period=f"{days}d")
        
        if hist.empty:
            print("No recent data received")
            return None
        
        records = []
        for date, row in hist.iterrows():
            open_price = row['Open']
            close_price = row['Close']
            change_pct = ((close_price - open_price) / open_price) * 100
            
            record = {
                'Date': date.strftime('%m/%d/%Y'),
                'Price': round(close_price, 4),
                'Open': round(open_price, 4),
                'High': round(row['High'], 4),
                'Low': round(row['Low'], 4),
                'Vol.': format_volume(row['Volume']),
                'Change %': f"{change_pct:.2f}%"
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
        
        return df
        
    except Exception as e:
        print(f"Error fetching recent XRP data: {e}")
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

def update_xrp_prices_yahoo():
    """
    Update XRP price data using Yahoo Finance API
    If no file exists, download complete history
    """
    # Define file paths
    xrp_data_file = "Crypto/Ripple/Data/XRP Historical Data.csv"
    
    # Check if the file exists
    if not os.path.exists(xrp_data_file):
        print(f"XRP data file not found at {xrp_data_file}")
        print("Downloading complete XRP history from Yahoo Finance...")
        
        # Download complete history
        df = fetch_complete_xrp_history_yahoo()
        
        if df is None:
            print("Failed to download XRP data from Yahoo Finance.")
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
    
    # Calculate days since last update
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    days_since_update = (datetime.now() - last_date_dt).days
    
    print(f"Days since last update: {days_since_update}")
    
    # Fetch recent data (last 30 days to be safe)
    recent_data = fetch_recent_xrp_data_yahoo(days=30)
    
    if recent_data is None:
        print("No recent data was fetched.")
        return
    
    # Filter to only new data
    recent_data['Date'] = pd.to_datetime(recent_data['Date'], format='%m/%d/%Y')
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    new_data = recent_data[recent_data['Date'] > last_date_dt]
    
    if new_data.empty:
        print("No new data to add.")
        return
    
    # Convert back to string format
    new_data['Date'] = new_data['Date'].dt.strftime('%m/%d/%Y')
    
    print(f"Found {len(new_data)} new days of data")
    print(f"Date range: {new_data['Date'].iloc[0]} to {new_data['Date'].iloc[-1]}")
    print(f"Price range: ${new_data['Price'].min():.4f} to ${new_data['Price'].max():.4f}")
    
    # Read existing data
    existing_df = pd.read_csv(xrp_data_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%m/%d/%Y')
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Convert Date back to M/D/YYYY string format for consistency
    combined_df['Date'] = combined_df['Date'].dt.strftime('%m/%d/%Y')
    
    # Save updated data
    combined_df.to_csv(xrp_data_file, index=False)
    
    print(f"\nSuccessfully updated XRP data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_data)} new records")
    
    return combined_df

def main():
    """Main function to run the XRP price update using Yahoo Finance."""
    
    print("XRP Price Data Update (Yahoo Finance)")
    print("=" * 45)
    
    try:
        updated_df = update_xrp_prices_yahoo()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
            print("📊 Yahoo Finance Benefits:")
            print("   - Completely free (no API key required)")
            print("   - No rate limits for historical data")
            print("   - Complete historical data back to 2013")
            print("   - Reliable and stable data source")
            print("   - Professional-grade financial data")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")

if __name__ == "__main__":
    main()
