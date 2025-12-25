import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_latest_bitcoin_data():
    """
    Fetch the latest Bitcoin data from Alpha Vantage
    """
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
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
        
        # Convert to DataFrame with correct headers
        records = []
        for date, values in time_series.items():
            record = {
                'Date': date,
                'Price': float(values['4. close']),  # Use Close as Price
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Vol.': float(values['5. volume'])  # Use Vol. to match existing format
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching latest data: {e}")
        return None

def load_cleaned_data():
    """
    Load the cleaned comprehensive Bitcoin data
    """
    try:
        # Load the cleaned data file
        df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Cleaned_Complete_Data_20250719.csv')
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

def combine_and_update_data(existing_df, latest_df):
    """
    Combine existing data with latest data, updating overlapping dates
    """
    if existing_df is None or latest_df is None:
        return None
    
    # Find the latest date in existing data
    existing_latest_date = existing_df['Date'].max()
    print(f"Existing data ends at: {existing_latest_date.strftime('%Y-%m-%d')}")
    
    # Find the latest date in new data
    latest_latest_date = latest_df['Date'].max()
    print(f"Latest API data ends at: {latest_latest_date.strftime('%Y-%m-%d')}")
    
    # Filter new data to only include dates after the existing data ends
    new_data = latest_df[latest_df['Date'] > existing_latest_date].copy()
    
    if len(new_data) == 0:
        print("No new data to add - existing data is already up to date!")
        return existing_df
    
    print(f"Adding {len(new_data)} new days of data")
    
    # Combine the data
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Remove any duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    return combined_df

def save_final_data(df, filename=None):
    """
    Save the final combined Bitcoin data
    """
    if filename is None:
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Final data saved to: {filename}")
    
    return filename

def main():
    """
    Main function to create the final complete Bitcoin dataset
    """
    print("="*60)
    print("FINAL BITCOIN DATA COMBINER")
    print("="*60)
    print("This script will combine your cleaned Bitcoin data")
    print("with the latest Alpha Vantage data for the final dataset.")
    print("="*60)
    
    # Load cleaned data
    print("Loading cleaned comprehensive Bitcoin data...")
    existing_df = load_cleaned_data()
    
    if existing_df is None:
        print("Failed to load cleaned data.")
        return
    
    print(f"✓ Loaded {len(existing_df):,} days of cleaned data")
    print(f"  Date range: {existing_df['Date'].min().strftime('%Y-%m-%d')} to {existing_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {list(existing_df.columns)}")
    
    # Fetch latest data
    print("\nFetching latest Bitcoin data from Alpha Vantage...")
    latest_df = fetch_latest_bitcoin_data()
    
    if latest_df is None:
        print("Failed to fetch latest data. Using cleaned data as final dataset.")
        filename = save_final_data(existing_df)
        print(f"Final dataset saved: {filename}")
        return
    
    print(f"✓ Fetched {len(latest_df):,} days of latest data")
    print(f"  Date range: {latest_df['Date'].min().strftime('%Y-%m-%d')} to {latest_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {list(latest_df.columns)}")
    
    # Combine and update data
    print("\nCombining and updating data...")
    final_df = combine_and_update_data(existing_df, latest_df)
    
    if final_df is None:
        print("Failed to combine data.")
        return
    
    # Save final data
    filename = save_final_data(final_df)
    
    # Display summary
    print("\n" + "="*50)
    print("FINAL DATASET SUMMARY")
    print("="*50)
    print(f"Total days: {len(final_df):,}")
    print(f"Date range: {final_df['Date'].min().strftime('%Y-%m-%d')} to {final_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Days of data: {(final_df['Date'].max() - final_df['Date'].min()).days:,}")
    print(f"Current price: ${final_df['Price'].iloc[-1]:,.2f}")
    print(f"All-time high: ${final_df['Price'].max():,.2f}")
    print(f"All-time low: ${final_df['Price'].min():,.2f}")
    
    # Check for any missing data
    expected_days = (final_df['Date'].max() - final_df['Date'].min()).days + 1
    missing_days = expected_days - len(final_df)
    if missing_days > 0:
        print(f"Missing days: {missing_days}")
    else:
        print("No missing days detected")
    
    print(f"\n🎉 FINAL COMPLETE DATASET READY! 🎉")
    print(f"File: {filename}")
    print("Column structure: Date, Price, Open, High, Low, Vol.")
    print("This is your complete Bitcoin dataset from 2010 to present!")
    print("You can now use this for your rainbow chart analysis!")

if __name__ == "__main__":
    main() 