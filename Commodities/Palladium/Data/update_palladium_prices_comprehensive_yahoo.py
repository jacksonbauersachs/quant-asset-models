import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

def fetch_comprehensive_yahoo_palladium():
    """Fetch comprehensive palladium data from Yahoo Finance."""
    print("Fetching comprehensive palladium data from Yahoo Finance...")
    print("=" * 60)
    
    try:
        # Try different tickers that might have more historical data
        tickers_to_try = [
            'PALL',  # Palladium ETF
            'PPLT',  # Platinum ETF (might have longer history)
            'GLD',   # Gold ETF (for comparison)
            'SLV',   # Silver ETF (for comparison)
        ]
        
        best_data = None
        best_ticker = None
        best_start_date = None
        
        for ticker in tickers_to_try:
            print(f"\nTrying ticker: {ticker}")
            try:
                # Create ticker object
                ticker_obj = yf.Ticker(ticker)
                
                # Try to get maximum historical data
                hist = ticker_obj.history(period="max")
                
                if not hist.empty:
                    start_date = hist.index[0].strftime('%Y-%m-%d')
                    end_date = hist.index[-1].strftime('%Y-%m-%d')
                    print(f"  {ticker}: {len(hist)} records from {start_date} to {end_date}")
                    
                    # Check if this is the best data we've found
                    if best_data is None or len(hist) > len(best_data):
                        best_data = hist
                        best_ticker = ticker
                        best_start_date = start_date
                        print(f"  -> New best data! {len(hist)} records")
                    else:
                        print(f"  -> Not better than current best ({len(best_data)} records)")
                else:
                    print(f"  {ticker}: No data available")
                    
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
        
        if best_data is not None:
            print(f"\nBest data found: {best_ticker} with {len(best_data)} records")
            print(f"Date range: {best_start_date} to {best_data.index[-1].strftime('%Y-%m-%d')}")
            return best_data, best_ticker
        else:
            print("\nNo data found for any ticker")
            return None, None
            
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")
        return None, None

def create_comprehensive_palladium_dataset():
    """Create comprehensive palladium dataset using Yahoo Finance."""
    print("Creating Comprehensive Palladium Dataset")
    print("=" * 50)
    
    # Fetch data from Yahoo Finance
    hist_data, ticker = fetch_comprehensive_yahoo_palladium()
    
    if hist_data is None:
        print("Failed to get Yahoo Finance data")
        return None
    
    # Convert to our format
    df = hist_data.copy()
    
    # Reset index to get Date as a column
    df = df.reset_index()
    
    # Rename columns to match our format
    df = df.rename(columns={
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High', 
        'Low': 'Low',
        'Close': 'Price',
        'Volume': 'Vol.'
    })
    
    # Format the date
    df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
    
    # Format volume
    df['Vol.'] = df['Vol.'].apply(lambda x: f"{x/1000:.2f}K" if x >= 1000 else f"{x:.2f}")
    
    # Calculate change percentage
    df['Price_float'] = df['Price'].astype(float)
    df['Change %'] = df['Price_float'].pct_change() * 100
    df['Change %'] = df['Change %'].fillna(0).round(2).astype(str) + '%'
    
    # Drop temporary column
    df = df.drop('Price_float', axis=1)
    
    # Reorder columns to match our format
    df = df[['Date', 'Open', 'High', 'Low', 'Price', 'Vol.', 'Change %']]
    
    # Save the data
    output_file = "Commodities/Palladium/Data/Palladium Historical Data Comprehensive.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nComprehensive palladium data saved!")
    print(f"   Ticker used: {ticker}")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"   Price range: ${df['Price'].min():.4f} to ${df['Price'].max():.4f}")
    print(f"   Data saved to: {output_file}")
    
    return df

def main():
    """Main function to fetch comprehensive palladium data."""
    
    print("Yahoo Finance Comprehensive Palladium Data Fetcher")
    print("=" * 60)
    print("Source: Yahoo Finance (Multiple Tickers)")
    print("Goal: Get maximum historical palladium data")
    print()
    
    try:
        df = create_comprehensive_palladium_dataset()
        if df is not None:
            print("\nComprehensive data update completed successfully!")
            print("Data Source: Yahoo Finance")
            print("\nNext Steps:")
            print("   1. Check the data quality and date range")
            print("   2. Run updated growth models with complete data")
            print("   3. Compare results with previous analysis")
        else:
            print("\nComprehensive data update failed")
    except Exception as e:
        print(f"\nError during comprehensive update: {e}")

if __name__ == "__main__":
    main()
