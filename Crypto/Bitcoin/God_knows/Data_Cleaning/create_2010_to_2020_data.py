import pandas as pd
from datetime import datetime

def create_2010_to_2020_data():
    """
    Create a version of the Bitcoin data that goes from the first price (2010-07-18) 
    to October 20, 2020.
    """
    
    # Read the original data
    input_file = '2010_2025_Daily_Data_(BTC).csv'
    output_file = '2010_2020_Daily_Data_(BTC).csv'
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter data from 2010-07-18 to 2020-10-20
    start_date = '2010-07-18'
    end_date = '2020-10-20'
    
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    print(f"Original data: {len(df)} rows")
    print(f"Filtered data: {len(filtered_df)} rows")
    print(f"Date range: {filtered_df['Date'].min()} to {filtered_df['Date'].max()}")
    
    # Save the filtered data
    filtered_df.to_csv(output_file, index=False)
    print(f"Saved filtered data to {output_file}")
    
    # Display some statistics
    print(f"\nPrice statistics:")
    print(f"First price: ${filtered_df['Price'].iloc[0]:.2f}")
    print(f"Last price: ${filtered_df['Price'].iloc[-1]:.2f}")
    print(f"Price range: ${filtered_df['Price'].min():.2f} - ${filtered_df['Price'].max():.2f}")
    
    return filtered_df

if __name__ == "__main__":
    create_2010_to_2020_data() 