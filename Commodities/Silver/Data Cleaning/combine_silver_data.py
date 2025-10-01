import pandas as pd
import os
from datetime import datetime

def combine_silver_data():
    """
    Combine three silver price CSV files in chronological order.
    Files to combine:
    1. Silver Prices 2_4_1970 to 1_12_1990.csv (1970-1990)
    2. Silver Price 1_15_1990 to 11_25_2009.csv (1990-2009)
    3. Silver Price 11_26_2009 to 08_05_2025.csv (2009-2025)
    """
    
    # Define file paths
    data_dir = "Metals/Silver/Data"
    output_dir = "Metals/Silver/Data"
    
    file1 = os.path.join(data_dir, "Silver Prices 2_4_1970 to 1_12_1990.csv")
    file2 = os.path.join(data_dir, "Silver Price 1_15_1990 to 11_25_2009.csv")
    file3 = os.path.join(data_dir, "Silver Price 11_26_2009 to 08_05_2025.csv")
    
    # Check if files exist
    for file_path in [file1, file2, file3]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    print("Reading silver price data files...")
    
    # Read each CSV file
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        
        print(f"File 1: {len(df1)} rows (1970-1990)")
        print(f"File 2: {len(df2)} rows (1990-2009)")
        print(f"File 3: {len(df3)} rows (2009-2025)")
        
        # Combine all dataframes
        combined_df = pd.concat([df1, df2, df3], ignore_index=True)
        
        print(f"Combined dataset: {len(combined_df)} rows")
        
        # Convert Date column to datetime for proper sorting
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        # Sort by date to ensure chronological order
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        # Convert Date back to string format for consistency
        combined_df['Date'] = combined_df['Date'].dt.strftime('%m/%d/%Y')
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"Silver_Price_Complete_{timestamp}.csv")
        
        # Save combined data
        combined_df.to_csv(output_file, index=False)
        
        print(f"Combined data saved to: {output_file}")
        print(f"Date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
        
        # Display some statistics
        print("\nData Summary:")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Check for any missing values
        missing_values = combined_df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values per column:")
            print(missing_values[missing_values > 0])
        
        return output_file
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

if __name__ == "__main__":
    output_file = combine_silver_data()
    if output_file:
        print(f"\nSuccessfully combined silver price data into: {output_file}")
    else:
        print("Failed to combine silver price data.") 