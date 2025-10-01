import pandas as pd
import os
from datetime import datetime

def combine_gold_data():
    """
    Combine three gold price CSV files in chronological order.
    Files to combine:
    1. Gold Price 1_3_75 to 11_7_94.csv (1975-1994)
    2. Gold Price 11_8_94 to 10_3_2014.csv (1994-2014)
    3. Gold Price 10_4_2014 to 8_6_2025.csv (2014-2025)
    """
    
    # Define file paths
    data_dir = "Gold/Data"
    output_dir = "Gold/Data"
    
    file1 = os.path.join(data_dir, "Gold Price 1_3_75 to 11_7_94.csv")
    file2 = os.path.join(data_dir, "Gold Price 11_8_94 to 10_3_2014.csv")
    file3 = os.path.join(data_dir, "Gold Price 10_4_2014 to 8_6_2025.csv")
    
    # Check if files exist
    for file_path in [file1, file2, file3]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    print("Reading gold price data files...")
    
    # Read each CSV file
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        
        print(f"File 1: {len(df1)} rows (1975-1994)")
        print(f"File 2: {len(df2)} rows (1994-2014)")
        print(f"File 3: {len(df3)} rows (2014-2025)")
        
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
        output_file = os.path.join(output_dir, f"Gold_Price_Complete_{timestamp}.csv")
        
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
    output_file = combine_gold_data()
    if output_file:
        print(f"\nSuccessfully combined gold price data into: {output_file}")
    else:
        print("Failed to combine gold price data.") 