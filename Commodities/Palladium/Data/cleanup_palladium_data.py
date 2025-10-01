import pandas as pd
import numpy as np

def clean_palladium_data():
    """Clean the palladium data by removing unrealistic values."""
    print("Cleaning Palladium Data")
    print("=" * 30)
    
    # Load the data
    df = pd.read_csv("Commodities/Palladium/Data/palladium-prices-historical-chart-data.csv")
    
    print(f"Original data: {len(df)} records")
    print(f"Column names: {list(df.columns)}")
    
    # Check if the value column exists and has the right name
    value_col = None
    for col in df.columns:
        if 'value' in col.lower():
            value_col = col
            break
    
    if value_col is None:
        print("Error: Could not find value column")
        return None
    
    print(f"Using column: {value_col}")
    print(f"Original value range: ${df[value_col].min():.2f} to ${df[value_col].max():.2f}")
    
    # Convert date column - handle both M/D/YYYY and MM/DD/YYYY formats
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Identify problematic values based on time period
    # Early palladium prices (1970s-1980s) were legitimately low (50-200 range)
    # But recent prices (2000s+) should be much higher (500+ range)
    
    # Create a more sophisticated filter
    df['year'] = df['date'].dt.year
    
    # Debug: Check what years we have
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"Records from 2024+: {(df['year'] >= 2024).sum()}")
    print(f"Records from 2025: {(df['year'] == 2025).sum()}")
    
    # For 2024+, values below 500 are suspicious (palladium should be 500+)
    # For older years (before 2024), keep all values as they're likely legitimate
    recent_mask = df['year'] >= 2024
    old_mask = df['year'] < 2024
    
    # Find suspicious values in 2024+ data
    recent_suspicious = recent_mask & (df[value_col] < 500)
    
    # No suspicious values in old data (keep everything before 2024)
    old_suspicious = pd.Series([False] * len(df), index=df.index)
    
    # Combine both conditions
    problematic_mask = recent_suspicious | old_suspicious
    
    print(f"\nFound {problematic_mask.sum()} suspicious records:")
    print(f"  2024+ data: {recent_suspicious.sum()} records below $500")
    print(f"  Pre-2024 data: {old_suspicious.sum()} records (keeping all)")
    
    if problematic_mask.sum() > 0:
        print("\nProblematic records:")
        problematic_records = df[problematic_mask]
        for idx, row in problematic_records.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row[value_col]:.2f}")
    
    # Remove problematic records
    df_clean = df[~problematic_mask].copy()
    
    print(f"\nAfter cleaning:")
    print(f"  Records removed: {len(df) - len(df_clean)}")
    print(f"  Remaining records: {len(df_clean)}")
    print(f"  New value range: ${df_clean[value_col].min():.2f} to ${df_clean[value_col].max():.2f}")
    # Handle potential NaT values in dates
    valid_dates = df_clean['date'].dropna()
    if len(valid_dates) > 0:
        print(f"  Date range: {valid_dates.iloc[0].strftime('%Y-%m-%d')} to {valid_dates.iloc[-1].strftime('%Y-%m-%d')}")
    else:
        print("  Date range: Unable to determine (NaT values present)")
    
    # Save cleaned data
    output_file = "Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv"
    df_clean.to_csv(output_file, index=False)
    
    print(f"\nCleaned data saved to: {output_file}")
    
    return df_clean

def main():
    """Main function to clean palladium data."""
    try:
        df_clean = clean_palladium_data()
        print("\nData cleaning completed successfully!")
        print("You can now use the cleaned data for growth model analysis.")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()
