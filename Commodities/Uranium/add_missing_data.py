"""
Script to add missing Uranium spot price data.
You can manually enter the data or provide it in a CSV file.
"""

import pandas as pd
import os
from datetime import datetime

# File paths
data_file = "Uranium_historical_data.csv"

def add_manual_data():
    """Add data manually entered by user."""
    print("=" * 60)
    print("Add Missing Uranium Spot Price Data")
    print("=" * 60)
    print("\nYour current data ends at: 2025-06-01")
    print("You need data from: 2025-07-01 to 2026-01-01")
    print("\nIMPORTANT: You need U3O8 SPOT prices, NOT SPUT ETF prices!")
    print("\nSources for actual Uranium Spot Prices (U3O8):")
    print("  1. Cameco Monthly Reports (look for 'Spot Price' section):")
    print("     https://www.cameco.com/invest/markets/uranium-price")
    print("     - They publish monthly spot prices in their investor reports")
    print("  2. UxC (Uranium Exchange Corporation) - usually requires subscription")
    print("  3. Nuclear industry reports/newsletters")
    print("\nNOTE: Many free sources show SPUT ETF prices, not actual spot prices.")
    print("      Make sure you're getting U3O8 spot price, not ETF prices.")
    print("\n" + "=" * 60)
    
    # Read existing data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max()
    print(f"\nLast date in file: {last_date.strftime('%Y-%m-%d')}")
    
    # Get new data from user
    new_data = []
    months = [
        ("2025-07-01", "July 2025"),
        ("2025-08-01", "August 2025"),
        ("2025-09-01", "September 2025"),
        ("2025-10-01", "October 2025"),
        ("2025-11-01", "November 2025"),
        ("2025-12-01", "December 2025"),
        ("2026-01-01", "January 2026"),
    ]
    
    print("\nEnter spot prices for each month (press Enter to skip a month):")
    print("-" * 60)
    
    for date_str, month_name in months:
        # Check if date is after last date
        date_obj = pd.Timestamp(date_str)
        if date_obj <= last_date:
            continue
            
        while True:
            try:
                price_input = input(f"{month_name} ({date_str}): $").strip()
                if not price_input:
                    print(f"  Skipped {month_name}")
                    break
                price = float(price_input.replace('$', '').replace(',', ''))
                new_data.append({'Date': date_str, 'Price': price})
                print(f"  ✓ Added {month_name}: ${price:.2f}")
                break
            except ValueError:
                print("  Invalid price. Enter a number (e.g., 75.50)")
            except KeyboardInterrupt:
                print("\n\nCancelled.")
                return None
    
    if not new_data:
        print("\nNo new data added.")
        return None
    
    # Create DataFrame from new data
    new_df = pd.DataFrame(new_data)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    
    # Combine with existing data
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Remove duplicates and sort
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Format dates back to YYYY-MM-DD
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Save
    combined_df.to_csv(data_file, index=False)
    
    print(f"\n✓ Successfully added {len(new_data)} months of data!")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    
    return combined_df

def add_from_csv(csv_file):
    """Add data from a CSV file with Date,Price columns."""
    try:
        new_df = pd.read_csv(csv_file)
        
        # Ensure Date and Price columns exist
        if 'Date' not in new_df.columns or 'Price' not in new_df.columns:
            print("Error: CSV must have 'Date' and 'Price' columns")
            return None
        
        # Convert dates
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Read existing data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter to only dates after last date in existing file
        last_date = df['Date'].max()
        new_df = new_df[new_df['Date'] > last_date]
        
        if new_df.empty:
            print("No new data found (all dates already exist or are before last date)")
            return None
        
        # Combine
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save
        combined_df.to_csv(data_file, index=False)
        
        print(f"✓ Added {len(new_df)} records from {csv_file}")
        return combined_df
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Enter data manually")
    print("2. Import from CSV file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        add_manual_data()
    elif choice == "2":
        csv_file = input("Enter path to CSV file: ").strip().strip('"')
        if os.path.exists(csv_file):
            add_from_csv(csv_file)
        else:
            print(f"File not found: {csv_file}")
    else:
        print("Invalid choice")

