"""
Check Monte Carlo Data
=====================

Simple script to check the actual Monte Carlo simulation data files.
"""

import pandas as pd
import numpy as np

def check_file(filename):
    """Check a Monte Carlo data file"""
    print(f"\nChecking: {filename}")
    print("="*50)
    
    try:
        df = pd.read_csv(filename)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns[:5])}...")  # First 5 columns
        
        # Check first row (starting prices)
        first_row = df.iloc[0, :].values
        print(f"First row (starting prices):")
        print(f"  Min: ${first_row.min():,.2f}")
        print(f"  Max: ${first_row.max():,.2f}")
        print(f"  Mean: ${first_row.mean():,.2f}")
        print(f"  Std: ${first_row.std():,.2f}")
        
        # Check last row (final prices)
        last_row = df.iloc[-1, :].values
        print(f"Last row (final prices):")
        print(f"  Min: ${last_row.min():,.2f}")
        print(f"  Max: ${last_row.max():,.2f}")
        print(f"  Mean: ${last_row.mean():,.2f}")
        print(f"  Std: ${last_row.std():,.2f}")
        
        # Check for any zero or negative values
        zero_count = np.sum(last_row <= 0)
        if zero_count > 0:
            print(f"⚠️  Found {zero_count} zero or negative final prices")
        else:
            print("✅ All final prices are positive")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def main():
    """Main function"""
    print("MONTE CARLO DATA CHECK")
    print("="*50)
    
    files_to_check = [
        'Results/Bitcoin/bitcoin_monte_carlo_simple_paths_20250720.csv',
        'Results/Bitcoin/bitcoin_monte_carlo_fixed_paths_20250720.csv'
    ]
    
    for filename in files_to_check:
        check_file(filename)

if __name__ == "__main__":
    main() 