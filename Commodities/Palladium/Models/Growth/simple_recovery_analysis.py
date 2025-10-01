import pandas as pd
import numpy as np

def analyze_palladium_recovery():
    """Simple analysis of palladium recovery times."""
    print("Palladium Recovery Analysis")
    print("=" * 40)
    
    # Load the cleaned data
    df = pd.read_csv("Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Find the value column
    value_col = None
    for col in df.columns:
        if 'value' in col.lower():
            value_col = col
            break
    
    if value_col is None:
        print("Error: Could not find value column")
        return
    
    # Rename for consistency
    df = df.rename(columns={value_col: 'value'})
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Calculate rolling statistics
    window = 252  # 1 year of trading days
    df['rolling_mean'] = df['value'].rolling(window=window, min_periods=30).mean()
    df['rolling_std'] = df['value'].rolling(window=window, min_periods=30).std()
    
    # Calculate z-scores
    df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
    
    # Current situation
    current_price = df['value'].iloc[-1]
    current_mean = df['rolling_mean'].iloc[-1]
    current_std = df['rolling_std'].iloc[-1]
    current_z = df['z_score'].iloc[-1]
    
    print(f"\nCurrent Situation:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Rolling Mean (1-year): ${current_mean:.2f}")
    print(f"Rolling Std Dev: ${current_std:.2f}")
    print(f"Z-Score: {current_z:.2f} ({abs(current_z):.1f} standard deviations {'below' if current_z < 0 else 'above'} mean)")
    
    # Find significant drops (at least 2 standard deviations below mean)
    significant_drops = df[df['z_score'] <= -2.0].copy()
    
    print(f"\nFound {len(significant_drops)} significant drops (2+ std dev below mean)")
    
    if len(significant_drops) == 0:
        print("No significant historical drops found")
        return
    
    # Analyze a few major drops
    print(f"\nMajor Historical Drops:")
    print("-" * 50)
    
    # Get the most severe drops
    major_drops = significant_drops.nsmallest(10, 'z_score')
    
    for i, (idx, row) in enumerate(major_drops.iterrows()):
        drop_date = row['date']
        drop_price = row['value']
        drop_z = row['z_score']
        
        print(f"{i+1}. {drop_date.strftime('%Y-%m-%d')}: ${drop_price:.2f} (Z: {drop_z:.2f})")
        
        # Look for recovery
        future_data = df[df['date'] > drop_date].copy()
        if len(future_data) > 0:
            recovery_mask = future_data['value'] >= future_data['rolling_mean']
            recovery_points = future_data[recovery_mask]
            
            if len(recovery_points) > 0:
                recovery_date = recovery_points.iloc[0]['date']
                recovery_days = (recovery_date - drop_date).days
                recovery_years = recovery_days / 365.25
                print(f"   -> Recovery: {recovery_date.strftime('%Y-%m-%d')} ({recovery_days} days, {recovery_years:.1f} years)")
            else:
                print(f"   -> No recovery found in data")
        print()

def main():
    try:
        analyze_palladium_recovery()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
