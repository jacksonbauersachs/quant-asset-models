import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def analyze_palladium_recovery_times():
    """Analyze historical recovery times for palladium price drops."""
    print("Palladium Recovery Time Analysis")
    print("=" * 50)
    
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
    print(f"Price range: ${df['value'].min():.2f} to ${df['value'].max():.2f}")
    
    # Calculate rolling statistics
    window = 252  # 1 year of trading days
    df['rolling_mean'] = df['value'].rolling(window=window, min_periods=30).mean()
    df['rolling_std'] = df['value'].rolling(window=window, min_periods=30).std()
    
    # Calculate z-scores (how many standard deviations from mean)
    df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
    
    # Find current situation
    current_price = df['value'].iloc[-1]
    current_mean = df['rolling_mean'].iloc[-1]
    current_std = df['rolling_std'].iloc[-1]
    current_z = df['z_score'].iloc[-1]
    
    print(f"\nCurrent Situation (as of {df['date'].iloc[-1].strftime('%Y-%m-%d')}):")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Rolling Mean (1-year): ${current_mean:.2f}")
    print(f"Rolling Std Dev: ${current_std:.2f}")
    print(f"Z-Score: {current_z:.2f} ({abs(current_z):.1f} standard deviations {'below' if current_z < 0 else 'above'} mean)")
    
    # Find significant drops (at least 1.5 standard deviations below mean)
    # and filter for meaningful drops (not just daily noise)
    significant_drops = df[df['z_score'] <= -1.5].copy()
    
    # Remove consecutive drops to avoid double-counting
    significant_drops = significant_drops[~significant_drops.index.isin(significant_drops.index - 1)]
    
    # Focus on drops similar to or more severe than current
    current_severity = abs(current_z) if current_z < 0 else 0
    severe_drops = significant_drops[significant_drops['z_score'] <= current_z].copy()
    
    print(f"\nFound {len(severe_drops)} periods with drops of similar or greater severity")
    
    if len(severe_drops) == 0:
        print("No historical drops of similar magnitude found")
        return
    
    # Analyze recovery times
    recovery_analysis = []
    
    for idx, row in severe_drops.iterrows():
        drop_date = row['date']
        drop_price = row['value']
        drop_z = row['z_score']
        
        # Look for recovery (price back to or above the rolling mean)
        future_data = df[df['date'] > drop_date].copy()
        
        if len(future_data) > 0:
            # Find when price recovers to rolling mean
            recovery_mask = future_data['value'] >= future_data['rolling_mean']
            recovery_points = future_data[recovery_mask]
            
            if len(recovery_points) > 0:
                recovery_date = recovery_points.iloc[0]['date']
                recovery_price = recovery_points.iloc[0]['value']
                recovery_days = (recovery_date - drop_date).days
                recovery_years = recovery_days / 365.25
                
                # Only include recoveries that took at least 30 days (filter out noise)
                if recovery_days >= 30:
                    recovery_analysis.append({
                        'drop_date': drop_date,
                        'drop_price': drop_price,
                        'drop_z_score': drop_z,
                        'recovery_date': recovery_date,
                        'recovery_price': recovery_price,
                        'recovery_days': recovery_days,
                        'recovery_years': recovery_years
                    })
    
    if len(recovery_analysis) == 0:
        print("No complete recoveries found in historical data")
        return
    
    # Convert to DataFrame for analysis
    recovery_df = pd.DataFrame(recovery_analysis)
    
    print(f"\nRecovery Analysis ({len(recovery_df)} complete recoveries found):")
    print("=" * 60)
    
    for i, recovery in recovery_df.iterrows():
        print(f"Drop: {recovery['drop_date'].strftime('%Y-%m-%d')} (Z: {recovery['drop_z_score']:.2f})")
        print(f"  -> Recovery: {recovery['recovery_date'].strftime('%Y-%m-%d')} ({recovery['recovery_days']:.0f} days, {recovery['recovery_years']:.1f} years)")
        print()
    
    # Statistical summary
    recovery_times = recovery_df['recovery_years']
    
    print("Recovery Time Statistics:")
    print(f"  Mean: {recovery_times.mean():.1f} years")
    print(f"  Median: {recovery_times.median():.1f} years")
    print(f"  Min: {recovery_times.min():.1f} years")
    print(f"  Max: {recovery_times.max():.1f} years")
    print(f"  Std Dev: {recovery_times.std():.1f} years")
    
    # Percentiles
    print(f"\nRecovery Time Percentiles:")
    print(f"  25th percentile: {recovery_times.quantile(0.25):.1f} years")
    print(f"  75th percentile: {recovery_times.quantile(0.75):.1f} years")
    print(f"  90th percentile: {recovery_times.quantile(0.90):.1f} years")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Recovery times histogram
    plt.subplot(2, 2, 1)
    plt.hist(recovery_times, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Recovery Time (Years)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Recovery Times')
    plt.axvline(recovery_times.mean(), color='red', linestyle='--', label=f'Mean: {recovery_times.mean():.1f}y')
    plt.axvline(recovery_times.median(), color='green', linestyle='--', label=f'Median: {recovery_times.median():.1f}y')
    plt.legend()
    
    # Plot 2: Recovery times over time
    plt.subplot(2, 2, 2)
    plt.scatter(recovery_df['drop_date'], recovery_times, alpha=0.7, color='blue')
    plt.xlabel('Drop Date')
    plt.ylabel('Recovery Time (Years)')
    plt.title('Recovery Times Over Time')
    plt.xticks(rotation=45)
    
    # Plot 3: Z-score vs recovery time
    plt.subplot(2, 2, 3)
    plt.scatter(recovery_df['drop_z_score'], recovery_times, alpha=0.7, color='red')
    plt.xlabel('Drop Z-Score (Severity)')
    plt.ylabel('Recovery Time (Years)')
    plt.title('Drop Severity vs Recovery Time')
    plt.axvline(current_z, color='black', linestyle='--', label=f'Current: {current_z:.2f}')
    plt.legend()
    
    # Plot 4: Price recovery examples
    plt.subplot(2, 2, 4)
    for i, recovery in recovery_df.head(3).iterrows():
        # Get data around the drop and recovery
        start_date = recovery['drop_date'] - timedelta(days=365)
        end_date = recovery['recovery_date'] + timedelta(days=365)
        
        period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if len(period_data) > 0:
            plt.plot(period_data['date'], period_data['value'], alpha=0.7, 
                    label=f"{recovery['drop_date'].strftime('%Y')} (Z: {recovery['drop_z_score']:.1f})")
    
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('Example Recovery Patterns')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the analysis chart
    chart_path = "Commodities/Palladium/Models/Growth/palladium_recovery_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\nRecovery analysis chart saved: {chart_path}")
    
    plt.show()
    
    # Save detailed results
    results_file = "Commodities/Palladium/Models/Growth/palladium_recovery_analysis.txt"
    with open(results_file, 'w') as f:
        f.write("Palladium Recovery Time Analysis\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Current Situation:\n")
        f.write(f"Date: {df['date'].iloc[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Current Price: ${current_price:.2f}\n")
        f.write(f"Rolling Mean: ${current_mean:.2f}\n")
        f.write(f"Rolling Std Dev: ${current_std:.2f}\n")
        f.write(f"Z-Score: {current_z:.2f}\n\n")
        
        f.write(f"Recovery Statistics:\n")
        f.write(f"Mean Recovery Time: {recovery_times.mean():.1f} years\n")
        f.write(f"Median Recovery Time: {recovery_times.median():.1f} years\n")
        f.write(f"Min Recovery Time: {recovery_times.min():.1f} years\n")
        f.write(f"Max Recovery Time: {recovery_times.max():.1f} years\n")
        f.write(f"Std Dev: {recovery_times.std():.1f} years\n\n")
        
        f.write("Historical Recovery Examples:\n")
        for i, recovery in recovery_df.iterrows():
            f.write(f"{recovery['drop_date'].strftime('%Y-%m-%d')} → {recovery['recovery_date'].strftime('%Y-%m-%d')} ({recovery['recovery_years']:.1f} years)\n")
    
    print(f"Detailed results saved: {results_file}")

def main():
    """Main function to analyze palladium recovery times."""
    try:
        analyze_palladium_recovery_times()
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
