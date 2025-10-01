import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_undervaluation_recovery():
    """Analyze how long it takes palladium to recover from undervaluation relative to fair value."""
    print("Palladium Undervaluation Recovery Analysis")
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
    
    # Calculate days from start for fair value calculation
    start_date = df['date'].iloc[0]
    df['days'] = (df['date'] - start_date).dt.days
    
    # Calculate fair value using exponential model (from our growth analysis)
    # Formula: y = 50.8749 * e^(0.000203 * x)
    df['fair_value'] = 50.8749 * np.exp(0.000203 * df['days'])
    
    # Calculate undervaluation percentage
    df['undervaluation_pct'] = ((df['fair_value'] - df['value']) / df['fair_value']) * 100
    
    # Current situation
    current_price = df['value'].iloc[-1]
    current_fair_value = df['fair_value'].iloc[-1]
    current_undervaluation = df['undervaluation_pct'].iloc[-1]
    
    print(f"\nCurrent Situation:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Fair Value: ${current_fair_value:.2f}")
    print(f"Undervaluation: {current_undervaluation:.1f}%")
    
    if current_undervaluation < 0:
        print(f"Note: Palladium is currently OVERVALUED by {abs(current_undervaluation):.1f}%")
    else:
        print(f"Palladium is currently UNDERVALUED by {current_undervaluation:.1f}%")
    
    # Find periods of significant undervaluation (at least 10% below fair value)
    significant_undervaluation = df[df['undervaluation_pct'] >= 10].copy()
    
    print(f"\nFound {len(significant_undervaluation)} periods of significant undervaluation (10%+ below fair value)")
    
    if len(significant_undervaluation) == 0:
        print("No significant undervaluation periods found")
        return
    
    # Analyze recovery times for different undervaluation levels
    undervaluation_brackets = [
        (10, 20, "10-20% undervalued"),
        (20, 30, "20-30% undervalued"), 
        (30, 40, "30-40% undervalued"),
        (40, 50, "40-50% undervalued"),
        (50, 100, "50%+ undervalued")
    ]
    
    print(f"\nRecovery Analysis by Undervaluation Level:")
    print("=" * 60)
    
    for min_pct, max_pct, label in undervaluation_brackets:
        bracket_data = significant_undervaluation[
            (significant_undervaluation['undervaluation_pct'] >= min_pct) & 
            (significant_undervaluation['undervaluation_pct'] < max_pct)
        ].copy()
        
        if len(bracket_data) == 0:
            continue
            
        print(f"\n{label}:")
        print(f"Found {len(bracket_data)} periods")
        
        recovery_times = []
        
        for idx, row in bracket_data.iterrows():
            drop_date = row['date']
            drop_price = row['value']
            drop_fair_value = row['fair_value']
            drop_undervaluation = row['undervaluation_pct']
            
            # Look for recovery (price back to or above fair value)
            future_data = df[df['date'] > drop_date].copy()
            
            if len(future_data) > 0:
                # Find when price recovers to fair value
                recovery_mask = future_data['value'] >= future_data['fair_value']
                recovery_points = future_data[recovery_mask]
                
                if len(recovery_points) > 0:
                    recovery_date = recovery_points.iloc[0]['date']
                    recovery_price = recovery_points.iloc[0]['value']
                    recovery_days = (recovery_date - drop_date).days
                    recovery_years = recovery_days / 365.25
                    
                    recovery_times.append(recovery_years)
                    
                    print(f"  {drop_date.strftime('%Y-%m-%d')}: {drop_undervaluation:.1f}% undervalued")
                    print(f"    -> Recovery: {recovery_date.strftime('%Y-%m-%d')} ({recovery_days} days, {recovery_years:.1f} years)")
        
        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            median_recovery = np.median(recovery_times)
            min_recovery = np.min(recovery_times)
            max_recovery = np.max(recovery_times)
            
            print(f"  Average recovery time: {avg_recovery:.1f} years")
            print(f"  Median recovery time: {median_recovery:.1f} years")
            print(f"  Range: {min_recovery:.1f} - {max_recovery:.1f} years")
    
    # Find periods similar to current undervaluation level
    if current_undervaluation > 0:
        similar_undervaluation = significant_undervaluation[
            (significant_undervaluation['undervaluation_pct'] >= current_undervaluation - 5) & 
            (significant_undervaluation['undervaluation_pct'] <= current_undervaluation + 5)
        ].copy()
        
        if len(similar_undervaluation) > 0:
            print(f"\nPeriods Similar to Current Undervaluation ({current_undervaluation:.1f}%):")
            print("-" * 50)
            
            recovery_times_similar = []
            
            for idx, row in similar_undervaluation.iterrows():
                drop_date = row['date']
                drop_price = row['value']
                drop_undervaluation = row['undervaluation_pct']
                
                # Look for recovery
                future_data = df[df['date'] > drop_date].copy()
                
                if len(future_data) > 0:
                    recovery_mask = future_data['value'] >= future_data['fair_value']
                    recovery_points = future_data[recovery_mask]
                    
                    if len(recovery_points) > 0:
                        recovery_date = recovery_points.iloc[0]['date']
                        recovery_days = (recovery_date - drop_date).days
                        recovery_years = recovery_days / 365.25
                        
                        recovery_times_similar.append(recovery_years)
                        
                        print(f"  {drop_date.strftime('%Y-%m-%d')}: {drop_undervaluation:.1f}% undervalued")
                        print(f"    -> Recovery: {recovery_date.strftime('%Y-%m-%d')} ({recovery_days} days, {recovery_years:.1f} years)")
            
            if recovery_times_similar:
                avg_recovery_similar = np.mean(recovery_times_similar)
                median_recovery_similar = np.median(recovery_times_similar)
                
                print(f"\n  Average recovery time for similar undervaluation: {avg_recovery_similar:.1f} years")
                print(f"  Median recovery time for similar undervaluation: {median_recovery_similar:.1f} years")
    
    # Save detailed results
    results_file = "Commodities/Palladium/Models/Growth/undervaluation_recovery_analysis.txt"
    with open(results_file, 'w') as f:
        f.write("Palladium Undervaluation Recovery Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Current Situation:\n")
        f.write(f"Date: {df['date'].iloc[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Current Price: ${current_price:.2f}\n")
        f.write(f"Fair Value: ${current_fair_value:.2f}\n")
        f.write(f"Undervaluation: {current_undervaluation:.1f}%\n\n")
        
        f.write("Recovery Time Analysis by Undervaluation Level:\n")
        for min_pct, max_pct, label in undervaluation_brackets:
            bracket_data = significant_undervaluation[
                (significant_undervaluation['undervaluation_pct'] >= min_pct) & 
                (significant_undervaluation['undervaluation_pct'] < max_pct)
            ].copy()
            
            if len(bracket_data) > 0:
                f.write(f"\n{label}: {len(bracket_data)} periods\n")
    
    print(f"\nDetailed results saved: {results_file}")

def main():
    try:
        analyze_undervaluation_recovery()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
