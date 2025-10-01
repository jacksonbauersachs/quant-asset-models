import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.optimize import curve_fit

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['date'].min()
        df['days'] = (df['date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def identify_lows(df, window=90, z_threshold=-1.0, percentile_threshold=0.25, rolling_tolerance=1.15):
    """Identify significant lows in the data using multiple methods."""
    
    # Method 1: Rolling minimum approach (more aggressive)
    df['rolling_min'] = df[' value'].rolling(window=window, center=True).min()
    df['is_low_rolling'] = df[' value'] <= df['rolling_min'] * rolling_tolerance  # Within 15% of rolling minimum
    
    # Method 2: Z-score approach (more aggressive)
    df['rolling_mean'] = df[' value'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df[' value'].rolling(window=window, center=True).std()
    df['z_score'] = (df[' value'] - df['rolling_mean']) / df['rolling_std']
    df['is_low_zscore'] = df['z_score'] < z_threshold  # More than 1.0 standard deviations below mean
    
    # Method 3: Percentile approach (more aggressive)
    df['percentile'] = df[' value'].rolling(window=window*2, center=True).rank(pct=True)
    df['is_low_percentile'] = df['percentile'] < percentile_threshold  # Bottom 25% of values
    
    # Method 4: Local minima detection (new method)
    df['is_local_min'] = False
    for i in range(window, len(df) - window):
        current_val = df[' value'].iloc[i]
        left_window = df[' value'].iloc[i-window:i].min()
        right_window = df[' value'].iloc[i+1:i+window+1].min()
        if current_val <= left_window and current_val <= right_window:
            df.loc[df.index[i], 'is_local_min'] = True
    
    # Combine methods (including local minima)
    df['is_low'] = df['is_low_rolling'] | df['is_low_zscore'] | df['is_low_percentile'] | df['is_local_min']
    
    # Get lows data
    lows = df[df['is_low']].copy()
    
    # Remove consecutive lows (keep only the lowest of each cluster) - more aggressive
    lows = lows.sort_values('date')
    lows['days_diff'] = lows['days'].diff()
    lows = lows[(lows['days_diff'] > 15) | (lows['days_diff'].isna())]  # At least 15 days apart (was 30)
    
    print(f"\nAggressive Low Points Analysis:")
    print(f"  Found {len(lows)} significant lows")
    print(f"  Low price range: ${lows[' value'].min():.2f} to ${lows[' value'].max():.2f}")
    print(f"  Date range: {lows['date'].min().strftime('%Y-%m-%d')} to {lows['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Parameters used:")
    print(f"    - Window: {window} days")
    print(f"    - Z-score threshold: {z_threshold}")
    print(f"    - Percentile threshold: {percentile_threshold*100}%")
    print(f"    - Rolling tolerance: {rolling_tolerance}")
    
    return lows

def fit_lows_model(lows):
    """Fit an exponential model to the lows data."""
    
    x = lows['days'].values
    y = lows[' value'].values
    
    # Exponential model
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        # Try multiple initial guesses
        initial_guesses = [
            [10, 0.0001, 50],
            [20, 0.0002, 30],
            [5, 0.00005, 100],
            [50, 0.0003, 0]
        ]
        
        best_model = None
        best_r2 = -np.inf
        
        for p0 in initial_guesses:
            try:
                popt, _ = curve_fit(exponential_func, x, y, p0=p0, maxfev=10000)
                a, b, c = popt
                
                # Calculate R²
                y_pred = exponential_func(x, a, b, c)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = {
                        'name': 'Lows Model',
                        'params': popt,
                        'r2': r2,
                        'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}',
                        'growth_rate': b
                    }
            except:
                continue
        
        return best_model
        
    except Exception as e:
        print(f"Error fitting lows model: {e}")
        return None

def create_lows_chart(df, lows, lows_model, output_dir):
    """Create a chart showing the lows model."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.3, s=1, color='lightblue', label='All Data')
    ax1.scatter(lows['years'], lows[' value'], alpha=0.8, s=30, color='red', label='Significant Lows')
    
    if lows_model:
        # Plot the lows model
        x_all = df['days'].values
        y_lows_model = lows_model['params'][0] * np.exp(lows_model['params'][1] * x_all) + lows_model['params'][2]
        ax1.plot(df['years'], y_lows_model, 'r-', linewidth=3, label=f'Lows Model (R² = {lows_model["r2"]:.3f})')
        
        # Highlight the model as a "floor"
        ax1.fill_between(df['years'], y_lows_model, y_lows_model * 0.1, alpha=0.1, color='red', label='Historical Floor')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Lows Model - Historical Floor Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.3, s=1, color='lightblue', label='All Data')
    ax2.scatter(lows['years'], lows[' value'], alpha=0.8, s=30, color='red', label='Significant Lows')
    
    if lows_model:
        ax2.plot(df['years'], y_lows_model, 'r-', linewidth=3, label=f'Lows Model')
        ax2.fill_between(df['years'], y_lows_model, y_lows_model * 0.1, alpha=0.1, color='red', label='Historical Floor')
    
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Lows Model - Log Scale (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    if lows_model:
        formula_text = f"""Lows Model Formula:
{lows_model['formula']}
R² = {lows_model['r2']:.4f}
Growth Rate = {lows_model['growth_rate']:.6f}

This model represents the historical floor
that palladium has maintained over time."""
        
        ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_lows_model.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_lows_analysis(df, lows_model):
    """Calculate analysis for the lows model."""
    
    if not lows_model:
        return None
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    # Calculate model floor value
    floor_value = lows_model['params'][0] * np.exp(lows_model['params'][1] * current_days) + lows_model['params'][2]
    
    # Calculate how far above the floor we are
    floor_premium = ((current_price - floor_value) / floor_value) * 100
    
    # Calculate historical floor analysis
    x_all = df['days'].values
    y_all = df[' value'].values
    floor_predictions = lows_model['params'][0] * np.exp(lows_model['params'][1] * x_all) + lows_model['params'][2]
    
    # Count how many times price was below the floor
    below_floor = np.sum(y_all < floor_predictions)
    total_points = len(y_all)
    floor_violations = (below_floor / total_points) * 100
    
    return {
        'floor_value': floor_value,
        'current_price': current_price,
        'floor_premium': floor_premium,
        'floor_violations': floor_violations,
        'growth_rate': lows_model['growth_rate']
    }

def main():
    print("Palladium Lows Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Identify lows
    print("\nIdentifying significant lows...")
    lows = identify_lows(df)
    
    if len(lows) < 3:
        print("Not enough lows found for analysis")
        return
    
    # Fit lows model
    print("\nFitting lows model...")
    lows_model = fit_lows_model(lows)
    
    if lows_model:
        print(f"Lows Model: {lows_model['formula']}")
        print(f"  R² = {lows_model['r2']:.4f}")
        print(f"  Growth Rate = {lows_model['growth_rate']:.6f}")
    else:
        print("Failed to fit lows model")
        return
    
    # Calculate analysis
    print("\nCalculating lows analysis...")
    analysis = calculate_lows_analysis(df, lows_model)
    
    if analysis:
        print(f"\nFloor Analysis:")
        print(f"  Current Price: ${analysis['current_price']:.2f}")
        print(f"  Model Floor: ${analysis['floor_value']:.2f}")
        print(f"  Floor Premium: {analysis['floor_premium']:.1f}%")
        print(f"  Historical Floor Violations: {analysis['floor_violations']:.1f}%")
        print(f"  Floor Growth Rate: {analysis['growth_rate']:.6f}")
    
    # Create chart
    print("\nCreating lows model chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_lows_chart(df, lows, lows_model, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_lows_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM LOWS MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Identified significant lows using multiple methods:\n")
        f.write("   - Rolling minimum approach\n")
        f.write("   - Z-score analysis (1.5+ standard deviations below mean)\n")
        f.write("   - Percentile analysis (bottom 15% of values)\n")
        f.write("2. Fitted exponential model to these lows\n")
        f.write("3. This represents the historical floor that palladium maintains\n\n")
        
        f.write(f"Lows Model: {lows_model['formula']}\n")
        f.write(f"  R² = {lows_model['r2']:.6f}\n")
        f.write(f"  Growth Rate = {lows_model['growth_rate']:.6f}\n")
        f.write(f"  Number of Lows: {len(lows)}\n\n")
        
        if analysis:
            f.write("Floor Analysis:\n")
            f.write(f"  Current Price: ${analysis['current_price']:.2f}\n")
            f.write(f"  Model Floor: ${analysis['floor_value']:.2f}\n")
            f.write(f"  Floor Premium: {analysis['floor_premium']:.1f}%\n")
            f.write(f"  Historical Floor Violations: {analysis['floor_violations']:.1f}%\n\n")
        
        f.write("Key Insights:\n")
        f.write("- This model represents the historical floor for palladium\n")
        f.write("- It shows the minimum price level palladium has maintained\n")
        f.write("- Current price vs floor shows how much premium exists\n")
        f.write("- Floor violations show how often price broke below this level\n")
        f.write("- This provides a conservative baseline for palladium valuation\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("LOWS MODEL ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
