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

def filter_middle_data(df, window=365, percentile_range=(0.2, 0.8)):
    """Filter out spikes and troughs to get middle data."""
    
    print(f"\nFiltering middle data...")
    print(f"  Window: {window} days")
    print(f"  Percentile range: {percentile_range[0]*100}% to {percentile_range[1]*100}%")
    
    # Calculate rolling statistics
    df['rolling_mean'] = df[' value'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df[' value'].rolling(window=window, center=True).std()
    
    # Calculate rolling percentiles
    df['rolling_p20'] = df[' value'].rolling(window=window, center=True).quantile(percentile_range[0])
    df['rolling_p80'] = df[' value'].rolling(window=window, center=True).quantile(percentile_range[1])
    
    # Filter for middle data (between 20th and 80th percentiles)
    df['is_middle'] = (df[' value'] >= df['rolling_p20']) & (df[' value'] <= df['rolling_p80'])
    
    # Get middle data
    middle_data = df[df['is_middle']].copy()
    
    print(f"  Original data points: {len(df)}")
    print(f"  Middle data points: {len(middle_data)}")
    print(f"  Filtered out: {len(df) - len(middle_data)} points ({((len(df) - len(middle_data)) / len(df)) * 100:.1f}%)")
    print(f"  Middle price range: ${middle_data[' value'].min():.2f} to ${middle_data[' value'].max():.2f}")
    
    return middle_data

def fit_middle_model(middle_data):
    """Fit an exponential model to the middle data."""
    
    x = middle_data['days'].values
    y = middle_data[' value'].values
    
    # Exponential model
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        # Try multiple initial guesses
        initial_guesses = [
            [50, 0.0002, 100],
            [100, 0.0001, 50],
            [20, 0.0003, 200],
            [80, 0.00015, 150]
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
                        'name': 'Middle Data Model',
                        'params': popt,
                        'r2': r2,
                        'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}',
                        'growth_rate': b
                    }
            except:
                continue
        
        return best_model
        
    except Exception as e:
        print(f"Error fitting middle model: {e}")
        return None

def create_middle_data_chart(df, middle_data, middle_model, output_dir):
    """Create a chart showing the middle data model."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.2, s=1, color='lightgray', label='All Data (Filtered)')
    ax1.scatter(middle_data['years'], middle_data[' value'], alpha=0.6, s=2, color='blue', label='Middle Data (20th-80th Percentiles)')
    
    if middle_model:
        # Plot the middle model
        x_all = df['days'].values
        y_middle_model = middle_model['params'][0] * np.exp(middle_model['params'][1] * x_all) + middle_model['params'][2]
        ax1.plot(df['years'], y_middle_model, 'r-', linewidth=3, label=f'Middle Model (R² = {middle_model["r2"]:.3f})')
        
        # Add confidence bands (show the 20th-80th percentile range)
        if 'rolling_p20' in df.columns and 'rolling_p80' in df.columns:
            ax1.fill_between(df['years'], df['rolling_p20'], df['rolling_p80'], alpha=0.1, color='blue', label='Normal Trading Range')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Middle Data Model - Filtered Spikes and Troughs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.2, s=1, color='lightgray', label='All Data (Filtered)')
    ax2.scatter(middle_data['years'], middle_data[' value'], alpha=0.6, s=2, color='blue', label='Middle Data')
    
    if middle_model:
        ax2.plot(df['years'], y_middle_model, 'r-', linewidth=3, label=f'Middle Model')
        if 'rolling_p20' in df.columns and 'rolling_p80' in df.columns:
            ax2.fill_between(df['years'], df['rolling_p20'], df['rolling_p80'], alpha=0.1, color='blue', label='Normal Trading Range')
    
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Middle Data Model - Log Scale (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    if middle_model:
        formula_text = f"""Middle Data Model Formula:
{middle_model['formula']}
R² = {middle_model['r2']:.4f}
Growth Rate = {middle_model['growth_rate']:.6f}

This model represents the normal trading range
excluding extreme spikes and troughs."""
        
        ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_middle_data_model.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_middle_analysis(df, middle_model):
    """Calculate analysis for the middle data model."""
    
    if not middle_model:
        return None
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    # Calculate model fair value
    fair_value = middle_model['params'][0] * np.exp(middle_model['params'][1] * current_days) + middle_model['params'][2]
    
    # Calculate how far from fair value we are
    deviation = ((current_price - fair_value) / fair_value) * 100
    
    # Calculate where current price sits in the normal range
    if 'rolling_p20' in df.columns and 'rolling_p80' in df.columns:
        current_p20 = df['rolling_p20'].iloc[-1]
        current_p80 = df['rolling_p80'].iloc[-1]
        
        if current_price < current_p20:
            range_position = "Below Normal Range"
        elif current_price > current_p80:
            range_position = "Above Normal Range"
        else:
            range_position = "Within Normal Range"
    else:
        range_position = "Unknown"
    
    return {
        'fair_value': fair_value,
        'current_price': current_price,
        'deviation': deviation,
        'range_position': range_position,
        'growth_rate': middle_model['growth_rate']
    }

def main():
    print("Palladium Middle Data Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Filter middle data
    middle_data = filter_middle_data(df)
    
    if len(middle_data) < 100:
        print("Not enough middle data for analysis")
        return
    
    # Fit middle model
    print("\nFitting middle data model...")
    middle_model = fit_middle_model(middle_data)
    
    if middle_model:
        print(f"Middle Model: {middle_model['formula']}")
        print(f"  R² = {middle_model['r2']:.4f}")
        print(f"  Growth Rate = {middle_model['growth_rate']:.6f}")
    else:
        print("Failed to fit middle model")
        return
    
    # Calculate analysis
    print("\nCalculating middle data analysis...")
    analysis = calculate_middle_analysis(df, middle_model)
    
    if analysis:
        print(f"\nMiddle Data Analysis:")
        print(f"  Current Price: ${analysis['current_price']:.2f}")
        print(f"  Model Fair Value: ${analysis['fair_value']:.2f}")
        print(f"  Deviation: {analysis['deviation']:.1f}%")
        print(f"  Range Position: {analysis['range_position']}")
        print(f"  Growth Rate: {analysis['growth_rate']:.6f}")
    
    # Create chart
    print("\nCreating middle data model chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_middle_data_chart(df, middle_data, middle_model, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_middle_data_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM MIDDLE DATA MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Calculated rolling percentiles (20th-80th) over 365-day windows\n")
        f.write("2. Filtered out data points outside this range (extreme spikes/troughs)\n")
        f.write("3. Fitted exponential model to the remaining 'middle' data\n")
        f.write("4. This represents the normal trading range excluding volatility\n\n")
        
        f.write(f"Middle Data Model: {middle_model['formula']}\n")
        f.write(f"  R² = {middle_model['r2']:.6f}\n")
        f.write(f"  Growth Rate = {middle_model['growth_rate']:.6f}\n")
        f.write(f"  Middle Data Points: {len(middle_data)}\n")
        f.write(f"  Filtered Out: {len(df) - len(middle_data)} points\n\n")
        
        if analysis:
            f.write("Fair Value Analysis:\n")
            f.write(f"  Current Price: ${analysis['current_price']:.2f}\n")
            f.write(f"  Model Fair Value: ${analysis['fair_value']:.2f}\n")
            f.write(f"  Deviation: {analysis['deviation']:.1f}%\n")
            f.write(f"  Range Position: {analysis['range_position']}\n\n")
        
        f.write("Key Insights:\n")
        f.write("- This model represents the 'normal' palladium price trend\n")
        f.write("- It excludes extreme spikes and crashes\n")
        f.write("- Shows the underlying growth rate without volatility\n")
        f.write("- Provides a more realistic baseline for valuation\n")
        f.write("- Current price position shows if we're in normal range\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("MIDDLE DATA MODEL ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
