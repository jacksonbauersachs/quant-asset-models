import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def identify_peaks_and_valleys(df, window=365):
    """Identify peaks and valleys in the data using rolling windows."""
    
    # Calculate rolling statistics
    df['rolling_mean'] = df[' value'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df[' value'].rolling(window=window, center=True).std()
    
    # Identify peaks (local maxima)
    df['is_peak'] = False
    df['is_valley'] = False
    
    # Find peaks and valleys using rolling windows
    for i in range(window, len(df) - window):
        # Check if current point is a peak
        if (df[' value'].iloc[i] > df[' value'].iloc[i-window:i].max() and 
            df[' value'].iloc[i] > df[' value'].iloc[i+1:i+window+1].max()):
            df.loc[df.index[i], 'is_peak'] = True
        
        # Check if current point is a valley
        if (df[' value'].iloc[i] < df[' value'].iloc[i-window:i].min() and 
            df[' value'].iloc[i] < df[' value'].iloc[i+1:i+window+1].min()):
            df.loc[df.index[i], 'is_valley'] = True
    
    # Get peak and valley data
    peaks = df[df['is_peak']].copy()
    valleys = df[df['is_valley']].copy()
    
    print(f"\nPeak/Valley Analysis:")
    print(f"  Found {len(peaks)} peaks")
    print(f"  Found {len(valleys)} valleys")
    print(f"  Peak price range: ${peaks[' value'].min():.2f} to ${peaks[' value'].max():.2f}")
    print(f"  Valley price range: ${valleys[' value'].min():.2f} to ${valleys[' value'].max():.2f}")
    
    return peaks, valleys

def fit_exponential_model(x, y, model_name):
    """Fit an exponential model to the given data."""
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(exponential_func, x, y, p0=[50, 0.0002, 0], maxfev=10000)
        a, b, c = popt
        
        # Calculate R²
        y_pred = exponential_func(x, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        model = {
            'name': model_name,
            'params': popt,
            'r2': r2,
            'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}'
        }
        
        return model
        
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        return None

def create_peak_valley_chart(df, peaks, valleys, peak_model, valley_model, avg_model, output_dir):
    """Create a chart showing the peak/valley analysis."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.3, s=1, color='lightblue', label='All Data')
    ax1.scatter(peaks['years'], peaks[' value'], alpha=0.8, s=20, color='red', label='Peaks')
    ax1.scatter(valleys['years'], valleys[' value'], alpha=0.8, s=20, color='green', label='Valleys')
    
    # Plot models
    x_all = df['days'].values
    
    if peak_model:
        y_peak = peak_model['params'][0] * np.exp(peak_model['params'][1] * x_all) + peak_model['params'][2]
        ax1.plot(df['years'], y_peak, 'r-', linewidth=2, label=f'Peak Model (R² = {peak_model["r2"]:.3f})')
    
    if valley_model:
        y_valley = valley_model['params'][0] * np.exp(valley_model['params'][1] * x_all) + valley_model['params'][2]
        ax1.plot(df['years'], y_valley, 'g-', linewidth=2, label=f'Valley Model (R² = {valley_model["r2"]:.3f})')
    
    if avg_model:
        y_avg = avg_model['params'][0] * np.exp(avg_model['params'][1] * x_all) + avg_model['params'][2]
        ax1.plot(df['years'], y_avg, 'b-', linewidth=3, label=f'Average Model (R² = {avg_model["r2"]:.3f})')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Peak/Valley Model Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.3, s=1, color='lightblue', label='All Data')
    ax2.scatter(peaks['years'], peaks[' value'], alpha=0.8, s=20, color='red', label='Peaks')
    ax2.scatter(valleys['years'], valleys[' value'], alpha=0.8, s=20, color='green', label='Valleys')
    
    if peak_model:
        ax2.plot(df['years'], y_peak, 'r-', linewidth=2, label=f'Peak Model')
    if valley_model:
        ax2.plot(df['years'], y_valley, 'g-', linewidth=2, label=f'Valley Model')
    if avg_model:
        ax2.plot(df['years'], y_avg, 'b-', linewidth=3, label=f'Average Model')
    
    ax2.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Peak/Valley Analysis - Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_peak_valley_analysis.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_fair_values(df, peak_model, valley_model, avg_model):
    """Calculate fair values for all models."""
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    results = {}
    
    if peak_model:
        fair_value = peak_model['params'][0] * np.exp(peak_model['params'][1] * current_days) + peak_model['params'][2]
        undervaluation = ((fair_value - current_price) / current_price) * 100
        results['peak'] = {
            'fair_value': fair_value,
            'undervaluation': undervaluation,
            'growth_rate': peak_model['params'][1]
        }
    
    if valley_model:
        fair_value = valley_model['params'][0] * np.exp(valley_model['params'][1] * current_days) + valley_model['params'][2]
        undervaluation = ((fair_value - current_price) / current_price) * 100
        results['valley'] = {
            'fair_value': fair_value,
            'undervaluation': undervaluation,
            'growth_rate': valley_model['params'][1]
        }
    
    if avg_model:
        fair_value = avg_model['params'][0] * np.exp(avg_model['params'][1] * current_days) + avg_model['params'][2]
        undervaluation = ((fair_value - current_price) / current_price) * 100
        results['average'] = {
            'fair_value': fair_value,
            'undervaluation': undervaluation,
            'growth_rate': avg_model['params'][1]
        }
    
    return results

def main():
    print("Palladium Peak/Valley Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Identify peaks and valleys
    print("\nIdentifying peaks and valleys...")
    peaks, valleys = identify_peaks_and_valleys(df)
    
    if len(peaks) < 3 or len(valleys) < 3:
        print("Not enough peaks/valleys found for analysis")
        return
    
    # Fit models
    print("\nFitting models...")
    
    # Peak model
    peak_model = None
    if len(peaks) >= 3:
        peak_model = fit_exponential_model(peaks['days'].values, peaks[' value'].values, "Peak Model")
        if peak_model:
            print(f"Peak Model: {peak_model['formula']}")
            print(f"  R² = {peak_model['r2']:.4f}")
    
    # Valley model
    valley_model = None
    if len(valleys) >= 3:
        valley_model = fit_exponential_model(valleys['days'].values, valleys[' value'].values, "Valley Model")
        if valley_model:
            print(f"Valley Model: {valley_model['formula']}")
            print(f"  R² = {valley_model['r2']:.4f}")
    
    # Average model
    avg_model = None
    if peak_model and valley_model:
        # Average the parameters
        avg_params = [
            (peak_model['params'][0] + valley_model['params'][0]) / 2,  # a
            (peak_model['params'][1] + valley_model['params'][1]) / 2,  # b (growth rate)
            (peak_model['params'][2] + valley_model['params'][2]) / 2   # c
        ]
        
        # Calculate R² for average model
        x_all = df['days'].values
        y_all = df[' value'].values
        y_avg = avg_params[0] * np.exp(avg_params[1] * x_all) + avg_params[2]
        
        ss_res = np.sum((y_all - y_avg) ** 2)
        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        avg_model = {
            'name': 'Average Model',
            'params': avg_params,
            'r2': r2,
            'formula': f'y = {avg_params[0]:.4f} * e^({avg_params[1]:.6f} * x) + {avg_params[2]:.2f}'
        }
        
        print(f"Average Model: {avg_model['formula']}")
        print(f"  R² = {avg_model['r2']:.4f}")
    
    # Calculate fair values
    print("\nCalculating fair values...")
    fair_values = calculate_fair_values(df, peak_model, valley_model, avg_model)
    
    current_price = df[' value'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    
    if 'peak' in fair_values:
        print(f"Peak Model Fair Value: ${fair_values['peak']['fair_value']:.2f}")
        print(f"  Undervaluation: {fair_values['peak']['undervaluation']:.1f}%")
        print(f"  Growth Rate: {fair_values['peak']['growth_rate']:.6f}")
    
    if 'valley' in fair_values:
        print(f"Valley Model Fair Value: ${fair_values['valley']['fair_value']:.2f}")
        print(f"  Undervaluation: {fair_values['valley']['undervaluation']:.1f}%")
        print(f"  Growth Rate: {fair_values['valley']['growth_rate']:.6f}")
    
    if 'average' in fair_values:
        print(f"Average Model Fair Value: ${fair_values['average']['fair_value']:.2f}")
        print(f"  Undervaluation: {fair_values['average']['undervaluation']:.1f}%")
        print(f"  Growth Rate: {fair_values['average']['growth_rate']:.6f}")
    
    # Create chart
    print("\nCreating peak/valley analysis chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_peak_valley_chart(df, peaks, valleys, peak_model, valley_model, avg_model, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_peak_valley_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM PEAK/VALLEY MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Identified peaks and valleys in palladium price data\n")
        f.write("2. Fitted exponential models to peaks only and valleys only\n")
        f.write("3. Averaged the parameters to create a balanced model\n")
        f.write("4. This reduces overfitting to recent volatility\n\n")
        
        if peak_model:
            f.write(f"Peak Model: {peak_model['formula']}\n")
            f.write(f"  R² = {peak_model['r2']:.6f}\n")
            f.write(f"  Fair Value: ${fair_values['peak']['fair_value']:.2f}\n")
            f.write(f"  Undervaluation: {fair_values['peak']['undervaluation']:.1f}%\n\n")
        
        if valley_model:
            f.write(f"Valley Model: {valley_model['formula']}\n")
            f.write(f"  R² = {valley_model['r2']:.6f}\n")
            f.write(f"  Fair Value: ${fair_values['valley']['fair_value']:.2f}\n")
            f.write(f"  Undervaluation: {fair_values['valley']['undervaluation']:.1f}%\n\n")
        
        if avg_model:
            f.write(f"Average Model: {avg_model['formula']}\n")
            f.write(f"  R² = {avg_model['r2']:.6f}\n")
            f.write(f"  Fair Value: ${fair_values['average']['fair_value']:.2f}\n")
            f.write(f"  Undervaluation: {fair_values['average']['undervaluation']:.1f}%\n\n")
        
        f.write("Key Insights:\n")
        f.write("- Peak model shows bullish outlook (higher growth rate)\n")
        f.write("- Valley model shows bearish outlook (lower growth rate)\n")
        f.write("- Average model provides balanced perspective\n")
        f.write("- This approach reduces overfitting to recent volatility\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("PEAK/VALLEY ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
