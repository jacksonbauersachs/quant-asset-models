"""
Template script for optimizing intercept with a fixed slope.
This script reads configuration from config.py.

Usage:
    1. Edit config.py with your currency information
    2. Set FIXED_SLOPE in config.py (typically from running interactive_rainbow.py first)
    3. Run: python optimize_intercept.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent file handle issues
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
import sys
from datetime import datetime
import gc

# Import configuration
try:
    from config import (
        CURRENCY_NAME, CURRENCY_NAME_LOWER, DATA_FOLDER, HISTORICAL_DATA_FILE,
        OUTPUT_FOLDER, START_DATE, FIXED_SLOPE,
        INTERCEPT_MIN, INTERCEPT_MAX, get_output_files, validate_config
    )
    # GENESIS_DATE is optional - will auto-detect from data if not provided
    try:
        from config import GENESIS_DATE
    except ImportError:
        GENESIS_DATE = None  # Will auto-detect from data
except ImportError:
    print("Error: config.py not found! Make sure you're running from the clone folder.")
    sys.exit(1)

# Validate config
validate_config()
output_files = get_output_files()

def load_data():
    """Load and prepare price data for analysis."""
    
    data_file = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILE)
    
    if not os.path.exists(data_file):
        print(f"Error: {CURRENCY_NAME} data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime - handle multiple formats
    try:
        # Try M/D/YYYY format first (original format)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    except (ValueError, TypeError):
        try:
            # Try YYYY-MM-DD format
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        except (ValueError, TypeError):
            # Last resort: let pandas auto-detect
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Determine genesis date (use earliest date in data if not specified)
    if GENESIS_DATE is None or GENESIS_DATE == "":
        genesis_date = df['Date'].min()
        print(f"Auto-detected genesis date from data: {genesis_date.strftime('%Y-%m-%d')}")
    else:
        genesis_date = pd.Timestamp(GENESIS_DATE)
        print(f"Using configured genesis date: {genesis_date.strftime('%Y-%m-%d')}")
    
    # Create days since genesis for rainbow chart model
    genesis = pd.Timestamp(genesis_date)
    df['Days_Since_Genesis'] = (df['Date'] - genesis).dt.days
    
    # Ensure days_since_genesis is at least 1 to avoid log(0) errors
    df['Days_Since_Genesis'] = df['Days_Since_Genesis'].clip(lower=1)
    
    # Filter from start date (auto-detect from first date if not specified)
    if START_DATE is None or START_DATE == "":
        start_date = df['Date'].min()
        print(f"Auto-detected start date from data: {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = pd.Timestamp(START_DATE)
        print(f"Using configured start date: {start_date.strftime('%Y-%m-%d')}")
    
    df_filtered = df[df['Date'] >= start_date].copy()
    
    # Filter out any rows with zero or negative prices (can cause log errors)
    df_filtered = df_filtered[df_filtered['Price'] > 0].copy()
    
    print(f"{CURRENCY_NAME} data loaded: {len(df_filtered)} records from {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df_filtered['Price'].min():.2f} to ${df_filtered['Price'].max():,.2f}")
    
    return df_filtered

def rainbow_model(x, slope, intercept):
    """Rainbow model: price = 10^(slope * ln(days_since_genesis) + intercept)"""
    # Ensure x is at least 1 to avoid log(0) errors
    x = np.maximum(x, 1)
    return 10**(slope * np.log(x) + intercept)

def calculate_r2_log_space(y_actual, y_pred):
    """Calculate R² score in log space (like the original Bitcoin model)."""
    # Filter out invalid values (zeros, negatives, NaN, Inf)
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    
    if not np.any(valid_mask):
        return 0.0  # No valid data
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Convert to log space for R² calculation (matching original model)
    log_actual = np.log10(y_actual_valid)
    log_pred = np.log10(y_pred_valid)
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def objective_function(intercept):
    """Objective function to minimize (negative R² to maximize R²)."""
    global df_filtered, fixed_slope
    
    # Calculate model predictions
    y_pred = rainbow_model(df_filtered['Days_Since_Genesis'], fixed_slope, intercept)
    
    # Calculate R² in log space
    r2 = calculate_r2_log_space(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def optimize_intercept_for_r2(df, slope):
    """Find the optimal intercept that maximizes R² for a given fixed slope."""
    global df_filtered, fixed_slope
    df_filtered = df
    fixed_slope = slope
    
    print(f"Optimizing intercept for fixed slope: {slope}")
    print("Searching for optimal intercept...")
    
    # Use very wide bounds - let optimization find the best fit
    wide_min = -50
    wide_max = 10
    
    try:
        result = minimize_scalar(objective_function, bounds=(wide_min, wide_max), method='bounded')
        optimal_intercept = result.x
        optimal_r2 = -result.fun  # Convert back to positive R²
        
        # If we hit a boundary, try even wider
        if abs(optimal_intercept - wide_min) < 0.1 or abs(optimal_intercept - wide_max) < 0.1:
            print(f"Intercept hit boundary. Trying wider range...")
            wider_min = -100
            wider_max = 20
            result = minimize_scalar(objective_function, bounds=(wider_min, wider_max), method='bounded')
            optimal_intercept = result.x
            optimal_r2 = -result.fun
        
        # Validate the result makes sense
        test_pred = rainbow_model(df['Days_Since_Genesis'].iloc[-1], slope, optimal_intercept)
        if not np.isfinite(test_pred) or test_pred > df['Price'].max() * 1000:
            print(f"Warning: Optimal intercept produces unrealistic predictions. Using conservative default.")
            optimal_intercept = np.log10(df['Price'].iloc[0]) - slope * np.log(df['Days_Since_Genesis'].iloc[0])
            optimal_r2 = -objective_function(optimal_intercept)
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using default intercept.")
        optimal_intercept = np.log10(df['Price'].iloc[0]) - slope * np.log(df['Days_Since_Genesis'].iloc[0])
        optimal_r2 = -objective_function(optimal_intercept)
    
    print(f"Optimal intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_intercept, optimal_r2

def create_optimization_chart(df, slope, optimal_intercept, optimal_r2):
    """Create a chart showing the optimization results."""
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(lambda x: rainbow_model(x, slope, optimal_intercept))
    
    # Cap unrealistic predictions to prevent display issues
    max_reasonable = df['Price'].max() * 100  # Cap at 100x max price
    df['Model_Price'] = df['Model_Price'].clip(upper=max_reasonable)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual price on both subplots
    ax1.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label=f'Actual {CURRENCY_NAME} Price')
    ax2.plot(df['Date'], df['Price'], color='blue', linewidth=2, label=f'Actual {CURRENCY_NAME} Price')
    
    # Plot optimized model line
    ax1.semilogy(df['Date'], df['Model_Price'], 
                color='red', linestyle='--', linewidth=2, label='Optimized Model')
    ax2.plot(df['Date'], df['Model_Price'], 
            color='red', linestyle='--', linewidth=2, label='Optimized Model')
    
    # Highlight current position
    current_price = df['Price'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    ax1.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    ax2.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    
    # Formatting for both subplots
    for ax, ylabel_suffix in [(ax1, 'Log Scale'), (ax2, 'Normal Scale')]:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{CURRENCY_NAME} Price (USD) - {ylabel_suffix}', fontsize=12)
        ax.set_title(f'{CURRENCY_NAME} Optimized Model - {ylabel_suffix}\nSlope: {slope:.6f}, Intercept: {optimal_intercept:.6f}, R²: {optimal_r2:.6f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Custom y-axis formatting
        def price_formatter(x, pos):
            if x >= 1000000:
                return f'${x/1000000:.1f}M'
            elif x >= 1000:
                return f'${x/1000:.0f}K'
            else:
                return f'${x:.0f}'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
        
        # Add legend
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set y-axis limits
        if ax == ax2:  # Normal scale
            max_price = max(df['Price'].max(), df['Model_Price'].max())
            ax.set_ylim(0, min(max_price * 2, df['Price'].max() * 3))
        else:  # Log scale
            ax.set_ylim(0.01, max(df['Price'].max(), df['Model_Price'].max()) * 2)
    
    plt.tight_layout()
    
    # Save the chart
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    chart_file = os.path.join(OUTPUT_FOLDER, output_files['intercept_optimization_png'])
    
    # Save with proper flushing
    fig.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Properly close and release file handles
    fig.clear()
    plt.close(fig)
    plt.close('all')  # Close all matplotlib figures
    
    # Force garbage collection to release file handles (important on Windows)
    gc.collect()
    
    return chart_file

def calculate_projections(df, slope, intercept):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    
    # Calculate current fair value
    current_fair_price = rainbow_model(current_days, slope, intercept)
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = rainbow_model(future_days, slope, intercept)
        
        # Model-based projection (from fair value)
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection (includes undervaluation benefit)
        if undervaluation < 0:  # If undervalued
            undervaluation_benefit = abs(undervaluation)
            current_return = model_return + undervaluation_benefit
        else:  # If overvalued
            current_return = model_return - abs(undervaluation)
        
        current_based_price = current_price * (1 + current_return / 100)
        
        projections[years] = {
            'model_price': model_future_price,
            'current_based_price': current_based_price,
            'model_return': model_return,
            'current_return': current_return
        }
    
    return {
        'current_price': current_price,
        'current_fair_price': current_fair_price,
        'undervaluation': undervaluation,
        'projections': projections
    }

def save_optimization_results(slope, optimal_intercept, optimal_r2, projections):
    """Save optimization results to a text file."""
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    results_file = os.path.join(OUTPUT_FOLDER, output_files['intercept_optimization_txt'])
    
    with open(results_file, 'w') as f:
        f.write(f"{CURRENCY_NAME.upper()} INTERCEPT OPTIMIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fixed Slope: {slope:.6f}\n")
        f.write(f"Optimal Intercept: {optimal_intercept:.6f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        f.write(f"price = 10^({slope:.6f} * ln(days_since_genesis) + {optimal_intercept:.6f})\n\n")
        
        f.write("CURRENT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Current Price: ${projections['current_price']:,.2f}\n")
        f.write(f"Model Fair Price: ${projections['current_fair_price']:,.2f}\n")
        f.write(f"Undervaluation: {projections['undervaluation']:+.2f}%\n\n")
        
        f.write("FORWARD PROJECTIONS\n")
        f.write("-" * 25 + "\n")
        f.write("Model-Based Projections (from fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['model_price']:,.0f} | Growth: {p['model_return']:+.2f}%\n")
        
        f.write("\nCurrent Price-Based Projections (includes undervaluation benefit):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['current_based_price']:,.0f} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections start from the model's fair value\n")
        f.write("- Current price-based projections start from the current (undervalued) price\n")
        f.write("- Undervaluation benefit is added to current price-based returns\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def main():
    """Main function to optimize intercept."""
    
    print(f"{CURRENCY_NAME} Intercept Optimization")
    print("=" * 40)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Use fixed slope from config
    print(f"\nUsing fixed slope: {FIXED_SLOPE}")
    print("Intercept will be optimized to maximize R²")
    optimal_intercept, optimal_r2 = optimize_intercept_for_r2(df, FIXED_SLOPE)
    
    # Create optimization chart
    print("\nCreating optimization chart...")
    chart_file = create_optimization_chart(df, FIXED_SLOPE, optimal_intercept, optimal_r2)
    print(f"Chart saved: {chart_file}")
    
    # Calculate projections
    print("\nCalculating forward projections...")
    projections = calculate_projections(df, FIXED_SLOPE, optimal_intercept)
    
    # Save results
    results_file = save_optimization_results(FIXED_SLOPE, optimal_intercept, optimal_r2, projections)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Fixed Slope: {FIXED_SLOPE:.6f}")
    print(f"Optimal Intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    print(f"Current Price: ${projections['current_price']:,.2f}")
    print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
    print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

