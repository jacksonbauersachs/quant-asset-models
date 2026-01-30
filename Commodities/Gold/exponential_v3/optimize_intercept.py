"""
Template script for optimizing growth rate and creating projections.
This script reads configuration from config.py.

Usage:
    1. Edit config.py with your currency information
    2. Optionally set FIXED_GROWTH_RATE in config.py (from running interactive_rainbow.py first)
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
import subprocess
import platform
from pathlib import Path

# Import configuration
try:
    from config import (
        CURRENCY_NAME, CURRENCY_NAME_LOWER, DATA_FOLDER, HISTORICAL_DATA_FILE,
        OUTPUT_FOLDER, START_DATE,
        get_output_files, validate_config
    )
    # GENESIS_DATE is optional - will auto-detect from data if not provided
    try:
        from config import GENESIS_DATE
    except ImportError:
        GENESIS_DATE = None  # Will auto-detect from data
    # FIXED_GROWTH_RATE is optional - will optimize if not provided
    try:
        from config import FIXED_GROWTH_RATE
    except ImportError:
        FIXED_GROWTH_RATE = None  # Will optimize if not set
except ImportError:
    print("Error: config.py not found! Make sure you're running from the clone folder.")
    sys.exit(1)

# Validate config
validate_config()
output_files = get_output_files()

def load_data():
    """Load and prepare price data for analysis."""
    
    # Resolve DATA_FOLDER relative to script location, not cwd
    script_dir = Path(__file__).parent.resolve()
    
    # If DATA_FOLDER is absolute, use it as-is
    if os.path.isabs(DATA_FOLDER):
        data_folder_path = Path(DATA_FOLDER)
    else:
        # DATA_FOLDER is like "Crypto/Doge" - need to find project root
        # Project root is the directory that CONTAINS the "Crypto" folder
        current = script_dir
        project_root = None
        
        # Walk up until we find a directory that contains "Crypto" folder
        while current.parent != current:
            # Check if current directory contains "Crypto" folder
            crypto_path = current / "Crypto"
            if crypto_path.exists() and crypto_path.is_dir():
                # Found project root (directory containing Crypto folder)
                project_root = current
                break
            current = current.parent
        
        if project_root:
            data_folder_path = project_root / DATA_FOLDER
        else:
            # Fallback: assume DATA_FOLDER is relative to script
            data_folder_path = script_dir / DATA_FOLDER
    
    data_file = data_folder_path / HISTORICAL_DATA_FILE
    
    if not os.path.exists(data_file):
        print(f"Error: {CURRENCY_NAME} data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(str(data_file))
    
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
    
    # Handle different column names for price (Price, Value, Close, etc.)
    price_column = None
    for col in ['Price', 'Value', 'Close', 'Adj Close']:
        if col in df.columns:
            price_column = col
            break
    
    if price_column is None:
        print(f"Error: Could not find price column. Available columns: {list(df.columns)}")
        return None
    
    # Rename to 'Price' for consistency
    if price_column != 'Price':
        df['Price'] = df[price_column]
    
    # Clean Price column - remove commas and convert to float
    # Handle comma-separated numbers (e.g., "1,002.60") safely
    if df['Price'].dtype == 'object':
        # Remove commas from string values before converting
        df['Price'] = df['Price'].str.replace(',', '', regex=False)
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

def rainbow_model(x, growth_rate, first_price, first_days, intercept=0):
    """Exponential growth model: price = first_price * e^(growth_rate * (days - first_days) + intercept)"""
    # Ensure x is at least first_days to avoid issues
    x = np.maximum(x, first_days)
    return first_price * np.exp(growth_rate * (x - first_days) + intercept)

def calculate_r2(y_actual, y_pred):
    """Calculate R² score for exponential growth model."""
    # Filter out invalid values (zeros, negatives, NaN, Inf)
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    
    if not np.any(valid_mask):
        return 0.0  # No valid data
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate R² in normal space (not log space for exponential model)
    ss_res = np.sum((y_actual_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_actual_valid - np.mean(y_actual_valid)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def objective_function(intercept):
    """Objective function to minimize (negative R² to maximize R²)."""
    global df_filtered, fixed_growth_rate, first_price, first_days
    
    # Calculate model predictions with fixed growth rate and variable intercept
    y_pred = rainbow_model(df_filtered['Days_Since_Genesis'], fixed_growth_rate, first_price, first_days, intercept)
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def optimize_intercept_for_r2(df, growth_rate):
    """Find the optimal intercept that maximizes R² for a given fixed growth rate.
    The intercept is anchored at the first data point (base intercept = 0),
    then a small vertical offset is optimized to minimize R² without changing
    the fundamental position of the curve."""
    global df_filtered, fixed_growth_rate, first_price, first_days
    df_filtered = df
    fixed_growth_rate = growth_rate
    first_price = df['Price'].iloc[0]
    first_days = df['Days_Since_Genesis'].iloc[0]
    
    print(f"Optimizing intercept for fixed growth rate: {growth_rate:.6f} per day")
    annualized_rate = growth_rate * 365.25 * 100
    print(f"Annualized growth rate: {annualized_rate:.2f}% per year")
    print("Anchoring curve at first data point, then optimizing small vertical offset...")
    
    def objective_function(vertical_offset):
        """Objective function: minimize negative R² with small vertical offset.
        The offset is in log space, so it's a multiplicative adjustment."""
        # Base intercept = 0 makes model pass through first point exactly
        base_intercept = 0.0
        intercept = base_intercept + vertical_offset
        
        # Calculate model predictions
        y_pred = rainbow_model(df_filtered['Days_Since_Genesis'], fixed_growth_rate, 
                              first_price, first_days, intercept)
        
        # Calculate R²
        r2 = calculate_r2(df_filtered['Price'], y_pred)
        
        return -r2
    
    # Use much smaller bounds for vertical offset (just fine-tuning)
    # This keeps the curve anchored at the first point
    offset_min = -2.0  # Small adjustment range
    offset_max = 2.0
    
    try:
        result = minimize_scalar(objective_function, bounds=(offset_min, offset_max), method='bounded')
        optimal_offset = result.x
        optimal_intercept = 0.0 + optimal_offset  # Base intercept (0) + offset
        optimal_r2 = -result.fun
        
        # Validate the result makes sense
        test_pred = rainbow_model(df['Days_Since_Genesis'].iloc[-1], growth_rate, 
                                 first_price, first_days, optimal_intercept)
        if not np.isfinite(test_pred) or test_pred > df['Price'].max() * 1000:
            print(f"Warning: Optimal intercept produces unrealistic predictions. Using intercept=0.")
            optimal_intercept = 0.0
            optimal_r2 = -objective_function(0.0)
            optimal_offset = 0.0
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using intercept=0.")
        optimal_intercept = 0.0
        optimal_r2 = -objective_function(0.0)
        optimal_offset = 0.0
    
    print(f"Optimal intercept: {optimal_intercept:.6f} (base: 0.0, offset: {optimal_offset:.6f})")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_intercept, optimal_r2

def create_optimization_chart(df, growth_rate, optimal_intercept, optimal_r2):
    """Create a chart showing the optimization results."""
    
    # Get first point values
    first_price = df['Price'].iloc[0]
    first_days = df['Days_Since_Genesis'].iloc[0]
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, growth_rate, first_price, first_days, optimal_intercept)
    )
    
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
        annualized_rate = growth_rate * 365.25 * 100
        ax.set_title(f'{CURRENCY_NAME} Optimized Model - {ylabel_suffix}\nGrowth Rate: {growth_rate:.6f} ({annualized_rate:.2f}%/yr), Intercept: {optimal_intercept:.6f}, R²: {optimal_r2:.6f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Dynamic y-axis formatting based on data range
        min_price = min(df['Price'].min(), df['Model_Price'].min())
        max_price = max(df['Price'].max(), df['Model_Price'].max())
        price_range = max_price - min_price
        
        # Create dynamic price formatter based on data range
        if max_price >= 1000000:
            def price_formatter(x, pos):
                return f'${x/1000000:.2f}M'
        elif max_price >= 1000:
            def price_formatter(x, pos):
                return f'${x/1000:.1f}K'
        elif max_price >= 1:
            def price_formatter(x, pos):
                return f'${x:.2f}'
        else:
            def price_formatter(x, pos):
                return f'${x:.4f}'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
        
        # Set dynamic tick spacing based on price range
        from matplotlib.ticker import MaxNLocator
        if ax == ax2:  # Normal scale
            # Use MaxNLocator to get reasonable number of ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            # Set y-axis limits with some padding
            padding = price_range * 0.1  # 10% padding
            ax.set_ylim(max(0, min_price - padding), max_price + padding)
        else:  # Log scale
            # For log scale, let matplotlib handle it automatically
            ax.set_ylim(max(0.01, min_price * 0.5), max_price * 2)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))
    
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
    
    # Open the image file automatically
    try:
        if platform.system() == 'Windows':
            os.startfile(chart_file)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', chart_file])
        else:  # Linux and other Unix-like systems
            subprocess.run(['xdg-open', chart_file])
        print(f"Opened image: {chart_file}")
    except Exception as e:
        print(f"Note: Could not auto-open image: {e}")
        print(f"Image saved at: {chart_file}")
    
    return chart_file

def calculate_projections(df, growth_rate, intercept):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    first_price = df['Price'].iloc[0]
    first_days = df['Days_Since_Genesis'].iloc[0]
    
    # Calculate current fair value using exponential model
    current_fair_price = rainbow_model(current_days, growth_rate, first_price, first_days, intercept)
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = rainbow_model(future_days, growth_rate, first_price, first_days, intercept)
        
        # Model-based projection: percentage relative to fair value
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection: same dollar value, but percentage relative to current actual price
        current_based_price = model_future_price  # Same dollar value as model
        current_return = ((current_based_price - current_price) / current_price) * 100
        
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
        'projections': projections,
        'first_price': first_price,
        'first_days': first_days
    }

def format_price(price):
    """Format price with appropriate decimal places based on value."""
    if price < 0.01:
        return f"${price:.6f}"  # 6 decimals for very small prices (< 1 cent)
    elif price < 0.1:
        return f"${price:.5f}"  # 5 decimals for prices < 10 cents
    elif price < 1.0:
        return f"${price:.4f}"  # 4 decimals for prices < $1
    elif price < 1000:
        return f"${price:.2f}"  # 2 decimals for prices < $1000
    else:
        return f"${price:,.0f}"  # No decimals, with commas for prices >= $1000

def save_optimization_results(growth_rate, optimal_intercept, optimal_r2, projections):
    """Save optimization results to a text file."""
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    results_file = os.path.join(OUTPUT_FOLDER, output_files['intercept_optimization_txt'])
    
    annualized_rate = growth_rate * 365.25 * 100
    first_price = projections.get('first_price', 'N/A')
    first_days = projections.get('first_days', 'N/A')
    
    with open(results_file, 'w') as f:
        f.write(f"{CURRENCY_NAME.upper()} INTERCEPT OPTIMIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fixed Growth Rate: {growth_rate:.6f} per day\n")
        f.write(f"Annualized Growth Rate: {annualized_rate:.2f}% per year\n")
        f.write(f"Optimal Intercept: {optimal_intercept:.6f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        if first_price != 'N/A' and first_days != 'N/A':
            f.write(f"price = {first_price:.2f} × e^({growth_rate:.6f} × (day - {first_days:.0f}) + {optimal_intercept:.6f})\n")
        else:
            f.write(f"price = first_price × e^(growth_rate × (day - first_day) + {optimal_intercept:.6f})\n")
        f.write("\n")
        
        f.write("CURRENT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Current Price: {format_price(projections['current_price'])}\n")
        f.write(f"Model Fair Price: {format_price(projections['current_fair_price'])}\n")
        f.write(f"Undervaluation: {projections['undervaluation']:+.2f}%\n\n")
        
        f.write("FORWARD PROJECTIONS\n")
        f.write("-" * 25 + "\n")
        f.write("Model-Based Projections (from fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: {format_price(p['model_price'])} | Growth: {p['model_return']:+.2f}%\n")
        
        f.write("\nCurrent Price-Based Projections (same dollar values, percentage relative to current price):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: {format_price(p['current_based_price'])} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections: percentage relative to model's fair value\n")
        f.write("- Current price-based projections: same dollar values, percentage relative to current actual price\n")
        f.write("- Both show the model's predicted future price, just from different reference points\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def main():
    """Main function to optimize growth rate."""
    
    print(f"{CURRENCY_NAME} Growth Rate Optimization")
    print("=" * 40)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Use fixed growth rate from config if provided, otherwise optimize
    if FIXED_GROWTH_RATE is not None:
        print(f"\nUsing fixed growth rate from config: {FIXED_GROWTH_RATE:.6f} per day")
        fixed_growth_rate = FIXED_GROWTH_RATE
        annualized_rate = fixed_growth_rate * 365.25 * 100
        print(f"Annualized growth rate: {annualized_rate:.2f}% per year")
    else:
        print("\nOptimizing growth rate to maximize R²...")
        fixed_growth_rate, _ = optimize_growth_rate_for_r2(df)
        annualized_rate = fixed_growth_rate * 365.25 * 100
        print(f"Using optimized growth rate: {fixed_growth_rate:.6f} per day ({annualized_rate:.2f}% per year)")
    
    # Now optimize intercept for the fixed growth rate
    print("\nOptimizing intercept to maximize R² for the fixed growth rate...")
    optimal_intercept, optimal_r2 = optimize_intercept_for_r2(df, fixed_growth_rate)
    
    # Create optimization chart
    print("\nCreating optimization chart...")
    chart_file = create_optimization_chart(df, fixed_growth_rate, optimal_intercept, optimal_r2)
    print(f"Chart saved: {chart_file}")
    
    # Calculate projections
    print("\nCalculating forward projections...")
    projections = calculate_projections(df, fixed_growth_rate, optimal_intercept)
    
    # Save results
    print("\nSaving optimization results...")
    results_file = save_optimization_results(fixed_growth_rate, optimal_intercept, optimal_r2, projections)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Fixed Growth Rate: {fixed_growth_rate:.6f} per day ({annualized_rate:.2f}% per year)")
    print(f"Optimal Intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    print(f"Current Price: ${projections['current_price']:,.2f}")
    print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
    print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

