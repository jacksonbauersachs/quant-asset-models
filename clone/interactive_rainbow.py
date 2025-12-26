"""
Template script for creating an interactive rainbow chart.
This script reads configuration from config.py.

Usage:
    1. Edit config.py with your currency information
    2. Run: python interactive_rainbow.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize_scalar
import os
import sys
from datetime import datetime
import gc

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
except ImportError:
    print("Error: config.py not found! Make sure config.py is in the same folder as this script.")
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

def rainbow_model(x, growth_rate, first_price, first_days):
    """Exponential growth model: price = first_price * e^(growth_rate * (days - first_days))"""
    # Ensure x is at least first_days to avoid issues
    x = np.maximum(x, first_days)
    return first_price * np.exp(growth_rate * (x - first_days))

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

def find_optimal_growth_rate(df):
    """Find the optimal growth rate that maximizes R² for exponential growth model."""
    first_price = df['Price'].iloc[0]
    first_days = df['Days_Since_Genesis'].iloc[0]
    days = df['Days_Since_Genesis'].values
    prices = df['Price'].values
    
    def r2_for_growth_rate(growth_rate):
        """Calculate R² for a given growth rate."""
        y_pred = rainbow_model(days, growth_rate, first_price, first_days)
        
        # Check if predictions are reasonable (not astronomical)
        max_pred = np.max(y_pred)
        max_actual = np.max(prices)
        # If predictions are more than 1000x the max actual price, penalize heavily
        if max_pred > max_actual * 1000:
            return -10000
        
        r2 = calculate_r2(prices, y_pred)
        # Return a large negative value if R² is invalid to avoid it
        if not np.isfinite(r2):
            return -10000
        return r2
    
    # Use reasonable bounds for exponential growth rate
    # Growth rate is typically small (e.g., 0.0001 to 0.01 for daily growth)
    wide_min = -0.01  # Allow negative growth (decay)
    wide_max = 0.01   # Max daily growth rate (about 3.65x per year)
    
    try:
        result = minimize_scalar(lambda g: -r2_for_growth_rate(g), bounds=(wide_min, wide_max), method='bounded')
        optimal_growth = result.x
        optimal_r2 = -result.fun
        
        # If we hit a boundary, try wider
        if (abs(optimal_growth - wide_min) < 0.0001 or abs(optimal_growth - wide_max) < 0.0001):
            print(f"Growth rate hit boundary at {optimal_growth:.6f}. Trying wider range...")
            wider_min = -0.05
            wider_max = 0.05
            result = minimize_scalar(lambda g: -r2_for_growth_rate(g), bounds=(wider_min, wider_max), method='bounded')
            optimal_growth = result.x
            optimal_r2 = -result.fun
        
        # Validate the result makes sense
        test_pred = rainbow_model(days[-1], optimal_growth, first_price, first_days)
        if not np.isfinite(test_pred) or test_pred > np.max(prices) * 1000:
            print(f"Warning: Optimal growth rate produces unrealistic predictions. Using conservative default.")
            optimal_growth = 0.001  # 0.1% daily growth
            optimal_r2 = r2_for_growth_rate(optimal_growth)
            
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using default growth rate.")
        optimal_growth = 0.001  # 0.1% daily growth
        optimal_r2 = r2_for_growth_rate(optimal_growth)
    
    return optimal_growth, optimal_r2

def update_model(val):
    """Update the model when slider changes."""
    growth_rate = slider_growth_rate.val
    # Get first point values
    first_price = df_filtered['Price'].iloc[0]
    first_days = df_filtered['Days_Since_Genesis'].iloc[0]
    
    # Calculate model predictions using exponential growth
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, growth_rate, first_price, first_days)
    )
    
    # Cap unrealistic predictions to keep curve on the chart
    # Use tighter bounds based on actual data range
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    price_range = max_price - min_price
    
    # Cap at 5x the max price and 0.1x the min price (tighter bounds)
    max_reasonable = max_price + price_range * 4  # Allow up to 5x max, but based on range
    min_reasonable = max(min_price * 0.1, min_price - price_range * 0.5)  # Don't go too low
    
    df_filtered['Model_Price'] = df_filtered['Model_Price'].clip(lower=min_reasonable, upper=max_reasonable)
    
    # Update the model line on both subplots
    line1.set_ydata(df_filtered['Model_Price'])
    line2.set_ydata(df_filtered['Model_Price'])
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'], df_filtered['Model_Price'])
    
    # Update titles
    ax1.set_title(f'{CURRENCY_NAME} Rainbow Chart - Log Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    ax2.set_title(f'{CURRENCY_NAME} Rainbow Chart - Normal Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    
    # Update info text
    current_price = df_filtered['Price'].iloc[-1]
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    # Format prices appropriately (handle small values)
    if current_price < 1:
        current_str = f"${current_price:.4f}"
    elif current_price < 1000:
        current_str = f"${current_price:.2f}"
    else:
        current_str = f"${current_price:,.0f}"
    
    if model_fair_value < 1:
        model_str = f"${model_fair_value:.4f}"
    elif model_fair_value < 1000:
        model_str = f"${model_fair_value:.2f}"
    else:
        model_str = f"${model_fair_value:,.0f}"
    
    # Calculate annualized growth rate for display
    annualized_rate = growth_rate * 365.25 * 100  # Convert to percentage
    
    info_text = f"Growth Rate: {growth_rate:.6f} ({annualized_rate:.2f}%/yr) | R²: {r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
    
    ax_info.texts[0].set_text(info_text)
    
    # Redraw
    fig.canvas.draw_idle()

def main():
    """Create interactive rainbow chart."""
    
    print(f"{CURRENCY_NAME} Interactive Rainbow Chart")
    print("=" * 40)
    
    # Load data
    global df_filtered, ax1, ax2, line1, line2, ax_info, slider_growth_rate, fig
    df_filtered = load_data()
    if df_filtered is None:
        return
    
    # Find optimal growth rate that maximizes R²
    print("\nFinding optimal growth rate to maximize R²...")
    first_price = df_filtered['Price'].iloc[0]
    first_days = df_filtered['Days_Since_Genesis'].iloc[0]
    initial_growth_rate, optimal_r2 = find_optimal_growth_rate(df_filtered)
    print(f"Optimal growth rate: {initial_growth_rate:.6f} per day")
    annualized_rate = initial_growth_rate * 365.25 * 100
    print(f"Annualized growth rate: {annualized_rate:.2f}% per year")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    # Create figure with two subplots
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual price on both subplots
    ax1.semilogy(df_filtered['Date'], df_filtered['Price'], color='blue', linewidth=2, label=f'Actual {CURRENCY_NAME} Price')
    ax2.plot(df_filtered['Date'], df_filtered['Price'], color='blue', linewidth=2, label=f'Actual {CURRENCY_NAME} Price')
    
    # Calculate initial model prices using exponential growth
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, initial_growth_rate, first_price, first_days)
    )
    
    # Cap unrealistic predictions to keep curve on the chart
    # Use tighter bounds based on actual data range
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    price_range = max_price - min_price
    
    # Cap at 5x the max price and 0.1x the min price (tighter bounds)
    max_reasonable = max_price + price_range * 4  # Allow up to 5x max, but based on range
    min_reasonable = max(min_price * 0.1, min_price - price_range * 0.5)  # Don't go too low
    
    df_filtered['Model_Price'] = df_filtered['Model_Price'].clip(lower=min_reasonable, upper=max_reasonable)
    
    # Plot model line
    line1, = ax1.semilogy(df_filtered['Date'], df_filtered['Model_Price'], 
                color='black', linestyle='--', linewidth=2, label='Growth Model')
    line2, = ax2.plot(df_filtered['Date'], df_filtered['Model_Price'], 
            color='black', linestyle='--', linewidth=2, label='Growth Model')
    
    # Highlight current position
    current_price = df_filtered['Price'].iloc[-1]
    current_date = df_filtered['Date'].iloc[-1]
    
    # Format current price label appropriately
    if current_price < 1:
        current_label = f'Current: ${current_price:.4f}'
    elif current_price < 1000:
        current_label = f'Current: ${current_price:.2f}'
    else:
        current_label = f'Current: ${current_price:,.0f}'
    
    ax1.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=current_label)
    ax2.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=current_label)
    
    # Add model information
    initial_r2 = calculate_r2(df_filtered['Price'], df_filtered['Model_Price'])
    annualized_rate = initial_growth_rate * 365.25 * 100
    formula_text = f'Model: price = {first_price:.2f} × e^({initial_growth_rate:.6f} × (day - {first_days:.0f}))\nGrowth: {annualized_rate:.2f}%/yr | R² = {initial_r2:.4f}'
    ax1.text(0.02, 0.02, formula_text, transform=ax1.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    ax2.text(0.02, 0.02, formula_text, transform=ax2.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Formatting for both subplots
    for ax, ylabel_suffix in [(ax1, 'Log Scale'), (ax2, 'Normal Scale')]:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{CURRENCY_NAME} Price (USD) - {ylabel_suffix}', fontsize=12)
        ax.set_title(f'{CURRENCY_NAME} Rainbow Chart - {ylabel_suffix}\nR² = {initial_r2:.6f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Dynamic y-axis formatting based on data range
        min_price = min(df_filtered['Price'].min(), df_filtered['Model_Price'].min())
        max_price = max(df_filtered['Price'].max(), df_filtered['Model_Price'].max())
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
    
    # Create slider - wider range for more sensitivity
    ax_growth = plt.axes([0.1, 0.02, 0.8, 0.03])
    # Make the range wider for more sensitivity - ±150% of the growth rate
    slider_range = max(abs(initial_growth_rate) * 1.5, 0.01)  # ±150% of growth rate, or 1% minimum
    slider_min = initial_growth_rate - slider_range
    slider_max = initial_growth_rate + slider_range
    
    # Use a very small step size for fine-grained control
    # Divide range into many small steps so tiny movements = tiny changes
    step_size = (slider_max - slider_min) / 10000.0  # 10,000 steps for very fine control
    
    slider_growth_rate = Slider(ax_growth, 'Growth Rate (per day)', 
                         slider_min, 
                         slider_max, 
                         valinit=initial_growth_rate, 
                         valfmt='%.6f',
                         valstep=step_size)  # Very small step size = less sensitive
    slider_growth_rate.on_changed(update_model)
    
    # Add info panel at the top
    ax_info = plt.axes([0.02, 0.88, 0.96, 0.08])
    ax_info.axis('off')
    
    # Initial info text - format prices appropriately
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    # Format prices appropriately (handle small values)
    if current_price < 1:
        current_str = f"${current_price:.4f}"
    elif current_price < 1000:
        current_str = f"${current_price:.2f}"
    else:
        current_str = f"${current_price:,.0f}"
    
    if model_fair_value < 1:
        model_str = f"${model_fair_value:.4f}"
    elif model_fair_value < 1000:
        model_str = f"${model_fair_value:.2f}"
    else:
        model_str = f"${model_fair_value:,.0f}"
    
    # Calculate annualized growth rate for display
    annualized_rate = initial_growth_rate * 365.25 * 100
    
    info_text = f"Growth Rate: {initial_growth_rate:.6f} ({annualized_rate:.2f}%/yr) | R²: {initial_r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
    
    ax_info.text(0.02, 0.5, info_text, transform=ax_info.transAxes, fontsize=11, 
                verticalalignment='center', bbox=dict(facecolor='lightgray', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Instructions - format first price appropriately
    if first_price < 1:
        first_price_str = f"${first_price:.4f}"
    elif first_price < 1000:
        first_price_str = f"${first_price:.2f}"
    else:
        first_price_str = f"${first_price:,.0f}"
    
    instructions = f"Adjust slider to change growth rate. Model starts at first day price ({first_price_str})"
    
    ax_info.text(0.52, 0.5, instructions, transform=ax_info.transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the interactive chart
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    chart_file = os.path.join(OUTPUT_FOLDER, output_files['interactive_rainbow_png'])
    fig.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    # Flush to ensure file is written (important on Windows)
    plt.draw()
    gc.collect()
    print(f"Interactive chart saved: {chart_file}")
    
    print("\n" + "=" * 60)
    print(f"INTERACTIVE {CURRENCY_NAME.upper()} RAINBOW CHART")
    print("=" * 60)
    print("Use the slider to adjust the model parameters:")
    print(f"• Growth Rate: Controls exponential growth ({annualized_rate:.2f}% per year)")
    print("• Model starts at first day's price automatically")
    print("• Watch R² score to find the best fit")
    print(f"• Current deviation shows if {CURRENCY_NAME} is over/undervalued")
    print(f"\n💡 TIP: Note the optimal growth rate ({initial_growth_rate:.6f}) for future reference")
    
    plt.show()

if __name__ == "__main__":
    main()

