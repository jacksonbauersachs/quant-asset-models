"""
Interactive rainbow chart with fixed power and slope - only intercept is adjustable.
Use this after finding optimal power and slope values in interactive_rainbow_v2.py.

Set POWER and FIXED_GROWTH_RATE (slope) in config.py, then run this script.
Only the intercept slider will be available to move the curve up/down.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import sys

# Import configuration
try:
    from config import (
        CURRENCY_NAME, DATA_FOLDER, HISTORICAL_DATA_FILE,
        OUTPUT_FOLDER, START_DATE, get_output_files, validate_config
    )
    try:
        from config import GENESIS_DATE
    except ImportError:
        GENESIS_DATE = None
    
    # Get fixed power and slope from config
    try:
        from config import POWER
    except ImportError:
        POWER = 1.0  # Default to linear
    
    try:
        from config import FIXED_GROWTH_RATE
        FIXED_SLOPE = FIXED_GROWTH_RATE  # This is the slope
    except ImportError:
        FIXED_SLOPE = None
        print("Warning: FIXED_GROWTH_RATE not set in config.py!")
        print("Please set FIXED_GROWTH_RATE (slope) in config.py before running this script.")
        sys.exit(1)
        
except ImportError:
    print("Error: config.py not found! Make sure you're running from the log_model folder.")
    sys.exit(1)

# Fixed base for easier interpretation
BASE = 10.0

def load_data():
    """Load and prepare historical price data."""
    # Validate config first
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    
    # Get data file path - handle relative paths (go up one level if running from log_model folder)
    if os.path.basename(os.getcwd()) == 'log_model':
        data_file = os.path.join('..', DATA_FOLDER, HISTORICAL_DATA_FILE)
    else:
        data_file = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILE)
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found - {data_file}")
        return None
    
    try:
        df = pd.read_csv(data_file)
        
        # Handle different date formats
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            print("Error: Could not find date column in CSV")
            return None
        
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Handle price column (remove commas if present)
        price_col = None
        for col in df.columns:
            if 'price' in col.lower() and 'close' not in col.lower():
                price_col = col
                break
        if price_col is None:
            for col in df.columns:
                if 'close' in col.lower():
                    price_col = col
                    break
        
        if price_col is None:
            print("Error: Could not find price column in CSV")
            return None
        
        # Remove commas and convert to float
        if df[price_col].dtype == 'object':
            df['Price'] = df[price_col].str.replace(',', '').astype(float)
        else:
            df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        df = df.dropna(subset=['Price', 'Date'])
        df = df[df['Price'] > 0]  # Remove zero/negative prices
        
        # Auto-detect genesis date if not provided
        if GENESIS_DATE is None:
            genesis_date = df['Date'].min()
            print(f"Auto-detected genesis date from data: {genesis_date.date()}")
        else:
            genesis_date = pd.to_datetime(GENESIS_DATE)
        
        # Auto-detect start date if not provided
        if START_DATE is None or START_DATE == "":
            start_date = df['Date'].min()
            print(f"Auto-detected start date from data: {start_date.date()}")
        else:
            start_date = pd.to_datetime(START_DATE)
        
        # Filter data from start date
        df = df[df['Date'] >= start_date].copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate days since genesis
        df['Days_Since_Genesis'] = (df['Date'] - genesis_date).dt.days
        df['Days_Since_Genesis'] = df['Days_Since_Genesis'].clip(lower=1)  # Ensure at least 1
        
        print(f"{CURRENCY_NAME} data loaded: {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Price range: ${df['Price'].min():,.2f} to ${df['Price'].max():,.2f}")
        
        return df[['Date', 'Price', 'Days_Since_Genesis']]
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def rainbow_model(x, slope, intercept, base=10, power=1.0):
    """Growth model: price = base^(slope * (days^power) + intercept)"""
    x = np.maximum(x, 1)
    exponent = slope * (x**power) + intercept
    exponent = np.clip(exponent, -30, 30)  # Prevent overflow
    result = base**exponent
    result = np.where(np.isfinite(result), result, np.nan)
    return result

def calculate_r2(y_actual, y_pred, base=10):
    """Calculate R² score in log space."""
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    if np.sum(valid_mask) < 2:
        return 0.0
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    log_base = np.log(base)
    log_actual = np.log(y_actual_valid) / log_base
    log_pred = np.log(y_pred_valid) / log_base
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def calculate_intercept_for_first_point(df, slope, power, base=10.0):
    """Calculate intercept so model passes through first data point."""
    first_price = df['Price'].iloc[0]
    first_days = df['Days_Since_Genesis'].iloc[0]
    log_base = np.log(base)
    intercept = np.log(first_price) / log_base - slope * (first_days**power)
    return intercept

def update_model(val):
    """Update the model when intercept slider changes."""
    intercept = slider_intercept.val
    
    # Use fixed slope and power from config
    slope = FIXED_SLOPE
    power = POWER
    
    # Calculate model predictions
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, slope, intercept, BASE, power)
    )
    
    # Cap unrealistic predictions
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    max_reasonable = max_price * 2.0
    min_reasonable = min_price * 0.5
    df_filtered['Model_Price'] = df_filtered['Model_Price'].clip(lower=min_reasonable, upper=max_reasonable)
    
    # Update plots
    line1.set_ydata(df_filtered['Model_Price'])
    line2.set_ydata(df_filtered['Model_Price'])
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'].values, df_filtered['Model_Price'].values, BASE)
    
    # Update titles
    ax1.set_title(f'{CURRENCY_NAME} Rainbow Chart - Log Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    ax2.set_title(f'{CURRENCY_NAME} Rainbow Chart - Normal Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    
    # Update formula text
    formula_text = f'Model: price = {BASE:.0f}^({slope:.6f} × (day^{power:.2f}) + {intercept:.6f})\nR² = {r2:.4f}'
    formula_text1.set_text(formula_text)
    formula_text2.set_text(formula_text)
    
    # Update info text
    current_price = df_filtered['Price'].iloc[-1]
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
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
    
    info_text = f"Base: {BASE:.0f} (fixed) | Power: {power:.2f} (fixed) | Slope: {slope:.6f} (fixed) | Intercept: {intercept:.6f} | R²: {r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
    ax_info.texts[0].set_text(info_text)
    
    # Update y-axis limits
    actual_min_price = df_filtered['Price'].min()
    actual_max_price = df_filtered['Price'].max()
    actual_price_range = actual_max_price - actual_min_price
    
    if ax2:
        padding = actual_price_range * 0.1
        ax2.set_ylim(max(0, actual_min_price - padding), actual_max_price + padding)
    if ax1:
        ax1.set_ylim(max(0.01, actual_min_price * 0.5), actual_max_price * 2)
    
    fig.canvas.draw_idle()

def main():
    """Create interactive rainbow chart with only intercept adjustable."""
    print(f"{CURRENCY_NAME} Interactive Rainbow Chart (Intercept Only)")
    print("=" * 40)
    print(f"Using fixed parameters from config:")
    print(f"  Power: {POWER:.2f}")
    print(f"  Slope: {FIXED_SLOPE:.6f}")
    print(f"  Base: {BASE:.0f}")
    
    global df_filtered, ax1, ax2, line1, line2, ax_info, slider_intercept, fig, formula_text1, formula_text2
    df_filtered = load_data()
    if df_filtered is None:
        return
    
    # Calculate initial intercept to anchor at first point
    initial_intercept = calculate_intercept_for_first_point(df_filtered, FIXED_SLOPE, POWER, BASE)
    
    print(f"\nInitial intercept (anchored to first point): {initial_intercept:.6f}")
    
    # Calculate initial model prices
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, FIXED_SLOPE, initial_intercept, BASE, POWER)
    )
    
    # Cap predictions
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    max_reasonable = max_price * 2.0
    min_reasonable = min_price * 0.5
    df_filtered['Model_Price'] = df_filtered['Model_Price'].clip(lower=min_reasonable, upper=max_reasonable)
    
    # Calculate initial R²
    initial_r2 = calculate_r2(df_filtered['Price'].values, df_filtered['Model_Price'].values, BASE)
    
    # Create figure with two subplots
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual prices
    ax1.semilogy(df_filtered['Date'], df_filtered['Price'], 
                color='blue', linewidth=2, label='Actual Price')
    ax2.plot(df_filtered['Date'], df_filtered['Price'], 
            color='blue', linewidth=2, label='Actual Price')
    
    # Plot model line
    line1, = ax1.semilogy(df_filtered['Date'], df_filtered['Model_Price'], 
                color='black', linestyle='--', linewidth=2, label='Growth Model')
    line2, = ax2.plot(df_filtered['Date'], df_filtered['Model_Price'], 
            color='black', linestyle='--', linewidth=2, label='Growth Model')
    
    # Highlight current position
    current_price = df_filtered['Price'].iloc[-1]
    current_date = df_filtered['Date'].iloc[-1]
    
    if current_price < 1:
        current_label = f'Current: ${current_price:.4f}'
    elif current_price < 1000:
        current_label = f'Current: ${current_price:.2f}'
    else:
        current_label = f'Current: ${current_price:,.0f}'
    
    ax1.scatter(current_date, current_price, color='red', s=100, zorder=10, label=current_label)
    ax2.scatter(current_date, current_price, color='red', s=100, zorder=10, label=current_label)
    
    # Format both subplots
    for ax, ylabel_suffix in [(ax1, 'Log Scale'), (ax2, 'Normal Scale')]:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{CURRENCY_NAME} Price (USD) - {ylabel_suffix}', fontsize=12)
        ax.set_title(f'{CURRENCY_NAME} Rainbow Chart - {ylabel_suffix}\nR² = {initial_r2:.6f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Dynamic y-axis formatting
        min_price = df_filtered['Price'].min()
        max_price = df_filtered['Price'].max()
        price_range = max_price - min_price
        
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
        
        from matplotlib.ticker import MaxNLocator
        if ax == ax2:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
            padding = price_range * 0.1
            ax.set_ylim(max(0, min_price - padding), max_price + padding)
        else:
            ax.set_ylim(max(0.01, min_price * 0.5), max_price * 2)
        
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add formula text
    formula_text = f'Model: price = {BASE:.0f}^({FIXED_SLOPE:.6f} × (day^{POWER:.2f}) + {initial_intercept:.6f})\nR² = {initial_r2:.4f}'
    formula_text1 = ax1.text(0.02, 0.02, formula_text, transform=ax1.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    formula_text2 = ax2.text(0.02, 0.02, formula_text, transform=ax2.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Create slider for intercept only
    ax_intercept = plt.axes([0.1, 0.02, 0.8, 0.03])
    # Use ±10 range around initial intercept
    intercept_range = 10.0
    intercept_min = initial_intercept - intercept_range
    intercept_max = initial_intercept + intercept_range
    intercept_step = (intercept_max - intercept_min) / 1000.0
    
    slider_intercept = Slider(ax_intercept, 'Intercept (Move Up/Down)', 
                             intercept_min, 
                             intercept_max, 
                             valinit=initial_intercept, 
                             valfmt='%.6f',
                             valstep=intercept_step)
    slider_intercept.on_changed(update_model)
    
    # Add info panel
    ax_info = plt.axes([0.02, 0.88, 0.96, 0.08])
    ax_info.axis('off')
    
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
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
    
    info_text = f"Base: {BASE:.0f} (fixed) | Power: {POWER:.2f} (fixed) | Slope: {FIXED_SLOPE:.6f} (fixed) | Intercept: {initial_intercept:.6f} | R²: {initial_r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes, 
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='lightgray', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save chart
    output_files = get_output_files()
    # Handle output folder path
    if os.path.basename(os.getcwd()) == 'log_model':
        chart_file = os.path.join('.', output_files['interactive_rainbow_png'])
    else:
        chart_file = os.path.join(OUTPUT_FOLDER, output_files['interactive_rainbow_png'])
    os.makedirs(os.path.dirname(chart_file), exist_ok=True)
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nInteractive chart saved: {chart_file}")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE RAINBOW CHART (INTERCEPT ONLY)")
    print("=" * 60)
    print("Power and Slope are fixed from config.py")
    print("Use the Intercept slider to move the curve up/down")
    print("This lets you align the curve with low points or other reference points")
    
    plt.show()

if __name__ == "__main__":
    main()

