"""
Simplified interactive rainbow chart for logarithmic growth model.
Two-step approach:
1. Adjust power (acceleration/shape) with slider
2. Slope is optimized analytically using linear regression
Base is fixed at 10.0 for easier interpretation.
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

def optimize_slope_analytically(df, power, base=10.0):
    """
    Optimize slope and intercept analytically using linear regression.
    For fixed power and base, the model becomes linear in log space:
    log(price) = slope * (days^power) + intercept
    """
    days = df['Days_Since_Genesis'].values
    prices = df['Price'].values
    first_price = prices[0]
    first_days = days[0]
    
    # Transform: z = days^power
    z = days**power
    
    # Transform to log space: log(price) = slope * z + intercept
    log_base = np.log(base)
    log_prices = np.log(prices) / log_base
    
    # Anchor intercept to first point:
    # intercept = log(first_price) / log_base - slope * (first_days^power)
    # This gives us: log(price) = slope * z + log(first_price)/log_base - slope * (first_days^power)
    # Rearranging: log(price) - log(first_price)/log_base = slope * (z - first_days^power)
    
    z_centered = z - (first_days**power)
    log_prices_centered = log_prices - (np.log(first_price) / log_base)
    
    # Linear regression: log_prices_centered = slope * z_centered
    # slope = sum(z_centered * log_prices_centered) / sum(z_centered^2)
    denominator = np.sum(z_centered**2)
    if denominator == 0:
        slope = 0.0
    else:
        slope = np.sum(z_centered * log_prices_centered) / denominator
    
    # Calculate intercept to anchor at first point
    intercept = np.log(first_price) / log_base - slope * (first_days**power)
    
    # Calculate R²
    y_pred = rainbow_model(days, slope, intercept, base, power)
    r2 = calculate_r2(prices, y_pred, base)
    
    return slope, intercept, r2

def update_model(val):
    """Update the model when power slider changes."""
    power = slider_power.val
    
    # Optimize slope analytically for this power
    slope, intercept, r2 = optimize_slope_analytically(df_filtered, power, BASE)
    
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
    
    info_text = f"Base: {BASE:.0f} (fixed) | Power: {power:.2f} | Slope: {slope:.6f} | Intercept: {intercept:.6f} | R²: {r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
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
    """Create interactive rainbow chart."""
    print(f"{CURRENCY_NAME} Interactive Rainbow Chart (Simplified Model)")
    print("=" * 40)
    
    global df_filtered, ax1, ax2, line1, line2, ax_info, slider_power, fig, formula_text1, formula_text2
    df_filtered = load_data()
    if df_filtered is None:
        return
    
    # Initial power value (start with 1.0 for linear, user can adjust)
    initial_power = 1.0
    
    # Optimize slope for initial power
    initial_slope, initial_intercept, initial_r2 = optimize_slope_analytically(df_filtered, initial_power, BASE)
    
    print(f"\nInitial parameters (power={initial_power:.2f}):")
    print(f"  Slope: {initial_slope:.6f}")
    print(f"  Intercept: {initial_intercept:.6f}")
    print(f"  R²: {initial_r2:.6f}")
    
    # Calculate initial model prices
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, initial_slope, initial_intercept, BASE, initial_power)
    )
    
    # Cap predictions
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    max_reasonable = max_price * 2.0
    min_reasonable = min_price * 0.5
    df_filtered['Model_Price'] = df_filtered['Model_Price'].clip(lower=min_reasonable, upper=max_reasonable)
    
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
    formula_text = f'Model: price = {BASE:.0f}^({initial_slope:.6f} × (day^{initial_power:.2f}) + {initial_intercept:.6f})\nR² = {initial_r2:.4f}'
    formula_text1 = ax1.text(0.02, 0.02, formula_text, transform=ax1.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    formula_text2 = ax2.text(0.02, 0.02, formula_text, transform=ax2.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Create slider for power (acceleration)
    ax_power = plt.axes([0.1, 0.02, 0.8, 0.03])
    power_min = -2.0  # Allow negative values for inverse relationships
    power_max = 2.0
    power_step = (power_max - power_min) / 1000.0
    
    slider_power = Slider(ax_power, 'Power (Acceleration)', 
                         power_min, 
                         power_max, 
                         valinit=initial_power, 
                         valfmt='%.2f',
                         valstep=power_step)
    slider_power.on_changed(update_model)
    
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
    
    info_text = f"Base: {BASE:.0f} (fixed) | Power: {initial_power:.2f} | Slope: {initial_slope:.6f} | Intercept: {initial_intercept:.6f} | R²: {initial_r2:.6f} | Current: {current_str} | Model: {model_str} | Deviation: {deviation:+.1f}%"
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
    print("INTERACTIVE RAINBOW CHART (SIMPLIFIED MODEL)")
    print("=" * 60)
    print("Use the Power slider to adjust the acceleration pattern:")
    print("• Power < 1: Fast early growth, then flattens (log-like)")
    print("• Power = 1: Linear growth")
    print("• Power > 1: Slow early, then accelerates")
    print("• Slope is automatically optimized for each power value")
    print("• Base is fixed at 10.0 for easier interpretation")
    print("\nWatch R² score to find the best fit!")
    print(f"\n💡 TIP: Note the optimal power value for future reference")
    
    plt.show()

if __name__ == "__main__":
    main()

