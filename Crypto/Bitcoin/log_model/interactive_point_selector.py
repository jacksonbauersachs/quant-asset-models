"""
Interactive point selector for rainbow chart model.
Click on the price graph to select specific points, then click "Calculate R²" 
to find the optimal parameters that fit just those selected points.

This lets you use specific reference points (like lows) instead of all prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import minimize
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

# Fixed base
BASE = 10.0

# Global variables for selected points
selected_points = []  # List of (date, price, days) tuples
selected_markers = []  # List of marker objects

def load_data():
    """Load and prepare historical price data."""
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    
    # Get data file path - handle relative paths
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
        
        # Handle price column
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
        df = df[df['Price'] > 0]
        
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
        df['Days_Since_Genesis'] = df['Days_Since_Genesis'].clip(lower=1)
        
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
    exponent = np.clip(exponent, -30, 30)
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

def find_closest_data_point(event, df):
    """Find the closest data point to the click location."""
    if event.inaxes != ax:
        return None
    
    click_date = pd.to_datetime(event.xdata, unit='D', origin='1970-01-01')
    click_price = event.ydata
    
    # Find closest point by date
    df['Date_Diff'] = (df['Date'] - click_date).abs()
    closest_idx = df['Date_Diff'].idxmin()
    
    return closest_idx

def on_click(event):
    """Handle mouse clicks to add/remove points."""
    if event.inaxes != ax or event.button != 1:  # Left click only
        return
    
    # Don't process clicks if user is using zoom/pan tools
    if event.inaxes is not None and hasattr(event.inaxes.figure.canvas, 'toolbar'):
        toolbar = event.inaxes.figure.canvas.toolbar
        if toolbar is not None and toolbar.mode != '':
            return  # User is using a toolbar tool (zoom, pan, etc.)
    
    closest_idx = find_closest_data_point(event, df_filtered)
    if closest_idx is None:
        return
    
    point = df_filtered.iloc[closest_idx]
    point_tuple = (point['Date'], point['Price'], point['Days_Since_Genesis'])
    
    # Check if clicking on an existing selected point (remove it)
    clicked_on_point = False
    for i, (sel_date, sel_price, sel_days) in enumerate(selected_points):
        # Check if click is close to this point (more lenient when zoomed in)
        date_diff = abs((sel_date - point['Date']).total_seconds())
        price_diff = abs(np.log10(sel_price) - np.log10(point['Price']))
        # Use a threshold based on current zoom level
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        date_range = (xlim[1] - xlim[0]) * 86400  # Convert to seconds
        price_range_log = np.log10(ylim[1]) - np.log10(ylim[0])
        
        # Threshold: 2% of visible range
        date_threshold = date_range * 0.02
        price_threshold = price_range_log * 0.02
        
        if date_diff < date_threshold and price_diff < price_threshold:
            # Remove this point
            selected_points.pop(i)
            marker = selected_markers.pop(i)
            marker.remove()
            clicked_on_point = True
            break
    
    if clicked_on_point:
        update_status_text()
        fig.canvas.draw()
        return
    
    # Add new point
    selected_points.append(point_tuple)
    marker = ax.scatter(point['Date'], point['Price'], 
                       color='red', s=30, zorder=10, marker='o', 
                       edgecolors='black', linewidths=1)
    selected_markers.append(marker)
    
    # Update status text
    update_status_text()
    fig.canvas.draw()

def update_status_text():
    """Update the status text showing number of selected points."""
    status_text.set_text(f"Selected points: {len(selected_points)} | Use zoom tool or mouse wheel to zoom in for precise placement | Click 'Calculate R²'")
    fig.canvas.draw_idle()

def optimize_for_selected_points():
    """Optimize power, slope, and intercept to fit selected points."""
    if len(selected_points) < 3:
        print(f"Error: Need at least 3 points selected. Currently have {len(selected_points)}")
        status_text.set_text(f"Error: Need at least 3 points! Currently have {len(selected_points)}")
        fig.canvas.draw_idle()
        return
    
    # Extract selected data
    selected_days = np.array([p[2] for p in selected_points])
    selected_prices = np.array([p[1] for p in selected_points])
    
    print(f"\nOptimizing for {len(selected_points)} selected points...")
    print(f"Selected days: {selected_days}")
    print(f"Selected prices: {selected_prices}")
    
    # Calculate smart initial guess based on selected points
    first_price = selected_prices[0]
    first_days = selected_days[0]
    last_price = selected_prices[-1]
    last_days = selected_days[-1]
    
    log_base = np.log(BASE)
    
    # Estimate power - try a few values and pick best
    test_powers = [0.5, 1.0, 1.5]
    best_power = 1.0
    best_r2 = -1000
    
    for test_power in test_powers:
        try:
            z = selected_days**test_power
            log_prices = np.log(selected_prices) / log_base
            z_centered = z - z[0]
            log_prices_centered = log_prices - log_prices[0]
            
            if np.sum(z_centered**2) > 0:
                test_slope = np.sum(z_centered * log_prices_centered) / np.sum(z_centered**2)
                test_intercept = np.log(first_price) / log_base - test_slope * (first_days**test_power)
                test_pred = rainbow_model(selected_days, test_slope, test_intercept, BASE, test_power)
                test_r2 = calculate_r2(selected_prices, test_pred, BASE)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_power = test_power
        except:
            continue
    
    # Use best power, calculate slope and intercept
    z = selected_days**best_power
    log_prices = np.log(selected_prices) / log_base
    z_centered = z - z[0]
    log_prices_centered = log_prices - log_prices[0]
    
    if np.sum(z_centered**2) > 0:
        estimated_slope = np.sum(z_centered * log_prices_centered) / np.sum(z_centered**2)
    else:
        estimated_slope = 0.5
    
    estimated_intercept = np.log(first_price) / log_base - estimated_slope * (first_days**best_power)
    
    initial_guess = [best_power, estimated_slope, estimated_intercept]
    print(f"Initial guess: power={best_power:.3f}, slope={estimated_slope:.3f}, intercept={estimated_intercept:.3f}")
    
    def r2_for_params(params):
        """Calculate R² for given power, slope, intercept."""
        power, slope, intercept = params
        
        # Tighter bounds checking
        if power < -2.0 or power > 2.0 or slope < -100 or slope > 100 or abs(intercept) > 50:
            return 10000
        
        try:
            y_pred = rainbow_model(selected_days, slope, intercept, BASE, power)
            
            if not np.all(np.isfinite(y_pred)) or np.any(y_pred <= 0):
                return 10000
            
            r2 = calculate_r2(selected_prices, y_pred, BASE)
            if not np.isfinite(r2) or r2 < -1 or r2 > 1:
                return 10000
            
            return -r2
        except:
            return 10000
    
    # Tighter bounds
    bounds = [(-2.0, 2.0), (-100, 100), (-50, 50)]
    
    try:
        result = minimize(r2_for_params, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Check if optimization succeeded
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        optimal_power = result.x[0]
        optimal_slope = result.x[1]
        optimal_intercept = result.x[2]
        optimal_r2 = -result.fun
        
        print(f"\nOptimization Results:")
        print(f"  Power: {optimal_power:.6f}")
        print(f"  Slope: {optimal_slope:.6f}")
        print(f"  Intercept: {optimal_intercept:.6f}")
        print(f"  R²: {optimal_r2:.6f}")
        print(f"  Success: {result.success}")
        
        # Plot the fitted curve
        plot_fitted_curve(optimal_power, optimal_slope, optimal_intercept, optimal_r2)
        
        # Update status
        status_text.set_text(f"R² = {optimal_r2:.6f} | Power: {optimal_power:.6f} | Slope: {optimal_slope:.6f} | Intercept: {optimal_intercept:.6f}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        status_text.set_text(f"Optimization failed: {e}")
    
    fig.canvas.draw_idle()

def plot_fitted_curve(power, slope, intercept, r2):
    """Plot the fitted curve on the graph."""
    global curve_line
    # Remove existing curve if any
    if curve_line is not None:
        try:
            curve_line.remove()
        except:
            pass
    
    # Save current y-axis limits to preserve view
    current_ylim = ax.get_ylim()
    
    # Calculate curve for full date range
    days = df_filtered['Days_Since_Genesis'].values
    model_prices = rainbow_model(days, slope, intercept, BASE, power)
    
    # Ensure prices are valid and within reasonable range
    model_prices = np.where(np.isfinite(model_prices) & (model_prices > 0), model_prices, np.nan)
    
    # Check if curve is visible
    valid_prices = model_prices[np.isfinite(model_prices)]
    if len(valid_prices) == 0:
        print("Warning: Model produces no valid prices!")
        return
    
    # Plot curve on log scale
    curve_line, = ax.semilogy(df_filtered['Date'], model_prices, 
                             color='green', linestyle='--', linewidth=2, 
                             label=f'Fitted Model (R² = {r2:.4f})', zorder=5)
    
    # Restore y-axis limits to preserve view
    ax.set_ylim(current_ylim)
    
    # Update legend
    ax.legend(loc='upper left')
    fig.canvas.draw()

def clear_last_point(event):
    """Remove the last selected point."""
    global selected_points, selected_markers
    if len(selected_points) > 0:
        # Remove last point
        selected_points.pop()
        marker = selected_markers.pop()
        marker.remove()
        
        # Remove curve if exists (need to recalculate)
        global curve_line
        if 'curve_line' in globals() and curve_line is not None:
            curve_line.remove()
            curve_line = None
        
        update_status_text()
        print(f"\nRemoved last point. {len(selected_points)} points remaining.")
        fig.canvas.draw_idle()
    else:
        print("\nNo points to remove.")

def clear_points(event):
    """Clear all selected points."""
    global selected_points, selected_markers
    selected_points = []
    for marker in selected_markers:
        marker.remove()
    selected_markers = []
    
    # Remove curve if exists
    global curve_line
    if 'curve_line' in globals() and curve_line is not None:
        curve_line.remove()
        curve_line = None
    
    update_status_text()
    print("\nCleared all selected points")

def main():
    """Create interactive point selector."""
    print(f"{CURRENCY_NAME} Interactive Point Selector")
    print("=" * 40)
    print("Instructions:")
    print("1. Use the zoom tool (magnifying glass icon) or mouse wheel to zoom in for precise placement")
    print("2. Left-click on the price graph to select points (click again to remove)")
    print("3. Use the pan tool (hand icon) to move around when zoomed in")
    print("4. Select at least 3 points")
    print("5. Click 'Calculate R²' to find optimal parameters")
    print("6. Click 'Clear Points' to start over")
    print("\nTip: Zoom in on local maxima/minima for more precise point selection!")
    
    global df_filtered, ax, fig, status_text, curve_line
    curve_line = None
    
    df_filtered = load_data()
    if df_filtered is None:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot price data on log scale
    ax.semilogy(df_filtered['Date'], df_filtered['Price'], 
               color='blue', linewidth=2, label='Actual Price', alpha=0.7)
    
    # Format axes
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{CURRENCY_NAME} Price (USD)', fontsize=12)
    ax.set_title(f'{CURRENCY_NAME} - Click to Select Points for R² Calculation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Dynamic price formatting
    min_price = df_filtered['Price'].min()
    max_price = df_filtered['Price'].max()
    
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
    ax.legend(loc='upper left')
    
    # Add status text
    status_text = ax.text(0.02, 0.98, "Selected points: 0 | Click on chart to add/remove points, then click 'Calculate R²'", 
                         transform=ax.transAxes, fontsize=10, 
                         verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add buttons
    ax_calculate = plt.axes([0.7, 0.02, 0.12, 0.04])
    btn_calculate = Button(ax_calculate, 'Calculate R²')
    btn_calculate.on_clicked(lambda x: optimize_for_selected_points())
    
    ax_clear_last = plt.axes([0.83, 0.02, 0.08, 0.04])
    btn_clear_last = Button(ax_clear_last, 'Undo')
    btn_clear_last.on_clicked(clear_last_point)
    
    ax_clear = plt.axes([0.92, 0.02, 0.06, 0.04])
    btn_clear = Button(ax_clear, 'Clear All')
    btn_clear.on_clicked(clear_points)
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Enable zoom with mouse wheel
    def on_scroll(event):
        """Handle mouse wheel zoom."""
        if event.inaxes != ax:
            return
        
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
        
        # Ensure valid ydata
        if ydata <= 0 or not np.isfinite(ydata):
            return
        
        # Ensure valid current limits
        if cur_ylim[0] <= 0 or cur_ylim[1] <= 0:
            return
        
        # Zoom factor
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        # Calculate new limits centered on mouse position
        new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        
        # Handle log scale for y-axis
        try:
            log_ydata = np.log10(ydata)
            log_ylim_low = np.log10(cur_ylim[0])
            log_ylim_high = np.log10(cur_ylim[1])
            log_height = log_ylim_high - log_ylim_low
            new_log_height = log_height * zoom_factor
            
            # Calculate relative position in log space
            if log_height > 0:
                rely = (log_ydata - log_ylim_low) / log_height
            else:
                rely = 0.5
            
            # Calculate new log limits
            new_log_ylim_low = log_ydata - new_log_height * rely
            new_log_ylim_high = log_ydata + new_log_height * (1 - rely)
            
            # Convert back to linear scale with bounds checking
            new_ylim_low = 10**np.clip(new_log_ylim_low, -10, 10)
            new_ylim_high = 10**np.clip(new_log_ylim_high, -10, 10)
            
            # Ensure valid, finite values
            if not (np.isfinite(new_ylim_low) and np.isfinite(new_ylim_high) and 
                    new_ylim_low > 0 and new_ylim_high > new_ylim_low):
                return
            
            new_ylim = [new_ylim_low, new_ylim_high]
        except (ValueError, OverflowError):
            return
        
        # Calculate new x limits
        xlim_range = cur_xlim[1] - cur_xlim[0]
        if xlim_range > 0:
            relx = (xdata - cur_xlim[0]) / xlim_range
        else:
            relx = 0.5
        
        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        
        # Ensure valid x limits
        if not (np.isfinite(new_xlim[0]) and np.isfinite(new_xlim[1]) and 
                new_xlim[1] > new_xlim[0]):
            return
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Set figure title (works across all backends)
    try:
        fig.canvas.manager.set_window_title(f'{CURRENCY_NAME} Point Selector - Use zoom tool or mouse wheel to zoom in')
    except AttributeError:
        # Some backends don't support set_window_title, that's okay
        pass
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

