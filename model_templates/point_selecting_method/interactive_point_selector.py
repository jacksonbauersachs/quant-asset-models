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
from scipy.interpolate import PchipInterpolator
import os
import sys
import json
from datetime import datetime

# Import configuration
try:
    from config import (
        CURRENCY_NAME, CURRENCY_NAME_LOWER, DATA_FOLDER, HISTORICAL_DATA_FILE,
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

# Global variables for curves
curve_line = None  # Parametric model curve
powerlaw_line = None  # Power law curve
exponential_line = None  # Exponential model curve
pchip_line = None  # PCHIP interpolation curve
current_model_type = None  # 'parametric', 'powerlaw', 'exponential', or 'pchip'

# Global variable for scale
use_log_scale = True  # True for log scale, False for normal scale
ax = None  # Will be set in main()
fig = None  # Will be set in main()
price_line = None  # Main price data line
df_filtered = None  # Will be set in main()
status_text = None  # Will be set in main()

# Store last model parameters for re-plotting after scale toggle
last_model_params = None  # Dict with model_type and parameters

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
        # Save points
        save_points()
        update_status_text()
        fig.canvas.draw()
        return
    
    # Add new point
    selected_points.append(point_tuple)
    marker = ax.scatter(point['Date'], point['Price'], 
                       color='red', s=30, zorder=10, marker='o', 
                       edgecolors='black', linewidths=1)
    selected_markers.append(marker)
    
    # Save points
    save_points()
    
    # Update status text
    update_status_text()
    fig.canvas.draw()

def get_save_file_path():
    """Get the path to save selected points."""
    # ONLY use script directory - never OUTPUT_FOLDER or other locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_file = os.path.join(script_dir, f'{CURRENCY_NAME_LOWER}_selected_points.json')
    return save_file

def get_model_save_file_path():
    """Get the path to save last selected model."""
    # Save in the same directory as the script (where points are saved)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_file = os.path.join(script_dir, f'{CURRENCY_NAME_LOWER}_last_model.json')
    return save_file

def save_last_model(model_type, **params):
    """Save the last selected model type and parameters to JSON file."""
    save_file = get_model_save_file_path()
    try:
        model_data = {
            'model_type': model_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add model-specific parameters
        if model_type == 'powerlaw' and 'a' in params and 'b' in params and 'first_day_offset' in params:
            model_data['powerlaw_a'] = float(params['a'])
            model_data['powerlaw_b'] = float(params['b'])
            model_data['powerlaw_first_day_offset'] = float(params['first_day_offset'])
        elif model_type == 'exponential' and 'a' in params and 'b' in params and 'first_day_offset' in params:
            model_data['exponential_a'] = float(params['a'])
            model_data['exponential_b'] = float(params['b'])
            model_data['exponential_first_day_offset'] = float(params['first_day_offset'])
        elif model_type == 'parametric' and 'power' in params and 'slope' in params and 'intercept' in params:
            model_data['power'] = float(params['power'])
            model_data['slope'] = float(params['slope'])
            model_data['intercept'] = float(params['intercept'])
        
        with open(save_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Saved last model type: {model_type} to {save_file}")
    except Exception as e:
        print(f"Error saving model type: {e}")

def load_last_model():
    """Load the last selected model type from JSON file."""
    save_file = get_model_save_file_path()
    if not os.path.exists(save_file):
        return None
    
    try:
        with open(save_file, 'r') as f:
            model_data = json.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model type: {e}")
        return None

def save_points():
    """Save selected points to JSON file."""
    if len(selected_points) == 0:
        return
    
    save_file = get_save_file_path()
    try:
        # Convert dates to strings for JSON serialization
        points_data = []
        for date, price, days in selected_points:
            points_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price),
                'days': int(days)
            })
        
        with open(save_file, 'w') as f:
            json.dump(points_data, f, indent=2)
        print(f"Saved {len(selected_points)} points to {save_file}")
    except Exception as e:
        print(f"Error saving points: {e}")

def load_points():
    """Load selected points from JSON file."""
    save_file = get_save_file_path()
    if not os.path.exists(save_file):
        return []
    
    try:
        with open(save_file, 'r') as f:
            points_data = json.load(f)
        
        loaded_points = []
        for point in points_data:
            date = pd.to_datetime(point['date'])
            price = float(point['price'])
            days = int(point['days'])
            loaded_points.append((date, price, days))
        
        print(f"Loaded {len(loaded_points)} points from {save_file}")
        return loaded_points
    except Exception as e:
        print(f"Error loading points: {e}")
        return []

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
    print(f"Data filtered from START_DATE: {START_DATE if START_DATE else 'None (using all data)'}")
    
    # Calculate smart initial guess based on selected points
    first_price = selected_prices[0]
    first_days = selected_days[0]
    last_price = selected_prices[-1]
    last_days = selected_days[-1]
    
    log_base = np.log(BASE)
    
    # Estimate power - try a wider range of values for better initial guess
    test_powers = np.linspace(0.1, 2.0, 20)
    best_power = 1.0
    best_sse = 1e10
    
    for test_power in test_powers:
        try:
            z = selected_days**test_power
            log_prices = np.log(selected_prices) / log_base
            z_centered = z - z[0]
            log_prices_centered = log_prices - log_prices[0]
            
            if np.sum(z_centered**2) > 0:
                test_slope = np.sum(z_centered * log_prices_centered) / np.sum(z_centered**2)
                # Constrain intercept to pass through first point
                test_intercept = np.log(first_price) / log_base - test_slope * (first_days**test_power)
                test_pred = rainbow_model(selected_days, test_slope, test_intercept, BASE, test_power)
                
                # Use sum of squared errors in log space for better fit
                log_actual = np.log(selected_prices) / log_base
                log_pred = np.log(test_pred) / log_base
                test_sse = np.sum((log_actual - log_pred) ** 2)
                
                if test_sse < best_sse:
                    best_sse = test_sse
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
    
    # Estimate intercept (not constrained to first point)
    estimated_intercept = np.log(first_price) / log_base - estimated_slope * (first_days**best_power)
    
    # Optimize all three parameters: power, slope, and intercept
    initial_guess = [best_power, estimated_slope, estimated_intercept]
    print(f"Initial guess: power={best_power:.3f}, slope={estimated_slope:.3f}, intercept={estimated_intercept:.3f}")
    
    def objective_function(params):
        """Minimize sum of squared errors in log space."""
        power, slope, intercept = params
        
        # Bounds checking
        if power < -2.0 or power > 2.0 or slope < -100 or slope > 100 or abs(intercept) > 50:
            return 1e10
        
        try:
            y_pred = rainbow_model(selected_days, slope, intercept, BASE, power)
            
            if not np.all(np.isfinite(y_pred)) or np.any(y_pred <= 0):
                return 1e10
            
            # Use sum of squared errors in log space (forces line closer to points)
            log_actual = np.log(selected_prices) / log_base
            log_pred = np.log(y_pred) / log_base
            sse = np.sum((log_actual - log_pred) ** 2)
            
            if not np.isfinite(sse):
                return 1e10
            
            return sse
        except:
            return 1e10
    
    # Optimize all three parameters: power, slope, and intercept
    bounds = [(-2.0, 2.0), (-100, 100), (-50, 50)]
    
    try:
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Check if optimization succeeded
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        optimal_power = result.x[0]
        optimal_slope = result.x[1]
        optimal_intercept = result.x[2]
        
        # Calculate R² and errors for reporting
        optimal_pred = rainbow_model(selected_days, optimal_slope, optimal_intercept, BASE, optimal_power)
        optimal_r2 = calculate_r2(selected_prices, optimal_pred, BASE)
        
        # Calculate actual errors at each point
        log_actual = np.log(selected_prices) / log_base
        log_pred = np.log(optimal_pred) / log_base
        errors = log_actual - log_pred
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))
        
        print(f"\nOptimization Results:")
        print(f"  Power: {optimal_power:.6f}")
        print(f"  Slope: {optimal_slope:.6f}")
        print(f"  Intercept: {optimal_intercept:.6f}")
        print(f"  R²: {optimal_r2:.6f}")
        print(f"  Mean absolute log error: {mean_error:.6f}")
        print(f"  Max absolute log error: {max_error:.6f}")
        print(f"  Success: {result.success}")
        
        # Show errors for each point
        print(f"\n  Point-by-point errors:")
        for i, (days, price) in enumerate(zip(selected_days, selected_prices)):
            pred_price = rainbow_model(np.array([days]), optimal_slope, optimal_intercept, BASE, optimal_power)[0]
            error_pct = abs(pred_price - price) / price * 100
            print(f"    Point {i+1}: {error_pct:.2f}% error")
        
        # Plot the fitted curve
        plot_fitted_curve(optimal_power, optimal_slope, optimal_intercept, optimal_r2, model_type='parametric')
        
        # Save the model type
        save_last_model('parametric', power=optimal_power, slope=optimal_slope, intercept=optimal_intercept)
        
        # Update status
        status_text.set_text(f"Parametric Model | R² = {optimal_r2:.4f} | Power: {optimal_power:.6f} | Slope: {optimal_slope:.6f} | Intercept: {optimal_intercept:.6f}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        status_text.set_text(f"Optimization failed: {e}")
    
    fig.canvas.draw_idle()

def calculate_powerlaw_fit():
    """Calculate and plot power law fit: y = a * x^b through selected points."""
    if len(selected_points) < 2:
        print(f"Error: Need at least 2 points for power law fit. Currently have {len(selected_points)}")
        status_text.set_text(f"Error: Need at least 2 points! Currently have {len(selected_points)}")
        fig.canvas.draw_idle()
        return
    
    # Extract selected data
    selected_days = np.array([p[2] for p in selected_points])
    selected_prices = np.array([p[1] for p in selected_points])
    
    print(f"\nCalculating power law fit for {len(selected_points)} selected points...")
    print(f"Selected days: {selected_days}")
    print(f"Selected prices: {selected_prices}")
    
    # NORMALIZE X-AXIS: Use time since first point instead of days since genesis
    first_day = np.min(selected_days)
    normalized_days = selected_days - first_day + 1  # +1 to avoid log(0)
    
    print(f"Normalized days (time since first point): {normalized_days}")
    
    # Power law: y = a * x^b
    # In log space: log(y) = log(a) + b * log(x)
    # This is linear in log-log space
    # CONSTRAIN: 0 < b < 1 for concave (flattening) shape
    
    # Filter out zero or negative values
    valid_mask = (normalized_days > 0) & (selected_prices > 0)
    if np.sum(valid_mask) < 2:
        print("Error: Need at least 2 valid points (days > 0, prices > 0)")
        status_text.set_text("Error: Need valid points!")
        fig.canvas.draw_idle()
        return
    
    valid_normalized_days = normalized_days[valid_mask]
    valid_prices = selected_prices[valid_mask]
    valid_original_days = selected_days[valid_mask]
    
    # Fit in log-log space using constrained optimization
    log_normalized_days = np.log(valid_normalized_days)
    log_prices = np.log(valid_prices)
    
    def objective(params):
        """Minimize sum of squared errors with constraint 0 < b < 1"""
        log_a, b = params
        if b <= 0 or b >= 1:
            return 1e10  # Penalty for violating constraint
        a = np.exp(log_a)
        predicted = a * (valid_normalized_days**b)
        if np.any(predicted <= 0) or not np.all(np.isfinite(predicted)):
            return 1e10
        log_predicted = np.log(predicted)
        sse = np.sum((log_prices - log_predicted)**2)
        return sse if np.isfinite(sse) else 1e10
    
    # Initial guess using unconstrained linear regression
    n = len(valid_normalized_days)
    sum_log_x = np.sum(log_normalized_days)
    sum_log_y = np.sum(log_prices)
    sum_log_x_sq = np.sum(log_normalized_days**2)
    sum_log_xy = np.sum(log_normalized_days * log_prices)
    
    denominator = n * sum_log_x_sq - sum_log_x**2
    if abs(denominator) < 1e-10:
        print("Error: Cannot fit power law (points are collinear in log space)")
        status_text.set_text("Error: Cannot fit power law!")
        fig.canvas.draw_idle()
        return
    
    b_unconstrained = (n * sum_log_xy - sum_log_x * sum_log_y) / denominator
    log_a_unconstrained = (sum_log_y - b_unconstrained * sum_log_x) / n
    
    # Clip b to valid range if needed
    if b_unconstrained <= 0:
        b_initial = 0.1
    elif b_unconstrained >= 1:
        b_initial = 0.9
    else:
        b_initial = b_unconstrained
    
    # Optimize with constraint
    result = minimize(objective, [log_a_unconstrained, b_initial], 
                     bounds=[(None, None), (0.001, 0.999)], method='L-BFGS-B')
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Using initial guess.")
        b = b_initial
        log_a = log_a_unconstrained
    else:
        log_a = result.x[0]
        b = result.x[1]
    
    a = np.exp(log_a)
    
    # Calculate R² using normalized days
    predicted_prices = a * (valid_normalized_days**b)
    log_predicted = np.log(predicted_prices)
    ss_res = np.sum((log_prices - log_predicted)**2)
    ss_tot = np.sum((log_prices - np.mean(log_prices))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate errors
    errors = np.abs(predicted_prices - valid_prices) / valid_prices * 100
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    # Verify first point (to show it's NOT constrained to pass through)
    first_normalized_day = valid_normalized_days[0]
    first_predicted = a * (first_normalized_day**b)
    first_error = abs(first_predicted - valid_prices[0]) / valid_prices[0] * 100
    
    print(f"\nPower Law Fit Results (Normalized, Constrained 0 < b < 1):")
    print(f"  Formula: y = {a:.6f} * (x - {first_day:.1f} + 1)^{b:.6f}")
    print(f"  Coefficient (a): {a:.6f}")
    print(f"  Exponent (b): {b:.6f} (constrained to 0 < b < 1 for concave shape)")
    print(f"  First point offset: {first_day:.1f} days")
    print(f"  R²: {r2:.6f}")
    print(f"  Mean error: {mean_error:.6f}%")
    print(f"  Max error: {max_error:.6f}%")
    print(f"  First point error: {first_error:.2f}% (NOT constrained - this shows it's free to fit)")
    print(f"  Point-by-point errors:")
    for i, (days, price, pred_price, error) in enumerate(zip(valid_original_days, valid_prices, predicted_prices, errors)):
        print(f"    Point {i+1}: {error:.2f}% error")
    
    # Plot the power law curve (pass normalized info)
    plot_fitted_curve(None, None, None, r2, model_type='powerlaw', a=a, b=b, 
                     selected_days=valid_original_days, first_day_offset=first_day)
    
    # Save the model type
    save_last_model('powerlaw', a=a, b=b, first_day_offset=first_day)
    
    # Update status - use scientific notation if a is very small
    if abs(a) < 0.01 or abs(a) > 1000:
        a_str = f"{a:.4e}"
    else:
        a_str = f"{a:.6f}"
    status_text.set_text(f"Power Law: y = {a_str} * x^{b:.4f} | R² = {r2:.4f} | Mean error: {mean_error:.2f}%")
    
    fig.canvas.draw_idle()

def calculate_exponential_fit():
    """Calculate and plot exponential fit: y = a * e^(b * (x - offset)) through selected points."""
    if len(selected_points) < 2:
        print(f"Error: Need at least 2 points for exponential fit. Currently have {len(selected_points)}")
        status_text.set_text(f"Error: Need at least 2 points! Currently have {len(selected_points)}")
        fig.canvas.draw_idle()
        return
    
    # Extract selected data
    selected_days = np.array([p[2] for p in selected_points])
    selected_prices = np.array([p[1] for p in selected_points])
    
    print(f"\nCalculating exponential fit for {len(selected_points)} selected points...")
    print(f"Selected days: {selected_days}")
    print(f"Selected prices: {selected_prices}")
    
    # NORMALIZE X-AXIS: Use time since first point instead of days since genesis
    first_day = np.min(selected_days)
    normalized_days = selected_days - first_day  # No +1 needed for exponential
    
    print(f"Normalized days (time since first point): {normalized_days}")
    
    # Exponential: y = a * e^(b * x)
    # In log space: ln(y) = ln(a) + b * x
    # This is linear in log-linear space
    
    # Filter out zero or negative values
    valid_mask = (selected_prices > 0)
    if np.sum(valid_mask) < 2:
        print("Error: Need at least 2 valid points (prices > 0)")
        status_text.set_text("Error: Need valid points!")
        fig.canvas.draw_idle()
        return
    
    valid_normalized_days = normalized_days[valid_mask]
    valid_prices = selected_prices[valid_mask]
    valid_original_days = selected_days[valid_mask]
    
    # Fit in log-linear space using linear regression
    log_prices = np.log(valid_prices)
    
    # Linear regression: ln(y) = ln(a) + b * x
    n = len(valid_normalized_days)
    sum_x = np.sum(valid_normalized_days)
    sum_y = np.sum(log_prices)
    sum_x_sq = np.sum(valid_normalized_days**2)
    sum_xy = np.sum(valid_normalized_days * log_prices)
    
    denominator = n * sum_x_sq - sum_x**2
    if abs(denominator) < 1e-10:
        print("Error: Cannot fit exponential (points are collinear)")
        status_text.set_text("Error: Cannot fit exponential!")
        fig.canvas.draw_idle()
        return
    
    b = (n * sum_xy - sum_x * sum_y) / denominator
    log_a = (sum_y - b * sum_x) / n
    a = np.exp(log_a)
    
    # Calculate R² using normalized days
    predicted_prices = a * np.exp(b * valid_normalized_days)
    log_predicted = np.log(predicted_prices)
    ss_res = np.sum((log_prices - log_predicted)**2)
    ss_tot = np.sum((log_prices - np.mean(log_prices))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate errors
    errors = np.abs(predicted_prices - valid_prices) / valid_prices * 100
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    # Verify first point (to show it's NOT constrained to pass through)
    first_normalized_day = valid_normalized_days[0]
    first_predicted = a * np.exp(b * first_normalized_day)
    first_error = abs(first_predicted - valid_prices[0]) / valid_prices[0] * 100
    
    print(f"\nExponential Fit Results (Normalized):")
    print(f"  Formula: y = {a:.6f} * e^({b:.6f} * (x - {first_day:.1f}))")
    print(f"  Coefficient (a): {a:.6f}")
    print(f"  Growth Rate (b): {b:.6f} ({'growth' if b > 0 else 'decay'})")
    print(f"  First point offset: {first_day:.1f} days")
    print(f"  R²: {r2:.6f}")
    print(f"  Mean error: {mean_error:.6f}%")
    print(f"  Max error: {max_error:.6f}%")
    print(f"  First point error: {first_error:.2f}% (NOT constrained - this shows it's free to fit)")
    print(f"  Point-by-point errors:")
    for i, (days, price, pred_price, error) in enumerate(zip(valid_original_days, valid_prices, predicted_prices, errors)):
        print(f"    Point {i+1}: {error:.2f}% error")
    
    # Plot the exponential curve (pass normalized info)
    plot_fitted_curve(None, None, None, r2, model_type='exponential', a=a, b=b, 
                     selected_days=valid_original_days, first_day_offset=first_day)
    
    # Save the model type
    save_last_model('exponential', a=a, b=b, first_day_offset=first_day)
    
    # Update status - use scientific notation if a is very small
    if abs(a) < 0.01 or abs(a) > 1000:
        a_str = f"{a:.4e}"
    else:
        a_str = f"{a:.6f}"
    growth_str = "growth" if b > 0 else "decay"
    status_text.set_text(f"Exponential: y = {a_str} * e^({b:.4f}*x) ({growth_str}) | R² = {r2:.4f} | Mean error: {mean_error:.2f}%")
    
    fig.canvas.draw_idle()

def calculate_pchip_interpolation():
    """Calculate and plot PCHIP interpolation through all selected points."""
    if len(selected_points) < 2:
        print(f"Error: Need at least 2 points for PCHIP interpolation. Currently have {len(selected_points)}")
        status_text.set_text(f"Error: Need at least 2 points! Currently have {len(selected_points)}")
        fig.canvas.draw_idle()
        return
    
    # Extract selected data
    selected_days = np.array([p[2] for p in selected_points])
    selected_prices = np.array([p[1] for p in selected_points])
    selected_dates = np.array([p[0] for p in selected_points])
    
    print(f"\nCalculating PCHIP interpolation for {len(selected_points)} selected points...")
    print(f"Selected days: {selected_days}")
    print(f"Selected prices: {selected_prices}")
    
    # Sort by days
    sort_idx = np.argsort(selected_days)
    sorted_days = selected_days[sort_idx]
    sorted_prices = selected_prices[sort_idx]
    sorted_dates = selected_dates[sort_idx]
    
    # Create PCHIP interpolator (works in linear space, preserves shape)
    pchip = PchipInterpolator(sorted_days, sorted_prices)
    
    # Verify PCHIP passes through all points
    prices_check = pchip(sorted_days)
    errors = np.abs(prices_check - sorted_prices) / sorted_prices * 100
    
    print(f"\nPCHIP Interpolation Results:")
    print(f"  Points: {len(selected_points)}")
    print(f"  Interpolation range: {sorted_days[0]:.1f} to {sorted_days[-1]:.1f} days")
    print(f"  Point-by-point verification:")
    for i, (days, price, error) in enumerate(zip(sorted_days, sorted_prices, errors)):
        print(f"    Point {i+1}: {error:.6f}% error (should be ~0%)")
    
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    print(f"  Mean error: {mean_error:.6f}%")
    print(f"  Max error: {max_error:.6f}%")
    print(f"  Shape: Steep early, flattens later (preserves monotonicity)")
    
    # Plot the PCHIP curve
    plot_fitted_curve(None, None, None, None, model_type='pchip', 
                     selected_days=sorted_days, selected_dates=sorted_dates)
    
    # Update status
    status_text.set_text(f"PCHIP Interpolation | Passes through all {len(selected_points)} points | Mean error: {mean_error:.4f}%")
    
    fig.canvas.draw_idle()

def toggle_scale(event):
    """Toggle between log scale and normal scale."""
    global use_log_scale, price_line, curve_line, powerlaw_line, exponential_line, pchip_line, last_model_params, current_model_type
    
    # Toggle scale
    use_log_scale = not use_log_scale
    
    # Clear all lines
    if price_line is not None:
        try:
            price_line.remove()
        except:
            pass
        price_line = None
    
    # Remove all curve lines
    if curve_line is not None:
        try:
            curve_line.remove()
        except:
            pass
        curve_line = None
    if powerlaw_line is not None:
        try:
            powerlaw_line.remove()
        except:
            pass
        powerlaw_line = None
    if exponential_line is not None:
        try:
            exponential_line.remove()
        except:
            pass
        exponential_line = None
    if pchip_line is not None:
        try:
            pchip_line.remove()
        except:
            pass
        pchip_line = None
    
    # Set y-axis scale
    if use_log_scale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    
    # Re-plot price data
    if use_log_scale:
        price_line, = ax.semilogy(df_filtered['Date'], df_filtered['Price'], 
                                 color='blue', linewidth=2, label='Actual Price', alpha=0.7)
    else:
        price_line, = ax.plot(df_filtered['Date'], df_filtered['Price'], 
                             color='blue', linewidth=2, label='Actual Price', alpha=0.7)
    
    # Re-plot selected points
    for marker in selected_markers:
        try:
            marker.remove()
        except:
            pass
    selected_markers.clear()
    for date, price, days in selected_points:
        marker = ax.scatter(date, price, 
                          color='red', s=30, zorder=10, marker='o', 
                          edgecolors='black', linewidths=1)
        selected_markers.append(marker)
    
    # Re-plot current model if one exists
    if last_model_params is not None:
        model_type = last_model_params.get('model_type')
        if model_type == 'parametric':
            plot_fitted_curve(
                last_model_params.get('power'),
                last_model_params.get('slope'),
                last_model_params.get('intercept'),
                last_model_params.get('r2'),
                model_type='parametric'
            )
        elif model_type == 'powerlaw':
            plot_fitted_curve(
                None, None, None,
                last_model_params.get('r2'),
                model_type='powerlaw',
                a=last_model_params.get('a'),
                b=last_model_params.get('b'),
                first_day_offset=last_model_params.get('first_day_offset')
            )
        elif model_type == 'exponential':
            plot_fitted_curve(
                None, None, None,
                last_model_params.get('r2'),
                model_type='exponential',
                a=last_model_params.get('a'),
                b=last_model_params.get('b'),
                first_day_offset=last_model_params.get('first_day_offset')
            )
        elif model_type == 'pchip' and len(selected_points) >= 2:
            calculate_pchip_interpolation()
    
    # Update button label
    btn_toggle.label.set_text('Normal\nScale' if use_log_scale else 'Log\nScale')
    
    # Update legend
    ax.legend(loc='upper left')
    fig.canvas.draw()

def plot_fitted_curve(power, slope, intercept, r2, model_type='parametric', a=None, b=None, 
                      selected_days=None, first_day_offset=None, selected_dates=None):
    """Plot the fitted curve on the graph."""
    global curve_line, powerlaw_line, exponential_line, pchip_line, current_model_type, use_log_scale, last_model_params
    
    # Hide the other curve types
    if model_type == 'parametric':
        # Remove other curves if they exist
        if powerlaw_line is not None:
            try:
                powerlaw_line.remove()
                powerlaw_line = None
            except:
                pass
        if exponential_line is not None:
            try:
                exponential_line.remove()
                exponential_line = None
            except:
                pass
        if pchip_line is not None:
            try:
                pchip_line.remove()
                pchip_line = None
            except:
                pass
        # Remove existing parametric curve if any
        if curve_line is not None:
            try:
                curve_line.remove()
            except:
                pass
    elif model_type == 'powerlaw':
        # Remove other curves if they exist
        if curve_line is not None:
            try:
                curve_line.remove()
                curve_line = None
            except:
                pass
        if exponential_line is not None:
            try:
                exponential_line.remove()
                exponential_line = None
            except:
                pass
        if pchip_line is not None:
            try:
                pchip_line.remove()
                pchip_line = None
            except:
                pass
        # Remove existing power law if any
        if powerlaw_line is not None:
            try:
                powerlaw_line.remove()
            except:
                pass
    elif model_type == 'exponential':
        # Remove other curves if they exist
        if curve_line is not None:
            try:
                curve_line.remove()
                curve_line = None
            except:
                pass
        if powerlaw_line is not None:
            try:
                powerlaw_line.remove()
                powerlaw_line = None
            except:
                pass
        if pchip_line is not None:
            try:
                pchip_line.remove()
                pchip_line = None
            except:
                pass
        # Remove existing exponential if any
        if exponential_line is not None:
            try:
                exponential_line.remove()
            except:
                pass
    else:  # pchip
        # Remove other curves if they exist
        if curve_line is not None:
            try:
                curve_line.remove()
                curve_line = None
            except:
                pass
        if powerlaw_line is not None:
            try:
                powerlaw_line.remove()
                powerlaw_line = None
            except:
                pass
        if exponential_line is not None:
            try:
                exponential_line.remove()
                exponential_line = None
            except:
                pass
        # Remove existing PCHIP if any
        if pchip_line is not None:
            try:
                pchip_line.remove()
            except:
                pass
    
    # Save current y-axis limits to preserve view
    current_ylim = ax.get_ylim()
    
    if model_type == 'parametric':
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
        
        # Plot curve using appropriate scale
        if use_log_scale:
            curve_line, = ax.semilogy(df_filtered['Date'], model_prices, 
                                     color='green', linestyle='--', linewidth=2, 
                                     label=f'Parametric Model (R² = {r2:.4f})', zorder=5)
        else:
            curve_line, = ax.plot(df_filtered['Date'], model_prices, 
                                 color='green', linestyle='--', linewidth=2, 
                                 label=f'Parametric Model (R² = {r2:.4f})', zorder=5)
        current_model_type = 'parametric'
        # Restore y-axis limits to preserve view
        ax.set_ylim(current_ylim)
    elif model_type == 'powerlaw':
        # Power law: y = a * (x - offset + 1)^b (normalized x-axis)
        if a is None or b is None or first_day_offset is None:
            print("Error: Power law parameters (a, b, first_day_offset) required!")
            return
        
        # Always use full date range for power law
        days = df_filtered['Days_Since_Genesis'].values
        mask = days > 0
        days_valid = days[mask]
        
        if len(days_valid) > 0:
            # Normalize: x_norm = x - first_day_offset + 1
            normalized_days = days_valid - first_day_offset + 1
            
            # Power law: y = a * (normalized_days)^b
            prices_powerlaw = a * (normalized_days**b)
            
            # Ensure prices are valid
            prices_powerlaw = np.where(np.isfinite(prices_powerlaw) & (prices_powerlaw > 0), prices_powerlaw, np.nan)
            
            # Format coefficient for label
            if abs(a) < 0.01 or abs(a) > 1000:
                a_label = f"{a:.4e}"
            else:
                a_label = f"{a:.6f}"
            
            # Plot power law curve over full timeline
            dates_valid = df_filtered['Date'].values[mask]
            if use_log_scale:
                powerlaw_line, = ax.semilogy(dates_valid, prices_powerlaw, 
                                           color='orange', linestyle='-', linewidth=2, 
                                           label=f'Power Law (b={b:.4f}, R²={r2:.4f})', zorder=5)
            else:
                powerlaw_line, = ax.plot(dates_valid, prices_powerlaw, 
                                       color='orange', linestyle='-', linewidth=2, 
                                       label=f'Power Law (b={b:.4f}, R²={r2:.4f})', zorder=5)
            current_model_type = 'powerlaw'
            # Restore y-axis limits to preserve view (don't auto-zoom)
            ax.set_ylim(current_ylim)
        else:
            print("Warning: No valid days for power law!")
            return
    elif model_type == 'exponential':
        # Exponential: y = a * e^(b * (x - offset))
        if a is None or b is None or first_day_offset is None:
            print("Error: Exponential parameters (a, b, first_day_offset) required!")
            return
        
        # Always use full date range for exponential
        days = df_filtered['Days_Since_Genesis'].values
        mask = days > 0
        days_valid = days[mask]
        
        if len(days_valid) > 0:
            # Normalize: x_norm = x - first_day_offset
            normalized_days = days_valid - first_day_offset
            
            # Exponential: y = a * e^(b * normalized_days)
            prices_exponential = a * np.exp(b * normalized_days)
            
            # Ensure prices are valid
            prices_exponential = np.where(np.isfinite(prices_exponential) & (prices_exponential > 0), prices_exponential, np.nan)
            
            # Format coefficient for label
            if abs(a) < 0.01 or abs(a) > 1000:
                a_label = f"{a:.4e}"
            else:
                a_label = f"{a:.6f}"
            
            # Plot exponential curve over full timeline
            dates_valid = df_filtered['Date'].values[mask]
            if use_log_scale:
                exponential_line, = ax.semilogy(dates_valid, prices_exponential, 
                                              color='red', linestyle='-', linewidth=2, 
                                              label=f'Exponential (b={b:.4f}, R²={r2:.4f})', zorder=5)
            else:
                exponential_line, = ax.plot(dates_valid, prices_exponential, 
                                          color='red', linestyle='-', linewidth=2, 
                                          label=f'Exponential (b={b:.4f}, R²={r2:.4f})', zorder=5)
            current_model_type = 'exponential'
            # Restore y-axis limits to preserve view (don't auto-zoom)
            ax.set_ylim(current_ylim)
        else:
            print("Warning: No valid days for exponential!")
            return
    else:  # pchip
        # PCHIP interpolation
        if selected_days is None or selected_dates is None or len(selected_days) < 2:
            print("Error: Need selected days and dates for PCHIP!")
            return
        
        # Get sorted prices
        sort_idx = np.argsort(selected_days)
        sorted_prices = np.array([p[1] for p in selected_points])[sort_idx]
        
        # Create PCHIP interpolator
        pchip = PchipInterpolator(selected_days, sorted_prices)
        
        # Interpolate for range between first and last selected points with padding
        min_days = np.min(selected_days)
        max_days = np.max(selected_days)
        day_range = max_days - min_days
        min_days_plot = max(1, min_days - day_range * 0.1)
        max_days_plot = max_days + day_range * 0.1
        
        # Create fine grid for smooth curve
        days_interp = np.linspace(min_days_plot, max_days_plot, 500)
        prices_interp = pchip(days_interp)
        
        # Ensure prices are valid
        prices_interp = np.where(np.isfinite(prices_interp) & (prices_interp > 0), prices_interp, np.nan)
        
        # Convert days back to dates for plotting
        # Find closest dates in df_filtered
        dates_interp = []
        for day in days_interp:
            closest_idx = (df_filtered['Days_Since_Genesis'] - day).abs().idxmin()
            dates_interp.append(df_filtered.loc[closest_idx, 'Date'])
        dates_interp = np.array(dates_interp)
        
        # Plot PCHIP curve
        if use_log_scale:
            pchip_line, = ax.semilogy(dates_interp, prices_interp, 
                                     color='purple', linestyle='-', linewidth=2, 
                                     label='PCHIP Interpolation (steep early, flattens later)', zorder=5)
        else:
            pchip_line, = ax.plot(dates_interp, prices_interp, 
                                 color='purple', linestyle='-', linewidth=2, 
                                 label='PCHIP Interpolation (steep early, flattens later)', zorder=5)
        current_model_type = 'pchip'
        
        # Auto-zoom to selected points
        min_date = np.min(selected_dates)
        max_date = np.max(selected_dates)
        date_range = (max_date - min_date).total_seconds() / 86400
        min_date_plot = min_date - pd.Timedelta(days=date_range * 0.1)
        max_date_plot = max_date + pd.Timedelta(days=date_range * 0.1)
        
        min_price = np.min(sorted_prices)
        max_price = np.max(sorted_prices)
        price_range_log = np.log10(max_price) - np.log10(min_price)
        min_price_plot = 10**(np.log10(min_price) - price_range_log * 0.2)
        max_price_plot = 10**(np.log10(max_price) + price_range_log * 0.2)
        
        ax.set_xlim([min_date_plot, max_date_plot])
        ax.set_ylim([min_price_plot, max_price_plot])
    
    # Store model parameters for re-plotting after scale toggle
    if model_type == 'parametric':
        last_model_params = {
            'model_type': 'parametric',
            'power': power,
            'slope': slope,
            'intercept': intercept,
            'r2': r2
        }
    elif model_type == 'powerlaw':
        last_model_params = {
            'model_type': 'powerlaw',
            'a': a,
            'b': b,
            'first_day_offset': first_day_offset,
            'r2': r2
        }
    elif model_type == 'exponential':
        last_model_params = {
            'model_type': 'exponential',
            'a': a,
            'b': b,
            'first_day_offset': first_day_offset,
            'r2': r2
        }
    elif model_type == 'pchip':
        last_model_params = {
            'model_type': 'pchip'
        }
    
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
        
        # Remove curves if exist (need to recalculate)
        global curve_line, powerlaw_line, exponential_line, pchip_line
        if 'curve_line' in globals() and curve_line is not None:
            try:
                curve_line.remove()
            except:
                pass
            curve_line = None
        if 'powerlaw_line' in globals() and powerlaw_line is not None:
            try:
                powerlaw_line.remove()
            except:
                pass
            powerlaw_line = None
        if 'exponential_line' in globals() and exponential_line is not None:
            try:
                exponential_line.remove()
            except:
                pass
            exponential_line = None
        if 'pchip_line' in globals() and pchip_line is not None:
            try:
                pchip_line.remove()
            except:
                pass
            pchip_line = None
        
        # Save points
        save_points()
        
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
    
    # Remove curves if exist
    global curve_line, powerlaw_line, exponential_line, pchip_line
    if 'curve_line' in globals() and curve_line is not None:
        try:
            curve_line.remove()
        except:
            pass
        curve_line = None
    if 'powerlaw_line' in globals() and powerlaw_line is not None:
        try:
            powerlaw_line.remove()
        except:
            pass
        powerlaw_line = None
    if 'exponential_line' in globals() and exponential_line is not None:
        try:
            exponential_line.remove()
        except:
            pass
        exponential_line = None
    if 'pchip_line' in globals() and pchip_line is not None:
        try:
            pchip_line.remove()
        except:
            pass
        pchip_line = None
    
    # Save points (empty list)
    save_points()
    
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
    print("4. Select at least 2 points for power law/exponential/PCHIP, or 3+ points for parametric model")
    print("5. Click 'Calculate R²' for PARAMETRIC model: price = 10^(slope × (days^power) + intercept)")
    print("6. Click 'Power Law' for POWER LAW model: price = a × (days - offset + 1)^b (0 < b < 1, concave)")
    print("7. Click 'Exponential' for EXPONENTIAL model: price = a × e^(b × (days - offset)) (accelerating growth)")
    print("8. Click 'PCHIP' for PCHIP interpolation: smooth curve through all points (steep early, flattens)")
    print("9. You can switch between models to compare!")
    print("10. Selected points are automatically saved and will load next time")
    print("11. Click 'Clear Points' to start over")
    print("\nTip: Zoom in on local maxima/minima for more precise point selection!")
    
    global df_filtered, ax, fig, status_text, curve_line, powerlaw_line, exponential_line, pchip_line, current_model_type, selected_points, selected_markers
    curve_line = None
    powerlaw_line = None
    exponential_line = None
    pchip_line = None
    current_model_type = None
    
    # Declare global variables at the start
    global fig, ax, price_line, df_filtered, status_text
    
    df_filtered = load_data()
    if df_filtered is None:
        return
    
    # Load saved points
    loaded_points = load_points()
    if loaded_points:
        selected_points = loaded_points
        print(f"Restored {len(selected_points)} saved points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set initial scale to log
    ax.set_yscale('log')
    
    # Plot price data on log scale
    price_line, = ax.semilogy(df_filtered['Date'], df_filtered['Price'], 
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
    
    # Plot loaded points if any
    if loaded_points:
        selected_markers = []
        for date, price, days in selected_points:
            marker = ax.scatter(date, price, 
                              color='red', s=30, zorder=10, marker='o', 
                              edgecolors='black', linewidths=1)
            selected_markers.append(marker)
        print(f"Plotted {len(selected_points)} saved points on graph")
    
    # Add status text
    status_text = ax.text(0.02, 0.98, f"Selected points: {len(selected_points)} | Click to add points, then calculate: Parametric (R²), Power Law, Exponential, or PCHIP", 
                         transform=ax.transAxes, fontsize=10, 
                         verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add buttons
    ax_toggle = plt.axes([0.30, 0.02, 0.09, 0.04])
    global btn_toggle
    btn_toggle = Button(ax_toggle, 'Normal\nScale')
    btn_toggle.on_clicked(toggle_scale)
    
    ax_calculate = plt.axes([0.40, 0.02, 0.09, 0.04])
    btn_calculate = Button(ax_calculate, 'Parametric\n(R²)')
    btn_calculate.on_clicked(lambda x: optimize_for_selected_points())
    
    ax_powerlaw = plt.axes([0.50, 0.02, 0.09, 0.04])
    btn_powerlaw = Button(ax_powerlaw, 'Power Law\n(Concave)')
    btn_powerlaw.on_clicked(lambda x: calculate_powerlaw_fit())
    
    ax_exponential = plt.axes([0.60, 0.02, 0.09, 0.04])
    btn_exponential = Button(ax_exponential, 'Exponential\n(Growth)')
    btn_exponential.on_clicked(lambda x: calculate_exponential_fit())
    
    ax_pchip = plt.axes([0.70, 0.02, 0.09, 0.04])
    btn_pchip = Button(ax_pchip, 'PCHIP\n(Interp)')
    btn_pchip.on_clicked(lambda x: calculate_pchip_interpolation())
    
    ax_clear_last = plt.axes([0.80, 0.02, 0.07, 0.04])
    btn_clear_last = Button(ax_clear_last, 'Undo')
    btn_clear_last.on_clicked(clear_last_point)
    
    ax_clear = plt.axes([0.88, 0.02, 0.07, 0.04])
    btn_clear = Button(ax_clear, 'Clear')
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

