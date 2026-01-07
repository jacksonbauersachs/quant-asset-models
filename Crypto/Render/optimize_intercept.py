"""
Template script for optimizing intercept and creating projections with LOGARITHMIC growth model.
This script reads configuration from config.py.

Usage:
    1. Edit config.py with your currency information
    2. Optionally set FIXED_GROWTH_RATE in config.py as FIXED_SLOPE (from running interactive_rainbow.py first)
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
    # FIXED_GROWTH_RATE is used as FIXED_SLOPE for logarithmic models - will optimize if not provided
    try:
        from config import FIXED_GROWTH_RATE
        FIXED_SLOPE = FIXED_GROWTH_RATE  # For logarithmic model, this is the slope
    except ImportError:
        FIXED_SLOPE = None  # Will optimize if not set
    
    # POWER is optional - default to 1.0 if not set
    try:
        from config import POWER
    except ImportError:
        POWER = 1.0  # Default to linear (power = 1)
    
    # MODEL_TYPE is optional - default to 'parametric'
    try:
        from config import MODEL_TYPE
    except ImportError:
        MODEL_TYPE = 'parametric'  # Default to parametric model
    
    # Power law parameters (only used if MODEL_TYPE == 'powerlaw')
    try:
        from config import POWER_LAW_A, POWER_LAW_B, POWER_LAW_FIRST_DAY_OFFSET
    except ImportError:
        POWER_LAW_A = None
        POWER_LAW_B = None
        POWER_LAW_FIRST_DAY_OFFSET = None
    
    # Exponential parameters (only used if MODEL_TYPE == 'exponential')
    try:
        from config import EXPONENTIAL_A, EXPONENTIAL_B, EXPONENTIAL_FIRST_DAY_OFFSET
    except ImportError:
        EXPONENTIAL_A = None
        EXPONENTIAL_B = None
        EXPONENTIAL_FIRST_DAY_OFFSET = None
except ImportError:
    print("Error: config.py not found! Make sure you're running from the log_model folder.")
    sys.exit(1)

# Validate config
validate_config()
output_files = get_output_files()

def load_saved_model_type():
    """Load the last selected model type from interactive_point_selector.py."""
    import json
    import os
    
    # PRIORITY: Check script directory FIRST (where the model was saved)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_file = os.path.join(script_dir, f'{CURRENCY_NAME_LOWER}_last_model.json')
    
    if os.path.exists(script_file):
        try:
            with open(script_file, 'r') as f:
                model_data = json.load(f)
            print(f"Loaded saved model from: {script_file}")
            return model_data
        except Exception as e:
            print(f"Note: Could not load saved model type from {script_file}: {e}")
    
    # No fallback - ONLY use script directory
    return None

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

def rainbow_model(x, slope, intercept, power=1.0):
    """Parametric growth model: price = 10^(slope * (days^power) + intercept)"""
    # Ensure x is at least 1 to avoid issues
    x = np.maximum(x, 1)
    # Calculate exponent and prevent overflow
    exponent = slope * (x**power) + intercept
    exponent = np.clip(exponent, -30, 30)  # Prevent overflow
    result = 10**exponent
    result = np.where(np.isfinite(result), result, np.nan)
    return result

def powerlaw_model(x, a, b, first_day_offset):
    """Power law model: price = a * (days - first_day_offset + 1)^b"""
    # Normalize: x_norm = x - first_day_offset + 1
    x_norm = x - first_day_offset + 1
    # Ensure normalized x is positive
    x_norm = np.maximum(x_norm, 0.001)
    # Power law: y = a * x_norm^b
    result = a * (x_norm**b)
    result = np.where(np.isfinite(result) & (result > 0), result, np.nan)
    return result

def exponential_model(x, a, b, first_day_offset):
    """Exponential model: price = a * e^(b * (days - first_day_offset))"""
    # Normalize: x_norm = x - first_day_offset
    x_norm = x - first_day_offset
    # Exponential: y = a * e^(b * x_norm)
    result = a * np.exp(b * x_norm)
    result = np.where(np.isfinite(result) & (result > 0), result, np.nan)
    return result

def calculate_r2(y_actual, y_pred):
    """Calculate R² score for logarithmic/parametric growth model (in log10 space)."""
    # Filter out invalid values (zeros, negatives, NaN, Inf)
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    
    if not np.any(valid_mask):
        return 0.0  # No valid data
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate R² in log space (standard for logarithmic models)
    log_actual = np.log10(y_actual_valid)
    log_pred = np.log10(y_pred_valid)
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def calculate_r2_exponential(y_actual, y_pred):
    """Calculate R² score for exponential model (in natural log space)."""
    # Filter out invalid values (zeros, negatives, NaN, Inf)
    valid_mask = (y_actual > 0) & (y_pred > 0) & np.isfinite(y_actual) & np.isfinite(y_pred)
    
    if not np.any(valid_mask):
        return 0.0  # No valid data
    
    y_actual_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate R² in natural log space (standard for exponential models)
    log_actual = np.log(y_actual_valid)
    log_pred = np.log(y_pred_valid)
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def objective_function_param(intercept):
    """Objective function for parametric model (minimize negative R² to maximize R²)."""
    global df_filtered, fixed_slope
    
    # Calculate model predictions with fixed slope, fixed power, and variable intercept
    y_pred = rainbow_model(df_filtered['Days_Since_Genesis'], fixed_slope, intercept, POWER)
    
    # Validate predictions are reasonable before calculating R²
    max_pred = np.max(y_pred)
    max_actual = np.max(df_filtered['Price'])
    min_pred = np.min(y_pred)
    min_actual = np.min(df_filtered['Price'])
    
    # If predictions are way off, return a very bad score
    if not np.all(np.isfinite(y_pred)) or max_pred > max_actual * 1000 or min_pred < min_actual * 0.001:
        return 1e10  # Very bad score (we're minimizing, so return large positive)
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def objective_function_powerlaw(a):
    """Objective function for power law model (minimize negative R² to maximize R²)."""
    global df_filtered, powerlaw_b, powerlaw_first_day_offset
    
    # Calculate model predictions with variable coefficient a, fixed b and first_day_offset
    y_pred = powerlaw_model(df_filtered['Days_Since_Genesis'], a, powerlaw_b, powerlaw_first_day_offset)
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def objective_function_exponential(a):
    """Objective function for exponential model (minimize negative R² to maximize R²)."""
    global df_filtered, exponential_b, exponential_first_day_offset
    
    # Calculate model predictions with variable coefficient a, fixed b and first_day_offset
    y_pred = exponential_model(df_filtered['Days_Since_Genesis'], a, exponential_b, exponential_first_day_offset)
    
    # Calculate R² (use natural log for exponential)
    r2 = calculate_r2_exponential(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def optimize_intercept_for_r2(df, slope):
    """Find the optimal intercept that maximizes R² for a given fixed slope (parametric model)."""
    global df_filtered, fixed_slope
    df_filtered = df
    fixed_slope = slope
    
    # Validate slope is reasonable for the power and data range
    max_days = df['Days_Since_Genesis'].max()
    max_exponent = slope * (max_days**POWER)
    
    if abs(max_exponent) > 25:  # Before clipping at 30
        print(f"WARNING: Slope {slope:.6f} with power {POWER:.6f} produces very large exponents (max: {max_exponent:.2f})")
        print(f"  This may cause numerical issues. Consider using a smaller slope or different power.")
        print(f"  Suggested: slope should be < {25 / (max_days**POWER):.6f} for power {POWER:.6f}")
    
    print(f"Optimizing intercept for fixed slope: {slope:.6f}, power: {POWER:.6f}")
    print("Searching for optimal intercept...")
    
    # Use wide bounds for intercept (in log space, so can be negative or positive)
    wide_min = -20.0
    wide_max = 20.0
    
    try:
        result = minimize_scalar(objective_function_param, bounds=(wide_min, wide_max), method='bounded')
        optimal_intercept = result.x
        optimal_r2 = -result.fun  # Convert back to positive R²
        
        # If we hit a boundary, try even wider
        if abs(optimal_intercept - wide_min) < 0.1 or abs(optimal_intercept - wide_max) < 0.1:
            print(f"Intercept hit boundary. Trying wider range...")
            wider_min = -50.0
            wider_max = 50.0
            result = minimize_scalar(objective_function_param, bounds=(wider_min, wider_max), method='bounded')
            optimal_intercept = result.x
            optimal_r2 = -result.fun
        
        # Validate the result makes sense
        test_pred = rainbow_model(df['Days_Since_Genesis'].iloc[-1], slope, optimal_intercept, POWER)
        max_actual = df['Price'].max()
        if not np.isfinite(test_pred) or test_pred > max_actual * 1000 or test_pred < max_actual * 0.001:
            print(f"Warning: Optimal intercept produces unrealistic predictions (predicted: {test_pred:.2e}, actual max: {max_actual:.2f}).")
            print(f"  This suggests the slope/power combination may be incorrect.")
            print(f"  Recalculating intercept from first data point...")
            # Calculate default intercept from first point
            first_price = df['Price'].iloc[0]
            first_days = df['Days_Since_Genesis'].iloc[0]
            optimal_intercept = np.log10(first_price) - slope * (first_days**POWER)
            optimal_r2 = -objective_function_param(optimal_intercept)
            
            # Validate again
            test_pred2 = rainbow_model(df['Days_Since_Genesis'].iloc[-1], slope, optimal_intercept, POWER)
            if not np.isfinite(test_pred2) or test_pred2 > max_actual * 1000:
                print(f"ERROR: Even default intercept produces unrealistic predictions!")
                print(f"  The slope ({slope:.6f}) and power ({POWER:.6f}) combination is incompatible with the data.")
                print(f"  Please check your saved model parameters or config settings.")
                raise ValueError(f"Slope {slope:.6f} and power {POWER:.6f} produce invalid predictions")
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using default intercept.")
        # Calculate default intercept from first point
        first_price = df['Price'].iloc[0]
        first_days = df['Days_Since_Genesis'].iloc[0]
        optimal_intercept = np.log10(first_price) - slope * (first_days**POWER)
        optimal_r2 = -objective_function_param(optimal_intercept)
    
    print(f"Optimal intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_intercept, optimal_r2

def optimize_coefficient_for_r2(df, b, first_day_offset, initial_a=None):
    """Find the optimal coefficient a that maximizes R² for power law model (fixed b and first_day_offset).
    
    This optimizes a to fit ALL data points, not just selected points.
    Uses b and first_day_offset from config (from selected points) as fixed shape parameters.
    """
    global df_filtered, powerlaw_b, powerlaw_first_day_offset
    df_filtered = df
    powerlaw_b = b
    powerlaw_first_day_offset = first_day_offset
    
    print(f"Optimizing coefficient a for power law with b={b:.6f}, first_day_offset={first_day_offset:.1f}")
    print(f"Fitting to ALL {len(df)} price points in dataset...")
    
    # Calculate initial guess if not provided
    if initial_a is None:
        # Use first point to estimate initial a
        first_price = df['Price'].iloc[0]
        first_days = df['Days_Since_Genesis'].iloc[0]
        first_normalized = first_days - first_day_offset + 1
        initial_a = first_price / (first_normalized**b) if first_normalized > 0 else 1.0
        print(f"Initial coefficient estimate: {initial_a:.6f}")
    else:
        print(f"Using initial coefficient from config: {initial_a:.6f}")
    
    # Use wide bounds for coefficient a (must be positive)
    wide_min = 1e-10
    wide_max = 1e10
    
    try:
        # Try starting from initial guess if provided
        if initial_a is not None and wide_min < initial_a < wide_max:
            # Use initial_a as starting point, but search around it
            result = minimize_scalar(objective_function_powerlaw, bounds=(wide_min, wide_max), method='bounded')
        else:
            result = minimize_scalar(objective_function_powerlaw, bounds=(wide_min, wide_max), method='bounded')
        
        optimal_a = result.x
        optimal_r2 = -result.fun  # Convert back to positive R²
        
        # Validate the result makes sense
        test_pred = powerlaw_model(df['Days_Since_Genesis'].iloc[-1], optimal_a, b, first_day_offset)
        if not np.isfinite(test_pred) or test_pred > df['Price'].max() * 1000:
            print(f"Warning: Optimal coefficient produces unrealistic predictions. Using initial estimate.")
            optimal_a = initial_a
            optimal_r2 = -objective_function_powerlaw(optimal_a)
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using initial estimate.")
        optimal_a = initial_a
        optimal_r2 = -objective_function_powerlaw(optimal_a)
    
    print(f"Optimal coefficient a: {optimal_a:.6f} (optimized to fit all {len(df)} data points)")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_a, optimal_r2

def optimize_coefficient_for_r2_exponential(df, b, first_day_offset, initial_a=None):
    """Find the optimal coefficient a that maximizes R² for exponential model (fixed b and first_day_offset).
    
    This optimizes a to fit ALL data points, not just selected points.
    Uses b and first_day_offset from config (from selected points) as fixed shape parameters.
    """
    global df_filtered, exponential_b, exponential_first_day_offset
    df_filtered = df
    exponential_b = b
    exponential_first_day_offset = first_day_offset
    
    print(f"Optimizing coefficient a for exponential with b={b:.6f}, first_day_offset={first_day_offset:.1f}")
    print(f"Fitting to ALL {len(df)} price points in dataset...")
    
    # Calculate initial guess if not provided
    if initial_a is None:
        # Use first point to estimate initial a
        first_price = df['Price'].iloc[0]
        first_days = df['Days_Since_Genesis'].iloc[0]
        first_normalized = first_days - first_day_offset
        initial_a = first_price / np.exp(b * first_normalized) if np.isfinite(np.exp(b * first_normalized)) else 1.0
        print(f"Initial coefficient estimate: {initial_a:.6f}")
    else:
        print(f"Using initial coefficient from config: {initial_a:.6f}")
    
    # Use wide bounds for coefficient a (must be positive)
    wide_min = 1e-10
    wide_max = 1e10
    
    try:
        result = minimize_scalar(objective_function_exponential, bounds=(wide_min, wide_max), method='bounded')
        
        optimal_a = result.x
        optimal_r2 = -result.fun  # Convert back to positive R²
        
        # Validate the result makes sense
        test_pred = exponential_model(df['Days_Since_Genesis'].iloc[-1], optimal_a, b, first_day_offset)
        if not np.isfinite(test_pred) or test_pred > df['Price'].max() * 1000:
            print(f"Warning: Optimal coefficient produces unrealistic predictions. Using initial estimate.")
            optimal_a = initial_a
            optimal_r2 = -objective_function_exponential(optimal_a)
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using initial estimate.")
        optimal_a = initial_a
        optimal_r2 = -objective_function_exponential(optimal_a)
    
    print(f"Optimal coefficient a: {optimal_a:.6f} (optimized to fit all {len(df)} data points)")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_a, optimal_r2

def create_optimization_chart(df, slope, optimal_intercept, optimal_r2):
    """Create a chart showing the optimization results."""
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, slope, optimal_intercept, POWER)
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
        ax.set_title(f'{CURRENCY_NAME} Optimized Model - {ylabel_suffix}\nPower: {POWER:.2f}, Slope: {slope:.6f}, Intercept: {optimal_intercept:.6f}, R²: {optimal_r2:.6f}', 
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
    
    # Save the chart - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_file = os.path.join(script_dir, output_files['intercept_optimization_png'])
    
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

def create_optimization_chart_powerlaw(df, a, b, first_day_offset, optimal_r2):
    """Create a chart showing the power law optimization results."""
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: powerlaw_model(x, a, b, first_day_offset)
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
                color='orange', linestyle='--', linewidth=2, label='Optimized Power Law Model')
    ax2.plot(df['Date'], df['Model_Price'], 
            color='orange', linestyle='--', linewidth=2, label='Optimized Power Law Model')
    
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
        ax.set_title(f'{CURRENCY_NAME} Optimized Power Law Model - {ylabel_suffix}\nCoefficient (a): {a:.6f}, Exponent (b): {b:.6f}, First Day Offset: {first_day_offset:.1f}, R²: {optimal_r2:.6f}', 
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
    
    # Save the chart - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_file = os.path.join(script_dir, output_files['intercept_optimization_png'])
    
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

def create_optimization_chart_exponential(df, a, b, first_day_offset, optimal_r2):
    """Create a chart showing the exponential optimization results."""
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: exponential_model(x, a, b, first_day_offset)
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
                color='red', linestyle='--', linewidth=2, label='Optimized Exponential Model')
    ax2.plot(df['Date'], df['Model_Price'], 
            color='red', linestyle='--', linewidth=2, label='Optimized Exponential Model')
    
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
        growth_str = "growth" if b > 0 else "decay"
        ax.set_title(f'{CURRENCY_NAME} Optimized Exponential Model - {ylabel_suffix}\nCoefficient (a): {a:.6f}, Growth Rate (b): {b:.6f} ({growth_str}), First Day Offset: {first_day_offset:.1f}, R²: {optimal_r2:.6f}', 
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
    
    # Save the chart - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_file = os.path.join(script_dir, output_files['intercept_optimization_png'])
    
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

def calculate_projections(df, slope, intercept):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years (parametric model)."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    max_actual = df['Price'].max()
    min_actual = df['Price'].min()
    
    # Calculate current fair value using parametric model
    current_fair_price = rainbow_model(current_days, slope, intercept, POWER)
    
    # Validate current fair price is reasonable
    if not np.isfinite(current_fair_price) or current_fair_price > max_actual * 1000 or current_fair_price < min_actual * 0.001:
        print(f"WARNING: Model produces unrealistic current fair price: {current_fair_price:.2e}")
        print(f"  Actual price range: ${min_actual:.2f} to ${max_actual:.2f}")
        print(f"  This suggests the model parameters (slope={slope:.6f}, intercept={intercept:.6f}, power={POWER:.6f}) are incorrect.")
        # Use current price as fallback
        current_fair_price = current_price
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = rainbow_model(future_days, slope, intercept, POWER)
        
        # Validate future price is reasonable
        if not np.isfinite(model_future_price) or model_future_price > max_actual * 1e6:
            print(f"WARNING: {years}-year projection is unrealistic: {model_future_price:.2e}")
            # Cap at reasonable maximum
            model_future_price = max_actual * 100
        
        # Both projections use the same dollar value (model_future_price)
        # Model-based projection: percentage relative to fair value
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection: same dollar value, percentage relative to current price
        current_based_price = model_future_price  # Same dollar value
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
        'projections': projections
    }

def calculate_projections_powerlaw(df, a, b, first_day_offset):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years (power law model)."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    
    # Calculate current fair value using power law model
    current_fair_price = powerlaw_model(current_days, a, b, first_day_offset)
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = powerlaw_model(future_days, a, b, first_day_offset)
        
        # Both projections use the same dollar value (model_future_price)
        # Model-based projection: percentage relative to fair value
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection: same dollar value, percentage relative to current price
        current_based_price = model_future_price  # Same dollar value
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
        'projections': projections
    }

def calculate_projections_exponential(df, a, b, first_day_offset):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years (exponential model)."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    
    # Calculate current fair value using exponential model
    current_fair_price = exponential_model(current_days, a, b, first_day_offset)
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = exponential_model(future_days, a, b, first_day_offset)
        
        # Both projections use the same dollar value (model_future_price)
        # Model-based projection: percentage relative to fair value
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection: same dollar value, percentage relative to current price
        current_based_price = model_future_price  # Same dollar value
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
        'projections': projections
    }

def calculate_annual_growth_rate(slope, power, current_days):
    """Calculate annual growth percentage from parametric model parameters.
    
    For parametric model: price = 10^(slope * (days^power) + intercept)
    The growth rate in log space is: d(log10(price))/d(days) = slope * power * (days^(power-1))
    Annual growth percentage = (10^(slope * power * (days^(power-1)) * 365.25) - 1) * 100
    """
    import math
    
    if power == 1.0:
        # Linear case: growth rate is constant
        # d(log10(price))/d(days) = slope
        # Annual growth = (10^(slope * 365.25) - 1) * 100
        annual_growth_factor = 10**(slope * 365.25)
        annual_growth_pct = (annual_growth_factor - 1) * 100
    else:
        # Non-linear case: growth rate depends on current day
        # Calculate growth rate at current day
        growth_rate_per_day = slope * power * (current_days**(power - 1))
        # Annual growth = (10^(growth_rate_per_day * 365.25) - 1) * 100
        annual_growth_factor = 10**(growth_rate_per_day * 365.25)
        annual_growth_pct = (annual_growth_factor - 1) * 100
    
    return annual_growth_pct

def save_optimization_results(slope, optimal_intercept, optimal_r2, projections, df=None):
    """Save optimization results to a text file (parametric model)."""
    
    # Calculate annual growth rate
    # Get current days from dataframe if available, otherwise estimate
    if df is not None and 'Days_Since_Genesis' in df.columns:
        current_days = df['Days_Since_Genesis'].iloc[-1]
    else:
        # Estimate from current price and model
        current_fair_price = projections['current_fair_price']
        if POWER == 1.0:
            # For power=1: log10(price) = slope * days + intercept
            # days = (log10(price) - intercept) / slope
            current_days = (np.log10(current_fair_price) - optimal_intercept) / slope
        else:
            # For other powers, use a reasonable estimate
            current_days = 5500  # Rough estimate for Bitcoin
    
    annual_growth_pct = calculate_annual_growth_rate(slope, POWER, current_days)
    
    # Save - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, output_files['intercept_optimization_txt'])
    
    with open(results_file, 'w') as f:
        f.write(f"{CURRENCY_NAME.upper()} INTERCEPT OPTIMIZATION RESULTS (PARAMETRIC MODEL)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fixed Slope: {slope:.6f}\n")
        f.write(f"Optimal Intercept: {optimal_intercept:.6f}\n")
        f.write(f"Power: {POWER:.6f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n")
        f.write(f"Annual Growth Rate: {annual_growth_pct:+.2f}%\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        f.write(f"price = 10^({slope:.6f} × (day^{POWER:.6f}) + {optimal_intercept:.6f})\n")
        f.write("\n")
        
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
        
        f.write("\nCurrent Price-Based Projections (converges toward model's future fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['current_based_price']:,.0f} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections start from the model's fair value\n")
        f.write("- Current price-based projections show convergence toward model's future fair value\n")
        f.write("- If undervalued: price converges upward then grows with model\n")
        f.write("- If overvalued: price converges downward to model's future fair value\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def calculate_annual_growth_rate_powerlaw(b, current_days, first_day_offset):
    """Calculate annual growth percentage from power law model parameters.
    
    For power law model: price = a * (days - first_day_offset + 1)^b
    The normalized days: x = days - first_day_offset + 1
    After 1 year: x_new = x + 365.25
    Annual growth = ((x + 365.25)/x)^b - 1 = (1 + 365.25/x)^b - 1
    """
    x = current_days - first_day_offset + 1
    if x <= 0:
        x = 1  # Avoid division by zero
    
    # Annual growth factor: (1 + 365.25/x)^b
    annual_growth_factor = (1 + 365.25 / x) ** b
    annual_growth_pct = (annual_growth_factor - 1) * 100
    
    return annual_growth_pct

def save_optimization_results_powerlaw(a, b, first_day_offset, optimal_r2, projections, df=None):
    """Save optimization results to a text file (power law model)."""
    
    # Calculate annual growth rate
    # Get current days from dataframe if available, otherwise estimate
    if df is not None and 'Days_Since_Genesis' in df.columns:
        current_days = df['Days_Since_Genesis'].iloc[-1]
    else:
        # Estimate from current price and model
        current_fair_price = projections['current_fair_price']
        # For power law: price = a * (days - first_day_offset + 1)^b
        # days = (price/a)^(1/b) + first_day_offset - 1
        if a > 0 and b > 0:
            x_est = (current_fair_price / a) ** (1 / b)
            current_days = x_est + first_day_offset - 1
        else:
            current_days = 1000  # Rough estimate
    
    annual_growth_pct = calculate_annual_growth_rate_powerlaw(b, current_days, first_day_offset)
    
    # Save - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, output_files['intercept_optimization_txt'])
    
    with open(results_file, 'w') as f:
        f.write(f"{CURRENCY_NAME.upper()} OPTIMIZATION RESULTS (POWER LAW MODEL)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Optimal Coefficient (a): {a:.6f}\n")
        f.write(f"Exponent (b): {b:.6f}\n")
        f.write(f"First Day Offset: {first_day_offset:.1f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n")
        f.write(f"Annual Growth Rate: {annual_growth_pct:+.2f}%\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        f.write(f"price = {a:.6f} × (day - {first_day_offset:.1f} + 1)^{b:.6f}\n")
        f.write("\n")
        
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
        
        f.write("\nCurrent Price-Based Projections (converges toward model's future fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['current_based_price']:,.0f} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections start from the model's fair value\n")
        f.write("- Current price-based projections show convergence toward model's future fair value\n")
        f.write("- If undervalued: price converges upward then grows with model\n")
        f.write("- If overvalued: price converges downward to model's future fair value\n")
        f.write(f"- Power law exponent b={b:.6f} is constrained to 0 < b < 1 for concave (flattening) shape\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def save_optimization_results_exponential(a, b, first_day_offset, optimal_r2, projections):
    """Save optimization results to a text file (exponential model)."""
    
    # Save - ONLY use script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, output_files['intercept_optimization_txt'])
    
    with open(results_file, 'w') as f:
        f.write(f"{CURRENCY_NAME.upper()} OPTIMIZATION RESULTS (EXPONENTIAL MODEL)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Optimal Coefficient (a): {a:.6f}\n")
        f.write(f"Growth Rate (b): {b:.6f} ({'growth' if b > 0 else 'decay'})\n")
        f.write(f"First Day Offset: {first_day_offset:.1f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        f.write(f"price = {a:.6f} × e^({b:.6f} × (day - {first_day_offset:.1f}))\n")
        f.write("\n")
        
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
        
        f.write("\nCurrent Price-Based Projections (converges toward model's future fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['current_based_price']:,.0f} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections start from the model's fair value\n")
        f.write("- Current price-based projections show convergence toward model's future fair value\n")
        f.write("- If undervalued: price converges upward then grows with model\n")
        f.write("- If overvalued: price converges downward to model's future fair value\n")
        f.write(f"- Exponential model shows {'accelerating growth' if b > 0 else 'decay'} over time\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def optimize_slope_for_r2(df):
    """Find the optimal slope that maximizes R² (optimizing intercept for each slope)."""
    days = df['Days_Since_Genesis'].values
    prices = df['Price'].values
    
    def r2_for_slope(slope):
        """Calculate R² for a given slope (optimizing intercept via linear regression)."""
        # Use linear regression in log space: log10(price) = slope * (days^power) + intercept
        log_prices = np.log10(prices)
        days_powered = days**POWER
        
        # Calculate intercept that minimizes error for this slope
        # intercept = mean(log_prices) - slope * mean(days_powered)
        intercept = np.mean(log_prices) - slope * np.mean(days_powered)
        
        y_pred = rainbow_model(days, slope, intercept, POWER)
        
        # Check if predictions are reasonable
        max_pred = np.max(y_pred)
        max_actual = np.max(prices)
        if max_pred > max_actual * 1000 or max_pred < max_actual * 0.001:
            return -10000
        
        r2 = calculate_r2(prices, y_pred)
        if not np.isfinite(r2):
            return -10000
        return r2
    
    # Use reasonable bounds for slope
    wide_min = 0.1
    wide_max = 10.0
    
    try:
        result = minimize_scalar(lambda s: -r2_for_slope(s), bounds=(wide_min, wide_max), method='bounded')
        optimal_slope = result.x
        optimal_r2 = -result.fun
        
        # If we hit a boundary, try wider
        if (abs(optimal_slope - wide_min) < 0.01 or abs(optimal_slope - wide_max) < 0.01):
            print(f"Slope hit boundary. Trying wider range...")
            wider_min = 0.01
            wider_max = 20.0
            result = minimize_scalar(lambda s: -r2_for_slope(s), bounds=(wider_min, wider_max), method='bounded')
            optimal_slope = result.x
            optimal_r2 = -result.fun
            
    except Exception as e:
        print(f"Warning: Optimization failed: {e}. Using default slope.")
        optimal_slope = 2.0
        optimal_r2 = r2_for_slope(optimal_slope)
    
    return optimal_slope, optimal_r2

def main():
    """Main function to optimize intercept (parametric) or coefficient (power law)."""
    
    # Declare all global variables at the start (before any use)
    global POWER_LAW_B, POWER_LAW_A, POWER_LAW_FIRST_DAY_OFFSET
    global EXPONENTIAL_B, EXPONENTIAL_A, EXPONENTIAL_FIRST_DAY_OFFSET
    global FIXED_SLOPE, POWER
    
    # Check for saved model type from interactive_point_selector.py
    # Only use it if config doesn't explicitly set MODEL_TYPE or parameters
    saved_model = load_saved_model_type()
    if saved_model:
        print(f"Found saved model: {saved_model.get('model_type', 'unknown')} from {saved_model.get('timestamp', 'unknown time')}")
    # Handle empty string MODEL_TYPE as "not set"
    model_type_to_use = MODEL_TYPE if MODEL_TYPE and MODEL_TYPE.strip() else None
    use_saved_model = False
    
    # ALWAYS prioritize saved model parameters - they were optimized together
    if saved_model and saved_model.get('model_type') == 'parametric':
        saved_slope = saved_model.get('slope')
        saved_power = saved_model.get('power')
        
        if saved_slope is not None:
            # Always use saved slope (override config if different)
            if FIXED_SLOPE is None or (FIXED_SLOPE is not None and abs(FIXED_SLOPE - saved_slope) > 0.0001):
                if FIXED_SLOPE is not None:
                    print(f"\nUsing saved slope: {saved_slope:.6f} (overriding config slope: {FIXED_SLOPE:.6f})")
                else:
                    print(f"\nUsing saved slope: {saved_slope:.6f}")
                FIXED_SLOPE = saved_slope
        
        if saved_power is not None:
            # Always use saved power (override config if different)
            if POWER is None or abs(POWER - saved_power) > 0.0001:
                if POWER is not None:
                    print(f"Using saved power: {saved_power:.6f} (overriding config power: {POWER:.6f})")
                else:
                    print(f"Using saved power: {saved_power:.6f}")
                POWER = saved_power
            else:
                print(f"Using saved power: {POWER:.6f}")
        
        print(f"  Note: Using saved model parameters (slope={FIXED_SLOPE:.6f}, power={POWER:.6f}) from interactive_point_selector.py")
    
    # Determine if config explicitly sets model type or parameters
    config_explicitly_sets_model = False
    
    # Check if config explicitly sets model-specific parameters
    # Empty string or None means not explicitly set
    if model_type_to_use:  # Not empty string or None
        if model_type_to_use == 'powerlaw' and POWER_LAW_B is not None:
            config_explicitly_sets_model = True
        elif model_type_to_use == 'exponential' and EXPONENTIAL_B is not None:
            config_explicitly_sets_model = True
        elif model_type_to_use == 'parametric' and FIXED_SLOPE is not None:
            # For parametric, we already handled power above, so just check slope
            config_explicitly_sets_model = True
    
    # If config doesn't explicitly set model, check for saved model
    if not config_explicitly_sets_model and saved_model and saved_model.get('model_type'):
        saved_type = saved_model['model_type']
        # Only use saved type if it's not 'pchip' (pchip is interpolation only)
        if saved_type in ['parametric', 'powerlaw', 'exponential']:
            model_type_to_use = saved_type
            use_saved_model = True
            print(f"\nUsing saved model type from interactive_point_selector.py: {saved_type}")
            if model_type_to_use:
                print(f"(Config MODEL_TYPE='{model_type_to_use}' and no explicit parameters set)")
            else:
                print(f"(Config MODEL_TYPE not set, using saved model)")
            
            # Update global variables with saved parameters (only if config doesn't set them)
            if saved_type == 'powerlaw' and 'powerlaw_b' in saved_model:
                if POWER_LAW_B is None:
                    POWER_LAW_B = saved_model.get('powerlaw_b')
                if POWER_LAW_A is None:
                    POWER_LAW_A = saved_model.get('powerlaw_a')
                if POWER_LAW_FIRST_DAY_OFFSET is None:
                    POWER_LAW_FIRST_DAY_OFFSET = saved_model.get('powerlaw_first_day_offset')
                print(f"  Using saved power law parameters: b={POWER_LAW_B}, a={POWER_LAW_A}, offset={POWER_LAW_FIRST_DAY_OFFSET}")
            elif saved_type == 'exponential' and 'exponential_b' in saved_model:
                if EXPONENTIAL_B is None:
                    EXPONENTIAL_B = saved_model.get('exponential_b')
                if EXPONENTIAL_A is None:
                    EXPONENTIAL_A = saved_model.get('exponential_a')
                if EXPONENTIAL_FIRST_DAY_OFFSET is None:
                    EXPONENTIAL_FIRST_DAY_OFFSET = saved_model.get('exponential_first_day_offset')
                print(f"  Using saved exponential parameters: b={EXPONENTIAL_B}, a={EXPONENTIAL_A}, offset={EXPONENTIAL_FIRST_DAY_OFFSET}")
            elif saved_type == 'parametric':
                # Already handled power above, just handle slope if not set
                if FIXED_SLOPE is None and 'slope' in saved_model:
                    FIXED_SLOPE = saved_model.get('slope')
                print(f"  Using saved parametric parameters: slope={FIXED_SLOPE}, power={POWER}")
    elif config_explicitly_sets_model:
        print(f"\nUsing model type and parameters from config.py")
        if model_type_to_use == 'parametric':
            print(f"  (Note: Power was set to {POWER:.6f} from saved model to match slope)")
    
    # If still no model type determined, default to parametric
    if not model_type_to_use:
        model_type_to_use = 'parametric'
        print(f"\nNo model type specified in config or saved. Defaulting to parametric model.")
    
    print(f"{CURRENCY_NAME} Optimization ({model_type_to_use.upper()} Model)")
    print("=" * 40)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    if model_type_to_use == 'powerlaw':
        # Power law model: optimize coefficient a
        if POWER_LAW_B is None or POWER_LAW_B <= 0 or POWER_LAW_B >= 1:
            print("Error: POWER_LAW_B must be set in config.py and must be 0 < b < 1")
            print("Run interactive_point_selector.py and click 'Power Law' to get the values.")
            return
        
        # Determine first_day_offset
        if POWER_LAW_FIRST_DAY_OFFSET is None:
            # Auto-detect: use minimum days since genesis in data
            first_day_offset = df['Days_Since_Genesis'].min()
            print(f"Auto-detected first_day_offset: {first_day_offset:.1f} days")
        else:
            first_day_offset = POWER_LAW_FIRST_DAY_OFFSET
            print(f"Using configured first_day_offset: {first_day_offset:.1f} days")
        
        # Always optimize coefficient a to maximize R² across ALL price points
        # Using b and first_day_offset from config (from interactive point selector) as fixed shape parameters
        print(f"\nUsing power law shape parameters from config (from selected points):")
        print(f"  Exponent (b): {POWER_LAW_B:.6f} (fixed - determines curve shape)")
        print(f"  First Day Offset: {first_day_offset:.1f} (fixed - normalization reference)")
        print(f"\nOptimizing coefficient a to maximize R² across ALL {len(df)} price points...")
        print(f"(This adjusts the vertical position to best fit all data, not just selected points)")
        optimal_a, optimal_r2 = optimize_coefficient_for_r2(df, POWER_LAW_B, first_day_offset, initial_a=POWER_LAW_A)
        
        # Create optimization chart
        print("\nCreating optimization chart...")
        chart_file = create_optimization_chart_powerlaw(df, optimal_a, POWER_LAW_B, first_day_offset, optimal_r2)
        print(f"Chart saved: {chart_file}")
        
        # Calculate projections
        print("\nCalculating forward projections...")
        projections = calculate_projections_powerlaw(df, optimal_a, POWER_LAW_B, first_day_offset)
        
        # Save results
        print("\nSaving optimization results...")
        results_file = save_optimization_results_powerlaw(optimal_a, POWER_LAW_B, first_day_offset, optimal_r2, projections, df=df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Optimal Coefficient (a): {optimal_a:.6f}")
        print(f"Exponent (b): {POWER_LAW_B:.6f}")
        print(f"First Day Offset: {first_day_offset:.1f}")
        print(f"Maximum R²: {optimal_r2:.6f}")
        print(f"Current Price: ${projections['current_price']:,.2f}")
        print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
        print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
        print(f"\nResults saved to: {results_file}")
    
    elif model_type_to_use == 'exponential':
        # Exponential model: optimize coefficient a
        if EXPONENTIAL_B is None:
            print("Error: EXPONENTIAL_B must be set in config.py")
            print("Run interactive_point_selector.py and click 'Exponential' to get the values.")
            return
        
        # Determine first_day_offset
        if EXPONENTIAL_FIRST_DAY_OFFSET is None:
            # Auto-detect: use minimum days since genesis in data
            first_day_offset = df['Days_Since_Genesis'].min()
            print(f"Auto-detected first_day_offset: {first_day_offset:.1f} days")
        else:
            first_day_offset = EXPONENTIAL_FIRST_DAY_OFFSET
            print(f"Using configured first_day_offset: {first_day_offset:.1f} days")
        
        # Always optimize coefficient a to maximize R² across ALL price points
        # Using b and first_day_offset from config (from interactive point selector) as fixed shape parameters
        growth_str = "growth" if EXPONENTIAL_B > 0 else "decay"
        print(f"\nUsing exponential shape parameters from config (from selected points):")
        print(f"  Growth Rate (b): {EXPONENTIAL_B:.6f} (fixed - determines {growth_str} rate)")
        print(f"  First Day Offset: {first_day_offset:.1f} (fixed - normalization reference)")
        print(f"\nOptimizing coefficient a to maximize R² across ALL {len(df)} price points...")
        print(f"(This adjusts the vertical position to best fit all data, not just selected points)")
        optimal_a, optimal_r2 = optimize_coefficient_for_r2_exponential(df, EXPONENTIAL_B, first_day_offset, initial_a=EXPONENTIAL_A)
        
        # Create optimization chart
        print("\nCreating optimization chart...")
        chart_file = create_optimization_chart_exponential(df, optimal_a, EXPONENTIAL_B, first_day_offset, optimal_r2)
        print(f"Chart saved: {chart_file}")
        
        # Calculate projections
        print("\nCalculating forward projections...")
        projections = calculate_projections_exponential(df, optimal_a, EXPONENTIAL_B, first_day_offset)
        
        # Save results
        print("\nSaving optimization results...")
        results_file = save_optimization_results_exponential(optimal_a, EXPONENTIAL_B, first_day_offset, optimal_r2, projections)
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Optimal Coefficient (a): {optimal_a:.6f}")
        print(f"Growth Rate (b): {EXPONENTIAL_B:.6f} ({growth_str})")
        print(f"First Day Offset: {first_day_offset:.1f}")
        print(f"Maximum R²: {optimal_r2:.6f}")
        print(f"Current Price: ${projections['current_price']:,.2f}")
        print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
        print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
        print(f"\nResults saved to: {results_file}")
    
    else:  # parametric model
        print(f"Using Power: {POWER:.6f}")
        
        # Use saved slope if available, otherwise use config, otherwise optimize
        if FIXED_SLOPE is not None:
            print(f"Using slope: {FIXED_SLOPE:.6f}")
            fixed_slope = FIXED_SLOPE
        else:
            print("\nOptimizing slope to maximize R²...")
            fixed_slope, _ = optimize_slope_for_r2(df)
            print(f"Using optimized slope: {fixed_slope:.6f}")
        
        # Now optimize intercept for the fixed slope
        print(f"\nOptimizing intercept to maximize R² for the fixed slope...")
        optimal_intercept, optimal_r2 = optimize_intercept_for_r2(df, fixed_slope)
        
        # Create optimization chart
        print("\nCreating optimization chart...")
        chart_file = create_optimization_chart(df, fixed_slope, optimal_intercept, optimal_r2)
        print(f"Chart saved: {chart_file}")
        
        # Calculate projections
        print("\nCalculating forward projections...")
        projections = calculate_projections(df, fixed_slope, optimal_intercept)
        
        # Save results
        print("\nSaving optimization results...")
        results_file = save_optimization_results(fixed_slope, optimal_intercept, optimal_r2, projections, df=df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Fixed Slope: {fixed_slope:.6f}")
        print(f"Optimal Intercept: {optimal_intercept:.6f}")
        print(f"Maximum R²: {optimal_r2:.6f}")
        print(f"Current Price: ${projections['current_price']:,.2f}")
        print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
        print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
        print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

