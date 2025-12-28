"""
TREND + CYCLE MODEL - MODERN REGIME VALUATION
=============================================
This script fits a modern trend (post-regime shift) and creates valuation bands
based on cyclical deviations. This gives actionable "cheap/fair/rich" signals.

Usage:
    python trend_cycle_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
import os
import sys

# Import configuration
try:
    from config import (
        CURRENCY_NAME, CURRENCY_NAME_LOWER, DATA_FOLDER, HISTORICAL_DATA_FILE,
        OUTPUT_FOLDER, START_DATE, GENESIS_DATE,
        get_output_files, validate_config
    )
except ImportError:
    print("Error: config.py not found! Make sure you're running from the log_model folder.")
    sys.exit(1)

validate_config()
output_files = get_output_files()

def load_data():
    """Load and prepare price data."""
    # Handle relative paths - go up one level if running from log_model folder
    if os.path.basename(os.getcwd()) == 'log_model':
        data_file = os.path.join('..', DATA_FOLDER, HISTORICAL_DATA_FILE)
    else:
        data_file = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILE)
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found - {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    
    # Handle different date formats
    date_cols = ['Date', 'date', 'DATE']
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Error: No date column found in data file")
        return None
    
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Handle price column
    price_cols = ['Price', 'price', 'PRICE', 'Close', 'close', 'CLOSE']
    price_col = None
    for col in price_cols:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print("Error: No price column found in data file")
        return None
    
    # Convert price to float (handle commas)
    if df[price_col].dtype == 'object':
        df[price_col] = df[price_col].astype(str).str.replace(',', '').astype(float)
    else:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    df = df.rename(columns={date_col: 'Date', price_col: 'Price'})
    df = df[['Date', 'Price']].copy()
    df = df.dropna()
    
    # Calculate days since genesis
    if GENESIS_DATE is not None and GENESIS_DATE != "":
        try:
            genesis = pd.to_datetime(GENESIS_DATE)
        except:
            genesis = df['Date'].min()
    else:
        genesis = df['Date'].min()
    
    df['Days_Since_Genesis'] = (df['Date'] - genesis).dt.days + 1
    df['Days_Since_Genesis'] = df['Days_Since_Genesis'].clip(lower=1)
    
    # Filter by START_DATE if specified
    if START_DATE and START_DATE != "":
        try:
            start = pd.to_datetime(START_DATE)
            df = df[df['Date'] >= start].copy()
        except:
            pass
    
    # Filter out invalid prices
    df = df[df['Price'] > 0].copy()
    
    return df

def rainbow_model(x, slope, intercept):
    """Logarithmic growth model: price = 10^(slope * ln(days) + intercept)"""
    x = np.maximum(x, 1)
    return 10**(slope * np.log(x) + intercept)

def fit_modern_trend(df, trend_start_date=None):
    """
    Fit a trend model starting from a specific date (modern regime).
    If trend_start_date is None, uses the most recent 5 years.
    """
    if trend_start_date is None:
        # Use most recent 5 years
        trend_start_date = df['Date'].max() - pd.Timedelta(days=5*365.25)
    
    trend_start = pd.to_datetime(trend_start_date)
    df_trend = df[df['Date'] >= trend_start].copy()
    
    if len(df_trend) < 30:
        print(f"Warning: Not enough data for trend fitting. Using all data.")
        df_trend = df.copy()
    
    days = df_trend['Days_Since_Genesis'].values
    prices = df_trend['Price'].values
    first_price = prices[0]
    first_days = days[0]
    
    def r2_for_slope(slope):
        intercept = np.log10(first_price) - slope * np.log(first_days)
        y_pred = rainbow_model(days, slope, intercept)
        
        # Calculate R² in log space
        log_actual = np.log10(prices)
        log_pred = np.log10(y_pred)
        
        ss_res = np.sum((log_actual - log_pred) ** 2)
        ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        if not np.isfinite(r2):
            return -10000
        return r2
    
    try:
        result = minimize_scalar(lambda s: -r2_for_slope(s), bounds=(0.1, 10.0), method='bounded')
        optimal_slope = result.x
        optimal_r2 = -result.fun
        optimal_intercept = np.log10(first_price) - optimal_slope * np.log(first_days)
        
        return optimal_slope, optimal_intercept, optimal_r2, trend_start
    except Exception as e:
        print(f"Warning: Optimization failed: {e}")
        # Fallback to default
        optimal_slope = 1.5
        optimal_intercept = np.log10(first_price) - optimal_slope * np.log(first_days)
        optimal_r2 = r2_for_slope(optimal_slope)
        return optimal_slope, optimal_intercept, optimal_r2, trend_start

def calculate_valuation_bands(df, slope, intercept):
    """Calculate valuation bands based on residuals from trend."""
    # Calculate model predictions
    df['Model_Price'] = df['Days_Since_Genesis'].apply(
        lambda x: rainbow_model(x, slope, intercept)
    )
    
    # Calculate residuals in log space
    df['Log_Price'] = np.log10(df['Price'])
    df['Log_Model'] = np.log10(df['Model_Price'])
    df['Log_Residual'] = df['Log_Price'] - df['Log_Model']
    
    # Calculate standard deviation of residuals
    std_residual = df['Log_Residual'].std()
    mean_residual = df['Log_Residual'].mean()
    
    # Create bands: ±1σ, ±2σ
    df['Band_Upper_2'] = 10**(df['Log_Model'] + mean_residual + 2*std_residual)
    df['Band_Upper_1'] = 10**(df['Log_Model'] + mean_residual + 1*std_residual)
    df['Band_Lower_1'] = 10**(df['Log_Model'] + mean_residual - 1*std_residual)
    df['Band_Lower_2'] = 10**(df['Log_Model'] + mean_residual - 2*std_residual)
    
    return df, std_residual, mean_residual

def get_valuation_status(current_price, model_price, std_residual, mean_residual):
    """Determine if current price is cheap/fair/rich."""
    log_current = np.log10(current_price)
    log_model = np.log10(model_price)
    log_residual = log_current - log_model
    
    # Calculate how many standard deviations away from mean
    z_score = (log_residual - mean_residual) / std_residual if std_residual > 0 else 0
    
    if z_score <= -2:
        status = "Firesale! (Very Undervalued)"
        color = "blue"
    elif z_score <= -1:
        status = "Buy (Undervalued)"
        color = "green"
    elif z_score <= 1:
        status = "Hold (Fair Value)"
        color = "yellow"
    elif z_score <= 2:
        status = "Sell (Overvalued)"
        color = "orange"
    else:
        status = "Danger! (Very Overvalued)"
        color = "red"
    
    deviation_pct = ((current_price - model_price) / model_price) * 100
    
    return status, color, z_score, deviation_pct

def create_visualization(df, slope, intercept, std_residual, mean_residual, trend_start):
    """Create visualization of trend + cycle model with valuation bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Log scale
    ax1.semilogy(df['Date'], df['Price'], 'b-', linewidth=1.5, label='Actual Price', alpha=0.7)
    ax1.semilogy(df['Date'], df['Model_Price'], 'r-', linewidth=2, label='Trend Model')
    ax1.fill_between(df['Date'], df['Band_Lower_2'], df['Band_Upper_2'], 
                     alpha=0.1, color='red', label='±2σ Band')
    ax1.fill_between(df['Date'], df['Band_Lower_1'], df['Band_Upper_1'], 
                     alpha=0.2, color='orange', label='±1σ Band')
    ax1.axvline(x=trend_start, color='purple', linestyle='--', linewidth=2, 
                label=f'Trend Start: {trend_start.strftime("%Y-%m-%d")}')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (Log Scale)', fontsize=12)
    ax1.set_title(f'{CURRENCY_NAME} Trend + Cycle Model (Log Scale)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Normal scale
    ax2.plot(df['Date'], df['Price'], 'b-', linewidth=1.5, label='Actual Price', alpha=0.7)
    ax2.plot(df['Date'], df['Model_Price'], 'r-', linewidth=2, label='Trend Model')
    ax2.fill_between(df['Date'], df['Band_Lower_2'], df['Band_Upper_2'], 
                     alpha=0.1, color='red', label='±2σ Band')
    ax2.fill_between(df['Date'], df['Band_Lower_1'], df['Band_Upper_1'], 
                     alpha=0.2, color='orange', label='±1σ Band')
    ax2.axvline(x=trend_start, color='purple', linestyle='--', linewidth=2, 
                label=f'Trend Start: {trend_start.strftime("%Y-%m-%d")}')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.set_title(f'{CURRENCY_NAME} Trend + Cycle Model (Normal Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis for normal scale
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    # Save figure - handle relative paths
    if os.path.basename(os.getcwd()) == 'log_model':
        output_file = os.path.join('.', f"{CURRENCY_NAME_LOWER}_trend_cycle_model.png")
    else:
        output_file = os.path.join(OUTPUT_FOLDER, f"{CURRENCY_NAME_LOWER}_trend_cycle_model.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved: {output_file}")
    return output_file

def generate_valuation_report(df, slope, intercept, std_residual, mean_residual, 
                             trend_start, status, z_score, deviation_pct):
    """Generate valuation report."""
    # Handle relative paths
    if os.path.basename(os.getcwd()) == 'log_model':
        report_file = os.path.join('.', f"{CURRENCY_NAME_LOWER}_valuation_report.txt")
    else:
        report_file = os.path.join(OUTPUT_FOLDER, f"{CURRENCY_NAME_LOWER}_valuation_report.txt")
    
    current_price = df['Price'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    model_price = df['Model_Price'].iloc[-1]
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BITCOIN TREND + CYCLE MODEL - VALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Current Date: {current_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Trend Period: {trend_start.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}\n\n")
        
        f.write("CURRENT VALUATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Current Price: ${current_price:,.2f}\n")
        f.write(f"Model Fair Value: ${model_price:,.2f}\n")
        f.write(f"Deviation from Trend: {deviation_pct:+.2f}%\n")
        f.write(f"Z-Score: {z_score:.2f} standard deviations\n")
        f.write(f"Valuation Status: {status}\n\n")
        
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Slope: {slope:.6f}\n")
        f.write(f"Intercept: {intercept:.6f}\n")
        f.write(f"Formula: price = 10^({slope:.6f} × ln(days) + {intercept:.6f})\n")
        f.write(f"Residual Std Dev: {std_residual:.6f} (log space)\n")
        f.write(f"Mean Residual: {mean_residual:.6f} (log space)\n\n")
        
        f.write("VALUATION BANDS\n")
        f.write("-" * 70 + "\n")
        f.write("Based on historical deviations from trend:\n")
        f.write(f"  • -2 std dev (Firesale!): ${df['Band_Lower_2'].iloc[-1]:,.2f}\n")
        f.write(f"  • -1 std dev (Buy): ${df['Band_Lower_1'].iloc[-1]:,.2f}\n")
        f.write(f"  • Trend (Fair): ${model_price:,.2f}\n")
        f.write(f"  • +1 std dev (Sell): ${df['Band_Upper_1'].iloc[-1]:,.2f}\n")
        f.write(f"  • +2 std dev (Danger!): ${df['Band_Upper_2'].iloc[-1]:,.2f}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("This model fits a trend to the MODERN REGIME (post-regime shift).\n")
        f.write("Valuation bands are based on historical cyclical deviations from this trend.\n\n")
        f.write("• The trend represents structural growth in the current regime\n")
        f.write("• Bands show where price typically cycles around the trend\n")
        f.write("• Current valuation is relative to this modern trend, not historical data\n")
        f.write("• This avoids the problem of anchoring to early low-liquidity years\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        if "Undervalued" in status or "Firesale" in status:
            f.write("• Current price is below trend - potential buying opportunity\n")
            f.write("• Consider accumulating if price approaches -2σ band\n")
        elif "Overvalued" in status or "Danger" in status:
            f.write("• Current price is above trend - consider taking profits\n")
            f.write("• Be cautious if price exceeds +2σ band\n")
        else:
            f.write("• Current price is near fair value - hold position\n")
            f.write("• Wait for clearer signals (move toward bands)\n")
        f.write("\n")
        
        f.write("IMPORTANT NOTES\n")
        f.write("-" * 70 + "\n")
        f.write("• This model uses only MODERN data (post-regime shift)\n")
        f.write("• It acknowledges that growth dynamics have changed\n")
        f.write("• Valuation is relative to CURRENT regime, not historical average\n")
        f.write("• Bands are based on recent volatility patterns\n")
        f.write("• This is more defensible than a single-slope model for all time\n")
    
    print(f"[OK] Valuation report saved: {report_file}")
    return report_file

def main():
    """Main analysis function."""
    print("=" * 70)
    print("BITCOIN TREND + CYCLE MODEL ANALYSIS")
    print("=" * 70)
    print("\nLoading data...")
    
    df = load_data()
    if df is None:
        return
    
    print(f"[OK] Loaded {len(df)} data points")
    print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n")
    
    # Use post-2020 data for modern trend (institutional era)
    # Or use START_DATE if specified
    if START_DATE and START_DATE != "":
        trend_start = pd.to_datetime(START_DATE)
        print(f"Using configured trend start date: {trend_start.strftime('%Y-%m-%d')}")
    else:
        # Default to 2020-03-01 (institutional wave start)
        trend_start = pd.to_datetime("2020-03-01")
        if trend_start < df['Date'].min():
            trend_start = df['Date'].min() + pd.Timedelta(days=365*3)  # Use 3 years from start
        print(f"Using default trend start date: {trend_start.strftime('%Y-%m-%d')}")
    
    print("\nFitting modern trend model...")
    slope, intercept, r2, trend_start = fit_modern_trend(df, trend_start)
    print(f"[OK] Trend model fitted")
    print(f"   Slope: {slope:.6f}")
    print(f"   Intercept: {intercept:.6f}")
    print(f"   R²: {r2:.6f}")
    print(f"   Trend period: {trend_start.strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n")
    
    print("Calculating valuation bands...")
    df, std_residual, mean_residual = calculate_valuation_bands(df, slope, intercept)
    print(f"[OK] Valuation bands calculated")
    print(f"   Residual std dev: {std_residual:.6f}\n")
    
    print("Determining current valuation status...")
    current_price = df['Price'].iloc[-1]
    model_price = df['Model_Price'].iloc[-1]
    status, color, z_score, deviation_pct = get_valuation_status(
        current_price, model_price, std_residual, mean_residual
    )
    print(f"[OK] Current valuation: {status}")
    print(f"   Current price: ${current_price:,.2f}")
    print(f"   Model fair value: ${model_price:,.2f}")
    print(f"   Deviation: {deviation_pct:+.2f}%\n")
    
    print("Creating visualizations...")
    create_visualization(df, slope, intercept, std_residual, mean_residual, trend_start)
    
    print("Generating valuation report...")
    generate_valuation_report(df, slope, intercept, std_residual, mean_residual, 
                            trend_start, status, z_score, deviation_pct)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nCheck the following files in {OUTPUT_FOLDER}:")
    print(f"  • {CURRENCY_NAME_LOWER}_trend_cycle_model.png (visualization)")
    print(f"  • {CURRENCY_NAME_LOWER}_valuation_report.txt (detailed report)")
    print(f"\nCurrent Valuation: {status}")

if __name__ == "__main__":
    main()

