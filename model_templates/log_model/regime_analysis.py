"""
REGIME SHIFT ANALYSIS - ROLLING REGRESSION
==========================================
This script analyzes how Bitcoin's growth model parameters (slope, intercept) 
change over time using rolling windows. This helps identify regime shifts.

Usage:
    python regime_analysis.py
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

def fit_model_to_window(df_window):
    """Fit logarithmic model to a data window. Returns slope, intercept, R²."""
    if len(df_window) < 30:  # Need at least 30 data points
        return None, None, None
    
    days = df_window['Days_Since_Genesis'].values
    prices = df_window['Price'].values
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
        
        return optimal_slope, optimal_intercept, optimal_r2
    except:
        return None, None, None

def rolling_regression_analysis(df, window_years=3):
    """Perform rolling regression analysis with specified window size."""
    window_days = int(window_years * 365.25)
    
    results = []
    
    # Start from the window size, slide by 30 days
    start_idx = 0
    end_idx = start_idx + window_days
    
    while end_idx <= len(df):
        window_df = df.iloc[start_idx:end_idx].copy()
        center_date = window_df['Date'].iloc[len(window_df) // 2]
        
        slope, intercept, r2 = fit_model_to_window(window_df)
        
        if slope is not None:
            results.append({
                'Date': center_date,
                'Slope': slope,
                'Intercept': intercept,
                'R2': r2,
                'Window_Start': window_df['Date'].iloc[0],
                'Window_End': window_df['Date'].iloc[-1]
            })
        
        # Slide window forward by 30 days
        start_idx += 30
        end_idx = start_idx + window_days
    
    return pd.DataFrame(results)

def identify_regime_shifts(results_df):
    """Identify potential regime shifts based on slope changes."""
    if len(results_df) < 2:
        return []
    
    # Calculate slope changes
    results_df = results_df.sort_values('Date').reset_index(drop=True)
    results_df['Slope_Change'] = results_df['Slope'].diff()
    results_df['Slope_Change_Pct'] = results_df['Slope'].pct_change() * 100
    
    # Identify significant changes (>20% change in slope)
    significant_changes = results_df[abs(results_df['Slope_Change_Pct']) > 20].copy()
    
    regime_shifts = []
    for _, row in significant_changes.iterrows():
        regime_shifts.append({
            'Date': row['Date'],
            'Slope_Before': results_df[results_df['Date'] < row['Date']]['Slope'].iloc[-1] if len(results_df[results_df['Date'] < row['Date']]) > 0 else None,
            'Slope_After': row['Slope'],
            'Change_Pct': row['Slope_Change_Pct']
        })
    
    return regime_shifts

def create_visualizations(df, results_3yr, results_5yr, regime_shifts):
    """Create visualization of rolling regression results."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Plot 1: Slope over time (3-year window)
    ax1 = axes[0]
    ax1.plot(results_3yr['Date'], results_3yr['Slope'], 'b-', linewidth=2, label='3-Year Window')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Slope', fontsize=12)
    ax1.set_title('Bitcoin Growth Model Slope Over Time (3-Year Rolling Window)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark regime shifts
    for shift in regime_shifts:
        ax1.axvline(x=shift['Date'], color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Slope over time (5-year window)
    ax2 = axes[1]
    ax2.plot(results_5yr['Date'], results_5yr['Slope'], 'g-', linewidth=2, label='5-Year Window')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Slope', fontsize=12)
    ax2.set_title('Bitcoin Growth Model Slope Over Time (5-Year Rolling Window)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: R² over time (both windows)
    ax3 = axes[2]
    ax3.plot(results_3yr['Date'], results_3yr['R2'], 'b-', linewidth=2, label='3-Year Window R²')
    ax3.plot(results_5yr['Date'], results_5yr['R2'], 'g-', linewidth=2, label='5-Year Window R²')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Model Fit Quality Over Time (R² Score)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure - handle relative paths
    if os.path.basename(os.getcwd()) == 'log_model':
        output_file = os.path.join('.', f"{CURRENCY_NAME_LOWER}_regime_analysis.png")
    else:
        output_file = os.path.join(OUTPUT_FOLDER, f"{CURRENCY_NAME_LOWER}_regime_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved: {output_file}")
    return output_file

def generate_findings_report(df, results_3yr, results_5yr, regime_shifts):
    """Generate a text report of findings."""
    # Handle relative paths
    if os.path.basename(os.getcwd()) == 'log_model':
        report_file = os.path.join('.', f"{CURRENCY_NAME_LOWER}_regime_findings.txt")
    else:
        report_file = os.path.join(OUTPUT_FOLDER, f"{CURRENCY_NAME_LOWER}_regime_findings.txt")
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BITCOIN REGIME SHIFT ANALYSIS - FINDINGS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write("This analysis uses rolling regression windows to track how Bitcoin's\n")
        f.write("growth model parameters (slope, intercept) change over time.\n")
        f.write("This helps identify regime shifts where growth dynamics fundamentally change.\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n\n")
        
        # 3-Year Window Summary
        f.write("3-YEAR ROLLING WINDOW ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of windows analyzed: {len(results_3yr)}\n")
        f.write(f"Average slope: {results_3yr['Slope'].mean():.4f}\n")
        f.write(f"Slope range: {results_3yr['Slope'].min():.4f} to {results_3yr['Slope'].max():.4f}\n")
        f.write(f"Average R²: {results_3yr['R2'].mean():.4f}\n")
        f.write(f"R² range: {results_3yr['R2'].min():.4f} to {results_3yr['R2'].max():.4f}\n\n")
        
        # 5-Year Window Summary
        f.write("5-YEAR ROLLING WINDOW ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of windows analyzed: {len(results_5yr)}\n")
        f.write(f"Average slope: {results_5yr['Slope'].mean():.4f}\n")
        f.write(f"Slope range: {results_5yr['Slope'].min():.4f} to {results_5yr['Slope'].max():.4f}\n")
        f.write(f"Average R²: {results_5yr['R2'].mean():.4f}\n")
        f.write(f"R² range: {results_5yr['R2'].min():.4f} to {results_5yr['R2'].max():.4f}\n\n")
        
        # Recent trends
        f.write("RECENT TRENDS (Last 2 Years)\n")
        f.write("-" * 70 + "\n")
        recent_3yr = results_3yr[results_3yr['Date'] >= results_3yr['Date'].max() - pd.Timedelta(days=730)]
        recent_5yr = results_5yr[results_5yr['Date'] >= results_5yr['Date'].max() - pd.Timedelta(days=730)]
        
        if len(recent_3yr) > 0:
            f.write(f"3-Year Window - Recent Average Slope: {recent_3yr['Slope'].mean():.4f}\n")
            f.write(f"3-Year Window - Recent Average R²: {recent_3yr['R2'].mean():.4f}\n")
        if len(recent_5yr) > 0:
            f.write(f"5-Year Window - Recent Average Slope: {recent_5yr['Slope'].mean():.4f}\n")
            f.write(f"5-Year Window - Recent Average R²: {recent_5yr['R2'].mean():.4f}\n")
        f.write("\n")
        
        # Regime shifts
        f.write("IDENTIFIED REGIME SHIFTS\n")
        f.write("-" * 70 + "\n")
        if len(regime_shifts) > 0:
            for i, shift in enumerate(regime_shifts, 1):
                f.write(f"\nShift #{i}:\n")
                f.write(f"  Date: {shift['Date'].strftime('%Y-%m-%d')}\n")
                if shift['Slope_Before'] is not None:
                    f.write(f"  Slope Before: {shift['Slope_Before']:.4f}\n")
                f.write(f"  Slope After: {shift['Slope_After']:.4f}\n")
                f.write(f"  Change: {shift['Change_Pct']:.2f}%\n")
        else:
            f.write("No significant regime shifts detected (>20% slope change).\n")
        f.write("\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("• A DECREASING slope indicates growth is slowing over time\n")
        f.write("• A STABLE slope suggests consistent growth dynamics\n")
        f.write("• A DECREASING R² suggests the model fits less well (regime shift)\n")
        f.write("• Significant slope changes (>20%) may indicate regime shifts\n")
        f.write("• 3-year windows are more sensitive to recent changes\n")
        f.write("• 5-year windows provide more stable, long-term trends\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        f.write("1. If slope is decreasing over time, consider using a more recent\n")
        f.write("   time period for the growth model (e.g., post-2020 data)\n")
        f.write("2. If R² is decreasing, the single-slope model may no longer\n")
        f.write("   be appropriate - consider piecewise or trend+cycle models\n")
        f.write("3. Use the identified regime shift dates as breakpoints for\n")
        f.write("   piecewise models if significant shifts are detected\n")
        f.write("4. Compare out-of-sample performance: fit model up to 2020,\n")
        f.write("   then test predictions on 2021-2025 data\n\n")
    
    print(f"[OK] Findings report saved: {report_file}")
    return report_file

def main():
    """Main analysis function."""
    print("=" * 70)
    print("BITCOIN REGIME SHIFT ANALYSIS")
    print("=" * 70)
    print("\nLoading data...")
    
    df = load_data()
    if df is None:
        return
    
    print(f"[OK] Loaded {len(df)} data points")
    print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n")
    
    print("Running 3-year rolling regression...")
    results_3yr = rolling_regression_analysis(df, window_years=3)
    print(f"[OK] Analyzed {len(results_3yr)} windows\n")
    
    print("Running 5-year rolling regression...")
    results_5yr = rolling_regression_analysis(df, window_years=5)
    print(f"[OK] Analyzed {len(results_5yr)} windows\n")
    
    print("Identifying regime shifts...")
    regime_shifts = identify_regime_shifts(results_3yr)
    print(f"[OK] Identified {len(regime_shifts)} potential regime shifts\n")
    
    print("Creating visualizations...")
    create_visualizations(df, results_3yr, results_5yr, regime_shifts)
    
    print("Generating findings report...")
    generate_findings_report(df, results_3yr, results_5yr, regime_shifts)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nCheck the following files in {OUTPUT_FOLDER}:")
    print(f"  • {CURRENCY_NAME_LOWER}_regime_analysis.png (visualizations)")
    print(f"  • {CURRENCY_NAME_LOWER}_regime_findings.txt (detailed report)")

if __name__ == "__main__":
    main()

