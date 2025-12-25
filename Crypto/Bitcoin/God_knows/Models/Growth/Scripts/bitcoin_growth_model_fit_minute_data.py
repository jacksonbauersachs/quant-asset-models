import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_minute_data():
    """Load and process the minute-by-minute Bitcoin data"""
    print("Loading minute-by-minute Bitcoin data...")
    
    # Load the minute data
    df = pd.read_csv('../../Data Sets/Bitcoin Data/btcusd_1-min_data.csv')
    print(f"Loaded {len(df):,} minute-by-minute data points")
    
    # Convert Unix timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Use Close price as our price data
    df['Price'] = df['Close']
    
    # Remove any invalid prices
    df = df[df['Price'] > 0].dropna(subset=['Price'])
    
    # Calculate days since start
    start_date = df['DateTime'].min()
    # Start from day 1 instead of actual time offset to avoid negative log values
    df['Days'] = (df['DateTime'] - start_date).dt.total_seconds() / (24 * 3600) + 1
    
    print(f"Data range: {start_date} to {df['DateTime'].max()}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    print(f"Time span: {df['Days'].max():.1f} days")
    
    return df

def downsample_data(df, method='daily_close'):
    """
    Downsample minute data to reduce computational load while preserving trend
    
    Options:
    - 'daily_close': One price per day (close of last minute of each day)
    - 'hourly_close': One price per hour 
    - 'sample_minutes': Sample every N minutes
    """
    print(f"Downsampling using method: {method}")
    
    if method == 'daily_close':
        # Group by date and take the last (close) price of each day
        df['Date'] = df['DateTime'].dt.date
        daily_df = df.groupby('Date').agg({
            'Price': 'last',  # Last price of the day
            'DateTime': 'last',  # Last timestamp of the day
            'Days': 'last'
        }).reset_index()
        print(f"Downsampled to {len(daily_df):,} daily data points")
        return daily_df
        
    elif method == 'hourly_close':
        # Group by hour and take the last price of each hour
        df['Hour'] = df['DateTime'].dt.floor('H')
        hourly_df = df.groupby('Hour').agg({
            'Price': 'last',
            'DateTime': 'last',
            'Days': 'last'
        }).reset_index()
        print(f"Downsampled to {len(hourly_df):,} hourly data points")
        return hourly_df
        
    elif method == 'sample_minutes':
        # Sample every 60 minutes (1 hour intervals)
        sampled_df = df.iloc[::60].copy()
        print(f"Sampled to {len(sampled_df):,} data points (every 60 minutes)")
        return sampled_df
        
    else:
        print("Using all minute data (may be slow)")
        return df

def fit_growth_model(df, start_day=None, model_type='log_log'):
    """
    Fit the growth model with different options:
    - 'log_log': log10(price) = a * ln(day) + b (original)
    - 'log_linear': log10(price) = a * day + b (alternative)
    """
    if start_day is None:
        start_day = df['Days'].min()
    
    print(f"Fitting growth model starting from day {start_day}...")
    print(f"Model type: {model_type}")
    
    # Filter data to start from specified day
    model_df = df[df['Days'] >= start_day].copy()
    
    if len(model_df) == 0:
        print(f"Error: No data available after day {start_day}")
        return None
    
    print(f"Using {len(model_df):,} data points for model fitting")
    
    # Create transformations based on model type
    if model_type == 'log_log':
        # Original model: log10(price) = a * ln(day) + b
        model_df['T'] = np.log(model_df['Days'])  # ln(day)
        model_df['U'] = np.log10(model_df['Price'])  # log10(price)
        formula_desc = "log10(price) = a * ln(day) + b"
    elif model_type == 'log_linear':
        # Alternative model: log10(price) = a * day + b
        model_df['T'] = model_df['Days']  # day (linear)
        model_df['U'] = np.log10(model_df['Price'])  # log10(price)
        formula_desc = "log10(price) = a * day + b"
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    # Remove any infinite or NaN values
    model_df = model_df[np.isfinite(model_df['T']) & np.isfinite(model_df['U'])]
    
    # Perform linear regression: U = a * T + b
    slope, intercept, r_value, p_value, std_err = linregress(model_df['T'], model_df['U'])
    
    # Calculate R-squared
    r_squared = r_value ** 2
    
    print("\n" + "="*50)
    print("BITCOIN GROWTH MODEL RESULTS")
    print("="*50)
    print(f"Formula: {formula_desc}")
    print(f"a (slope) = {slope:.10f}")
    print(f"b (intercept) = {intercept:.10f}")
    print(f"R² = {r_squared:.10f}")
    print(f"Correlation = {r_value:.10f}")
    print(f"P-value = {p_value:.2e}")
    print(f"Standard Error = {std_err:.10f}")
    print(f"Data points used: {len(model_df):,}")
    print(f"Start day: {start_day}")
    
    # Test predictions
    print("\n" + "="*30)
    print("MODEL VALIDATION")
    print("="*30)
    
    # Latest prediction
    latest_t = model_df['T'].iloc[-1]
    latest_u_pred = slope * latest_t + intercept
    latest_u_actual = model_df['U'].iloc[-1]
    
    latest_price_pred = 10 ** latest_u_pred
    latest_price_actual = model_df['Price'].iloc[-1]
    
    error_pct = abs(latest_price_pred - latest_price_actual) / latest_price_actual * 100
    
    print(f"Latest day: {model_df['Days'].iloc[-1]:.1f}")
    print(f"Predicted price: ${latest_price_pred:,.2f}")
    print(f"Actual price: ${latest_price_actual:,.2f}")
    print(f"Error: {error_pct:.2f}%")
    
    # Save coefficients
    output_file = f'bitcoin_growth_model_coefficients_minute_data_{model_type}.txt'
    with open(output_file, 'w') as f:
        f.write(f'a = {slope}\n')
        f.write(f'b = {intercept}\n')
        f.write(f'R2 = {r_squared}\n')
        f.write(f'correlation = {r_value}\n')
        f.write(f'p_value = {p_value}\n')
        f.write(f'std_err = {std_err}\n')
        f.write(f'data_points = {len(model_df)}\n')
        f.write(f'start_day = {start_day}\n')
        f.write(f'model_type = {model_type}\n')
        f.write(f'# Formula: {formula_desc} (day >= {start_day})\n')
        f.write(f'# Based on minute-by-minute data from {model_df["DateTime"].min()} to {model_df["DateTime"].max()}\n')
    
    print(f"\nCoefficients saved to: {output_file}")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'model_df': model_df,
        'data_points': len(model_df),
        'model_type': model_type,
        'formula_desc': formula_desc
    }

def compare_models(original_a=1.6329135221917355, original_b=-9.328646304661454, original_r2=0.9357851345169623):
    """Compare new model with original model"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print("Original Model (daily data):")
    print(f"  a = {original_a:.10f}")
    print(f"  b = {original_b:.10f}")
    print(f"  R² = {original_r2:.10f}")
    print()
    
def create_visualization(result):
    """Create visualization of the model fit"""
    if result is None:
        return
        
    model_df = result['model_df']
    slope = result['slope']
    intercept = result['intercept']
    r_squared = result['r_squared']
    
    # Create predicted values
    model_df['U_pred'] = slope * model_df['T'] + intercept
    model_df['Price_pred'] = 10 ** model_df['U_pred']
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Price vs Time (log scale)
    ax1.semilogy(model_df['Days'], model_df['Price'], 'b.', alpha=0.1, markersize=0.5, label='Actual Price')
    ax1.semilogy(model_df['Days'], model_df['Price_pred'], 'r-', linewidth=2, label=f'Model Fit (R² = {r_squared:.4f})')
    ax1.set_xlabel('Days Since Start')
    ax1.set_ylabel('Bitcoin Price (USD, log scale)')
    ax1.set_title('Bitcoin Growth Model - Minute Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Linear regression in log space
    ax2.plot(model_df['T'], model_df['U'], 'b.', alpha=0.1, markersize=0.5, label='log10(price) vs ln(day)')
    ax2.plot(model_df['T'], model_df['U_pred'], 'r-', linewidth=2, label=f'Linear Fit (R² = {r_squared:.4f})')
    ax2.set_xlabel('ln(Days)')
    ax2.set_ylabel('log10(Price)')
    ax2.set_title('Linear Regression in Log Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = '../../Results/Bitcoin/bitcoin_growth_model_minute_data_fit.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.show()

def main():
    """Main execution function"""
    print("BITCOIN GROWTH MODEL FITTING - MINUTE DATA")
    print("="*60)
    
    # Load minute data
    df = load_minute_data()
    
    # Downsample data (choose method based on computational needs)
    # Options: 'daily_close', 'hourly_close', 'sample_minutes'
    df_sampled = downsample_data(df, method='daily_close')  # Start with daily for speed
    
    # Fit growth model
    result = fit_growth_model(df_sampled)
    
    # Compare with original model
    compare_models()
    
    # Create visualization
    create_visualization(result)
    
    if result:
        print("\n" + "="*60)
        print("SUCCESS! New Bitcoin growth model fitted with minute data.")
        print(f"Improvement in data points: {result['data_points']:,} vs previous model")
        print("="*60)

if __name__ == "__main__":
    main() 