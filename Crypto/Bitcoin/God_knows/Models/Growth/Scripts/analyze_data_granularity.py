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
    df = pd.read_csv('Data Sets/Bitcoin Data/btcusd_1-min_data.csv')
    print(f"Loaded {len(df):,} minute-by-minute data points")
    
    # Convert Unix timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Use Close price as our price data
    df['Price'] = df['Close']
    
    # Remove any invalid prices
    df = df[df['Price'] > 0].dropna(subset=['Price'])
    
    # Calculate days since start
    start_date = df['DateTime'].min()
    df['Days'] = (df['DateTime'] - start_date).dt.total_seconds() / (24 * 3600) + 1
    
    print(f"Data range: {start_date} to {df['DateTime'].max()}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    print(f"Time span: {df['Days'].max():.1f} days")
    
    return df

def downsample_data(df, method='daily_close'):
    """
    Downsample minute data with different methods
    """
    if method == 'daily_close':
        # Group by date and take the last (close) price of each day
        df['Date'] = df['DateTime'].dt.date
        daily_df = df.groupby('Date').agg({
            'Price': 'last',
            'DateTime': 'last',
            'Days': 'last'
        }).reset_index()
        return daily_df
        
    elif method == 'daily_open':
        # Group by date and take the first (open) price of each day
        df['Date'] = df['DateTime'].dt.date
        daily_df = df.groupby('Date').agg({
            'Price': 'first',
            'DateTime': 'first',
            'Days': 'first'
        }).reset_index()
        return daily_df
        
    elif method == 'daily_avg':
        # Group by date and take the average price of each day
        df['Date'] = df['DateTime'].dt.date
        daily_df = df.groupby('Date').agg({
            'Price': 'mean',
            'DateTime': 'first',
            'Days': 'first'
        }).reset_index()
        return daily_df
        
    elif method == 'hourly_close':
        # Group by hour and take the last price of each hour
        df['Hour'] = df['DateTime'].dt.floor('H')
        hourly_df = df.groupby('Hour').agg({
            'Price': 'last',
            'DateTime': 'last',
            'Days': 'last'
        }).reset_index()
        return hourly_df
        
    elif method == 'sample_minutes':
        # Sample every N minutes
        N = 60  # Every hour
        sampled_df = df.iloc[::N].copy()
        return sampled_df
        
    else:
        return df

def fit_model(df, start_day=90, model_type='log_log'):
    """
    Fit the growth model and return R²
    """
    # Filter data to start from specified day
    model_df = df[df['Days'] >= start_day].copy()
    
    if len(model_df) == 0:
        return None
    
    # Create transformations based on model type
    if model_type == 'log_log':
        model_df['T'] = np.log(model_df['Days'])
        model_df['U'] = np.log10(model_df['Price'])
    elif model_type == 'log_linear':
        model_df['T'] = model_df['Days']
        model_df['U'] = np.log10(model_df['Price'])
    else:
        return None
    
    # Remove any infinite or NaN values
    model_df = model_df[np.isfinite(model_df['T']) & np.isfinite(model_df['U'])]
    
    if len(model_df) == 0:
        return None
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(model_df['T'], model_df['U'])
    r_squared = r_value ** 2
    
    return {
        'r_squared': r_squared,
        'data_points': len(model_df),
        'slope': slope,
        'intercept': intercept,
        'correlation': r_value
    }

def analyze_granularity_impact():
    """Analyze how different data granularities affect R²"""
    print("ANALYZING DATA GRANULARITY IMPACT ON BITCOIN GROWTH MODEL")
    print("="*70)
    
    # Load minute data
    df = load_minute_data()
    
    # Test different downsampling methods
    methods = [
        'daily_close',
        'daily_open', 
        'daily_avg',
        'hourly_close',
        'sample_minutes'
    ]
    
    results = []
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        # Downsample data
        df_sampled = downsample_data(df, method)
        print(f"  Data points: {len(df_sampled):,}")
        
        # Test both model types
        for model_type in ['log_log', 'log_linear']:
            result = fit_model(df_sampled, start_day=90, model_type=model_type)
            
            if result:
                results.append({
                    'method': method,
                    'model_type': model_type,
                    'r_squared': result['r_squared'],
                    'data_points': result['data_points'],
                    'slope': result['slope'],
                    'intercept': result['intercept'],
                    'correlation': result['correlation']
                })
                
                print(f"  {model_type}: R² = {result['r_squared']:.6f} ({result['data_points']:,} points)")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    # Sort by R²
    results_df_sorted = results_df.sort_values('r_squared', ascending=False)
    
    for idx, row in results_df_sorted.iterrows():
        print(f"{row['method']:15} | {row['model_type']:12} | R² = {row['r_squared']:.6f} | {row['data_points']:6,} points")
    
    # Find best model
    best_model = results_df.loc[results_df['r_squared'].idxmax()]
    print(f"\nBEST MODEL:")
    print(f"Method: {best_model['method']}")
    print(f"Model Type: {best_model['model_type']}")
    print(f"R² = {best_model['r_squared']:.6f}")
    print(f"Data Points: {best_model['data_points']:,}")
    print(f"Slope: {best_model['slope']:.10f}")
    print(f"Intercept: {best_model['intercept']:.10f}")
    print(f"Correlation: {best_model['correlation']:.6f}")
    
    # Create visualization
    create_comparison_plot(results_df)
    
    return results_df

def create_comparison_plot(results_df):
    """Create a plot comparing R² across different methods"""
    plt.figure(figsize=(12, 8))
    
    # Group by method and model type
    methods = results_df['method'].unique()
    model_types = results_df['model_type'].unique()
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, model_type in enumerate(model_types):
        subset = results_df[results_df['model_type'] == model_type]
        r_squared_values = [subset[subset['method'] == method]['r_squared'].iloc[0] 
                          if len(subset[subset['method'] == method]) > 0 else 0 
                          for method in methods]
        
        plt.bar(x + i*width, r_squared_values, width, label=model_type, alpha=0.8)
    
    plt.xlabel('Data Granularity Method')
    plt.ylabel('R²')
    plt.title('Bitcoin Growth Model R² by Data Granularity')
    plt.xticks(x + width/2, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = 'Results/Bitcoin/data_granularity_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    plt.show()

def main():
    """Main execution function"""
    results = analyze_granularity_impact()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. More data points don't always mean better R²")
    print("2. Noise in minute data can reduce model fit")
    print("3. Daily data often provides the best balance of data points vs noise")
    print("4. Different downsampling methods can significantly affect results")
    print("5. The log_linear model may perform better than log_log for some data")

if __name__ == "__main__":
    main() 