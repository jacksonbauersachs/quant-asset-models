import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['date'].min()
        df['days'] = (df['date'] - start_date).dt.days
        
        print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def fit_palladium_models(df):
    """Use our optimized exponential model with fixed growth rate and optimized intercept."""
    
    x = df['days'].values
    y = df[' value'].values
    
    # Our optimized parameters
    fixed_a = 54.36
    fixed_b = 0.000172  # Your chosen growth rate
    optimized_c = 143.98  # Optimized intercept for maximum R²
    
    # Calculate predictions using our optimized formula
    y_pred = fixed_a * np.exp(fixed_b * x) + optimized_c
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Create the model results in the same format as the existing script
    models = {
        'Optimized_Exponential': {
            'formula': f"Price = {fixed_a:.2f} * exp({fixed_b:.6f} * Days) + {optimized_c:.2f}",
            'params': [fixed_a, fixed_b, optimized_c],
            'r2': r2,
            'rmse': rmse
        }
    }
    
    # Set as best model
    models['Best_Model'] = {
        'name': 'Optimized_Exponential',
        'formula': f"Price = {fixed_a:.2f} * exp({fixed_b:.6f} * Days) + {optimized_c:.2f}",
        'r2': r2,
        'rmse': rmse,
        'params': [fixed_a, fixed_b, optimized_c]
    }
    
    print(f"Using optimized Palladium model: y = {fixed_a:.2f} * exp({fixed_b:.6f} * x) + {optimized_c:.2f}")
    print(f"R² = {r2:.6f}, RMSE = {rmse:.2f}")
    
    return models

def _predict_future_price(df, model_results, years_ahead):
    """Predict future price using our optimized exponential model."""
    days_in_year = 365
    last_day = int(df['days'].iloc[-1])
    future_day = last_day + years_ahead * days_in_year
    
    # Use our optimized exponential formula: y = a * exp(b * x) + c
    a, b, c = model_results['Best_Model']['params']
    return float(a * np.exp(b * future_day) + c)

def compute_projections(df, model_results):
    """Compute 1, 2, 3, 5, 10, 15, and 20-year projected returns using the fitted model."""
    if model_results is None:
        return None
    current_price = float(df[' value'].iloc[-1])
    
    # Model-implied fair price today using our optimized exponential formula
    last_day = int(df['days'].iloc[-1])
    a, b, c = model_results['Best_Model']['params']
    fair_price = float(a * np.exp(b * last_day) + c)
    
    horizons = [1, 2, 3, 5, 10, 15, 20]
    projections = {}
    for y in horizons:
        future_price = _predict_future_price(df, model_results, y)
        ret_pct = (future_price / current_price - 1.0) * 100.0
        projections[y] = {
            'future_price': future_price,
            'return_pct': ret_pct,
            'fair_return_pct': (future_price / fair_price - 1.0) * 100.0
        }
    return {
        'asset': 'Palladium',
        'as_of_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'current_price': current_price,
        'fair_price': fair_price,
        'projections': projections
    }

def save_projections(palladium_proj):
    """Write projections to palladium_projections.txt using the exact same format as the existing script."""
    os.makedirs("Commodities/Palladium/Models/Growth", exist_ok=True)
    results_file = "Commodities/Palladium/Models/Growth/palladium_projections.txt"
    with open(results_file, 'w') as f:
        f.write("PALLADIUM MODEL-BASED FORWARD PROJECTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if palladium_proj:
            f.write(f"{palladium_proj['asset'].upper()} PROJECTIONS\n")
            f.write("-" * (len(palladium_proj['asset']) + 12) + "\n")
            f.write(f"As of: {palladium_proj['as_of_date']}\n")
            f.write(f"Current Price: ${palladium_proj['current_price']:,.2f}\n")
            f.write(f"Model Fair Price (today): ${palladium_proj['fair_price']:,.2f}\n")
            for years in [1, 2, 3, 5, 10, 15, 20]:
                p = palladium_proj['projections'][years]
                f.write(f"{years}-Year Projected Price: ${p['future_price']:,.2f} | Return vs Current: {p['return_pct']:.2f}%\n")
            f.write("Fair-Based Growth Rates (from model fair price):\n")
            for years in [1, 2, 3, 5, 10, 15, 20]:
                p = palladium_proj['projections'][years]
                f.write(f"  {years}-Year Growth: {p['fair_return_pct']:.2f}%\n")
            f.write("\n")

        f.write("Notes:\n")
        f.write("- Projections use the latest fitted Palladium growth model.\n")
        f.write("- Model fair price represents the theoretical value based on historical trends.\n")
        f.write("- These are statistical projections, not financial advice.\n")

    print(f"Projections saved to: {results_file}")
    return results_file

def main():
    """Main function to create Palladium projections using the exact same logic as the existing script."""
    
    print("Palladium Growth Model Analysis")
    print("=" * 40)
    
    # Load data
    print("\nLoading Palladium data...")
    palladium_df = load_palladium_data()
    
    if palladium_df is None:
        print("Error: Could not load Palladium data file.")
        return
    
    # Fit models
    print("\nFitting Palladium growth models...")
    palladium_results = fit_palladium_models(palladium_df)
    
    # Compute and save projections
    print("\nComputing forward projections (1y, 2y, 3y, 5y, 10y, 15y, 20y)...")
    palladium_proj = compute_projections(palladium_df, palladium_results)
    projections_file = save_projections(palladium_proj)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PALLADIUM ANALYSIS COMPLETE")
    print("=" * 60)
    
    if palladium_results and 'Best_Model' in palladium_results:
        print(f"Best Palladium Model: {palladium_results['Best_Model']['name']} (R² = {palladium_results['Best_Model']['r2']:.6f})")
    
    print(f"\nProjections saved to: {projections_file}")

if __name__ == "__main__":
    main()
