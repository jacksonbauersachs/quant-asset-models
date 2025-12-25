"""
Fixed Bitcoin Monte Carlo Simulation
===================================

This script fixes the issues in the original Monte Carlo simulation:
1. Proper CSV saving with correct column indexing
2. Ensures correct starting price is used
3. Validates simulation results before saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load the complete Bitcoin dataset"""
    print("Loading complete Bitcoin dataset...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
    
    return df

def get_updated_models():
    """Get the updated growth and volatility models"""
    print("\n" + "="*60)
    print("LOADING UPDATED MODELS")
    print("="*60)
    
    # Load growth model coefficients
    try:
        with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
            lines = f.readlines()
            a = float(lines[0].split('=')[1].strip())
            b = float(lines[1].split('=')[1].strip())
            r2 = float(lines[2].split('=')[1].strip())
        
        print(f"Growth Model: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
        print(f"Growth Model R² = {r2:.6f}")
        
    except FileNotFoundError:
        print("⚠️  Growth model file not found, using default values")
        a, b, r2 = 1.827743, -10.880943, 0.940275
    
    # Load exponential volatility model
    try:
        with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Formula: volatility =' in line:
                    formula = line.split('=')[1].strip()
                    # Extract parameters: a * exp(-b * years) + c
                    try:
                        parts = formula.split('*')
                        a_vol = float(parts[0].strip())
                        
                        exp_part = parts[1].strip()
                        b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                        if b_vol < 0:
                            b_vol = abs(b_vol)
                        
                        c_vol = float(parts[2].split('+')[1].strip())
                        break
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing formula: {formula}")
                        print(f"Error details: {e}")
                        raise
            else:
                raise ValueError("Formula line not found in file")
        
        print(f"Volatility Model: volatility = {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
        print(f"Volatility Model R² = 0.2883")
        
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"⚠️  Volatility model file error: {e}")
        print("Using default exponential decay parameters")
        a_vol, b_vol, c_vol = 2.310879, 0.124138, 0.077392
    
    return {
        'growth': {'a': a, 'b': b, 'r2': r2},
        'volatility': {'a': a_vol, 'b': b_vol, 'c': c_vol}
    }

def calculate_volatility_at_time(years, vol_params):
    """Calculate volatility at a given time using exponential decay model"""
    a, b, c = vol_params['a'], vol_params['b'], vol_params['c']
    
    # Bitcoin's current age (years since 2010-07-18)
    bitcoin_current_age = 15.0  # 2025-07-19 minus 2010-07-18
    
    # Calculate volatility at Bitcoin's age + future years
    total_years = bitcoin_current_age + years
    volatility = a * np.exp(-b * total_years) + c
    
    # Ensure volatility stays within reasonable bounds
    volatility = np.maximum(volatility, 0.05)  # Minimum 5%
    volatility = np.minimum(volatility, 1.0)   # Maximum 100%
    
    return volatility

def fixed_monte_carlo_simulation(start_price, years, growth_params, vol_params, num_paths=1000):
    """
    Fixed Monte Carlo simulation using formula predictions as targets
    
    CRITICAL FIXES:
    ===============
    1. Ensure start_price is correctly used
    2. Validate simulation results
    3. Proper error handling
    """
    print(f"\n" + "="*60)
    print("RUNNING FIXED MONTE CARLO SIMULATION")
    print("="*60)
    print(f"Starting price: ${start_price:,.2f}")
    print(f"Simulation period: {years} years")
    print(f"Number of paths: {num_paths:,}")
    
    # Validate start price
    if start_price <= 0:
        raise ValueError(f"Invalid start price: ${start_price:,.2f}")
    
    # Time setup
    dt = 1/365.25  # Daily time step
    time_steps = int(years * 365.25)
    t = np.linspace(0, years, time_steps + 1)
    
    # Initialize price paths
    price_paths = np.zeros((num_paths, time_steps + 1))
    price_paths[:, 0] = start_price  # Set all paths to start at the correct price
    
    # Get model parameters
    a, b = growth_params['a'], growth_params['b']
    today_day = 6041  # Days since Bitcoin genesis (correct day numbering)
    
    print(f"Today's formula day: {today_day}")
    print(f"Formula prediction for today: ${10**(a * np.log(today_day) + b):,.2f}")
    print(f"Bitcoin is currently overvalued by {((start_price / 10**(a * np.log(today_day) + b) - 1) * 100):.1f}% relative to formula")
    
    # Log expected growth rates and volatility for key time periods
    print(f"\n" + "="*60)
    print("EXPECTED GROWTH RATES AND VOLATILITY BY YEAR")
    print("="*60)
    print("Year | Formula Day | Expected Price | Growth Rate | Volatility")
    print("-" * 70)
    
    for year in range(int(years) + 1):
        future_day = today_day + int(year * 365.25)
        expected_price = 10**(a * np.log(future_day) + b)
        
        # Calculate growth rate from today to this year
        if year == 0:
            growth_rate = 0.0
        else:
            growth_rate = np.log(expected_price / 10**(a * np.log(today_day) + b)) / year
        
        # Calculate volatility at this year
        volatility = calculate_volatility_at_time(year, vol_params)
        
        print(f"{year:4.0f} | {future_day:10.0f} | ${expected_price:12,.0f} | {growth_rate*100:8.1f}% | {volatility*100:8.1f}%")
    
    print("="*60)
    
    # Run simulation
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_paths):
        for j in range(1, time_steps + 1):
            # Calculate current time in years
            current_years = t[j]
            
            # Get volatility at this time
            volatility = calculate_volatility_at_time(current_years, vol_params)
            
            # Calculate expected price at this time using growth formula
            future_day = today_day + int(current_years * 365.25)
            expected_price = 10**(a * np.log(future_day) + b)
            
            # Simple approach: add random volatility around the expected price
            # The current price will naturally converge toward the formula's prediction
            random_factor = np.exp(np.random.normal(0, volatility * np.sqrt(dt)))
            
            if j == 1:
                # First step: start from current price
                price_paths[i, j] = price_paths[i, j-1] * random_factor
            else:
                # Subsequent steps: gradually move toward expected price
                # Use a weighted average between current path and expected price
                weight = min(0.1, dt)  # Small weight to gradually converge
                target_price = (1 - weight) * price_paths[i, j-1] + weight * expected_price
                price_paths[i, j] = target_price * random_factor
    
    # Validate simulation results
    print(f"\n" + "="*60)
    print("SIMULATION VALIDATION")
    print("="*60)
    
    # Check starting prices
    actual_start_prices = price_paths[:, 0]
    if not np.allclose(actual_start_prices, start_price):
        print(f"⚠️  WARNING: Starting prices don't match expected value!")
        print(f"Expected: ${start_price:,.2f}")
        print(f"Actual range: ${actual_start_prices.min():,.2f} - ${actual_start_prices.max():,.2f}")
    else:
        print(f"✅ Starting prices correctly set to ${start_price:,.2f}")
    
    # Check for any zero or negative prices
    min_price = price_paths.min()
    if min_price <= 0:
        print(f"⚠️  WARNING: Found {np.sum(price_paths <= 0)} zero or negative prices!")
        print(f"Minimum price: ${min_price:,.2f}")
    else:
        print(f"✅ All prices are positive (minimum: ${min_price:,.2f})")
    
    # Check final prices
    final_prices = price_paths[:, -1]
    print(f"Final price range: ${final_prices.min():,.2f} - ${final_prices.max():,.2f}")
    print(f"Average final price: ${final_prices.mean():,.2f}")
    
    return price_paths, t

def save_simulation_to_csv_fixed(price_paths, t, results, models, start_price, years):
    """Save Monte Carlo simulation results to CSV files (FIXED VERSION)"""
    print(f"\n" + "="*60)
    print("SAVING SIMULATION RESULTS TO CSV (FIXED)")
    print("="*60)
    
    # Validate price_paths shape
    num_paths, time_steps = price_paths.shape
    print(f"Price paths shape: {price_paths.shape}")
    print(f"Time steps: {len(t)}")
    
    # Save price paths (FIXED: Proper transposition and column naming)
    paths_filename = f'Results/Bitcoin/bitcoin_monte_carlo_fixed_paths_{datetime.now().strftime("%Y%m%d")}.csv'
    
    # Create DataFrame with time as index and paths as columns
    # price_paths is (num_paths, time_steps), we want (time_steps, num_paths)
    paths_df = pd.DataFrame(price_paths.T, index=t, columns=[f'Path_{i+1}' for i in range(num_paths)])
    paths_df.index.name = 'Years'
    
    # Validate the DataFrame before saving
    print(f"DataFrame shape: {paths_df.shape}")
    print(f"First row (starting prices): {paths_df.iloc[0, :5].values}")  # Show first 5 paths
    print(f"Last row (final prices): {paths_df.iloc[-1, :5].values}")    # Show first 5 paths
    
    paths_df.to_csv(paths_filename)
    print(f"Price paths saved to: {paths_filename}")
    
    # Save summary statistics
    summary_filename = f'Results/Bitcoin/bitcoin_monte_carlo_fixed_summary_{datetime.now().strftime("%Y%m%d")}.csv'
    
    summary_data = []
    for year in sorted(results.keys()):
        r = results[year]
        summary_data.append({
            'Year': year,
            'Mean_Price': r['mean'],
            'Median_Price': r['median'],
            'Std_Dev': r['std'],
            'P5_Price': r['p5'],
            'P25_Price': r['p25'],
            'P75_Price': r['p75'],
            'P95_Price': r['p95']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to: {summary_filename}")
    
    # Save formula predictions for comparison
    formula_filename = f'Results/Bitcoin/bitcoin_monte_carlo_fixed_formula_{datetime.now().strftime("%Y%m%d")}.csv'
    
    a, b = models['growth']['a'], models['growth']['b']
    today_day = 6041
    
    formula_data = []
    for year in sorted(results.keys()):
        future_day = today_day + int(year * 365.25)
        expected_price = 10**(a * np.log(future_day) + b)
        
        if year == 0:
            growth_rate = 0.0
        else:
            growth_rate = np.log(expected_price / 10**(a * np.log(today_day) + b)) / year
        
        volatility = calculate_volatility_at_time(year, models['volatility'])
        
        formula_data.append({
            'Year': year,
            'Formula_Day': future_day,
            'Formula_Price': expected_price,
            'Growth_Rate_Pct': growth_rate * 100,
            'Volatility_Pct': volatility * 100
        })
    
    formula_df = pd.DataFrame(formula_data)
    formula_df.to_csv(formula_filename, index=False)
    print(f"Formula predictions saved to: {formula_filename}")
    
    return paths_filename, summary_filename, formula_filename

def analyze_simulation_results(price_paths, t, start_price, years):
    """Analyze Monte Carlo simulation results"""
    print(f"\n" + "="*60)
    print("SIMULATION RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate statistics at different time points
    time_points = [0, 0.25, 0.5, 1, 2, 3, 5, 10]  # Years
    results = {}
    
    for year_point in time_points:
        if year_point <= years:
            day_index = int(year_point * 365.25)
            if day_index < len(t):
                prices_at_time = price_paths[:, day_index]
                
                mean_price = np.mean(prices_at_time)
                median_price = np.median(prices_at_time)
                std_price = np.std(prices_at_time)
                
                # Percentiles
                p5 = np.percentile(prices_at_time, 5)
                p25 = np.percentile(prices_at_time, 25)
                p75 = np.percentile(prices_at_time, 75)
                p95 = np.percentile(prices_at_time, 95)
                
                results[year_point] = {
                    'mean': mean_price,
                    'median': median_price,
                    'std': std_price,
                    'p5': p5,
                    'p25': p25,
                    'p75': p75,
                    'p95': p95
                }
                
                print(f"Year {year_point}: ${mean_price:,.0f} (${p5:,.0f} - ${p95:,.0f})")
    
    return results

def create_simulation_visualization(price_paths, t, start_price, results, models):
    """Create visualization for Monte Carlo simulation results"""
    print(f"\n" + "="*60)
    print("CREATING SIMULATION VISUALIZATION")
    print("="*60)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Monte Carlo Simulation Results (Fixed)', fontsize=16, fontweight='bold')
    
    # 1. Sample price paths
    sample_paths = min(50, len(price_paths))  # Show first 50 paths
    for i in range(sample_paths):
        ax1.plot(t, price_paths[i, :], alpha=0.3, linewidth=0.5)
    
    ax1.axhline(y=start_price, color='red', linestyle='--', linewidth=2, label=f'Start: ${start_price:,.0f}')
    ax1.set_title('Sample Price Paths (First 50)')
    ax1.set_ylabel('Bitcoin Price ($)')
    ax1.set_xlabel('Years')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Price distribution at different years
    years_to_plot = [1, 5, 10]
    colors = ['blue', 'green', 'red']
    
    for i, year in enumerate(years_to_plot):
        if year in results:
            day_index = int(year * 365.25)
            if day_index < len(t):
                prices_at_year = price_paths[:, day_index]
                ax2.hist(prices_at_year, bins=50, alpha=0.6, color=colors[i], 
                        label=f'{year} year(s)')
    
    ax2.set_title('Price Distribution at Different Years')
    ax2.set_xlabel('Bitcoin Price ($)')
    ax2.set_ylabel('Number of Paths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Volatility over time
    years_list = sorted(results.keys())
    volatilities = [results[y]['std'] / results[y]['mean'] for y in years_list]
    
    ax3.plot(years_list, volatilities, 'bo-', linewidth=2, markersize=8)
    ax3.set_title('Price Volatility Over Time')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Expected value growth vs formula prediction
    mean_prices = [results[y]['mean'] for y in years_list]
    
    ax4.plot(years_list, mean_prices, 'ro-', linewidth=2, markersize=8, label='Simulation Mean')
    ax4.fill_between(years_list, 
                     [results[y]['p5'] for y in years_list],
                     [results[y]['p95'] for y in years_list],
                     alpha=0.3, color='red')
    
    # Add formula predictions
    a, b = models['growth']['a'], models['growth']['b']
    today_day = 6041
    
    formula_years = [y for y in years_list if y > 0]
    formula_prices_at_years = []
    for year in formula_years:
        future_day = today_day + int(year * 365.25)
        formula_price = 10**(a * np.log(future_day) + b)
        formula_prices_at_years.append(formula_price)
    
    ax4.plot(formula_years, formula_prices_at_years, 'go-', linewidth=2, markersize=8, label='Formula Prediction')
    
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Expected Bitcoin Price ($)')
    ax4.set_title('Expected Value Growth vs Formula Prediction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'Results/Bitcoin/bitcoin_monte_carlo_fixed_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Simulation visualization saved to: {filename}")
    
    plt.show()

def main():
    """Main function to run fixed Monte Carlo simulation"""
    print("="*60)
    print("BITCOIN MONTE CARLO SIMULATION (FIXED VERSION)")
    print("="*60)
    print("This simulation fixes the issues in the original version:")
    print("- Proper CSV saving with correct column indexing")
    print("- Ensures correct starting price is used")
    print("- Validates simulation results before saving")
    print("="*60)
    
    # Load data and get current price
    df = load_bitcoin_data()
    start_price = df['Price'].iloc[-1]
    
    # Validate start price
    if start_price <= 0:
        raise ValueError(f"Invalid start price from data: ${start_price:,.2f}")
    
    print(f"✅ Validated start price: ${start_price:,.2f}")
    
    # Load updated models
    models = get_updated_models()
    
    # Simulation parameters
    years = 10  # 10 years simulation
    num_paths = 1000
    
    # Run fixed Monte Carlo simulation
    price_paths, t = fixed_monte_carlo_simulation(
        start_price, years, models['growth'], models['volatility'], num_paths
    )
    
    # Analyze results
    results = analyze_simulation_results(price_paths, t, start_price, years)
    
    # Create visualization
    create_simulation_visualization(price_paths, t, start_price, results, models)
    
    # Save results to CSV files (FIXED VERSION)
    paths_file, summary_file, formula_file = save_simulation_to_csv_fixed(price_paths, t, results, models, start_price, years)
    
    # Summary
    print(f"\n" + "="*50)
    print("FIXED SIMULATION SUMMARY")
    print("="*50)
    print(f"Starting price: ${start_price:,.2f}")
    print(f"Simulation period: {years} years")
    print(f"Number of paths: {num_paths:,}")
    
    if 1 in results:
        print(f"\nAfter 1 year:")
        print(f"  Expected price: ${results[1]['mean']:,.2f}")
        print(f"  90% confidence: ${results[1]['p5']:,.2f} - ${results[1]['p95']:,.2f}")
    
    if 5 in results:
        print(f"\nAfter 5 years:")
        print(f"  Expected price: ${results[5]['mean']:,.2f}")
        print(f"  90% confidence: ${results[5]['p5']:,.2f} - ${results[5]['p95']:,.2f}")
    
    if 10 in results:
        print(f"\nAfter 10 years:")
        print(f"  Expected price: ${results[10]['mean']:,.2f}")
        print(f"  90% confidence: ${results[10]['p5']:,.2f} - ${results[10]['p95']:,.2f}")
    
    print(f"\nCSV Files Created:")
    print(f"  Price paths: {paths_file}")
    print(f"  Summary stats: {summary_file}")
    print(f"  Formula predictions: {formula_file}")
    print("\n✅ Fixed simulation completed successfully!")

if __name__ == "__main__":
    main() 