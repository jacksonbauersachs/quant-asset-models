import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_bitcoin_data():
    """Load Bitcoin historical data"""
    print("Loading Bitcoin historical data...")
    # Use the same data file as the main GBM script
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
    return df

def dynamic_gbm_monte_carlo_with_save(S0, T, num_paths=1000, num_steps=3653):
    """Run Dynamic GBM simulation and save paths immediately"""
    print(f"\nRunning Dynamic GBM Monte Carlo Simulation with Immediate Save")
    print(f"="*60)
    print(f"Initial price: ${S0:,.2f}")
    print(f"Time horizon: {T} years")
    print(f"Number of paths: {num_paths:,}")
    print(f"Time steps: {num_steps:,} (daily)")
    
    # Load formula parameters
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Formula: volatility =' in line:
                formula = line.split('=')[1].strip()
                parts = formula.split('*')
                a_vol = float(parts[0].strip())
                exp_part = parts[1].strip()
                b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                if b_vol < 0:
                    b_vol = abs(b_vol)
                c_vol = float(parts[2].split('+')[1].strip())
                break
    
    # Bitcoin's current age
    bitcoin_current_age = 15.0
    today_day = 6041
    
    # Time setup
    dt = T / num_steps
    time_steps = np.linspace(0, T, num_steps + 1)
    
    # Create output file immediately
    paths_filename = f'Results/Bitcoin/bitcoin_gbm_paths_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    print(f"Creating output file: {paths_filename}")
    
    # Write header
    with open(paths_filename, 'w') as f:
        f.write('Years')
        for i in range(num_paths):
            f.write(f',Path_{i+1}')
        f.write('\n')
    
    # Initialize price paths (only keep current step in memory)
    current_prices = np.full(num_paths, S0)
    
    print(f"Dynamic Parameters:")
    print(f"  Growth formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
    print(f"  Volatility formula: {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
    print(f"  Bitcoin current age: {bitcoin_current_age:.1f} years")
    
    # Run simulation with immediate saving
    np.random.seed(42)
    
    print(f"\nRunning simulation and saving paths...")
    for j in range(num_steps + 1):
        current_time = time_steps[j]
        
        # Save current prices to file
        with open(paths_filename, 'a') as f:
            f.write(f'{current_time:.6f}')
            for price in current_prices:
                f.write(f',{price:.2f}')
            f.write('\n')
        
        # Progress indicator
        if j % 365 == 0:
            print(f"  Year {j//365}: ${np.mean(current_prices):,.0f}")
        
        # Calculate next step if not at the end
        if j < num_steps:
            # Calculate dynamic parameters
            future_age = bitcoin_current_age + current_time
            future_day = today_day + int(current_time * 365.25)
            
            # Dynamic growth rate
            current_fair_value = 10**(a * np.log(today_day) + b)
            future_fair_value = 10**(a * np.log(future_day) + b)
            mu = (future_fair_value / current_fair_value) ** (1/current_time) - 1 if current_time > 0 else 0
            
            # Dynamic volatility
            sigma = a_vol * np.exp(-b_vol * future_age) + c_vol
            
            # GBM parameters
            drift = (mu - 0.5 * sigma**2) * dt
            volatility = sigma * np.sqrt(dt)
            
            # Update prices
            random_shocks = np.random.normal(0, 1, num_paths)
            current_prices = current_prices * np.exp(drift + volatility * random_shocks)
    
    print(f"✅ Simulation complete! Paths saved to: {paths_filename}")
    return paths_filename

def analyze_gbm_results_from_file(paths_filename):
    """Analyze GBM results from saved file"""
    print(f"\nAnalyzing GBM results from {paths_filename}...")
    
    # Load the saved paths
    df = pd.read_csv(paths_filename, index_col=0)
    
    # Calculate statistics at different time points
    time_points = [0, 0.25, 0.5, 1, 2, 3, 5, 10]  # Years
    results = {}
    
    for year_point in time_points:
        if year_point <= 10:
            # Find closest time step
            time_steps = df.index.astype(float)
            step_index = np.argmin(np.abs(time_steps - year_point))
            
            prices_at_time = df.iloc[step_index, :].values
            
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

def save_summary_statistics(results, paths_filename):
    """Save summary statistics"""
    summary_filename = paths_filename.replace('_paths_', '_summary_')
    
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
    
    return summary_filename

def main():
    """Main function"""
    print("Bitcoin Dynamic GBM with Immediate Save and Analysis")
    print("="*60)
    
    # Load data
    df = load_bitcoin_data()
    current_price = df['Price'].iloc[-1]
    # Load growth model coefficients
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    today_day = 6041
    S0 = 10**(a * np.log(today_day) + b)
    print(f"Fair value (formula): ${S0:,.2f}")
    print(f"Current price: ${current_price:,.2f}")
    
    # For quick exploratory runs, use fewer paths (e.g., 100 instead of 1000)
    paths_file = dynamic_gbm_monte_carlo_with_save(S0, T=10, num_paths=100, num_steps=3653)
    
    # Analyze results
    results = analyze_gbm_results_from_file(paths_file)
    
    # Save summary statistics
    summary_file = save_summary_statistics(results, paths_file)
    
    print(f"\n" + "="*50)
    print("DYNAMIC GBM SIMULATION SUMMARY")
    print("="*50)
    print(f"Starting price (fair value): ${S0:,.2f}")
    print(f"Simulation period: 10 years")
    print(f"Number of paths: 100 (quick test)")
    print(f"Parameter update frequency: Every day (365 times per year)")
    print(f"Growth rate: Dynamic based on formula fair value")
    print(f"Volatility: Dynamic based on exponential decay formula")
    
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
    print(f"  File size: {os.path.getsize(paths_file) / (1024*1024):.1f} MB")
    print("\n✅ Dynamic GBM simulation completed successfully!")
    print("Note: Growth and volatility parameters updated every day during simulation")

if __name__ == "__main__":
    main() 