import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load Bitcoin historical data"""
    print("Loading Bitcoin historical data...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} days of Bitcoin data")
    return df

def load_mean_reversion_data():
    """Load fine-resolution mean reversion data"""
    print("Loading mean reversion data...")
    df = pd.read_csv('Results/Bitcoin/bitcoin_mean_reversion_daily_20250720_163955.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_formula_fair_values(df):
    """Calculate formula fair values for each date"""
    print("Calculating formula fair values...")
    
    # Load growth formula parameters
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    # Calculate fair values
    fair_values = []
    for _, row in df.iterrows():
        genesis_date = pd.to_datetime('2009-01-03')
        days_since_genesis = (row['Date'] - genesis_date).days
        
        if days_since_genesis > 0:
            fair_value = 10**(a * np.log(days_since_genesis) + b)
        else:
            fair_value = 0.01
        
        fair_values.append(fair_value)
    
    df['Fair_Value'] = fair_values
    return df

def fit_lambda_model(lambda_data):
    """Fit a model to the mean reversion speed evolution"""
    print("Fitting lambda evolution model...")
    
    # Remove extreme outliers
    clean_data = lambda_data[np.abs(lambda_data['lambda_annual']) < 500].copy()
    
    # Fit exponential decay model for lambda
    def lambda_model(age, lambda_0, alpha, lambda_inf):
        return lambda_0 * np.exp(-alpha * age) + lambda_inf
    
    try:
        popt, _ = curve_fit(lambda_model, clean_data['age'], clean_data['lambda_annual'],
                           p0=[100, 0.1, 20], maxfev=5000)
        lambda_0, alpha, lambda_inf = popt
        
        # Calculate R-squared
        y_pred = lambda_model(clean_data['age'], *popt)
        ss_res = np.sum((clean_data['lambda_annual'] - y_pred) ** 2)
        ss_tot = np.sum((clean_data['lambda_annual'] - clean_data['lambda_annual'].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Lambda model: λ(t) = {lambda_0:.2f} * exp(-{alpha:.3f} * t) + {lambda_inf:.2f}")
        print(f"R-squared: {r_squared:.3f}")
        
        return lambda_0, alpha, lambda_inf, r_squared
        
    except:
        print("Failed to fit lambda model, using constant")
        return 70, 0, 70, 0

def fit_lambda_volatility_model(lambda_data):
    """Fit a model to the volatility of lambda"""
    print("Fitting lambda volatility model...")
    
    # Calculate rolling volatility of lambda
    lambda_data['lambda_vol'] = lambda_data['lambda_annual'].rolling(window=30).std()
    clean_data = lambda_data.dropna()
    
    # Fit exponential decay model for lambda volatility
    def vol_model(age, vol_0, beta, vol_inf):
        return vol_0 * np.exp(-beta * age) + vol_inf
    
    try:
        popt, _ = curve_fit(vol_model, clean_data['age'], clean_data['lambda_vol'],
                           p0=[100, 0.1, 20], maxfev=5000)
        vol_0, beta, vol_inf = popt
        
        # Calculate R-squared
        y_pred = vol_model(clean_data['age'], *popt)
        ss_res = np.sum((clean_data['lambda_vol'] - y_pred) ** 2)
        ss_tot = np.sum((clean_data['lambda_vol'] - clean_data['lambda_vol'].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Lambda volatility model: σ_λ(t) = {vol_0:.2f} * exp(-{beta:.3f} * t) + {vol_inf:.2f}")
        print(f"R-squared: {r_squared:.3f}")
        
        return vol_0, beta, vol_inf, r_squared
        
    except:
        print("Failed to fit lambda volatility model, using constant")
        return 78, 0, 78, 0

def stochastic_mean_reversion_gbm_simulation(start_date, end_date, n_paths=1000, n_steps=365):
    """Simulate Bitcoin price with stochastic mean reversion"""
    print(f"Running stochastic mean reversion GBM simulation...")
    print(f"Paths: {n_paths}, Steps: {n_steps}")
    
    # Load data and fit models
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    lambda_data = load_mean_reversion_data()
    
    # Fit lambda models
    lambda_0, alpha, lambda_inf, lambda_r2 = fit_lambda_model(lambda_data)
    vol_0, beta, vol_inf, vol_r2 = fit_lambda_volatility_model(lambda_data)
    
    # Load volatility model
    with open('Models/Volatility Models/bitcoin_volatility_model_coefficients.txt', 'r') as f:
        lines = f.readlines()
        vol_a = float(lines[0].split('=')[1].strip())
        vol_b = float(lines[1].split('=')[1].strip())
        vol_c = float(lines[2].split('=')[1].strip())
    
    # Load growth model
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        growth_a = float(lines[0].split('=')[1].strip())
        growth_b = float(lines[1].split('=')[1].strip())
    
    # Initialize simulation
    genesis_date = pd.to_datetime('2009-01-03')
    start_age = (start_date - genesis_date).days / 365.25
    end_age = (end_date - genesis_date).days / 365.25
    
    dt = (end_age - start_age) / n_steps
    time_points = np.linspace(start_age, end_age, n_steps + 1)
    
    # Get starting price
    start_price = df[df['Date'] >= start_date]['Price'].iloc[0]
    start_fair_value = df[df['Date'] >= start_date]['Fair_Value'].iloc[0]
    
    print(f"Starting price: ${start_price:,.2f}")
    print(f"Starting fair value: ${start_fair_value:,.2f}")
    print(f"Starting price ratio: {start_price/start_fair_value:.3f}")
    
    # Initialize price paths
    price_paths = np.zeros((n_paths, n_steps + 1))
    lambda_paths = np.zeros((n_paths, n_steps + 1))
    fair_value_paths = np.zeros((n_steps + 1))
    
    # Set initial values
    price_paths[:, 0] = start_price
    lambda_paths[:, 0] = lambda_0 * np.exp(-alpha * start_age) + lambda_inf
    
    # Calculate fair value path
    for i, age in enumerate(time_points):
        days_since_genesis = age * 365.25
        fair_value_paths[i] = 10**(growth_a * np.log(days_since_genesis) + growth_b)
    
    # Simulation loop
    for i in range(n_steps):
        current_age = time_points[i]
        next_age = time_points[i + 1]
        
        # Current fair value
        current_fair_value = fair_value_paths[i]
        
        # Current lambda (mean reversion speed)
        current_lambda = lambda_paths[:, i]
        
        # Current volatility
        current_vol = vol_a * np.exp(-vol_b * current_age) + vol_c
        
        # Lambda volatility
        current_lambda_vol = vol_0 * np.exp(-beta * current_age) + vol_inf
        
        # Growth rate
        current_growth = growth_a / (current_age * np.log(10))
        
        # Stochastic mean reversion GBM
        for path in range(n_paths):
            # Update lambda (stochastic mean reversion speed)
            lambda_drift = alpha * (lambda_inf - current_lambda[path])
            lambda_noise = current_lambda_vol * np.sqrt(dt) * np.random.normal(0, 1)
            lambda_paths[path, i + 1] = current_lambda[path] + lambda_drift * dt + lambda_noise
            
            # Ensure lambda stays positive
            lambda_paths[path, i + 1] = max(lambda_paths[path, i + 1], 0.1)
            
            # Update price with stochastic mean reversion
            current_price = price_paths[path, i]
            current_lambda_path = lambda_paths[path, i + 1]
            
            # Mean reversion term
            mean_reversion_term = current_lambda_path * (current_fair_value - current_price) / current_price
            
            # Drift term (growth + mean reversion)
            drift = current_growth + mean_reversion_term
            
            # Diffusion term
            diffusion = current_vol * np.sqrt(dt) * np.random.normal(0, 1)
            
            # Update price
            price_paths[path, i + 1] = current_price * (1 + drift * dt + diffusion)
            
            # Ensure price stays positive
            price_paths[path, i + 1] = max(price_paths[path, i + 1], 0.01)
    
    return price_paths, lambda_paths, fair_value_paths, time_points

def create_stochastic_simulation_visualization(price_paths, lambda_paths, fair_value_paths, time_points):
    """Create visualization of stochastic mean reversion simulation"""
    print("Creating stochastic simulation visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin: Stochastic Mean Reversion GBM Simulation', fontsize=16, fontweight='bold')
    
    # 1. Price paths
    for i in range(min(50, len(price_paths))):
        ax1.plot(time_points, price_paths[i], alpha=0.3, linewidth=0.5, color='blue')
    ax1.plot(time_points, fair_value_paths, 'r-', linewidth=3, label='Fair Value')
    ax1.plot(time_points, np.percentile(price_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax1.set_title('Bitcoin Price Paths (Stochastic Mean Reversion)')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Price ($)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lambda paths
    for i in range(min(50, len(lambda_paths))):
        ax2.plot(time_points, lambda_paths[i], alpha=0.3, linewidth=0.5, color='red')
    ax2.plot(time_points, np.percentile(lambda_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax2.set_title('Mean Reversion Speed (λ) Paths')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Mean Reversion Speed (λ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Price distribution at end
    end_prices = price_paths[:, -1]
    ax3.hist(end_prices, bins=50, alpha=0.7, color='blue', density=True)
    ax3.axvline(fair_value_paths[-1], color='red', linestyle='--', linewidth=2, label='Fair Value')
    ax3.axvline(np.median(end_prices), color='green', linestyle='--', linewidth=2, label='Median')
    ax3.set_title('Price Distribution at End of Simulation')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Lambda distribution at end
    end_lambdas = lambda_paths[:, -1]
    ax4.hist(end_lambdas, bins=50, alpha=0.7, color='red', density=True)
    ax4.axvline(np.median(end_lambdas), color='green', linestyle='--', linewidth=2, label='Median')
    ax4.set_title('Mean Reversion Speed Distribution at End')
    ax4.set_xlabel('Mean Reversion Speed (λ)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_stochastic_mean_reversion_simulation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Stochastic simulation visualization saved to: {filename}")
    
    return filename

def save_stochastic_simulation_results(price_paths, lambda_paths, fair_value_paths, time_points):
    """Save stochastic simulation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save price paths
    price_df = pd.DataFrame(price_paths.T, columns=[f'Path_{i}' for i in range(len(price_paths))])
    price_df['Time_Years'] = time_points
    price_df['Fair_Value'] = fair_value_paths
    
    price_file = f'Results/Bitcoin/bitcoin_stochastic_price_paths_{timestamp}.csv'
    price_df.to_csv(price_file, index=False)
    print(f"Price paths saved to: {price_file}")
    
    # Save lambda paths
    lambda_df = pd.DataFrame(lambda_paths.T, columns=[f'Path_{i}' for i in range(len(lambda_paths))])
    lambda_df['Time_Years'] = time_points
    
    lambda_file = f'Results/Bitcoin/bitcoin_stochastic_lambda_paths_{timestamp}.csv'
    lambda_df.to_csv(lambda_file, index=False)
    print(f"Lambda paths saved to: {lambda_file}")
    
    # Calculate summary statistics
    end_prices = price_paths[:, -1]
    end_lambdas = lambda_paths[:, -1]
    
    summary_stats = {
        'Metric': [
            'Final Price - Mean', 'Final Price - Median', 'Final Price - Std',
            'Final Price - 5th Percentile', 'Final Price - 95th Percentile',
            'Final Lambda - Mean', 'Final Lambda - Median', 'Final Lambda - Std',
            'Price/Fair Value Ratio - Mean', 'Price/Fair Value Ratio - Median'
        ],
        'Value': [
            f"${end_prices.mean():,.2f}",
            f"${np.median(end_prices):,.2f}",
            f"${end_prices.std():,.2f}",
            f"${np.percentile(end_prices, 5):,.2f}",
            f"${np.percentile(end_prices, 95):,.2f}",
            f"{end_lambdas.mean():.2f}",
            f"{np.median(end_lambdas):.2f}",
            f"{end_lambdas.std():.2f}",
            f"{(end_prices / fair_value_paths[-1]).mean():.3f}",
            f"{np.median(end_prices / fair_value_paths[-1]):.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = f'Results/Bitcoin/bitcoin_stochastic_simulation_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    return price_file, lambda_file, summary_file

def main():
    """Main simulation function"""
    print("Bitcoin Stochastic Mean Reversion GBM Simulation")
    print("="*60)
    
    # Set simulation parameters
    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2030-01-01')
    n_paths = 1000
    n_steps = 365 * 5  # 5 years, daily steps
    
    # Run simulation
    price_paths, lambda_paths, fair_value_paths, time_points = stochastic_mean_reversion_gbm_simulation(
        start_date, end_date, n_paths, n_steps
    )
    
    # Create visualizations
    viz_file = create_stochastic_simulation_visualization(
        price_paths, lambda_paths, fair_value_paths, time_points
    )
    
    # Save results
    price_file, lambda_file, summary_file = save_stochastic_simulation_results(
        price_paths, lambda_paths, fair_value_paths, time_points
    )
    
    # Summary
    print(f"\n" + "="*60)
    print("STOCHASTIC MEAN REVERSION SIMULATION SUMMARY")
    print("="*60)
    
    end_prices = price_paths[:, -1]
    end_lambdas = lambda_paths[:, -1]
    final_fair_value = fair_value_paths[-1]
    
    print(f"Simulation Parameters:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"  Paths: {n_paths:,}")
    print(f"  Steps: {n_steps:,}")
    
    print(f"\nFinal Results:")
    print(f"  Final fair value: ${final_fair_value:,.2f}")
    print(f"  Final price - Mean: ${end_prices.mean():,.2f}")
    print(f"  Final price - Median: ${np.median(end_prices):,.2f}")
    print(f"  Final price - Std: ${end_prices.std():,.2f}")
    print(f"  Price/Fair Value ratio - Mean: {(end_prices / final_fair_value).mean():.3f}")
    print(f"  Price/Fair Value ratio - Median: {np.median(end_prices / final_fair_value):.3f}")
    
    print(f"\nMean Reversion Speed (Final):")
    print(f"  Mean λ: {end_lambdas.mean():.2f}")
    print(f"  Median λ: {np.median(end_lambdas):.2f}")
    print(f"  Std λ: {end_lambdas.std():.2f}")
    
    print(f"\nFiles created:")
    print(f"  Price paths: {price_file}")
    print(f"  Lambda paths: {lambda_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Visualization: {viz_file}")

if __name__ == "__main__":
    main() 