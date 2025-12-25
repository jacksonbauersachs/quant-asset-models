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
    df['Price_Ratio'] = df['Price'] / df['Fair_Value']
    return df

def realistic_mean_reversion_gbm_simulation(start_date, end_date, n_paths=1000, n_steps=365):
    """Simulate Bitcoin price with realistic mean reversion starting at market price"""
    print(f"Running realistic mean reversion GBM simulation...")
    print(f"Paths: {n_paths}, Steps: {n_steps}")
    
    # Load data and calculate fair values
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Load models
    with open('Models/Volatility Models/bitcoin_volatility_model_coefficients.txt', 'r') as f:
        lines = f.readlines()
        vol_a = float(lines[0].split('=')[1].strip())
        vol_b = float(lines[1].split('=')[1].strip())
        vol_c = float(lines[2].split('=')[1].strip())
    
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
    
    # Get starting conditions
    start_data = df[df['Date'] >= start_date].iloc[0]
    start_price = start_data['Price']
    start_fair_value = start_data['Fair_Value']
    start_price_ratio = start_data['Price_Ratio']
    
    print(f"Starting conditions:")
    print(f"  Market Price: ${start_price:,.2f}")
    print(f"  Fair Value: ${start_fair_value:,.2f}")
    print(f"  Price/Fair Value Ratio: {start_price_ratio:.3f}")
    print(f"  Overvalued by: {(start_price_ratio - 1) * 100:.1f}%")
    
    # Mean reversion parameters (conservative estimates)
    lambda_base = 30  # Base mean reversion speed
    lambda_vol = 10   # Volatility of mean reversion speed
    
    # Initialize price paths
    price_paths = np.zeros((n_paths, n_steps + 1))
    fair_value_paths = np.zeros((n_steps + 1))
    price_ratio_paths = np.zeros((n_paths, n_steps + 1))
    
    # Set initial values
    price_paths[:, 0] = start_price
    price_ratio_paths[:, 0] = start_price_ratio
    
    # Calculate fair value path
    for i, age in enumerate(time_points):
        days_since_genesis = age * 365.25
        fair_value_paths[i] = 10**(growth_a * np.log(days_since_genesis) + growth_b)
    
    # Simulation loop
    for i in range(n_steps):
        current_age = time_points[i]
        current_fair_value = fair_value_paths[i]
        
        # Current volatility
        current_vol = vol_a * np.exp(-vol_b * current_age) + vol_c
        
        # Growth rate
        current_growth = growth_a / (current_age * np.log(10))
        
        for path in range(n_paths):
            current_price = price_paths[path, i]
            current_price_ratio = price_ratio_paths[path, i]
            
            # Dynamic mean reversion speed based on price ratio
            # Higher correction when more overvalued/undervalued
            if current_price_ratio > 1.5:  # Very overvalued
                lambda_current = lambda_base * 2.0
            elif current_price_ratio > 1.2:  # Moderately overvalued
                lambda_current = lambda_base * 1.5
            elif current_price_ratio > 1.0:  # Slightly overvalued
                lambda_current = lambda_base * 1.0
            elif current_price_ratio > 0.8:  # Slightly undervalued
                lambda_current = lambda_base * 1.0
            elif current_price_ratio > 0.5:  # Moderately undervalued
                lambda_current = lambda_base * 1.5
            else:  # Very undervalued
                lambda_current = lambda_base * 2.0
            
            # Add some randomness to lambda
            lambda_current += np.random.normal(0, lambda_vol)
            lambda_current = max(lambda_current, 5)  # Minimum lambda
            
            # Mean reversion correction
            mean_reversion_correction = lambda_current * (current_fair_value - current_price) / current_price
            
            # Total drift (growth + mean reversion)
            total_drift = current_growth + mean_reversion_correction
            
            # Diffusion (random noise)
            diffusion = current_vol * np.sqrt(dt) * np.random.normal(0, 1)
            
            # Update price
            new_price = current_price * (1 + total_drift * dt + diffusion)
            new_price = max(new_price, 0.01)  # Ensure positive price
            
            price_paths[path, i + 1] = new_price
            price_ratio_paths[path, i + 1] = new_price / current_fair_value
    
    return price_paths, fair_value_paths, price_ratio_paths, time_points

def create_realistic_simulation_visualization(price_paths, fair_value_paths, price_ratio_paths, time_points):
    """Create visualization of realistic mean reversion simulation"""
    print("Creating realistic simulation visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin: Realistic Mean Reversion GBM (Starting at Market Price)', fontsize=16, fontweight='bold')
    
    # 1. Price paths
    for i in range(min(50, len(price_paths))):
        ax1.plot(time_points, price_paths[i], alpha=0.3, linewidth=0.5, color='blue')
    ax1.plot(time_points, fair_value_paths, 'r-', linewidth=3, label='Fair Value')
    ax1.plot(time_points, np.percentile(price_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax1.plot(time_points, np.percentile(price_paths, 25, axis=0), 'orange', linewidth=2, label='25th Percentile')
    ax1.plot(time_points, np.percentile(price_paths, 75, axis=0), 'purple', linewidth=2, label='75th Percentile')
    ax1.set_title('Bitcoin Price Paths (Realistic Mean Reversion)')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Price ($)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price ratio paths
    for i in range(min(50, len(price_ratio_paths))):
        ax2.plot(time_points, price_ratio_paths[i], alpha=0.3, linewidth=0.5, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Fair Value (1.0)')
    ax2.plot(time_points, np.percentile(price_ratio_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax2.set_title('Price/Fair Value Ratio Paths')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Price/Fair Value Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final price distribution
    end_prices = price_paths[:, -1]
    end_ratios = price_ratio_paths[:, -1]
    
    ax3.hist(end_prices, bins=50, alpha=0.7, color='blue', density=True)
    ax3.axvline(fair_value_paths[-1], color='red', linestyle='--', linewidth=2, label='Fair Value')
    ax3.axvline(np.median(end_prices), color='green', linestyle='--', linewidth=2, label='Median')
    ax3.set_title('Final Price Distribution')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final price ratio distribution
    ax4.hist(end_ratios, bins=50, alpha=0.7, color='green', density=True)
    ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Fair Value (1.0)')
    ax4.axvline(np.median(end_ratios), color='blue', linestyle='--', linewidth=2, label='Median')
    ax4.set_title('Final Price/Fair Value Ratio Distribution')
    ax4.set_xlabel('Price/Fair Value Ratio')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_realistic_mean_reversion_simulation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Realistic simulation visualization saved to: {filename}")
    
    return filename

def calculate_investment_recommendations(price_paths, fair_value_paths, price_ratio_paths):
    """Calculate investment recommendations based on simulation results"""
    print("Calculating investment recommendations...")
    
    end_prices = price_paths[:, -1]
    end_ratios = price_ratio_paths[:, -1]
    final_fair_value = fair_value_paths[-1]
    
    # Calculate probabilities
    prob_above_fair = np.mean(end_ratios > 1.0)
    prob_below_fair = np.mean(end_ratios < 1.0)
    prob_significant_drop = np.mean(end_ratios < 0.8)  # 20% below fair value
    prob_significant_gain = np.mean(end_ratios > 1.2)  # 20% above fair value
    
    # Calculate expected returns
    expected_return = np.mean(end_prices) / price_paths[0, 0] - 1
    median_return = np.median(end_prices) / price_paths[0, 0] - 1
    
    # Risk metrics
    volatility = np.std(end_prices) / price_paths[0, 0]
    var_95 = np.percentile(end_prices, 5) / price_paths[0, 0] - 1  # 95% VaR
    var_99 = np.percentile(end_prices, 1) / price_paths[0, 0] - 1  # 99% VaR
    
    # Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    recommendations = {
        'Probability Analysis': {
            'Probability of ending above fair value': f"{prob_above_fair:.1%}",
            'Probability of ending below fair value': f"{prob_below_fair:.1%}",
            'Probability of significant drop (>20% below fair)': f"{prob_significant_drop:.1%}",
            'Probability of significant gain (>20% above fair)': f"{prob_significant_gain:.1%}"
        },
        'Return Analysis': {
            'Expected return': f"{expected_return:.1%}",
            'Median return': f"{median_return:.1%}",
            'Volatility': f"{volatility:.1%}",
            'Sharpe ratio': f"{sharpe_ratio:.2f}"
        },
        'Risk Analysis': {
            '95% VaR (worst 5% outcome)': f"{var_95:.1%}",
            '99% VaR (worst 1% outcome)': f"{var_99:.1%}",
            'Final price - 5th percentile': f"${np.percentile(end_prices, 5):,.0f}",
            'Final price - 95th percentile': f"${np.percentile(end_prices, 95):,.0f}"
        },
        'Investment Recommendation': {
            'Current situation': f"Bitcoin is {(price_ratio_paths[0, 0] - 1) * 100:.1f}% above fair value",
            'Mean reversion effect': f"Model predicts {prob_below_fair:.1%} chance of ending below fair value",
            'Risk-adjusted return': f"Sharpe ratio of {sharpe_ratio:.2f}",
            'Recommendation': 'HOLD' if sharpe_ratio < 0.5 else 'CONSIDER BUYING' if sharpe_ratio > 1.0 else 'CAUTIOUS'
        }
    }
    
    return recommendations

def save_realistic_simulation_results(price_paths, fair_value_paths, price_ratio_paths, time_points, recommendations):
    """Save realistic simulation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save price paths
    price_df = pd.DataFrame(price_paths.T, columns=[f'Path_{i}' for i in range(len(price_paths))])
    price_df['Time_Years'] = time_points
    price_df['Fair_Value'] = fair_value_paths
    
    price_file = f'Results/Bitcoin/bitcoin_realistic_price_paths_{timestamp}.csv'
    price_df.to_csv(price_file, index=False)
    print(f"Price paths saved to: {price_file}")
    
    # Save recommendations
    rec_file = f'Results/Bitcoin/bitcoin_investment_recommendations_{timestamp}.txt'
    with open(rec_file, 'w', encoding='utf-8') as f:
        f.write("Bitcoin Investment Recommendations\n")
        f.write("="*50 + "\n\n")
        
        for category, items in recommendations.items():
            f.write(f"{category}:\n")
            f.write("-" * 30 + "\n")
            for key, value in items.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"Investment recommendations saved to: {rec_file}")
    
    return price_file, rec_file

def main():
    """Main simulation function"""
    print("Bitcoin Realistic Mean Reversion GBM Simulation")
    print("="*60)
    
    # Set simulation parameters
    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2030-01-01')
    n_paths = 1000
    n_steps = 365 * 5  # 5 years, daily steps
    
    # Run simulation
    price_paths, fair_value_paths, price_ratio_paths, time_points = realistic_mean_reversion_gbm_simulation(
        start_date, end_date, n_paths, n_steps
    )
    
    # Create visualizations
    viz_file = create_realistic_simulation_visualization(
        price_paths, fair_value_paths, price_ratio_paths, time_points
    )
    
    # Calculate investment recommendations
    recommendations = calculate_investment_recommendations(
        price_paths, fair_value_paths, price_ratio_paths
    )
    
    # Save results
    price_file, rec_file = save_realistic_simulation_results(
        price_paths, fair_value_paths, price_ratio_paths, time_points, recommendations
    )
    
    # Summary
    print(f"\n" + "="*60)
    print("REALISTIC MEAN REVERSION SIMULATION SUMMARY")
    print("="*60)
    
    end_prices = price_paths[:, -1]
    end_ratios = price_ratio_paths[:, -1]
    final_fair_value = fair_value_paths[-1]
    
    print(f"Simulation Parameters:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"  Paths: {n_paths:,}")
    print(f"  Steps: {n_steps:,}")
    
    print(f"\nStarting Conditions:")
    print(f"  Market Price: ${price_paths[0, 0]:,.2f}")
    print(f"  Fair Value: ${fair_value_paths[0]:,.2f}")
    print(f"  Price/Fair Value: {price_ratio_paths[0, 0]:.3f}")
    
    print(f"\nFinal Results:")
    print(f"  Final fair value: ${final_fair_value:,.2f}")
    print(f"  Final price - Mean: ${end_prices.mean():,.2f}")
    print(f"  Final price - Median: ${np.median(end_prices):,.2f}")
    print(f"  Final price ratio - Mean: {end_ratios.mean():.3f}")
    print(f"  Final price ratio - Median: {np.median(end_ratios):.3f}")
    
    print(f"\nInvestment Analysis:")
    for category, items in recommendations.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    print(f"\nFiles created:")
    print(f"  Price paths: {price_file}")
    print(f"  Recommendations: {rec_file}")
    print(f"  Visualization: {viz_file}")

if __name__ == "__main__":
    main() 