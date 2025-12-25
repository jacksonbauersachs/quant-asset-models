import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

def get_statistically_validated_mean_reversion_speed(price_ratio):
    """Get mean reversion speed based on statistically validated backtest results"""
    
    # Base mean reversion speed (conservative estimate)
    lambda_base = 30
    
    # Dynamic multipliers based on statistical significance
    if price_ratio > 1.5:  # Very Overvalued - HIGHLY SIGNIFICANT (p < 0.001)
        lambda_multiplier = 3.0  # Strong correction
        expected_return = -0.351  # -35.1% over 90 days
    elif price_ratio > 1.2:  # Moderately Overvalued - SIGNIFICANT (p = 0.027)
        lambda_multiplier = 2.0  # Moderate correction
        expected_return = -0.049  # -4.9% over 90 days
    elif price_ratio > 1.0:  # Slightly Overvalued - NOT SIGNIFICANT
        lambda_multiplier = 0.5  # Weak correction
        expected_return = 0.006  # +0.6% over 90 days
    elif price_ratio > 0.8:  # Fair Value - NOT SIGNIFICANT
        lambda_multiplier = 0.3  # Very weak correction
        expected_return = 0.038  # +3.8% over 90 days
    elif price_ratio > 0.5:  # Moderately Undervalued - HIGHLY SIGNIFICANT (p < 0.001)
        lambda_multiplier = 2.5  # Strong upward correction
        expected_return = 0.256  # +25.6% over 90 days
    else:  # Very Undervalued - HIGHLY SIGNIFICANT (p < 0.001)
        lambda_multiplier = 4.0  # Very strong upward correction
        expected_return = 0.994  # +99.4% over 90 days
    
    lambda_speed = lambda_base * lambda_multiplier
    
    return lambda_speed, expected_return

def statistically_validated_gbm_simulation(start_date, end_date, n_paths=1000, n_steps=365):
    """Simulate Bitcoin price with statistically validated mean reversion"""
    print(f"Running statistically validated GBM simulation...")
    print(f"Paths: {n_paths}, Steps: {n_steps}")
    
    # Load data and calculate fair values
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Load models
    with open('Models/Volatility Models/bitcoin_volatility_model_coefficients.txt', 'r') as f:
        lines = f.readlines()
        vol_a = float(lines[2].split('=')[1].strip())
        vol_b = float(lines[3].split('=')[1].strip())
        vol_c = 0  # No constant term in this model
    
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
    
    # Get initial mean reversion parameters
    initial_lambda, initial_expected_return = get_statistically_validated_mean_reversion_speed(start_price_ratio)
    print(f"  Initial mean reversion speed: {initial_lambda:.1f}")
    print(f"  Expected 90-day return (historical): {initial_expected_return:.1%}")
    
    # Initialize price paths
    price_paths = np.zeros((n_paths, n_steps + 1))
    fair_value_paths = np.zeros((n_steps + 1))
    price_ratio_paths = np.zeros((n_paths, n_steps + 1))
    lambda_paths = np.zeros((n_paths, n_steps + 1))
    
    # Set initial values
    price_paths[:, 0] = start_price
    price_ratio_paths[:, 0] = start_price_ratio
    lambda_paths[:, 0] = initial_lambda
    
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
            
            # Get statistically validated mean reversion speed
            lambda_speed, expected_return = get_statistically_validated_mean_reversion_speed(current_price_ratio)
            
            # Add some randomness to lambda (but keep it reasonable)
            lambda_volatility = lambda_speed * 0.2  # 20% volatility
            lambda_speed += np.random.normal(0, lambda_volatility)
            lambda_speed = max(lambda_speed, 5)  # Minimum lambda
            
            # Mean reversion correction
            mean_reversion_correction = lambda_speed * (current_fair_value - current_price) / current_price
            
            # Total drift (growth + mean reversion)
            total_drift = current_growth + mean_reversion_correction
            
            # Diffusion (random noise)
            diffusion = current_vol * np.sqrt(dt) * np.random.normal(0, 1)
            
            # Update price
            new_price = current_price * (1 + total_drift * dt + diffusion)
            new_price = max(new_price, 0.01)  # Ensure positive price
            
            price_paths[path, i + 1] = new_price
            price_ratio_paths[path, i + 1] = new_price / current_fair_value
            lambda_paths[path, i + 1] = lambda_speed
    
    return price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points

def create_statistically_validated_visualization(price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points):
    """Create visualization of statistically validated simulation"""
    print("Creating statistically validated visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin: Statistically Validated Mean Reversion GBM', fontsize=16, fontweight='bold')
    
    # 1. Price paths
    for i in range(min(50, len(price_paths))):
        ax1.plot(time_points, price_paths[i], alpha=0.3, linewidth=0.5, color='blue')
    ax1.plot(time_points, fair_value_paths, 'r-', linewidth=3, label='Fair Value')
    ax1.plot(time_points, np.percentile(price_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax1.plot(time_points, np.percentile(price_paths, 25, axis=0), 'orange', linewidth=2, label='25th Percentile')
    ax1.plot(time_points, np.percentile(price_paths, 75, axis=0), 'purple', linewidth=2, label='75th Percentile')
    ax1.set_title('Bitcoin Price Paths (Statistically Validated)')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Price ($)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price ratio paths with statistical thresholds
    for i in range(min(50, len(price_ratio_paths))):
        ax2.plot(time_points, price_ratio_paths[i], alpha=0.3, linewidth=0.5, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Fair Value (1.0)')
    ax2.axhline(y=1.2, color='orange', linestyle=':', linewidth=2, label='Moderately Overvalued (1.2)')
    ax2.axhline(y=1.5, color='red', linestyle=':', linewidth=2, label='Very Overvalued (1.5)')
    ax2.axhline(y=0.8, color='blue', linestyle=':', linewidth=2, label='Fair Value Lower (0.8)')
    ax2.axhline(y=0.5, color='purple', linestyle=':', linewidth=2, label='Very Undervalued (0.5)')
    ax2.plot(time_points, np.percentile(price_ratio_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax2.set_title('Price/Fair Value Ratio Paths (Statistical Thresholds)')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Price/Fair Value Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean reversion speed paths
    for i in range(min(50, len(lambda_paths))):
        ax3.plot(time_points, lambda_paths[i], alpha=0.3, linewidth=0.5, color='red')
    ax3.plot(time_points, np.percentile(lambda_paths, 50, axis=0), 'g-', linewidth=2, label='Median')
    ax3.set_title('Mean Reversion Speed (λ) Paths')
    ax3.set_xlabel('Bitcoin Age (years)')
    ax3.set_ylabel('Mean Reversion Speed (λ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final price ratio distribution with statistical zones
    end_ratios = price_ratio_paths[:, -1]
    
    # Create histogram with colored zones
    n, bins, patches = ax4.hist(end_ratios, bins=50, alpha=0.7, color='green', density=True)
    
    # Color zones based on statistical significance
    for i, patch in enumerate(patches):
        if bins[i] > 1.5:
            patch.set_color('red')  # Very overvalued - highly significant
        elif bins[i] > 1.2:
            patch.set_color('orange')  # Moderately overvalued - significant
        elif bins[i] > 1.0:
            patch.set_color('yellow')  # Slightly overvalued - not significant
        elif bins[i] > 0.8:
            patch.set_color('lightblue')  # Fair value - not significant
        elif bins[i] > 0.5:
            patch.set_color('blue')  # Moderately undervalued - highly significant
        else:
            patch.set_color('purple')  # Very undervalued - highly significant
    
    ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Fair Value (1.0)')
    ax4.axvline(np.median(end_ratios), color='blue', linestyle='--', linewidth=2, label='Median')
    ax4.set_title('Final Price/Fair Value Ratio Distribution\n(Colored by Statistical Significance)')
    ax4.set_xlabel('Price/Fair Value Ratio')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_statistically_validated_simulation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Statistically validated simulation saved to: {filename}")
    
    return filename

def calculate_statistically_validated_recommendations(price_paths, fair_value_paths, price_ratio_paths):
    """Calculate investment recommendations based on statistically validated results"""
    print("Calculating statistically validated recommendations...")
    
    end_prices = price_paths[:, -1]
    end_ratios = price_ratio_paths[:, -1]
    final_fair_value = fair_value_paths[-1]
    
    # Calculate probabilities by statistical zones
    prob_very_overvalued = np.mean(end_ratios > 1.5)
    prob_moderately_overvalued = np.mean((end_ratios > 1.2) & (end_ratios <= 1.5))
    prob_slightly_overvalued = np.mean((end_ratios > 1.0) & (end_ratios <= 1.2))
    prob_fair_value = np.mean((end_ratios > 0.8) & (end_ratios <= 1.0))
    prob_moderately_undervalued = np.mean((end_ratios > 0.5) & (end_ratios <= 0.8))
    prob_very_undervalued = np.mean(end_ratios <= 0.5)
    
    # Calculate expected returns
    expected_return = np.mean(end_prices) / price_paths[0, 0] - 1
    median_return = np.median(end_prices) / price_paths[0, 0] - 1
    
    # Risk metrics
    volatility = np.std(end_prices) / price_paths[0, 0]
    var_95 = np.percentile(end_prices, 5) / price_paths[0, 0] - 1
    var_99 = np.percentile(end_prices, 1) / price_paths[0, 0] - 1
    
    # Sharpe ratio
    risk_free_rate = 0.02
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Statistical significance analysis
    significant_negative_zones = prob_very_overvalued + prob_moderately_overvalued
    significant_positive_zones = prob_very_undervalued + prob_moderately_undervalued
    
    recommendations = {
        'Statistical Zone Probabilities': {
            'Very Overvalued (>1.5x) - Highly Significant': f"{prob_very_overvalued:.1%}",
            'Moderately Overvalued (1.2-1.5x) - Significant': f"{prob_moderately_overvalued:.1%}",
            'Slightly Overvalued (1.0-1.2x) - Not Significant': f"{prob_slightly_overvalued:.1%}",
            'Fair Value (0.8-1.0x) - Not Significant': f"{prob_fair_value:.1%}",
            'Moderately Undervalued (0.5-0.8x) - Highly Significant': f"{prob_moderately_undervalued:.1%}",
            'Very Undervalued (<0.5x) - Highly Significant': f"{prob_very_undervalued:.1%}"
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
        'Statistically Validated Recommendation': {
            'Current situation': f"Bitcoin is {(price_ratio_paths[0, 0] - 1) * 100:.1f}% above fair value",
            'Probability of significant negative zones': f"{significant_negative_zones:.1%}",
            'Probability of significant positive zones': f"{significant_positive_zones:.1%}",
            'Risk-adjusted return': f"Sharpe ratio of {sharpe_ratio:.2f}",
            'Recommendation': 'HOLD' if sharpe_ratio < 0.5 else 'CONSIDER BUYING' if sharpe_ratio > 1.0 else 'CAUTIOUS'
        }
    }
    
    return recommendations

def save_statistically_validated_results(price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points, recommendations):
    """Save statistically validated simulation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save price paths
    price_df = pd.DataFrame(price_paths.T, columns=[f'Path_{i}' for i in range(len(price_paths))])
    price_df['Time_Years'] = time_points
    price_df['Fair_Value'] = fair_value_paths
    
    price_file = f'Results/Bitcoin/bitcoin_statistically_validated_paths_{timestamp}.csv'
    price_df.to_csv(price_file, index=False)
    print(f"Statistically validated paths saved to: {price_file}")
    
    # Save recommendations
    rec_file = f'Results/Bitcoin/bitcoin_statistically_validated_recommendations_{timestamp}.txt'
    with open(rec_file, 'w', encoding='utf-8') as f:
        f.write("Bitcoin Statistically Validated Investment Recommendations\n")
        f.write("="*60 + "\n\n")
        
        f.write("Mean Reversion Factors Used:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Very Overvalued (>1.5x): λ = 90, Expected Return = -35.1% (p < 0.001)\n")
        f.write("2. Moderately Overvalued (1.2-1.5x): λ = 60, Expected Return = -4.9% (p = 0.027)\n")
        f.write("3. Slightly Overvalued (1.0-1.2x): λ = 15, Expected Return = +0.6% (p = 0.785)\n")
        f.write("4. Fair Value (0.8-1.0x): λ = 9, Expected Return = +3.8% (p = 0.785)\n")
        f.write("5. Moderately Undervalued (0.5-0.8x): λ = 75, Expected Return = +25.6% (p < 0.001)\n")
        f.write("6. Very Undervalued (<0.5x): λ = 120, Expected Return = +99.4% (p < 0.001)\n\n")
        
        for category, items in recommendations.items():
            f.write(f"{category}:\n")
            f.write("-" * 30 + "\n")
            for key, value in items.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"Statistically validated recommendations saved to: {rec_file}")
    
    return price_file, rec_file

def main():
    """Main simulation function"""
    print("Bitcoin Statistically Validated Mean Reversion GBM Simulation")
    print("="*70)
    
    # Set simulation parameters
    start_date = pd.to_datetime('2025-07-19')  # Use current date from data
    end_date = pd.to_datetime('2030-07-19')
    n_paths = 1000
    n_steps = 365 * 5  # 5 years, daily steps
    
    # Run simulation
    price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points = statistically_validated_gbm_simulation(
        start_date, end_date, n_paths, n_steps
    )
    
    # Create visualizations
    viz_file = create_statistically_validated_visualization(
        price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points
    )
    
    # Calculate recommendations
    recommendations = calculate_statistically_validated_recommendations(
        price_paths, fair_value_paths, price_ratio_paths
    )
    
    # Save results
    price_file, rec_file = save_statistically_validated_results(
        price_paths, fair_value_paths, price_ratio_paths, lambda_paths, time_points, recommendations
    )
    
    # Summary
    print(f"\n" + "="*70)
    print("STATISTICALLY VALIDATED SIMULATION SUMMARY")
    print("="*70)
    
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
    
    print(f"\nStatistically Validated Analysis:")
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