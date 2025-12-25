"""
Monte Carlo Simulation Accuracy Verification
===========================================

This script provides multiple methods to verify the accuracy of our
Bitcoin Monte Carlo simulation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_monte_carlo_data():
    """Load the Monte Carlo simulation data"""
    print("Loading Monte Carlo simulation data...")
    try:
        price_paths = pd.read_csv('Results/Bitcoin/bitcoin_monte_carlo_simple_paths_20250720.csv')
        print(f"✅ Loaded {len(price_paths.columns)} price paths")
        print(f"Time steps: {len(price_paths)}")
        return price_paths
    except FileNotFoundError:
        print("❌ Monte Carlo data not found. Please run bitcoin_monte_carlo_simple.py first.")
        return None

def get_formula_fair_value():
    """Get the formula's fair value for Bitcoin"""
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    today_day = 6041
    fair_value = 10**(a * np.log(today_day) + b)
    return fair_value

def verify_starting_conditions(price_paths):
    """Verify that all paths start at the correct price"""
    print("\n" + "="*80)
    print("VERIFICATION 1: STARTING CONDITIONS")
    print("="*80)
    
    # Get starting prices (first row, excluding the 'Years' column)
    starting_prices = price_paths.iloc[0, 1:].values
    
    print(f"Expected starting price: $118,075.00")
    print(f"Actual starting prices:")
    print(f"  Minimum: ${starting_prices.min():,.2f}")
    print(f"  Maximum: ${starting_prices.max():,.2f}")
    print(f"  Mean: ${starting_prices.mean():,.2f}")
    print(f"  Standard deviation: ${starting_prices.std():,.2f}")
    
    # Check if all paths start at the correct price
    expected_price = 118075.0
    tolerance = 0.01  # 1 cent tolerance
    
    if np.allclose(starting_prices, expected_price, atol=tolerance):
        print("✅ PASS: All paths start at the correct price")
        return True
    else:
        print("❌ FAIL: Paths do not start at the correct price")
        return False

def verify_formula_convergence(price_paths):
    """Verify that paths converge toward formula predictions"""
    print("\n" + "="*80)
    print("VERIFICATION 2: FORMULA CONVERGENCE")
    print("="*80)
    
    # Get formula predictions
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    today_day = 6041
    
    # Calculate formula predictions for key years
    years_to_check = [1, 5, 10]
    formula_predictions = {}
    
    for year in years_to_check:
        future_day = today_day + int(year * 365.25)
        formula_price = 10**(a * np.log(future_day) + b)
        formula_predictions[year] = formula_price
    
    print("Formula Predictions vs Simulation Results:")
    print("Year | Formula Prediction | Simulation Mean | Difference | % Error")
    print("-" * 75)
    
    convergence_ok = True
    
    for year in years_to_check:
        # Get simulation results for this year
        day_index = int(year * 365.25)
        if day_index < len(price_paths):
            prices_at_year = price_paths.iloc[day_index, 1:].values  # Skip 'Years' column
            sim_mean = prices_at_year.mean()
            
            formula_pred = formula_predictions[year]
            difference = sim_mean - formula_pred
            percent_error = abs(difference / formula_pred) * 100
            
            print(f"{year:4.0f} | ${formula_pred:14,.0f} | ${sim_mean:14,.0f} | ${difference:+10,.0f} | {percent_error:6.1f}%")
            
            # Check if convergence is reasonable (within 50% for year 1, 30% for year 5, 20% for year 10)
            max_error = 50 if year == 1 else (30 if year == 5 else 20)
            if percent_error > max_error:
                convergence_ok = False
                print(f"    ⚠️  Error exceeds {max_error}% threshold")
    
    if convergence_ok:
        print("✅ PASS: Paths converge reasonably toward formula predictions")
    else:
        print("❌ FAIL: Paths do not converge properly toward formula predictions")
    
    return convergence_ok

def verify_volatility_behavior(price_paths):
    """Verify that volatility decreases over time as expected"""
    print("\n" + "="*80)
    print("VERIFICATION 3: VOLATILITY BEHAVIOR")
    print("="*80)
    
    # Calculate volatility at different time points
    years_to_check = [0, 1, 2, 5, 10]
    volatilities = []
    
    print("Volatility Analysis:")
    print("Year | Mean Price | Std Dev | Coefficient of Variation")
    print("-" * 55)
    
    volatility_decreasing = True
    
    for year in years_to_check:
        day_index = int(year * 365.25)
        if day_index < len(price_paths):
            prices_at_year = price_paths.iloc[day_index, 1:].values  # Skip 'Years' column
            mean_price = prices_at_year.mean()
            std_dev = prices_at_year.std()
            cv = std_dev / mean_price * 100  # Coefficient of variation
            
            volatilities.append(cv)
            print(f"{year:4.0f} | ${mean_price:10,.0f} | ${std_dev:8,.0f} | {cv:6.1f}%")
    
    # Check if volatility decreases over time
    for i in range(1, len(volatilities)):
        if volatilities[i] > volatilities[i-1] * 1.1:  # Allow 10% increase due to randomness
            volatility_decreasing = False
            print(f"    ⚠️  Volatility increased from year {years_to_check[i-1]} to {years_to_check[i]}")
    
    if volatility_decreasing:
        print("✅ PASS: Volatility decreases over time as expected")
    else:
        print("❌ FAIL: Volatility does not decrease properly over time")
    
    return volatility_decreasing

def verify_economic_realism(price_paths):
    """Verify that the simulation produces economically realistic results"""
    print("\n" + "="*80)
    print("VERIFICATION 4: ECONOMIC REALISM")
    print("="*80)
    
    # Check final prices
    final_prices = price_paths.iloc[-1, 1:].values  # Skip 'Years' column
    
    print("Final Price Analysis (10 years):")
    print(f"  Minimum: ${final_prices.min():,.2f}")
    print(f"  Maximum: ${final_prices.max():,.2f}")
    print(f"  Mean: ${final_prices.mean():,.2f}")
    print(f"  Median: ${np.median(final_prices):,.2f}")
    print(f"  5th percentile: ${np.percentile(final_prices, 5):,.2f}")
    print(f"  95th percentile: ${np.percentile(final_prices, 95):,.2f}")
    
    # Economic realism checks
    checks_passed = 0
    total_checks = 5
    
    # Check 1: No negative prices
    if np.all(final_prices > 0):
        print("✅ Check 1 PASS: No negative prices")
        checks_passed += 1
    else:
        print("❌ Check 1 FAIL: Found negative prices")
    
    # Check 2: Reasonable price range (not too low, not too high)
    min_reasonable = 10000  # $10k minimum
    max_reasonable = 1000000  # $1M maximum
    
    if np.all(final_prices >= min_reasonable) and np.all(final_prices <= max_reasonable):
        print("✅ Check 2 PASS: Prices within reasonable range")
        checks_passed += 1
    else:
        print("❌ Check 2 FAIL: Prices outside reasonable range")
    
    # Check 3: Most paths show growth (Bitcoin's historical trend)
    growth_paths = np.sum(final_prices > 118075.0)  # Starting price
    growth_percentage = growth_paths / len(final_prices) * 100
    
    if growth_percentage >= 60:  # At least 60% should show growth
        print(f"✅ Check 3 PASS: {growth_percentage:.1f}% of paths show growth")
        checks_passed += 1
    else:
        print(f"❌ Check 3 FAIL: Only {growth_percentage:.1f}% of paths show growth")
    
    # Check 4: Reasonable annualized returns
    annualized_returns = (final_prices / 118075.0) ** (1/10) - 1
    mean_annual_return = annualized_returns.mean() * 100
    
    if -20 <= mean_annual_return <= 50:  # Between -20% and +50% annually
        print(f"✅ Check 4 PASS: Mean annual return {mean_annual_return:.1f}% is reasonable")
        checks_passed += 1
    else:
        print(f"❌ Check 4 FAIL: Mean annual return {mean_annual_return:.1f}% is unrealistic")
    
    # Check 5: Price distribution is log-normal (not too skewed)
    log_prices = np.log(final_prices)
    skewness = np.mean(((log_prices - log_prices.mean()) / log_prices.std()) ** 3)
    
    if abs(skewness) <= 2:  # Reasonable skewness
        print(f"✅ Check 5 PASS: Log-price skewness {skewness:.2f} is reasonable")
        checks_passed += 1
    else:
        print(f"❌ Check 5 FAIL: Log-price skewness {skewness:.2f} is too extreme")
    
    print(f"\nEconomic Realism Score: {checks_passed}/{total_checks}")
    
    return checks_passed >= 4  # Pass if at least 4/5 checks pass

def verify_statistical_properties(price_paths):
    """Verify statistical properties of the simulation"""
    print("\n" + "="*80)
    print("VERIFICATION 5: STATISTICAL PROPERTIES")
    print("="*80)
    
    # Calculate returns for each path
    starting_prices = price_paths.iloc[0, 1:].values  # Skip 'Years' column
    final_prices = price_paths.iloc[-1, 1:].values  # Skip 'Years' column
    total_returns = (final_prices / starting_prices) - 1
    
    print("Return Statistics:")
    print(f"  Mean return: {total_returns.mean()*100:.1f}%")
    print(f"  Median return: {np.median(total_returns)*100:.1f}%")
    print(f"  Standard deviation: {total_returns.std()*100:.1f}%")
    print(f"  Skewness: {np.mean(((total_returns - total_returns.mean()) / total_returns.std()) ** 3):.2f}")
    print(f"  Kurtosis: {np.mean(((total_returns - total_returns.mean()) / total_returns.std()) ** 4):.2f}")
    
    # Check for reasonable statistical properties
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Returns are not all identical
    if total_returns.std() > 0.01:  # At least 1% standard deviation
        print("✅ Check 1 PASS: Returns show reasonable variation")
        checks_passed += 1
    else:
        print("❌ Check 1 FAIL: Returns show insufficient variation")
    
    # Check 2: No extreme outliers (more than 5 standard deviations)
    z_scores = np.abs((total_returns - total_returns.mean()) / total_returns.std())
    extreme_outliers = np.sum(z_scores > 5)
    
    if extreme_outliers <= len(total_returns) * 0.01:  # Less than 1% extreme outliers
        print(f"✅ Check 2 PASS: Only {extreme_outliers} extreme outliers")
        checks_passed += 1
    else:
        print(f"❌ Check 2 FAIL: Too many extreme outliers ({extreme_outliers})")
    
    # Check 3: Returns distribution is reasonable
    positive_returns = np.sum(total_returns > 0)
    negative_returns = np.sum(total_returns < 0)
    
    if 0.3 <= positive_returns / len(total_returns) <= 0.9:  # Between 30% and 90% positive
        print(f"✅ Check 3 PASS: {positive_returns/len(total_returns)*100:.1f}% positive returns")
        checks_passed += 1
    else:
        print(f"❌ Check 3 FAIL: {positive_returns/len(total_returns)*100:.1f}% positive returns (unrealistic)")
    
    print(f"\nStatistical Properties Score: {checks_passed}/{total_checks}")
    
    return checks_passed >= 2  # Pass if at least 2/3 checks pass

def create_verification_visualization(price_paths):
    """Create visualization to help verify simulation accuracy"""
    print("\n" + "="*80)
    print("CREATING VERIFICATION VISUALIZATION")
    print("="*80)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Monte Carlo Simulation Verification', fontsize=16, fontweight='bold')
    
    # 1. Sample paths with formula prediction
    sample_paths = min(20, len(price_paths.columns))
    time_steps = len(price_paths)
    years = np.linspace(0, 10, time_steps)
    
    for i in range(sample_paths):
        ax1.plot(years, price_paths.iloc[:, i], alpha=0.5, linewidth=1)
    
    # Add formula prediction
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    today_day = 6041
    formula_prices = []
    for year in years:
        future_day = today_day + int(year * 365.25)
        formula_price = 10**(a * np.log(future_day) + b)
        formula_prices.append(formula_price)
    
    ax1.plot(years, formula_prices, 'r--', linewidth=3, label='Formula Prediction')
    ax1.axhline(y=118075, color='green', linestyle='-', linewidth=2, label='Starting Price')
    
    ax1.set_title('Sample Paths vs Formula Prediction')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Years')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Starting price distribution
    starting_prices = price_paths.iloc[0, :].values
    ax2.hist(starting_prices, bins=50, alpha=0.7, color='blue')
    ax2.axvline(x=118075, color='red', linestyle='--', linewidth=2, label='Expected: $118,075')
    ax2.set_title('Starting Price Distribution')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Number of Paths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final price distribution
    final_prices = price_paths.iloc[-1, :].values
    ax3.hist(final_prices, bins=50, alpha=0.7, color='green')
    ax3.set_title('Final Price Distribution (10 years)')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Number of Paths')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Volatility over time
    years_to_check = [0, 1, 2, 5, 10]
    volatilities = []
    
    for year in years_to_check:
        day_index = int(year * 365.25)
        if day_index < len(price_paths):
            prices_at_year = price_paths.iloc[day_index, :].values
            cv = prices_at_year.std() / prices_at_year.mean() * 100
            volatilities.append(cv)
    
    ax4.plot(years_to_check, volatilities, 'bo-', linewidth=2, markersize=8)
    ax4.set_title('Volatility Over Time')
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Results/Bitcoin/monte_carlo_verification_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Verification visualization saved to: {filename}")
    
    plt.show()

def main():
    """Main verification function"""
    print("MONTE CARLO SIMULATION ACCURACY VERIFICATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    price_paths = load_monte_carlo_data()
    if price_paths is None:
        return
    
    # Run all verification checks
    checks = []
    
    checks.append(("Starting Conditions", verify_starting_conditions(price_paths)))
    checks.append(("Formula Convergence", verify_formula_convergence(price_paths)))
    checks.append(("Volatility Behavior", verify_volatility_behavior(price_paths)))
    checks.append(("Economic Realism", verify_economic_realism(price_paths)))
    checks.append(("Statistical Properties", verify_statistical_properties(price_paths)))
    
    # Create visualization
    create_verification_visualization(price_paths)
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed_checks = sum(1 for _, passed in checks if passed)
    total_checks = len(checks)
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:25} | {status}")
    
    print("-" * 40)
    print(f"Overall Score: {passed_checks}/{total_checks}")
    
    if passed_checks >= 4:
        print("🎉 VERIFICATION PASSED: Simulation appears accurate!")
    elif passed_checks >= 3:
        print("⚠️  PARTIAL PASS: Some issues detected, but simulation may be usable")
    else:
        print("❌ VERIFICATION FAILED: Significant issues detected")
    
    print("="*80)

if __name__ == "__main__":
    main() 