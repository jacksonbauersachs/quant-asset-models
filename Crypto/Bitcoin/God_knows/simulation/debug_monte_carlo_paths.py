"""
Debug Monte Carlo Paths
======================

This script analyzes the Monte Carlo simulation paths to understand why
fair value drop strategies are showing such low returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_monte_carlo_data():
    """Load the Monte Carlo simulation data"""
    print("Loading Monte Carlo simulation data...")
    price_paths = pd.read_csv('Results/Bitcoin/bitcoin_monte_carlo_simple_paths_20250720.csv')
    print(f"Loaded {len(price_paths.columns)} price paths")
    print(f"Time steps: {len(price_paths)}")
    return price_paths

def get_formula_fair_value():
    """Get the formula's fair value for Bitcoin"""
    # Load growth model coefficients
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    # Calculate fair value for today (day 6041)
    today_day = 6041
    fair_value = 10**(a * np.log(today_day) + b)
    return fair_value

def analyze_path_behavior(price_paths):
    """Analyze the behavior of Monte Carlo paths"""
    print("\n" + "="*80)
    print("MONTE CARLO PATH ANALYSIS")
    print("="*80)
    
    fair_value = get_formula_fair_value()
    current_price = price_paths.iloc[0, 0]
    
    print(f"Formula Fair Value: ${fair_value:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Overvaluation: {((current_price - fair_value) / fair_value * 100):.1f}%")
    print()
    
    # Analyze all paths
    total_paths = len(price_paths.columns)
    paths_below_fair_value = 0
    paths_ever_below_fair = 0
    min_prices = []
    max_prices = []
    final_prices = []
    
    for path_num in range(total_paths):
        price_path = price_paths.iloc[:, path_num]
        
        # Check if path ever goes below fair value
        if (price_path < fair_value).any():
            paths_ever_below_fair += 1
        
        # Check if path ends below fair value
        if price_path.iloc[-1] < fair_value:
            paths_below_fair_value += 1
        
        min_prices.append(price_path.min())
        max_prices.append(price_path.max())
        final_prices.append(price_path.iloc[-1])
    
    min_prices = np.array(min_prices)
    max_prices = np.array(max_prices)
    final_prices = np.array(final_prices)
    
    print("PATH ANALYSIS SUMMARY:")
    print(f"Total paths: {total_paths:,}")
    print(f"Paths that ever go below fair value: {paths_ever_below_fair:,} ({paths_ever_below_fair/total_paths*100:.1f}%)")
    print(f"Paths ending below fair value: {paths_below_fair_value:,} ({paths_below_fair_value/total_paths*100:.1f}%)")
    print()
    
    print("PRICE STATISTICS:")
    print(f"Minimum price across all paths: ${min_prices.min():,.2f}")
    print(f"Maximum price across all paths: ${max_prices.max():,.2f}")
    print(f"Average minimum price: ${min_prices.mean():,.2f}")
    print(f"Average maximum price: ${max_prices.mean():,.2f}")
    print(f"Average final price: ${final_prices.mean():,.2f}")
    print()
    
    print("FAIR VALUE ANALYSIS:")
    print(f"Paths with minimum below fair value: {(min_prices < fair_value).sum():,} ({(min_prices < fair_value).sum()/total_paths*100:.1f}%)")
    print(f"Paths with maximum below fair value: {(max_prices < fair_value).sum():,} ({(max_prices < fair_value).sum()/total_paths*100:.1f}%)")
    print()
    
    # Analyze specific drop levels
    drop_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    for drop_level in drop_levels:
        drop_price = fair_value * (1 - drop_level)
        paths_with_drop = (min_prices <= drop_price).sum()
        print(f"Paths dropping {drop_level*100:.0f}% below fair value (${drop_price:,.0f}): {paths_with_drop:,} ({paths_with_drop/total_paths*100:.1f}%)")
    
    return {
        'fair_value': fair_value,
        'paths_ever_below_fair': paths_ever_below_fair,
        'paths_below_fair_value': paths_below_fair_value,
        'min_prices': min_prices,
        'max_prices': max_prices,
        'final_prices': final_prices
    }

def analyze_sample_paths(price_paths, analysis_results, num_samples=5):
    """Analyze a few sample paths in detail"""
    print("\n" + "="*80)
    print("SAMPLE PATH ANALYSIS")
    print("="*80)
    
    fair_value = analysis_results['fair_value']
    
    # Find paths that go below fair value
    min_prices = analysis_results['min_prices']
    paths_below_fair = np.where(min_prices < fair_value)[0]
    
    if len(paths_below_fair) > 0:
        print(f"Analyzing {min(num_samples, len(paths_below_fair))} paths that go below fair value:")
        for i, path_idx in enumerate(paths_below_fair[:num_samples]):
            price_path = price_paths.iloc[:, path_idx]
            min_price = price_path.min()
            min_idx = price_path.idxmin()
            final_price = price_path.iloc[-1]
            
            print(f"\nPath {path_idx}:")
            print(f"  Minimum price: ${min_price:,.2f} (at step {min_idx})")
            print(f"  Drop below fair value: {((fair_value - min_price) / fair_value * 100):.1f}%")
            print(f"  Final price: ${final_price:,.2f}")
            print(f"  Total return: {((final_price / price_path.iloc[0]) - 1) * 100:.1f}%")
    else:
        print("No paths go below fair value!")
    
    # Find paths that never go below fair value
    paths_above_fair = np.where(min_prices >= fair_value)[0]
    if len(paths_above_fair) > 0:
        print(f"\nAnalyzing {min(num_samples, len(paths_above_fair))} paths that never go below fair value:")
        for i, path_idx in enumerate(paths_above_fair[:num_samples]):
            price_path = price_paths.iloc[:, path_idx]
            min_price = price_path.min()
            final_price = price_path.iloc[-1]
            
            print(f"\nPath {path_idx}:")
            print(f"  Minimum price: ${min_price:,.2f}")
            print(f"  Closest to fair value: {((min_price - fair_value) / fair_value * 100):.1f}% above")
            print(f"  Final price: ${final_price:,.2f}")
            print(f"  Total return: {((final_price / price_path.iloc[0]) - 1) * 100:.1f}%")

def create_debug_visualization(price_paths, analysis_results):
    """Create debug visualization"""
    print("\nCreating debug visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Monte Carlo Path Debug Analysis', fontsize=16, fontweight='bold')
    
    fair_value = analysis_results['fair_value']
    min_prices = analysis_results['min_prices']
    max_prices = analysis_results['max_prices']
    final_prices = analysis_results['final_prices']
    
    # 1. Price distribution over time
    sample_paths = price_paths.iloc[:, :10]  # First 10 paths
    for i in range(len(sample_paths.columns)):
        ax1.plot(sample_paths.iloc[:, i], alpha=0.7, linewidth=1)
    ax1.axhline(y=fair_value, color='red', linestyle='--', label=f'Fair Value: ${fair_value:,.0f}')
    ax1.set_title('Sample Price Paths (First 10)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Time Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Minimum price distribution
    ax2.hist(min_prices, bins=50, alpha=0.7, color='blue')
    ax2.axvline(x=fair_value, color='red', linestyle='--', label=f'Fair Value: ${fair_value:,.0f}')
    ax2.set_title('Distribution of Minimum Prices')
    ax2.set_xlabel('Minimum Price ($)')
    ax2.set_ylabel('Number of Paths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final price distribution
    ax3.hist(final_prices, bins=50, alpha=0.7, color='green')
    ax3.axvline(x=fair_value, color='red', linestyle='--', label=f'Fair Value: ${fair_value:,.0f}')
    ax3.set_title('Distribution of Final Prices')
    ax3.set_xlabel('Final Price ($)')
    ax3.set_ylabel('Number of Paths')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Price range vs final price
    price_ranges = max_prices - min_prices
    ax4.scatter(price_ranges, final_prices, alpha=0.6, s=20)
    ax4.axhline(y=fair_value, color='red', linestyle='--', label=f'Fair Value: ${fair_value:,.0f}')
    ax4.set_title('Price Range vs Final Price')
    ax4.set_xlabel('Price Range (Max - Min) ($)')
    ax4.set_ylabel('Final Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Results/Bitcoin/monte_carlo_debug_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Debug visualization saved to: {filename}")
    
    plt.show()

def main():
    """Main function"""
    print("MONTE CARLO PATH DEBUG ANALYSIS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    price_paths = load_monte_carlo_data()
    
    # Analyze paths
    analysis_results = analyze_path_behavior(price_paths)
    
    # Analyze sample paths
    analyze_sample_paths(price_paths, analysis_results)
    
    # Create visualization
    create_debug_visualization(price_paths, analysis_results)
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 