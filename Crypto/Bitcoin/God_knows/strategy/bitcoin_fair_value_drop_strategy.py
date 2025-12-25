#!/usr/bin/env python3
"""
Bitcoin Fair Value Drop Strategy Analyzer
Purpose: Test drop strategies based on formula's fair value rather than current price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_monte_carlo_data():
    """Load Monte Carlo simulation data"""
    print("Loading Monte Carlo simulation data...")
    
    # Load price paths
    paths_file = 'Results/Bitcoin/bitcoin_monte_carlo_simple_paths_20250720.csv'
    if os.path.exists(paths_file):
        paths_df = pd.read_csv(paths_file, index_col=0)
        print(f"Loaded {len(paths_df.columns)} price paths")
        return paths_df
    else:
        print("Monte Carlo paths file not found!")
        return None

def get_formula_fair_value():
    """Get the formula's fair value for Bitcoin today"""
    # Load growth model coefficients (using the correct file)
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
    
    a = float(lines[0].split('=')[1].strip())
    b = float(lines[1].split('=')[1].strip())
    
    # Calculate today's fair value (day 6041)
    today_day = 6041
    fair_value = 10**(a * np.log(today_day) + b)
    
    return fair_value

def fair_value_drop_strategy(price_path, initial_investment=1000, 
                           drop_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
                           investment_ratios=[0.2, 0.2, 0.2, 0.2, 0.2],
                           fair_value=None):
    """
    Fair value drop strategy: invest portions when price drops below formula's fair value
    
    Args:
        price_path: Price path over time
        initial_investment: Total budget ($1000)
        drop_levels: List of drops below fair value to trigger investment (e.g., [0.1, 0.2, 0.3])
        investment_ratios: Portion of budget to invest at each drop level
        fair_value: Formula's fair value for Bitcoin
    """
    if fair_value is None:
        fair_value = get_formula_fair_value()
    
    bitcoin_owned = 0
    remaining_budget = initial_investment
    
    # Track investments made
    investments_made = []
    
    for i, price in enumerate(price_path):
        # Calculate how much the price has dropped below fair value
        if price < fair_value:
            current_drop = (fair_value - price) / fair_value
        else:
            current_drop = 0
        
        # Check if any drop level is triggered
        for drop_level, investment_ratio in zip(drop_levels, investment_ratios):
            if current_drop >= drop_level and investment_ratio > 0:
                # Calculate investment amount
                investment_amount = initial_investment * investment_ratio
                
                # Only invest if we have budget and haven't invested at this level yet
                if (remaining_budget >= investment_amount and 
                    drop_level not in [inv['drop_level'] for inv in investments_made]):
                    
                    # Buy Bitcoin
                    bitcoin_bought = investment_amount / price
                    bitcoin_owned += bitcoin_bought
                    remaining_budget -= investment_amount
                    
                    investments_made.append({
                        'drop_level': drop_level,
                        'price': price,
                        'fair_value': fair_value,
                        'drop_below_fair': current_drop,
                        'amount': investment_amount,
                        'bitcoin_bought': bitcoin_bought,
                        'year': i / 365.25  # Convert to years
                    })
                    
                    # Set this ratio to 0 to prevent double investment
                    investment_ratios[drop_levels.index(drop_level)] = 0
    
    # Final value
    final_price = price_path.iloc[-1]
    final_value = bitcoin_owned * final_price + remaining_budget
    
    # Calculate return
    total_return = (final_value - initial_investment) / initial_investment
    return total_return, investments_made

def calculate_strategy_returns(price_paths, strategy_name, strategy_func, **kwargs):
    """Calculate returns for a given investment strategy"""
    print(f"Calculating returns for {strategy_name}...")
    
    returns = []
    investment_details = []
    
    for path_num in range(len(price_paths.columns)):
        price_path = price_paths.iloc[:, path_num]
        path_return, investments = strategy_func(price_path, **kwargs)
        returns.append(path_return)
        investment_details.append(investments)
    
    return np.array(returns), investment_details

def analyze_fair_value_strategies(price_paths):
    """Analyze fair value drop strategies"""
    print("="*60)
    print("FAIR VALUE DROP STRATEGY ANALYSIS")
    print("="*60)
    
    fair_value = get_formula_fair_value()
    current_price = price_paths.iloc[0, 0]  # First price in first path
    
    print(f"Formula's Fair Value: ${fair_value:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Overvaluation: {((current_price - fair_value) / fair_value * 100):.1f}%")
    print()
    
    strategies = []
    
    # Define different fair value drop strategies
    fair_value_strategies = [
        {
            'name': 'Conservative Fair Value (10%, 20%, 30%)',
            'drop_levels': [0.1, 0.2, 0.3],
            'ratios': [0.3, 0.4, 0.3],
            'description': 'Invest when price drops 10%, 20%, 30% below fair value'
        },
        {
            'name': 'Moderate Fair Value (15%, 25%, 35%, 45%)',
            'drop_levels': [0.15, 0.25, 0.35, 0.45],
            'ratios': [0.25, 0.25, 0.25, 0.25],
            'description': 'Invest when price drops 15%, 25%, 35%, 45% below fair value'
        },
        {
            'name': 'Aggressive Fair Value (20%, 40%, 60%)',
            'drop_levels': [0.2, 0.4, 0.6],
            'ratios': [0.4, 0.4, 0.2],
            'description': 'Invest when price drops 20%, 40%, 60% below fair value'
        },
        {
            'name': 'Deep Value (30%, 50%, 70%)',
            'drop_levels': [0.3, 0.5, 0.7],
            'ratios': [0.4, 0.4, 0.2],
            'description': 'Invest when price drops 30%, 50%, 70% below fair value'
        }
    ]
    
    for strategy in fair_value_strategies:
        returns, investment_details = calculate_strategy_returns(
            price_paths,
            strategy['name'],
            fair_value_drop_strategy,
            drop_levels=strategy['drop_levels'],
            investment_ratios=strategy['ratios'],
            fair_value=fair_value
        )
        
        strategies.append({
            'name': strategy['name'],
            'returns': returns,
            'description': strategy['description'],
            'investment_details': investment_details
        })
    
    return strategies, fair_value

def create_fair_value_visualization(strategies, fair_value):
    """Create visualization for fair value strategies"""
    print("Creating fair value strategy visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Fair Value Drop Strategy Analysis ($1000 Budget)', fontsize=16, fontweight='bold')
    
    # 1. Box plot of returns
    strategy_names = [s['name'] for s in strategies]
    returns_data = [s['returns'] * 100 for s in strategies]  # Convert to percentage
    
    bp = ax1.boxplot(returns_data, labels=strategy_names, patch_artist=True)
    ax1.set_title('Return Distribution by Fair Value Strategy')
    ax1.set_ylabel('Total Return (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 2. Mean returns bar chart
    mean_returns = [np.mean(s['returns']) * 100 for s in strategies]
    bars = ax2.bar(strategy_names, mean_returns, color=colors)
    ax2.set_title('Average Returns by Fair Value Strategy')
    ax2.set_ylabel('Average Return (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Investment frequency analysis
    investment_frequencies = []
    for strategy in strategies:
        # Count how many paths had investments
        paths_with_investments = sum(1 for details in strategy['investment_details'] if len(details) > 0)
        frequency = paths_with_investments / len(strategy['investment_details']) * 100
        investment_frequencies.append(frequency)
    
    bars = ax3.bar(strategy_names, investment_frequencies, color=colors)
    ax3.set_title('Percentage of Paths with Investments')
    ax3.set_ylabel('Paths with Investments (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, investment_frequencies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Risk vs Return scatter plot
    mean_returns = [np.mean(s['returns']) * 100 for s in strategies]
    std_returns = [np.std(s['returns']) * 100 for s in strategies]
    
    scatter = ax4.scatter(std_returns, mean_returns, c=range(len(strategies)), 
                         cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Risk (Standard Deviation %)')
    ax4.set_ylabel('Average Return (%)')
    ax4.set_title('Risk vs Return Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, (x, y) in enumerate(zip(std_returns, mean_returns)):
        ax4.annotate(strategies[i]['name'][:15], (x, y), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f'Results/Bitcoin/bitcoin_fair_value_strategy_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Fair value strategy analysis saved to: {filename}")
    
    return fig

def generate_fair_value_report(strategies, fair_value):
    """Generate detailed report for fair value strategies"""
    print("="*60)
    print("FAIR VALUE STRATEGY ANALYSIS REPORT")
    print("="*60)
    
    report_data = []
    
    for strategy in strategies:
        returns = strategy['returns']
        
        # Calculate statistics
        mean_return = np.mean(returns) * 100
        median_return = np.median(returns) * 100
        std_return = np.std(returns) * 100
        positive_prob = np.mean(returns > 0) * 100
        max_return = np.max(returns) * 100
        min_return = np.min(returns) * 100
        
        # Calculate investment frequency
        paths_with_investments = sum(1 for details in strategy['investment_details'] if len(details) > 0)
        investment_frequency = paths_with_investments / len(strategy['investment_details']) * 100
        
        # Calculate average investment timing
        avg_investment_years = []
        for details in strategy['investment_details']:
            if details:
                avg_investment_years.extend([inv['year'] for inv in details])
        
        avg_timing = np.mean(avg_investment_years) if avg_investment_years else 0
        
        report_data.append({
            'Strategy': strategy['name'],
            'Description': strategy['description'],
            'Mean Return (%)': f'{mean_return:.1f}',
            'Median Return (%)': f'{median_return:.1f}',
            'Std Dev (%)': f'{std_return:.1f}',
            'Positive Probability (%)': f'{positive_prob:.1f}',
            'Investment Frequency (%)': f'{investment_frequency:.1f}',
            'Avg Investment Timing (Years)': f'{avg_timing:.1f}',
            'Max Return (%)': f'{max_return:.1f}',
            'Min Return (%)': f'{min_return:.1f}'
        })
    
    # Create DataFrame and save to CSV
    report_df = pd.DataFrame(report_data)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f'Results/Bitcoin/bitcoin_fair_value_strategy_report_{timestamp}.csv'
    report_df.to_csv(filename, index=False)
    print(f"Fair value strategy report saved to: {filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("FAIR VALUE STRATEGY SUMMARY")
    print("="*80)
    
    # Sort by mean return
    sorted_strategies = sorted(strategies, key=lambda x: np.mean(x['returns']), reverse=True)
    
    print(f"{'Rank':<4} {'Strategy':<35} {'Mean Return':<12} {'Risk':<12} {'Investment Freq':<15}")
    print("-" * 80)
    
    for i, strategy in enumerate(sorted_strategies, 1):
        returns = strategy['returns']
        mean_return = np.mean(returns) * 100
        risk = np.std(returns) * 100
        
        # Calculate investment frequency
        paths_with_investments = sum(1 for details in strategy['investment_details'] if len(details) > 0)
        frequency = paths_with_investments / len(strategy['investment_details']) * 100
        
        print(f"{i:<4} {strategy['name']:<35} {mean_return:>8.1f}% {risk:>10.1f}% {frequency:>13.1f}%")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_return = sorted_strategies[0]
    best_risk_adjusted = min(strategies, key=lambda x: np.std(x['returns']) / np.mean(x['returns']) if np.mean(x['returns']) > 0 else float('inf'))
    most_active = max(strategies, key=lambda x: sum(1 for details in x['investment_details'] if len(details) > 0))
    
    print(f"🏆 Best Average Return: {best_return['name']}")
    print(f"   Average Return: {np.mean(best_return['returns'])*100:.1f}%")
    print(f"   Investment Frequency: {sum(1 for details in best_return['investment_details'] if len(details) > 0) / len(best_return['investment_details']) * 100:.1f}%")
    
    print(f"\n⚖️  Best Risk-Adjusted Return: {best_risk_adjusted['name']}")
    print(f"   Sharpe Ratio: {np.mean(best_risk_adjusted['returns'])/np.std(best_risk_adjusted['returns']):.2f}")
    
    print(f"\n🎯 Most Active Strategy: {most_active['name']}")
    print(f"   Investment Frequency: {sum(1 for details in most_active['investment_details'] if len(details) > 0) / len(most_active['investment_details']) * 100:.1f}%")
    
    return report_df

def main():
    """Main function to run fair value drop strategy analysis"""
    print("BITCOIN FAIR VALUE DROP STRATEGY ANALYZER")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Budget: $1000")
    print("="*60)
    
    # Load Monte Carlo data
    price_paths = load_monte_carlo_data()
    if price_paths is None:
        print("Error: Could not load Monte Carlo data")
        return
    
    # Analyze fair value strategies
    strategies, fair_value = analyze_fair_value_strategies(price_paths)
    
    # Create visualization
    fig = create_fair_value_visualization(strategies, fair_value)
    
    # Generate report
    report_df = generate_fair_value_report(strategies, fair_value)
    
    print(f"\n" + "="*60)
    print("FAIR VALUE ANALYSIS COMPLETE")
    print("="*60)
    print("Files generated:")
    print(f"- Fair value strategy analysis: Results/Bitcoin/bitcoin_fair_value_strategy_analysis_*.png")
    print(f"- Fair value strategy report: Results/Bitcoin/bitcoin_fair_value_strategy_report_*.csv")
    
    return strategies, report_df

if __name__ == "__main__":
    main() 