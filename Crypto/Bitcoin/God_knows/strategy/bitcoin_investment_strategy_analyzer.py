#!/usr/bin/env python3
"""
Bitcoin Investment Strategy Analyzer
Purpose: Test different investment strategies with Monte Carlo simulations
Strategies tested:
1. Lump sum investment (all $1000 now)
2. Wait for price drops with multi-tiered investment
3. Dollar cost averaging (DCA)
4. Hybrid strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
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
        print("Monte Carlo paths file not found. Running simulation...")
        # Import and run the Monte Carlo simulation
        from bitcoin_monte_carlo_simple import simple_monte_carlo_simulation, get_updated_models
        models = get_updated_models()
        price_paths, t = simple_monte_carlo_simulation(118000, 10, models['growth'], models['volatility'])
        
        # Create DataFrame
        paths_df = pd.DataFrame(price_paths.T, index=t, columns=[f'Path_{i+1}' for i in range(len(price_paths))])
        paths_df.index.name = 'Years'
        return paths_df

def calculate_strategy_returns(price_paths, strategy_name, strategy_func, **kwargs):
    """Calculate returns for a given investment strategy"""
    print(f"Calculating returns for {strategy_name}...")
    
    returns = []
    for path_num in range(len(price_paths.columns)):
        price_path = price_paths.iloc[:, path_num]
        path_return = strategy_func(price_path, **kwargs)
        returns.append(path_return)
    
    return np.array(returns)

def lump_sum_strategy(price_path, initial_investment=1000):
    """Lump sum strategy: invest all money now"""
    initial_price = price_path.iloc[0]  # Current price
    final_price = price_path.iloc[-1]   # Price at end of simulation
    
    # Buy Bitcoin at current price
    bitcoin_owned = initial_investment / initial_price
    
    # Value at end of simulation
    final_value = bitcoin_owned * final_price
    
    # Calculate return
    total_return = (final_value - initial_investment) / initial_investment
    return total_return

def multi_tier_drop_strategy(price_path, initial_investment=1000, 
                           drop_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
                           investment_ratios=[0.2, 0.2, 0.2, 0.2, 0.2]):
    """
    Multi-tier drop strategy: invest portions when price drops below formula's fair value
    
    Args:
        price_path: Price path over time
        initial_investment: Total budget ($1000)
        drop_levels: List of drops below fair value to trigger investment (e.g., [0.1, 0.2, 0.3])
        investment_ratios: Portion of budget to invest at each drop level
    """
    # Get formula's fair value
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    today_day = 6041
    fair_value = 10**(a * np.log(today_day) + b)
    
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
    return total_return

def dollar_cost_averaging_strategy(price_path, initial_investment=1000, 
                                 num_investments=12, investment_interval=30):
    """
    Dollar cost averaging strategy: invest equal amounts at regular intervals
    
    Args:
        price_path: Price path over time
        initial_investment: Total budget ($1000)
        num_investments: Number of investments to make
        investment_interval: Days between investments
    """
    investment_amount = initial_investment / num_investments
    bitcoin_owned = 0
    
    # Make investments at regular intervals
    for i in range(num_investments):
        day_index = i * investment_interval
        if day_index < len(price_path):
            price = price_path.iloc[day_index]
            bitcoin_bought = investment_amount / price
            bitcoin_owned += bitcoin_bought
    
    # Final value
    final_price = price_path.iloc[-1]
    final_value = bitcoin_owned * final_price
    
    # Calculate return
    total_return = (final_value - initial_investment) / initial_investment
    return total_return

def hybrid_strategy(price_path, initial_investment=1000,
                   immediate_allocation=0.3,  # 30% invested immediately
                   drop_allocation=0.5,       # 50% for drop strategy
                   dca_allocation=0.2):       # 20% for DCA
    """
    Hybrid strategy: combine immediate investment, drop strategy, and DCA
    """
    immediate_investment = initial_investment * immediate_allocation
    drop_investment = initial_investment * drop_allocation
    dca_investment = initial_investment * dca_allocation
    
    # Immediate investment
    initial_price = price_path.iloc[0]
    bitcoin_immediate = immediate_investment / initial_price
    
    # Drop strategy (simplified - invest at 20% drop)
    bitcoin_drop = 0
    for i, price in enumerate(price_path):
        current_drop = (initial_price - price) / initial_price
        if current_drop >= 0.2:
            bitcoin_drop = drop_investment / price
            break
    
    # DCA strategy (4 equal investments)
    bitcoin_dca = 0
    dca_amount = dca_investment / 4
    for i in range(4):
        day_index = i * 90  # Every 90 days
        if day_index < len(price_path):
            price = price_path.iloc[day_index]
            bitcoin_dca += dca_amount / price
    
    # Total Bitcoin owned
    total_bitcoin = bitcoin_immediate + bitcoin_drop + bitcoin_dca
    
    # Final value
    final_price = price_path.iloc[-1]
    final_value = total_bitcoin * final_price
    
    # Calculate return
    total_return = (final_value - initial_investment) / initial_investment
    return total_return

def analyze_strategies(price_paths):
    """Analyze all investment strategies"""
    print("="*60)
    print("BITCOIN INVESTMENT STRATEGY ANALYSIS")
    print("="*60)
    
    strategies = []
    
    # 1. Lump sum strategy
    lump_sum_returns = calculate_strategy_returns(price_paths, "Lump Sum", lump_sum_strategy)
    strategies.append({
        'name': 'Lump Sum',
        'returns': lump_sum_returns,
        'description': 'Invest all $1000 immediately'
    })
    
    # 2. Multi-tier drop strategies
    drop_strategies = [
        {
            'name': 'Conservative Drop (10%, 20%, 30%)',
            'drop_levels': [0.1, 0.2, 0.3],
            'ratios': [0.3, 0.4, 0.3]
        },
        {
            'name': 'Aggressive Drop (20%, 40%, 60%)',
            'drop_levels': [0.2, 0.4, 0.6],
            'ratios': [0.4, 0.4, 0.2]
        },
        {
            'name': 'Moderate Drop (15%, 25%, 35%, 45%)',
            'drop_levels': [0.15, 0.25, 0.35, 0.45],
            'ratios': [0.25, 0.25, 0.25, 0.25]
        }
    ]
    
    for strategy in drop_strategies:
        returns = calculate_strategy_returns(
            price_paths, 
            strategy['name'], 
            multi_tier_drop_strategy,
            drop_levels=strategy['drop_levels'],
            investment_ratios=strategy['ratios']
        )
        strategies.append({
            'name': strategy['name'],
            'returns': returns,
            'description': f"Invest at {', '.join([f'{d*100}%' for d in strategy['drop_levels']])} drops"
        })
    
    # 3. DCA strategies
    dca_strategies = [
        {
            'name': 'Monthly DCA (12 investments)',
            'num_investments': 12,
            'interval': 30
        },
        {
            'name': 'Quarterly DCA (4 investments)',
            'num_investments': 4,
            'interval': 90
        },
        {
            'name': 'Weekly DCA (52 investments)',
            'num_investments': 52,
            'interval': 7
        }
    ]
    
    for strategy in dca_strategies:
        returns = calculate_strategy_returns(
            price_paths,
            strategy['name'],
            dollar_cost_averaging_strategy,
            num_investments=strategy['num_investments'],
            investment_interval=strategy['interval']
        )
        strategies.append({
            'name': strategy['name'],
            'returns': returns,
            'description': f"{strategy['num_investments']} equal investments"
        })
    
    # 4. Hybrid strategies
    hybrid_strategies = [
        {
            'name': 'Balanced Hybrid (30/50/20)',
            'immediate': 0.3,
            'drop': 0.5,
            'dca': 0.2
        },
        {
            'name': 'Conservative Hybrid (20/60/20)',
            'immediate': 0.2,
            'drop': 0.6,
            'dca': 0.2
        },
        {
            'name': 'Aggressive Hybrid (50/30/20)',
            'immediate': 0.5,
            'drop': 0.3,
            'dca': 0.2
        }
    ]
    
    for strategy in hybrid_strategies:
        returns = calculate_strategy_returns(
            price_paths,
            strategy['name'],
            hybrid_strategy,
            immediate_allocation=strategy['immediate'],
            drop_allocation=strategy['drop'],
            dca_allocation=strategy['dca']
        )
        strategies.append({
            'name': strategy['name'],
            'returns': returns,
            'description': f"Immediate: {strategy['immediate']*100}%, Drop: {strategy['drop']*100}%, DCA: {strategy['dca']*100}%"
        })
    
    return strategies

def create_strategy_comparison_visualization(strategies):
    """Create visualization comparing all strategies"""
    print("Creating strategy comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Investment Strategy Comparison ($1000 Budget)', fontsize=16, fontweight='bold')
    
    # 1. Box plot of returns
    strategy_names = [s['name'] for s in strategies]
    returns_data = [s['returns'] * 100 for s in strategies]  # Convert to percentage
    
    bp = ax1.boxplot(returns_data, labels=strategy_names, patch_artist=True)
    ax1.set_title('Return Distribution by Strategy')
    ax1.set_ylabel('Total Return (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 2. Mean returns bar chart
    mean_returns = [np.mean(s['returns']) * 100 for s in strategies]
    bars = ax2.bar(strategy_names, mean_returns, color=colors)
    ax2.set_title('Average Returns by Strategy')
    ax2.set_ylabel('Average Return (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Risk vs Return scatter plot
    mean_returns = [np.mean(s['returns']) * 100 for s in strategies]
    std_returns = [np.std(s['returns']) * 100 for s in strategies]
    
    scatter = ax3.scatter(std_returns, mean_returns, c=range(len(strategies)), 
                         cmap='viridis', s=100, alpha=0.7)
    ax3.set_xlabel('Risk (Standard Deviation %)')
    ax3.set_ylabel('Average Return (%)')
    ax3.set_title('Risk vs Return Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, (x, y) in enumerate(zip(std_returns, mean_returns)):
        ax3.annotate(strategies[i]['name'][:15], (x, y), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Probability of positive returns
    positive_prob = [np.mean(s['returns'] > 0) * 100 for s in strategies]
    bars = ax4.bar(strategy_names, positive_prob, color=colors)
    ax4.set_title('Probability of Positive Returns')
    ax4.set_ylabel('Probability (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, positive_prob):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f'Results/Bitcoin/bitcoin_investment_strategy_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Strategy comparison saved to: {filename}")
    
    return fig

def generate_strategy_report(strategies):
    """Generate a detailed strategy report"""
    print("="*60)
    print("INVESTMENT STRATEGY ANALYSIS REPORT")
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
        
        # Calculate percentiles
        p5 = np.percentile(returns, 5) * 100
        p25 = np.percentile(returns, 25) * 100
        p75 = np.percentile(returns, 75) * 100
        p95 = np.percentile(returns, 95) * 100
        
        report_data.append({
            'Strategy': strategy['name'],
            'Description': strategy['description'],
            'Mean Return (%)': f'{mean_return:.1f}',
            'Median Return (%)': f'{median_return:.1f}',
            'Std Dev (%)': f'{std_return:.1f}',
            'Positive Probability (%)': f'{positive_prob:.1f}',
            'Max Return (%)': f'{max_return:.1f}',
            'Min Return (%)': f'{min_return:.1f}',
            '5th Percentile (%)': f'{p5:.1f}',
            '25th Percentile (%)': f'{p25:.1f}',
            '75th Percentile (%)': f'{p75:.1f}',
            '95th Percentile (%)': f'{p95:.1f}'
        })
    
    # Create DataFrame and save to CSV
    report_df = pd.DataFrame(report_data)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f'Results/Bitcoin/bitcoin_investment_strategy_report_{timestamp}.csv'
    report_df.to_csv(filename, index=False)
    print(f"Strategy report saved to: {filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("STRATEGY SUMMARY")
    print("="*80)
    
    # Sort by mean return
    sorted_strategies = sorted(strategies, key=lambda x: np.mean(x['returns']), reverse=True)
    
    print(f"{'Rank':<4} {'Strategy':<25} {'Mean Return':<12} {'Risk':<12} {'Positive Prob':<12}")
    print("-" * 80)
    
    for i, strategy in enumerate(sorted_strategies[:10], 1):
        returns = strategy['returns']
        mean_return = np.mean(returns) * 100
        risk = np.std(returns) * 100
        positive_prob = np.mean(returns > 0) * 100
        
        print(f"{i:<4} {strategy['name']:<25} {mean_return:>8.1f}% {risk:>10.1f}% {positive_prob:>10.1f}%")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_return = sorted_strategies[0]
    best_risk_adjusted = min(strategies, key=lambda x: np.std(x['returns']) / np.mean(x['returns']) if np.mean(x['returns']) > 0 else float('inf'))
    most_consistent = max(strategies, key=lambda x: np.mean(x['returns'] > 0))
    
    print(f"🏆 Best Average Return: {best_return['name']}")
    print(f"   Average Return: {np.mean(best_return['returns'])*100:.1f}%")
    print(f"   Risk: {np.std(best_return['returns'])*100:.1f}%")
    
    print(f"\n⚖️  Best Risk-Adjusted Return: {best_risk_adjusted['name']}")
    print(f"   Sharpe Ratio: {np.mean(best_risk_adjusted['returns'])/np.std(best_risk_adjusted['returns']):.2f}")
    
    print(f"\n🎯 Most Consistent (Highest Positive Probability): {most_consistent['name']}")
    print(f"   Positive Probability: {np.mean(most_consistent['returns'] > 0)*100:.1f}%")
    
    return report_df

def main():
    """Main function to run the investment strategy analysis"""
    print("BITCOIN INVESTMENT STRATEGY ANALYZER")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Budget: $1000")
    print("="*60)
    
    # Load Monte Carlo data
    price_paths = load_monte_carlo_data()
    
    # Analyze strategies
    strategies = analyze_strategies(price_paths)
    
    # Create visualization
    fig = create_strategy_comparison_visualization(strategies)
    
    # Generate report
    report_df = generate_strategy_report(strategies)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files generated:")
    print(f"- Strategy comparison visualization: Results/Bitcoin/bitcoin_investment_strategy_comparison_*.png")
    print(f"- Strategy report: Results/Bitcoin/bitcoin_investment_strategy_report_*.csv")
    
    return strategies, report_df

if __name__ == "__main__":
    main() 