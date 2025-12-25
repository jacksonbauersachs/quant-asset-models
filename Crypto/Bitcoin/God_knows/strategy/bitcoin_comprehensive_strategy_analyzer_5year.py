import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

def load_gbm_paths():
    """Load the specified GBM simulation paths file and truncate to 5 years"""
    gbm_file = 'Results/Bitcoin/bitcoin_gbm_paths_20250723_165635.csv'
    print(f"Loading GBM paths from: {gbm_file}")
    df = pd.read_csv(gbm_file, index_col=0)
    
    # Truncate to first 5 years (1825 days)
    days_5_years = 1825
    if len(df) > days_5_years:
        df = df.iloc[:days_5_years]
        print(f"Truncated data to first {days_5_years} days (5 years)")
    
    return df

def calculate_lump_sum_strategy(prices, budget=1000):
    """Calculate lump sum strategy results"""
    initial_price = prices.iloc[0, 0]  # First price
    initial_coins = budget / initial_price
    final_value = initial_coins * prices.iloc[-1, :]
    
    return {
        'strategy': 'Lump Sum',
        'initial_coins': initial_coins,
        'final_value_mean': np.mean(final_value),
        'final_value_median': np.median(final_value),
        'final_value_std': np.std(final_value),
        'final_value_p5': np.percentile(final_value, 5),
        'final_value_p95': np.percentile(final_value, 95),
        'total_return_mean': (np.mean(final_value) / budget - 1) * 100,
        'total_return_median': (np.median(final_value) / budget - 1) * 100
    }

def calculate_dca_strategy(prices, monthly_amount=100, frequency='monthly'):
    """Calculate DCA strategy results"""
    if frequency == 'monthly':
        # Buy every 30 days
        buy_days = list(range(0, len(prices), 30))
    elif frequency == 'weekly':
        # Buy every 7 days
        buy_days = list(range(0, len(prices), 7))
    else:  # daily
        buy_days = list(range(len(prices)))
    
    total_invested = monthly_amount * len(buy_days)
    total_coins = np.zeros(len(prices.columns))
    
    for day in buy_days:
        if day < len(prices):
            day_prices = prices.iloc[day, :]
            coins_bought = monthly_amount / day_prices
            total_coins += coins_bought
    
    final_value = total_coins * prices.iloc[-1, :]
    
    return {
        'strategy': f'DCA ({frequency}, ${monthly_amount}/month)',
        'total_invested': total_invested,
        'total_coins_mean': np.mean(total_coins),
        'final_value_mean': np.mean(final_value),
        'final_value_median': np.median(final_value),
        'final_value_std': np.std(final_value),
        'final_value_p5': np.percentile(final_value, 5),
        'final_value_p95': np.percentile(final_value, 95),
        'total_return_mean': (np.mean(final_value) / total_invested - 1) * 100,
        'total_return_median': (np.median(final_value) / total_invested - 1) * 100
    }

def calculate_reserve_strategy(
    prices,
    monthly_amount=100,
    reserve_ratio=0.05,
    buy_thresholds=[0.9, 0.8, 0.75],
    buy_allocations=[0.2, 0.3, 0.5]
):
    # 3-tiered buy logic, no selling
    buy_allocations = np.array(buy_allocations) / np.sum(buy_allocations)
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    start_day = 6041
    buy_days = list(range(0, len(prices), 30))
    total_invested = monthly_amount * len(buy_days)
    final_values = []
    total_coins_list = []
    final_cash_list = []
    total_btc_bought_list = []
    total_dca_btc_list = []
    total_dca_invested_list = []
    total_reserve_invested_list = []

    for path_idx in range(len(prices.columns)):
        path_prices = prices.iloc[:, path_idx]
        current_coins = 0
        reserve_cash = 0
        dca_btc = 0
        dca_invested = 0
        reserve_invested = 0
        spent_this_path = [False, False, False]

        # Process each day sequentially, accumulating reserve and checking for opportunities
        monthly_interest_rate = 0.05 / 12  # 5% annual interest compounded monthly
        
        for day_idx, price in enumerate(path_prices):
            current_day = start_day + day_idx
            fair_value = 10**(a * np.log(current_day) + b)
            
            # Add monthly contribution and apply interest if this is a monthly interval
            if day_idx % 30 == 0:
                dca_amount = monthly_amount * (1 - reserve_ratio)
                reserve_amount = monthly_amount * reserve_ratio
                
                # Apply monthly interest to existing reserve cash
                if reserve_cash > 0:
                    interest_earned = reserve_cash * monthly_interest_rate
                    reserve_cash += interest_earned
                
                coins_bought = dca_amount / price
                current_coins += coins_bought
                dca_btc += coins_bought
                dca_invested += dca_amount
                reserve_cash += reserve_amount
            
            # Check thresholds from deepest to shallowest
            for i in reversed(range(3)):
                if not spent_this_path[i] and price < fair_value * buy_thresholds[i] and reserve_cash > 0:
                    buy_amt = reserve_cash * buy_allocations[i]
                    if buy_amt > 0:
                        coins_bought = buy_amt / price
                        current_coins += coins_bought
                        reserve_invested += buy_amt
                        reserve_cash -= buy_amt
                        spent_this_path[i] = True
        # At the end, invest any remaining reserve at the final price
        if reserve_cash > 0:
            coins_bought = reserve_cash / path_prices.iloc[-1]
            current_coins += coins_bought
            reserve_invested += reserve_cash
            reserve_cash = 0
        final_value = current_coins * path_prices.iloc[-1]
        final_values.append(final_value)
        total_coins_list.append(current_coins)
        final_cash_list.append(0)
        total_btc_bought_list.append(current_coins)
        total_dca_btc_list.append(dca_btc)
        total_dca_invested_list.append(dca_invested)
        total_reserve_invested_list.append(reserve_invested)

    print(f"[DEBUG] 3-Tier Reserve: rsv={reserve_ratio}, buy_thresholds={buy_thresholds}, buy_allocs={buy_allocations}")
    print(f"  Avg DCA invested: {np.mean(total_dca_invested_list):.2f}")
    print(f"  Avg reserve invested: {np.mean(total_reserve_invested_list):.2f}")
    print(f"  Avg total BTC bought: {np.mean(total_btc_bought_list):.6f}")
    print(f"  Sample (first path): BTC bought={total_btc_bought_list[0]:.6f}, reserve invested={total_reserve_invested_list[0]:.2f}")
    return {
        'strategy': f'3-Tier Reserve (rsv={int(reserve_ratio*100)}%, buy={[(int(t*100),int(a*100)) for t,a in zip(buy_thresholds,buy_allocations)]})',
        'total_invested': total_invested,
        'total_coins_mean': np.mean(total_coins_list),
        'final_value_mean': np.mean(final_values),
        'final_value_median': np.median(final_values),
        'final_value_std': np.std(final_values),
        'final_value_p5': np.percentile(final_values, 5),
        'final_value_p95': np.percentile(final_values, 95),
        'total_return_mean': (np.mean(final_values) / total_invested - 1) * 100,
        'total_return_median': (np.median(final_values) / total_invested - 1) * 100
    }

def analyze_all_strategies(prices, monthly_amount=100):
    print("Analyzing Investment Strategies with GBM Simulation (5-Year Analysis)")
    print("="*60)
    print(f"Monthly investment: ${monthly_amount:,}")
    print(f"Total investment over 5 years: ${monthly_amount * 60:,}")
    print(f"Number of paths: {len(prices.columns):,}")
    print(f"Time horizon: {len(prices)} days ({len(prices)/365:.1f} years)")
    print(f"Starting price: ${prices.iloc[0, 0]:,.2f}")

    results = []
    print("\n1. Analyzing DCA strategy...")
    dca_monthly = calculate_dca_strategy(prices, monthly_amount, 'monthly')
    results.append(dca_monthly)
    print("2. Analyzing sorted 3-tiered Reserve strategies (no selling)...")
    reserve_ratios = np.array([.1, .05, .01, .2, .3])
    buy_threshold_sets = [

        [.95, .9, .85],
        [.9, .85, .8],
        [.85, .8, .75],
        [.8, .75, .7],
        [.97, .95, .93],

    ]
    buy_allocation_sets = [
        [.2, .3, .5],
        [.1, .2, .7],
        [.5, .3, .2],
        [.7, .2, .1],
        [.9, .05, .05]
    ]
    # For each reserve ratio, select 5 evenly spaced threshold sets and 5 evenly spaced allocation sets
    combos = []
    n_thresholds = 5
    n_allocs = 5
    threshold_idxs = np.linspace(0, len(buy_threshold_sets)-1, n_thresholds, dtype=int)
    alloc_idxs = np.linspace(0, len(buy_allocation_sets)-1, n_allocs, dtype=int)
    for rsv in reserve_ratios:
        for t_idx in threshold_idxs:
            for a_idx in alloc_idxs:
                combos.append((rsv, buy_threshold_sets[t_idx], buy_allocation_sets[a_idx]))
    print(f"Testing {len(combos)} reserve strategy combinations...")
    for i, (rsv, thresholds, allocations) in enumerate(combos):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(combos)}")
        reserve_result = calculate_reserve_strategy(prices, monthly_amount, rsv, thresholds, allocations)
        results.append(reserve_result)
    print("3. Analyzing lump sum strategy...")
    lump_sum = calculate_lump_sum_strategy(prices, monthly_amount * 60)  # 5 years of monthly contributions
    results.append(lump_sum)
    return results

def create_strategy_comparison(results):
    """Create a comparison table of all strategies"""
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Strategy': result['strategy'],
            'Total Invested': f"${result['total_invested']:,.0f}",
            'Final Value (Mean)': f"${result['final_value_mean']:,.0f}",
            'Final Value (Median)': f"${result['final_value_median']:,.0f}",
            'Return (Mean)': f"{result['total_return_mean']:.1f}%",
            'Return (Median)': f"{result['total_return_median']:.1f}%",
            '5th Percentile': f"${result['final_value_p5']:,.0f}",
            '95th Percentile': f"${result['final_value_p95']:,.0f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df

def create_visualization(results, monthly_amount=100):
    """Create comprehensive visualization of strategy results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    strategies = [r['strategy'] for r in results]
    mean_returns = [r['total_return_mean'] for r in results]
    median_returns = [r['total_return_median'] for r in results]
    mean_values = [r['final_value_mean'] for r in results]
    total_invested = [r['total_invested'] for r in results]
    
    # 1. Mean vs Median Returns
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    ax1.bar(x_pos - width/2, mean_returns, width, label='Mean Return', alpha=0.8)
    ax1.bar(x_pos + width/2, median_returns, width, label='Median Return', alpha=0.8)
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Mean vs Median Returns by Strategy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s[:30] + '...' if len(s) > 30 else s for s in strategies], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Portfolio Values
    ax2.bar(range(len(strategies)), mean_values, alpha=0.8)
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Final Portfolio Value ($)')
    ax2.set_title('Mean Final Portfolio Values')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s[:30] + '...' if len(s) > 30 else s for s in strategies], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Return vs Risk (using 5th-95th percentile range as risk measure)
    risk_measures = [(r['final_value_p95'] - r['final_value_p5']) / r['final_value_mean'] * 100 for r in results]
    ax3.scatter(risk_measures, mean_returns, alpha=0.7, s=100)
    ax3.set_xlabel('Risk Measure (95th-5th percentile range as % of mean)')
    ax3.set_ylabel('Mean Return (%)')
    ax3.set_title('Risk vs Return')
    ax3.grid(True, alpha=0.3)
    
    # Add strategy labels to scatter plot
    for i, strategy in enumerate(strategies):
        ax3.annotate(strategy[:20] + '...' if len(strategy) > 20 else strategy, 
                    (risk_measures[i], mean_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Investment Efficiency (Return per dollar invested)
    efficiency = [r['total_return_mean'] / r['total_invested'] * 1000 for r in results]  # Return per $1000 invested
    ax4.bar(range(len(strategies)), efficiency, alpha=0.8)
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Return per $1000 Invested (%)')
    ax4.set_title('Investment Efficiency')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([s[:30] + '...' if len(s) > 30 else s for s in strategies], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Results/Bitcoin/comprehensive_strategy_analysis_5year_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")
    
    return fig

def save_results(results, monthly_amount=100):
    """Save detailed results to a text file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Results/Bitcoin/comprehensive_strategy_findings_5year_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("COMPREHENSIVE STRATEGY ANALYSIS RESULTS (5-YEAR ANALYSIS)\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Monthly investment: ${monthly_amount:,}\n")
        f.write(f"Total investment over 5 years: ${monthly_amount * 60:,}\n")
        f.write("="*80 + "\n\n")
        
        # Sort results by mean return
        sorted_results = sorted(results, key=lambda x: x['total_return_mean'], reverse=True)
        
        f.write("STRATEGIES RANKED BY MEAN RETURN:\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i:2d}. {result['strategy']}\n")
            f.write(f"    Mean Return: {result['total_return_mean']:.2f}%\n")
            f.write(f"    Median Return: {result['total_return_median']:.2f}%\n")
            f.write(f"    Final Value (Mean): ${result['final_value_mean']:,.2f}\n")
            f.write(f"    Final Value (Median): ${result['final_value_median']:,.2f}\n")
            f.write(f"    5th Percentile: ${result['final_value_p5']:,.2f}\n")
            f.write(f"    95th Percentile: ${result['final_value_p95']:,.2f}\n")
            f.write(f"    Total Invested: ${result['total_invested']:,.2f}\n")
            f.write("\n")
        
        # Find best strategies by category
        dca_strategies = [r for r in results if 'DCA' in r['strategy']]
        reserve_strategies = [r for r in results if 'Reserve' in r['strategy']]
        lump_sum_strategies = [r for r in results if 'Lump Sum' in r['strategy']]
        
        f.write("BEST STRATEGIES BY CATEGORY:\n")
        f.write("-"*80 + "\n")
        
        if dca_strategies:
            best_dca = max(dca_strategies, key=lambda x: x['total_return_mean'])
            f.write(f"Best DCA Strategy: {best_dca['strategy']}\n")
            f.write(f"  Mean Return: {best_dca['total_return_mean']:.2f}%\n\n")
        
        if reserve_strategies:
            best_reserve = max(reserve_strategies, key=lambda x: x['total_return_mean'])
            f.write(f"Best Reserve Strategy: {best_reserve['strategy']}\n")
            f.write(f"  Mean Return: {best_reserve['total_return_mean']:.2f}%\n\n")
        
        if lump_sum_strategies:
            best_lump = max(lump_sum_strategies, key=lambda x: x['total_return_mean'])
            f.write(f"Best Lump Sum Strategy: {best_lump['strategy']}\n")
            f.write(f"  Mean Return: {best_lump['total_return_mean']:.2f}%\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*80 + "\n")
        returns = [r['total_return_mean'] for r in results]
        f.write(f"Average Return (All Strategies): {np.mean(returns):.2f}%\n")
        f.write(f"Best Return: {max(returns):.2f}%\n")
        f.write(f"Worst Return: {min(returns):.2f}%\n")
        f.write(f"Return Standard Deviation: {np.std(returns):.2f}%\n")
    
    print(f"Detailed results saved to: {filename}")

def main():
    """Main function to run the comprehensive strategy analysis"""
    print("Loading GBM simulation data...")
    prices = load_gbm_paths()
    
    if prices is None:
        print("Failed to load GBM paths!")
        return
    
    print(f"Successfully loaded {len(prices.columns)} price paths")
    print(f"Data spans {len(prices)} days ({len(prices)/365:.1f} years)")
    
    # Analyze all strategies
    results = analyze_all_strategies(prices, monthly_amount=100)
    
    # Create comparison table
    comparison_df = create_strategy_comparison(results)
    print("\n" + "="*80)
    print("STRATEGY COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(results, monthly_amount=100)
    
    # Save detailed results
    print("\nSaving detailed results...")
    save_results(results, monthly_amount=100)
    
    print("\nAnalysis complete!")
    print("="*80)

if __name__ == "__main__":
    main() 