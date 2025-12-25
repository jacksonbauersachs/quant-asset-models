import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

def load_gbm_paths():
    """Load the specified GBM simulation paths file"""
    gbm_file = r'Results\Bitcoin\bitcoin_gbm_paths_5year_20250723_165635.csv'
    print(f"Loading GBM paths from: {gbm_file}")
    df = pd.read_csv(gbm_file, index_col=0)
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
    reserve_ratio=0.4,
    buy_thresholds=[0.96, 0.94, 0.92],
    buy_allocations=[0.2, 0.3, 0.5],
    sell_thresholds=[1.05, 1.10, 1.15],
    sell_allocations=[0.2, 0.3, 0.5]
):
    # 3-tiered buy/sell logic
    buy_allocations = np.array(buy_allocations) / np.sum(buy_allocations)
    sell_allocations = np.array(sell_allocations) / np.sum(sell_allocations)
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    start_day = 6041
    total_invested = monthly_amount * (len(prices) // 30)
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
        sold_this_path = [False, False, False]

        for day_idx, price in enumerate(path_prices):
            current_day = start_day + day_idx
            fair_value = 10**(a * np.log(current_day) + b)
            price_ratio = price / fair_value
            # Add monthly contribution
            if day_idx % 30 == 0:
                dca_amount = monthly_amount * (1 - reserve_ratio)
                reserve_amount = monthly_amount * reserve_ratio
                # Invest DCA portion immediately
                if dca_amount > 0:
                    dca_btc_amt = dca_amount / price
                    current_coins += dca_btc_amt
                    dca_btc += dca_btc_amt
                    dca_invested += dca_amount
                # No interest: just add reserve
                reserve_cash += reserve_amount
            # BUY LOGIC
            for i in range(3):
                if not spent_this_path[i] and price < fair_value * buy_thresholds[i] and reserve_cash > 0:
                    buy_amt = reserve_cash * buy_allocations[i]
                    if buy_amt > 0:
                        coins_bought = buy_amt / price
                        current_coins += coins_bought
                        reserve_invested += buy_amt
                        reserve_cash -= buy_amt
                        spent_this_path[i] = True
            # SELL LOGIC
            for j in range(3):
                if not sold_this_path[j] and price > fair_value * sell_thresholds[j] and current_coins > 0:
                    sell_btc = current_coins * sell_allocations[j]
                    proceeds = sell_btc * price
                    current_coins -= sell_btc
                    reserve_cash += proceeds
                    sold_this_path[j] = True
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

    print(f"[DEBUG] 3-Tier Reserve: rsv={reserve_ratio}, buy_thresholds={buy_thresholds}, buy_allocs={buy_allocations}, sell_thresholds={sell_thresholds}, sell_allocs={sell_allocations}")
    print(f"  Avg DCA invested: {np.mean(total_dca_invested_list):.2f}")
    print(f"  Avg reserve invested: {np.mean(total_reserve_invested_list):.2f}")
    print(f"  Avg total BTC bought: {np.mean(total_btc_bought_list):.6f}")
    print(f"  Sample (first path): BTC bought={total_btc_bought_list[0]:.6f}, reserve invested={total_reserve_invested_list[0]:.2f}")
    return {
        'strategy': f'3-Tier Reserve (rsv={int(reserve_ratio*100)}%, buy={[(int(t*100),int(a*100)) for t,a in zip(buy_thresholds,buy_allocations)]}, sell={[(int(t*100),int(a*100)) for t,a in zip(sell_thresholds,sell_allocations)]})',
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
    print("Analyzing Investment Strategies with GBM Simulation")
    print("="*60)
    print(f"Monthly investment: ${monthly_amount:,}")
    print(f"Total investment over 10 years: ${monthly_amount * 120:,}")
    print(f"Number of paths: {len(prices.columns):,}")
    print(f"Time horizon: {len(prices)} days ({len(prices)/365:.1f} years)")
    print(f"Starting price: ${prices.iloc[0, 0]:,.2f}")

    results = []
    print("\n1. Analyzing DCA strategy...")
    dca_monthly = calculate_dca_strategy(prices, monthly_amount, 'monthly')
    results.append(dca_monthly)
    print("2. Analyzing sorted 3-tiered Reserve strategies (with selling)...")

    reserve_ratio = 0.4
    buy_thresholds = [0.96, 0.94, 0.92]
    buy_allocations = [0.2, 0.3, 0.5]
    sell_thresholds = [1.05, 1.10, 1.15]
    sell_allocations = [0.2, 0.3, 0.5]

    for reserve_ratio in [reserve_ratio]:
        for buy_thresholds in [buy_thresholds]:
            for buy_allocations in [buy_allocations]:
                for sell_thresholds in [sell_thresholds]:
                    for sell_allocations in [sell_allocations]:
                        reserve = calculate_reserve_strategy(
                            prices,
                            monthly_amount,
                            reserve_ratio=reserve_ratio,
                            buy_thresholds=buy_thresholds,
                            buy_allocations=buy_allocations,
                            sell_thresholds=sell_thresholds,
                            sell_allocations=sell_allocations
                        )
                        results.append(reserve)
    return results

def create_strategy_comparison(results):
    """Create comparison table and visualization"""
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Strategy': result['strategy'],
            'Final_Value_Mean': result['final_value_mean'],
            'Final_Value_Median': result['final_value_median'],
            'Total_Return_Mean': result['total_return_mean'],
            'Total_Return_Median': result['total_return_median'],
            'Risk_Std': result['final_value_std'],
            'P5_Value': result['final_value_p5'],
            'P95_Value': result['final_value_p95']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Sort by mean return
    df = df.sort_values('Total_Return_Mean', ascending=False)
    
    # Display results
    print(f"\n{'Strategy':<50} {'Mean Return':<12} {'Median Return':<14} {'Risk':<10}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        strategy_short = row['Strategy'][:45] + "..." if len(row['Strategy']) > 45 else row['Strategy']
        print(f"{strategy_short:<50} {row['Total_Return_Mean']:<12.1f}% {row['Total_Return_Median']:<14.1f}% {row['Risk_Std']:<10.0f}")
    
    return df

def create_visualization(results, monthly_amount=100):
    """Create visualization of strategy results"""
    # Prepare data for plotting
    strategies = [r['strategy'] for r in results]
    mean_returns = [r['total_return_mean'] for r in results]
    median_returns = [r['total_return_median'] for r in results]
    risk_std = [r['final_value_std'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean Returns
    bars1 = ax1.bar(range(len(strategies)), mean_returns, color='skyblue', alpha=0.7)
    ax1.set_title('Mean Total Returns by Strategy')
    ax1.set_ylabel('Return (%)')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([s[:20] + "..." if len(s) > 20 else s for s in strategies], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Risk vs Return
    ax2.scatter(risk_std, mean_returns, s=100, alpha=0.7, c='red')
    ax2.set_xlabel('Risk (Standard Deviation)')
    ax2.set_ylabel('Mean Return (%)')
    ax2.set_title('Risk vs Return')
    ax2.grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        ax2.annotate(strategy[:15] + "..." if len(strategy) > 15 else strategy, 
                    (risk_std[i], mean_returns[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # 3. Final Value Distribution (top 5 strategies)
    top_5_strategies = sorted(results, key=lambda x: x['total_return_mean'], reverse=True)[:5]
    top_strategies = [r['strategy'][:20] + "..." if len(r['strategy']) > 20 else r['strategy'] for r in top_5_strategies]
    top_means = [r['final_value_mean'] for r in top_5_strategies]
    top_medians = [r['final_value_median'] for r in top_5_strategies]
    
    x = np.arange(len(top_strategies))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, top_means, width, label='Mean', color='lightgreen', alpha=0.7)
    bars4 = ax3.bar(x + width/2, top_medians, width, label='Median', color='orange', alpha=0.7)
    
    ax3.set_title('Top 5 Strategies: Final Value')
    ax3.set_ylabel('Final Value ($)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_strategies, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Strategy Categories
    categories = {
        'DCA': [],
        'Reserve': []
    }
    
    for result in results:
        if 'DCA' in result['strategy']:
            categories['DCA'].append(result['total_return_mean'])
        elif 'Reserve' in result['strategy']:
            categories['Reserve'].append(result['total_return_mean'])
    
    category_means = [np.mean(categories[cat]) if categories[cat] else 0 for cat in categories.keys()]
    bars5 = ax4.bar(categories.keys(), category_means, color=['blue', 'green', 'red'], alpha=0.7)
    ax4.set_title('Average Returns by Strategy Category')
    ax4.set_ylabel('Mean Return (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars5, category_means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'Results/Bitcoin/comprehensive_strategy_analysis_latest.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nStrategy analysis plot saved to: {plot_filename}")
    return plot_filename

def save_results(results, monthly_amount=100):
    """Save detailed results to CSV"""
    # Create detailed results DataFrame
    detailed_data = []
    for result in results:
        detailed_data.append({
            'Strategy': result['strategy'],
            'Monthly_Investment': monthly_amount,
            'Total_Invested': result.get('total_invested', monthly_amount * 120),
            'Final_Value_Mean': result['final_value_mean'],
            'Final_Value_Median': result['final_value_median'],
            'Final_Value_Std': result['final_value_std'],
            'Final_Value_P5': result['final_value_p5'],
            'Final_Value_P95': result['final_value_p95'],
            'Total_Return_Mean_Percent': result['total_return_mean'],
            'Total_Return_Median_Percent': result['total_return_median'],
            'Total_Coins_Mean': result.get('total_coins_mean', result.get('initial_coins', 0))
        })
    
    df = pd.DataFrame(detailed_data)
    
    # Save to CSV with custom filename for this range
    results_filename = 'Results/Bitcoin/comprehensive_strategy_results_rsv_60-80.csv'
    df.to_csv(results_filename, index=False)
    print(f"Detailed results saved to: {results_filename}")
    return results_filename

def main():
    """Main function"""
    print("Bitcoin Comprehensive Investment Strategy Analyzer")
    print("Using GBM Simulation with Dynamic Growth + Volatility")
    print("="*70)
    
    # Load GBM paths
    prices = load_gbm_paths()
    if prices is None:
        return
    
    # Analyze all strategies
    results = analyze_all_strategies(prices, monthly_amount=100)
    
    # Create comparison
    comparison_df = create_strategy_comparison(results)
    
    # Create visualization
    plot_file = create_visualization(results, monthly_amount=100)
    
    # Save results
    results_file = save_results(results, monthly_amount=100)
    
    # Summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE STRATEGY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Monthly investment: $100")
    print(f"Total investment over 10 years: $12,000")
    print(f"Time horizon: 10 years")
    print(f"Number of strategies tested: {len(results)}")
    print(f"Best strategy: {comparison_df.iloc[0]['Strategy']}")
    print(f"Best mean return: {comparison_df.iloc[0]['Total_Return_Mean']:.1f}%")
    print(f"Best median return: {comparison_df.iloc[0]['Total_Return_Median']:.1f}%")
    
    print(f"\nFiles created:")
    print(f"  Strategy comparison: {results_file}")
    print(f"  Visualization: {plot_file}")
    
    print(f"\n✅ Comprehensive strategy analysis completed!")

if __name__ == "__main__":
    main() 