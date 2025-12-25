import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
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

def backtest_mean_reversion_strategy(df, lookback_days=30, forward_days=90):
    """Backtest mean reversion strategy"""
    print(f"Backtesting mean reversion strategy...")
    print(f"Lookback: {lookback_days} days, Forward: {forward_days} days")
    
    results = []
    
    for i in range(lookback_days, len(df) - forward_days):
        # Current data
        current_date = df.iloc[i]['Date']
        current_price = df.iloc[i]['Price']
        current_ratio = df.iloc[i]['Price_Ratio']
        
        # Future data
        future_date = df.iloc[i + forward_days]['Date']
        future_price = df.iloc[i + forward_days]['Price']
        future_ratio = df.iloc[i + forward_days]['Price_Ratio']
        
        # Calculate returns
        price_return = (future_price - current_price) / current_price
        ratio_change = future_ratio - current_ratio
        
        # Categorize by overvaluation
        if current_ratio > 1.5:
            category = "Very Overvalued (>1.5x)"
        elif current_ratio > 1.2:
            category = "Moderately Overvalued (1.2-1.5x)"
        elif current_ratio > 1.0:
            category = "Slightly Overvalued (1.0-1.2x)"
        elif current_ratio > 0.8:
            category = "Fair Value (0.8-1.0x)"
        elif current_ratio > 0.5:
            category = "Moderately Undervalued (0.5-0.8x)"
        else:
            category = "Very Undervalued (<0.5x)"
        
        results.append({
            'current_date': current_date,
            'future_date': future_date,
            'current_price': current_price,
            'future_price': future_price,
            'current_ratio': current_ratio,
            'future_ratio': future_ratio,
            'price_return': price_return,
            'ratio_change': ratio_change,
            'category': category
        })
    
    return pd.DataFrame(results)

def analyze_backtest_results(results_df):
    """Analyze backtest results for statistical significance"""
    print("Analyzing backtest results...")
    
    # Group by category
    category_stats = results_df.groupby('category').agg({
        'price_return': ['count', 'mean', 'std', 'median'],
        'ratio_change': ['mean', 'std']
    }).round(4)
    
    # Calculate statistical significance
    significance_tests = {}
    
    # Test if overvalued categories have negative returns
    overvalued_categories = ['Very Overvalued (>1.5x)', 'Moderately Overvalued (1.2-1.5x)', 'Slightly Overvalued (1.0-1.2x)']
    
    for category in overvalued_categories:
        if category in results_df['category'].values:
            category_data = results_df[results_df['category'] == category]['price_return']
            
            # One-sample t-test: test if mean return is significantly different from 0
            t_stat, p_value = stats.ttest_1samp(category_data, 0)
            
            # Test if mean is significantly negative
            negative_p_value = stats.ttest_1samp(category_data, 0, alternative='less')[1]
            
            significance_tests[category] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'negative_p_value': negative_p_value,
                'significant_negative': negative_p_value < 0.05,
                'sample_size': len(category_data)
            }
    
    # Test if undervalued categories have positive returns
    undervalued_categories = ['Very Undervalued (<0.5x)', 'Moderately Undervalued (0.5-0.8x)']
    
    for category in undervalued_categories:
        if category in results_df['category'].values:
            category_data = results_df[results_df['category'] == category]['price_return']
            
            # One-sample t-test: test if mean return is significantly different from 0
            t_stat, p_value = stats.ttest_1samp(category_data, 0)
            
            # Test if mean is significantly positive
            positive_p_value = stats.ttest_1samp(category_data, 0, alternative='greater')[1]
            
            significance_tests[category] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'positive_p_value': positive_p_value,
                'significant_positive': positive_p_value < 0.05,
                'sample_size': len(category_data)
            }
    
    return category_stats, significance_tests

def create_backtest_visualization(results_df, category_stats, significance_tests):
    """Create visualization of backtest results"""
    print("Creating backtest visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Mean Reversion: Statistical Backtest Results', fontsize=16, fontweight='bold')
    
    # 1. Return distribution by category
    categories = results_df['category'].unique()
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    for i, category in enumerate(categories):
        category_data = results_df[results_df['category'] == category]['price_return']
        ax1.hist(category_data, bins=30, alpha=0.6, label=category, color=colors[i % len(colors)])
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Zero Return')
    ax1.set_title('Return Distribution by Price/Fair Value Category')
    ax1.set_xlabel('90-Day Return')
    ax1.set_ylabel('Frequency')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean returns by category
    mean_returns = []
    category_names = []
    
    for category in categories:
        category_data = results_df[results_df['category'] == category]['price_return']
        mean_returns.append(category_data.mean())
        category_names.append(category)
    
    bars = ax2.bar(range(len(categories)), mean_returns, color=colors[:len(categories)])
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_title('Mean 90-Day Returns by Category')
    ax2.set_xlabel('Price/Fair Value Category')
    ax2.set_ylabel('Mean Return')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(category_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add significance indicators
    for i, category in enumerate(categories):
        if category in significance_tests:
            test = significance_tests[category]
            if test.get('significant_negative', False) or test.get('significant_positive', False):
                ax2.text(i, mean_returns[i] + (0.01 if mean_returns[i] > 0 else -0.01), 
                        '*', ha='center', va='bottom' if mean_returns[i] > 0 else 'top', 
                        fontsize=16, fontweight='bold')
    
    # 3. Price ratio change over time
    for category in categories[:3]:  # Show first 3 categories
        category_data = results_df[results_df['category'] == category]
        ax3.scatter(category_data['current_ratio'], category_data['ratio_change'], 
                   alpha=0.6, label=category, s=20)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
    ax3.set_title('Price Ratio Change vs Starting Ratio')
    ax3.set_xlabel('Starting Price/Fair Value Ratio')
    ax3.set_ylabel('Change in Price/Fair Value Ratio (90 days)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical significance summary
    sig_data = []
    sig_labels = []
    
    for category, test in significance_tests.items():
        if test.get('significant_negative', False):
            sig_data.append(-test['t_statistic'])
            sig_labels.append(f"{category}\n(p={test['negative_p_value']:.3f})")
        elif test.get('significant_positive', False):
            sig_data.append(test['t_statistic'])
            sig_labels.append(f"{category}\n(p={test['positive_p_value']:.3f})")
        else:
            sig_data.append(test['t_statistic'])
            sig_labels.append(f"{category}\n(p={test['p_value']:.3f})")
    
    if sig_data:
        colors_sig = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in sig_data]
        bars = ax4.bar(range(len(sig_data)), sig_data, color=colors_sig)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax4.set_title('Statistical Significance (t-statistics)')
        ax4.set_xlabel('Category')
        ax4.set_ylabel('t-statistic')
        ax4.set_xticks(range(len(sig_data)))
        ax4.set_xticklabels(sig_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add significance thresholds
        ax4.axhline(y=1.96, color='red', linestyle=':', alpha=0.7, label='95% significance')
        ax4.axhline(y=-1.96, color='red', linestyle=':', alpha=0.7)
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_mean_reversion_backtest_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Backtest visualization saved to: {filename}")
    
    return filename

def save_backtest_results(results_df, category_stats, significance_tests):
    """Save backtest results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f'Results/Bitcoin/bitcoin_mean_reversion_backtest_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Backtest results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = f'Results/Bitcoin/bitcoin_mean_reversion_backtest_summary_{timestamp}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Bitcoin Mean Reversion Backtest Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Category Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(str(category_stats))
        f.write("\n\n")
        
        f.write("Statistical Significance Tests:\n")
        f.write("-" * 40 + "\n")
        for category, test in significance_tests.items():
            f.write(f"\n{category}:\n")
            f.write(f"  Sample size: {test['sample_size']}\n")
            f.write(f"  t-statistic: {test['t_statistic']:.3f}\n")
            f.write(f"  p-value: {test['p_value']:.3f}\n")
            
            if 'negative_p_value' in test:
                f.write(f"  Negative p-value: {test['negative_p_value']:.3f}\n")
                f.write(f"  Significantly negative: {test['significant_negative']}\n")
            elif 'positive_p_value' in test:
                f.write(f"  Positive p-value: {test['positive_p_value']:.3f}\n")
                f.write(f"  Significantly positive: {test['significant_positive']}\n")
    
    print(f"Backtest summary saved to: {summary_file}")
    
    return results_file, summary_file

def main():
    """Main backtest function"""
    print("Bitcoin Mean Reversion Statistical Backtest")
    print("="*50)
    
    # Load and prepare data
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Run backtest
    results_df = backtest_mean_reversion_strategy(df, lookback_days=30, forward_days=90)
    
    # Analyze results
    category_stats, significance_tests = analyze_backtest_results(results_df)
    
    # Create visualizations
    viz_file = create_backtest_visualization(results_df, category_stats, significance_tests)
    
    # Save results
    results_file, summary_file = save_backtest_results(results_df, category_stats, significance_tests)
    
    # Summary
    print(f"\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    
    print(f"Total observations: {len(results_df):,}")
    print(f"Date range: {results_df['current_date'].min()} to {results_df['current_date'].max()}")
    
    print(f"\nCategory Statistics:")
    print(category_stats)
    
    print(f"\nStatistical Significance:")
    for category, test in significance_tests.items():
        print(f"\n{category}:")
        print(f"  Sample size: {test['sample_size']}")
        print(f"  t-statistic: {test['t_statistic']:.3f}")
        print(f"  p-value: {test['p_value']:.3f}")
        
        if test.get('significant_negative', False):
            print(f"  ✅ SIGNIFICANTLY NEGATIVE (p={test['negative_p_value']:.3f})")
        elif test.get('significant_positive', False):
            print(f"  ✅ SIGNIFICANTLY POSITIVE (p={test['positive_p_value']:.3f})")
        else:
            print(f"  ❌ NOT STATISTICALLY SIGNIFICANT")
    
    print(f"\nKey Insights:")
    significant_categories = []
    for category, test in significance_tests.items():
        if test.get('significant_negative', False) or test.get('significant_positive', False):
            significant_categories.append(category)
    
    if significant_categories:
        print(f"✅ Statistically significant mean reversion found in: {', '.join(significant_categories)}")
    else:
        print(f"❌ No statistically significant mean reversion found")
    
    print(f"\nFiles created:")
    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Visualization: {viz_file}")

if __name__ == "__main__":
    main() 