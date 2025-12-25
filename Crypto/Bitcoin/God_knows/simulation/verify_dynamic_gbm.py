import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_formula_parameters():
    """Load growth and volatility formula parameters"""
    print("Loading formula parameters...")
    
    # Load growth formula parameters
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    # Load volatility formula parameters
    with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Formula: volatility =' in line:
                formula = line.split('=')[1].strip()
                parts = formula.split('*')
                a_vol = float(parts[0].strip())
                exp_part = parts[1].strip()
                b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                if b_vol < 0:
                    b_vol = abs(b_vol)
                c_vol = float(parts[2].split('+')[1].strip())
                break
    
    print(f"Growth formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
    print(f"Volatility formula: {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
    
    return a, b, a_vol, b_vol, c_vol

def calculate_formula_predictions(a, b, a_vol, b_vol, c_vol, years_ahead=10):
    """Calculate what our formulas predict for future prices and volatility"""
    print(f"\nCalculating formula predictions for {years_ahead} years...")
    
    # Bitcoin's current state
    bitcoin_current_age = 15.0
    today_day = 6041
    current_price = 118075  # Current Bitcoin price
    
    # Calculate formula predictions
    predictions = []
    
    for year in range(years_ahead + 1):
        future_age = bitcoin_current_age + year
        future_day = today_day + int(year * 365.25)
        
        # Growth formula prediction
        future_fair_value = 10**(a * np.log(future_day) + b)
        
        # Volatility formula prediction
        future_volatility = a_vol * np.exp(-b_vol * future_age) + c_vol
        
        # Annual growth rate from formula
        if year == 0:
            annual_growth_rate = 0
        else:
            current_fair_value = 10**(a * np.log(today_day) + b)
            annual_growth_rate = (future_fair_value / current_fair_value) ** (1/year) - 1
        
        predictions.append({
            'year': year,
            'age': future_age,
            'day': future_day,
            'fair_value': future_fair_value,
            'volatility': future_volatility,
            'annual_growth_rate': annual_growth_rate
        })
        
        print(f"Year {year}: Fair Value=${future_fair_value:,.0f}, Vol={future_volatility*100:.1f}%, Growth={annual_growth_rate*100:.1f}%")
    
    return predictions

def load_gbm_results():
    """Load the latest GBM simulation results"""
    print(f"\nLoading GBM simulation results...")
    
    # Find the latest GBM paths file
    import os
    results_dir = "Results/Bitcoin"
    gbm_files = [f for f in os.listdir(results_dir) if "bitcoin_gbm_paths_" in f and f.endswith('.csv')]
    
    if not gbm_files:
        print("❌ No GBM paths files found!")
        return None
    
    # Get the most recent file
    latest_file = sorted(gbm_files)[-1]
    paths_file = os.path.join(results_dir, latest_file)
    print(f"Using: {paths_file}")
    
    # Load the paths
    df = pd.read_csv(paths_file, index_col=0)
    print(f"Loaded {len(df.columns)} paths with {len(df)} time steps")
    
    return df

def analyze_gbm_consistency(df, predictions):
    """Analyze if GBM results are consistent with formula predictions"""
    print(f"\nAnalyzing GBM consistency with formulas...")
    
    # Calculate GBM statistics at each year
    gbm_stats = []
    
    for year in range(11):  # 0 to 10 years
        # Find closest time step
        time_steps = df.index.astype(float)
        step_index = np.argmin(np.abs(time_steps - year))
        
        prices_at_year = df.iloc[step_index, :].values
        
        mean_price = np.mean(prices_at_year)
        median_price = np.median(prices_at_year)
        std_price = np.std(prices_at_year)
        
        # Calculate annualized volatility
        if year == 0:
            annualized_vol = 0
        else:
            # Calculate volatility from price changes
            if step_index > 0:
                prev_prices = df.iloc[step_index-1, :].values
                returns = np.log(prices_at_year / prev_prices)
                annualized_vol = np.std(returns) * np.sqrt(365.25)
            else:
                annualized_vol = 0
        
        gbm_stats.append({
            'year': year,
            'mean_price': mean_price,
            'median_price': median_price,
            'std_price': std_price,
            'annualized_vol': annualized_vol
        })
        
        print(f"Year {year}: Mean=${mean_price:,.0f}, Vol={annualized_vol*100:.1f}%")
    
    return gbm_stats

def compare_formula_vs_gbm(predictions, gbm_stats):
    """Compare formula predictions with GBM results"""
    print(f"\nComparing Formula Predictions vs GBM Results")
    print("="*80)
    print(f"{'Year':<4} {'Formula Fair Value':<18} {'GBM Mean':<12} {'Diff %':<8} {'Formula Vol':<12} {'GBM Vol':<10} {'Vol Diff':<10}")
    print("-"*80)
    
    comparisons = []
    
    for i, (pred, gbm) in enumerate(zip(predictions, gbm_stats)):
        # Price comparison
        price_diff_pct = ((gbm['mean_price'] - pred['fair_value']) / pred['fair_value']) * 100
        
        # Volatility comparison
        vol_diff_pct = ((gbm['annualized_vol'] - pred['volatility']) / pred['volatility']) * 100 if pred['volatility'] > 0 else 0
        
        print(f"{pred['year']:<4} ${pred['fair_value']:<17,.0f} ${gbm['mean_price']:<11,.0f} {price_diff_pct:<7.1f}% {pred['volatility']*100:<11.1f}% {gbm['annualized_vol']*100:<9.1f}% {vol_diff_pct:<9.1f}%")
        
        comparisons.append({
            'year': pred['year'],
            'formula_fair_value': pred['fair_value'],
            'gbm_mean_price': gbm['mean_price'],
            'price_diff_pct': price_diff_pct,
            'formula_vol': pred['volatility'],
            'gbm_vol': gbm['annualized_vol'],
            'vol_diff_pct': vol_diff_pct
        })
    
    return comparisons

def create_consistency_visualization(predictions, gbm_stats, comparisons):
    """Create visualization of formula vs GBM consistency"""
    print(f"\nCreating consistency visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dynamic GBM Consistency Check: Formula vs Simulation', fontsize=16, fontweight='bold')
    
    years = [p['year'] for p in predictions]
    
    # 1. Price comparison
    formula_prices = [p['fair_value'] for p in predictions]
    gbm_prices = [g['mean_price'] for g in gbm_stats]
    
    ax1.plot(years, formula_prices, 'b-', linewidth=2, label='Formula Fair Value', marker='o')
    ax1.plot(years, gbm_prices, 'r--', linewidth=2, label='GBM Mean Price', marker='s')
    ax1.fill_between(years, [g['mean_price'] - g['std_price'] for g in gbm_stats], 
                     [g['mean_price'] + g['std_price'] for g in gbm_stats], 
                     alpha=0.3, color='red', label='GBM ±1 Std Dev')
    ax1.set_title('Price Evolution: Formula vs GBM')
    ax1.set_ylabel('Bitcoin Price ($)')
    ax1.set_xlabel('Years')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Volatility comparison
    formula_vols = [p['volatility'] * 100 for p in predictions]
    gbm_vols = [g['annualized_vol'] * 100 for g in gbm_stats]
    
    ax2.plot(years, formula_vols, 'b-', linewidth=2, label='Formula Volatility', marker='o')
    ax2.plot(years, gbm_vols, 'r--', linewidth=2, label='GBM Volatility', marker='s')
    ax2.set_title('Volatility Evolution: Formula vs GBM')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_xlabel('Years')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Price difference percentage
    price_diffs = [c['price_diff_pct'] for c in comparisons]
    ax3.bar(years, price_diffs, color='skyblue', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Price Difference: (GBM - Formula) / Formula')
    ax3.set_ylabel('Difference (%)')
    ax3.set_xlabel('Years')
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility difference percentage
    vol_diffs = [c['vol_diff_pct'] for c in comparisons]
    ax4.bar(years, vol_diffs, color='lightcoral', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Volatility Difference: (GBM - Formula) / Formula')
    ax4.set_ylabel('Difference (%)')
    ax4.set_xlabel('Years')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/gbm_consistency_check_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Consistency visualization saved to: {filename}")
    
    return filename

def main():
    """Main verification function"""
    print("Dynamic GBM Consistency Check")
    print("="*50)
    
    # Load formula parameters
    a, b, a_vol, b_vol, c_vol = load_formula_parameters()
    
    # Calculate formula predictions
    predictions = calculate_formula_predictions(a, b, a_vol, b_vol, c_vol)
    
    # Load GBM results
    df = load_gbm_results()
    if df is None:
        return
    
    # Analyze GBM consistency
    gbm_stats = analyze_gbm_consistency(df, predictions)
    
    # Compare formula vs GBM
    comparisons = compare_formula_vs_gbm(predictions, gbm_stats)
    
    # Create visualization
    viz_file = create_consistency_visualization(predictions, gbm_stats, comparisons)
    
    # Summary
    print(f"\n" + "="*50)
    print("CONSISTENCY CHECK SUMMARY")
    print("="*50)
    
    # Calculate average differences
    avg_price_diff = np.mean([abs(c['price_diff_pct']) for c in comparisons[1:]])  # Skip year 0
    avg_vol_diff = np.mean([abs(c['vol_diff_pct']) for c in comparisons[1:] if c['vol_diff_pct'] != 0])
    
    print(f"Average Price Difference: {avg_price_diff:.1f}%")
    print(f"Average Volatility Difference: {avg_vol_diff:.1f}%")
    
    if avg_price_diff < 10 and avg_vol_diff < 20:
        print("✅ GBM simulation is CONSISTENT with formulas!")
    elif avg_price_diff < 20 and avg_vol_diff < 30:
        print("⚠️  GBM simulation is MODERATELY consistent with formulas")
    else:
        print("❌ GBM simulation shows SIGNIFICANT deviations from formulas")
    
    print(f"\nVisualization saved to: {viz_file}")

if __name__ == "__main__":
    main() 