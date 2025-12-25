import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_growth_formula_coefficients():
    """Load the growth formula coefficients"""
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    return a, b

def calculate_growth_formula_predictions():
    """Calculate what the growth formula predicts for future years"""
    a, b = load_growth_formula_coefficients()
    
    # Current day (7/20/2025)
    today_day = 6041
    
    # Calculate predictions for different years
    years = [0, 0.25, 0.5, 1, 2, 3, 5, 10]
    predictions = {}
    
    for year in years:
        future_day = today_day + int(year * 365.25)
        predicted_price = 10**(a * np.log(future_day) + b)
        predictions[year] = predicted_price
    
    return predictions, a, b

def load_gbm_results():
    """Load the GBM simulation results"""
    # Find the most recent GBM fair value start results
    import glob
    import os
    
    gbm_files = glob.glob('Results/Bitcoin/bitcoin_gbm_fair_value_start_*_summary.csv')
    if not gbm_files:
        print("No GBM results found!")
        return None
    
    # Get the most recent file
    latest_file = max(gbm_files, key=os.path.getctime)
    print(f"Loading GBM results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df

def validate_gbm_against_formula():
    """Compare GBM results with growth formula predictions"""
    print("Validating GBM Simulation Against Growth Formula")
    print("="*60)
    
    # Get growth formula predictions
    formula_predictions, a, b = calculate_growth_formula_predictions()
    
    # Get GBM results
    gbm_df = load_gbm_results()
    if gbm_df is None:
        return
    
    # Create comparison table
    comparison_data = []
    
    for _, row in gbm_df.iterrows():
        year = row['Year']
        gbm_mean = row['Mean_Price']
        gbm_median = row['Median_Price']
        
        if year in formula_predictions:
            formula_price = formula_predictions[year]
            
            # Calculate differences
            mean_diff = ((gbm_mean - formula_price) / formula_price) * 100
            median_diff = ((gbm_median - formula_price) / formula_price) * 100
            
            comparison_data.append({
                'Year': year,
                'Formula_Prediction': formula_price,
                'GBM_Mean': gbm_mean,
                'GBM_Median': gbm_median,
                'Mean_Diff_Percent': mean_diff,
                'Median_Diff_Percent': median_diff
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display results
    print(f"\nGrowth Formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
    print(f"Starting day: 6041 (7/20/2025)")
    print(f"Starting price: ${formula_predictions[0]:,.2f}")
    
    print(f"\n{'Year':<6} {'Formula':<12} {'GBM Mean':<12} {'GBM Median':<12} {'Mean Diff%':<10} {'Median Diff%':<10}")
    print("-" * 70)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Year']:<6.1f} ${row['Formula_Prediction']:<11,.0f} ${row['GBM_Mean']:<11,.0f} ${row['GBM_Median']:<11,.0f} {row['Mean_Diff_Percent']:<10.1f}% {row['Median_Diff_Percent']:<10.1f}%")
    
    # Calculate overall statistics
    mean_abs_diff = np.mean(np.abs(comparison_df['Mean_Diff_Percent']))
    median_abs_diff = np.mean(np.abs(comparison_df['Median_Diff_Percent']))
    
    print(f"\nValidation Summary:")
    print(f"Average absolute difference (Mean): {mean_abs_diff:.1f}%")
    print(f"Average absolute difference (Median): {median_abs_diff:.1f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot formula predictions
    years = list(formula_predictions.keys())
    formula_prices = list(formula_predictions.values())
    plt.plot(years, formula_prices, 'b-', linewidth=3, label='Growth Formula', marker='o')
    
    # Plot GBM results
    plt.plot(comparison_df['Year'], comparison_df['GBM_Mean'], 'r--', linewidth=2, label='GBM Mean', marker='s')
    plt.plot(comparison_df['Year'], comparison_df['GBM_Median'], 'g--', linewidth=2, label='GBM Median', marker='^')
    
    # Add confidence intervals from GBM
    plt.fill_between(gbm_df['Year'], gbm_df['P5_Price'], gbm_df['P95_Price'], 
                    alpha=0.2, color='red', label='GBM 90% Confidence')
    
    plt.xlabel('Years from 7/20/2025')
    plt.ylabel('Bitcoin Price ($)')
    plt.title('GBM Simulation vs Growth Formula Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Save plot
    plot_filename = f'Results/Bitcoin/gbm_formula_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nValidation plot saved to: {plot_filename}")
    
    # Save comparison data
    comparison_filename = f'Results/Bitcoin/gbm_formula_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    comparison_df.to_csv(comparison_filename, index=False)
    print(f"Comparison data saved to: {comparison_filename}")
    
    # Assessment
    print(f"\nAssessment:")
    if mean_abs_diff < 5:
        print(f"✅ EXCELLENT: GBM closely follows growth formula (avg diff: {mean_abs_diff:.1f}%)")
    elif mean_abs_diff < 15:
        print(f"✅ GOOD: GBM reasonably follows growth formula (avg diff: {mean_abs_diff:.1f}%)")
    elif mean_abs_diff < 30:
        print(f"⚠️  ACCEPTABLE: GBM shows some deviation from formula (avg diff: {mean_abs_diff:.1f}%)")
    else:
        print(f"❌ POOR: GBM deviates significantly from formula (avg diff: {mean_abs_diff:.1f}%)")
    
    return comparison_df

if __name__ == "__main__":
    validate_gbm_against_formula() 