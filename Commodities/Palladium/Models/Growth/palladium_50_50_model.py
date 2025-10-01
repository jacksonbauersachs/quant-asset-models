import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.optimize import minimize_scalar

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['date'].min()
        df['days'] = (df['date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def calculate_above_below_ratio(x, y, a, b, c):
    """Calculate the ratio of points above vs below the model line."""
    
    # Get model predictions
    y_pred = exponential_func(x, a, b, c)
    
    # Count points above and below
    above = np.sum(y > y_pred)
    below = np.sum(y < y_pred)
    total = len(y)
    
    # Calculate percentages
    above_pct = (above / total) * 100
    below_pct = (below / total) * 100
    
    # Calculate deviation from 50/50
    deviation = abs(above_pct - 50.0)
    
    return above, below, above_pct, below_pct, deviation

def find_50_50_parameters(df, fixed_a=54.36):
    """Find both growth rate and intercept that give exactly 50/50 split."""
    
    x = df['days'].values
    y = df[' value'].values
    
    print(f"\nFinding 50/50 parameters:")
    print(f"  Fixed a (y-intercept): {fixed_a}")
    print(f"  Optimizing b (growth rate) and c (intercept)")
    
    def objective(params):
        """Objective function to minimize deviation from 50/50."""
        b, c = params
        above, below, above_pct, below_pct, deviation = calculate_above_below_ratio(x, y, fixed_a, b, c)
        return deviation
    
    # Search for optimal growth rate and intercept
    try:
        from scipy.optimize import minimize
        
        # Initial guess
        x0 = [0.000172, 20.66]  # Start with previous values
        
        # Bounds: growth rate between 0.00005 and 0.001, intercept between -500 and 500
        bounds = [(0.00005, 0.001), (-500, 500)]
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_b, optimal_c = result.x
            above, below, above_pct, below_pct, deviation = calculate_above_below_ratio(x, y, fixed_a, optimal_b, optimal_c)
            
            print(f"\n50/50 Optimization Results:")
            print(f"  Optimal growth rate (b): {optimal_b:.6f}")
            print(f"  Optimal intercept (c): {optimal_c:.2f}")
            print(f"  Points above line: {above} ({above_pct:.1f}%)")
            print(f"  Points below line: {below} ({below_pct:.1f}%)")
            print(f"  Deviation from 50/50: {deviation:.1f}%")
            
            return optimal_b, optimal_c, above, below, above_pct, below_pct, deviation
        else:
            print("Optimization failed")
            return None, None, None, None, None, None, None
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None, None, None, None, None, None, None

def create_50_50_chart(df, optimal_b, optimal_c, fixed_a, above, below, above_pct, below_pct, output_dir):
    """Create a chart showing the 50/50 model."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    x = df['days'].values
    y = df[' value'].values
    
    # Calculate model predictions
    y_pred = exponential_func(x, fixed_a, optimal_b, optimal_c)
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax1.plot(df['years'], y_pred, 'r-', linewidth=2, label=f'50/50 Model (R² = {calculate_r2(y, y_pred):.3f})')
    
    # Highlight points above and below
    above_mask = y > y_pred
    below_mask = y < y_pred
    
    ax1.scatter(df[above_mask]['years'], df[above_mask][' value'], alpha=0.8, s=2, color='green', label=f'Above Line ({above_pct:.1f}%)')
    ax1.scatter(df[below_mask]['years'], df[below_mask][' value'], alpha=0.8, s=2, color='red', label=f'Below Line ({below_pct:.1f}%)')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium 50/50 Model - Balanced Above/Below Split')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax2.plot(df['years'], y_pred, 'r-', linewidth=2, label=f'50/50 Model')
    
    ax2.scatter(df[above_mask]['years'], df[above_mask][' value'], alpha=0.8, s=2, color='green', label=f'Above Line ({above_pct:.1f}%)')
    ax2.scatter(df[below_mask]['years'], df[below_mask][' value'], alpha=0.8, s=2, color='red', label=f'Below Line ({below_pct:.1f}%)')
    
    ax2.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('50/50 Model - Log Scale (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f"""50/50 Model Formula:
y = {fixed_a:.2f} * e^({optimal_b:.6f} * x) + {optimal_c:.2f}

Balance Analysis:
Above Line: {above} points ({above_pct:.1f}%)
Below Line: {below} points ({below_pct:.1f}%)
Deviation from 50/50: {abs(above_pct - 50.0):.1f}%

This model balances time above/below
the line rather than minimizing errors."""
    
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_50_50_model.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_r2(y_actual, y_pred):
    """Calculate R² score."""
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_50_50_analysis(df, optimal_b, optimal_c, fixed_a):
    """Calculate analysis for the 50/50 model."""
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    # Calculate model fair value
    fair_value = exponential_func(current_days, fixed_a, optimal_b, optimal_c)
    
    # Calculate deviation
    deviation = ((current_price - fair_value) / fair_value) * 100
    
    # Calculate R²
    x_all = df['days'].values
    y_all = df[' value'].values
    y_pred_all = exponential_func(x_all, fixed_a, optimal_b, optimal_c)
    r2 = calculate_r2(y_all, y_pred_all)
    
    return {
        'fair_value': fair_value,
        'current_price': current_price,
        'deviation': deviation,
        'r2': r2,
        'growth_rate': optimal_b,
        'y_intercept': fixed_a,
        'intercept': optimal_c
    }

def main():
    print("Palladium 50/50 Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Find 50/50 parameters
    optimal_b, optimal_c, above, below, above_pct, below_pct, deviation = find_50_50_parameters(df)
    
    if optimal_b is None:
        print("Failed to find 50/50 parameters")
        return
    
    # Calculate analysis
    analysis = calculate_50_50_analysis(df, optimal_b, optimal_c, 54.36)
    
    print(f"\n50/50 Model Analysis:")
    print(f"  Current Price: ${analysis['current_price']:.2f}")
    print(f"  Model Fair Value: ${analysis['fair_value']:.2f}")
    print(f"  Deviation: {analysis['deviation']:.1f}%")
    print(f"  R² Score: {analysis['r2']:.4f}")
    print(f"  Balance: {above_pct:.1f}% above, {below_pct:.1f}% below")
    
    # Create chart
    print("\nCreating 50/50 model chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_50_50_chart(df, optimal_b, optimal_c, 54.36, above, below, above_pct, below_pct, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_50_50_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM 50/50 MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Fixed y-intercept at 54.36\n")
        f.write("2. Optimized growth rate (b) and intercept (c) to achieve 50/50 split\n")
        f.write("3. This balances time above/below the model line\n")
        f.write("4. More robust than R² for volatile data\n")
        f.write("5. Finds optimal combination of both parameters\n\n")
        
        f.write(f"50/50 Model: y = 54.36 * e^({optimal_b:.6f} * x) + {optimal_c:.2f}\n")
        f.write(f"  R² = {analysis['r2']:.6f}\n")
        f.write(f"  Points above line: {above} ({above_pct:.1f}%)\n")
        f.write(f"  Points below line: {below} ({below_pct:.1f}%)\n")
        f.write(f"  Deviation from 50/50: {deviation:.1f}%\n\n")
        
        f.write("Fair Value Analysis:\n")
        f.write(f"  Current Price: ${analysis['current_price']:.2f}\n")
        f.write(f"  Model Fair Value: ${analysis['fair_value']:.2f}\n")
        f.write(f"  Deviation: {analysis['deviation']:.1f}%\n\n")
        
        f.write("Key Insights:\n")
        f.write("- This model balances time above/below the line\n")
        f.write("- More robust to outliers than R² minimization\n")
        f.write("- Shows the 'true center' of palladium's behavior\n")
        f.write("- Better for volatile data like palladium\n")
        f.write("- Current price position relative to balanced model\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("50/50 MODEL ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
