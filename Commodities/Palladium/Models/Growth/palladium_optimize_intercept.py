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

def calculate_r2(x, y, a, b, c):
    """Calculate R² for given parameters."""
    y_pred = exponential_func(x, a, b, c)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def optimize_intercept(df, fixed_growth_rate=0.000172):
    """Optimize the y-intercept for a fixed growth rate."""
    
    x = df['days'].values
    y = df[' value'].values
    
    print(f"\nOptimizing y-intercept for fixed growth rate: {fixed_growth_rate}")
    
    # Test different y-intercepts (a parameter)
    a_values = np.linspace(1, 200, 100)  # Test 100 different a values
    r2_values = []
    
    for a in a_values:
        # For each a, find the best c (y-intercept offset)
        def objective(c):
            y_pred = exponential_func(x, a, fixed_growth_rate, c)
            ss_res = np.sum((y - y_pred) ** 2)
            return ss_res
        
        # Optimize c for this a
        try:
            result = minimize_scalar(objective, bounds=(-1000, 1000), method='bounded')
            if result.success:
                c_optimal = result.x
                r2 = calculate_r2(x, y, a, fixed_growth_rate, c_optimal)
                r2_values.append(r2)
            else:
                r2_values.append(-np.inf)
        except:
            r2_values.append(-np.inf)
    
    # Find the best parameters
    best_idx = np.argmax(r2_values)
    best_a = a_values[best_idx]
    best_r2 = r2_values[best_idx]
    
    # Find optimal c for the best a
    def objective(c):
        y_pred = exponential_func(x, best_a, fixed_growth_rate, c)
        ss_res = np.sum((y - y_pred) ** 2)
        return ss_res
    
    result = minimize_scalar(objective, bounds=(-1000, 1000), method='bounded')
    best_c = result.x if result.success else 0
    
    return a_values, r2_values, best_a, best_c, best_r2

def create_optimization_chart(df, a_values, r2_values, best_a, best_c, best_r2, fixed_growth_rate, output_dir):
    """Create a chart showing the optimization results."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: R² vs y-intercept (a parameter)
    ax1.plot(a_values, r2_values, 'b-', linewidth=2, label='R² vs y-intercept')
    ax1.axvline(x=best_a, color='red', linestyle='--', alpha=0.7, label=f'Optimal a = {best_a:.2f}')
    ax1.axhline(y=best_r2, color='red', linestyle='--', alpha=0.7, label=f'Max R² = {best_r2:.4f}')
    
    ax1.set_xlabel('y-intercept parameter (a)')
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'R² Optimization for Fixed Growth Rate = {fixed_growth_rate}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Model fit with optimal parameters
    x = df['days'].values
    y = df[' value'].values
    
    # Plot actual data
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    # Plot optimized model
    y_optimized = exponential_func(x, best_a, fixed_growth_rate, best_c)
    ax2.plot(df['years'], y_optimized, 'r-', linewidth=2, label=f'Optimized Model (R² = {best_r2:.4f})')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($)')
    ax2.set_title('Optimized Model Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add optimization info
    formula_text = f"""Optimized Model:
y = {best_a:.4f} * e^({fixed_growth_rate:.6f} * x) + {best_c:.2f}
R² = {best_r2:.4f}

Fixed Growth Rate: {fixed_growth_rate}
Optimal y-intercept: {best_a:.2f}
Optimal offset: {best_c:.2f}"""
    
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_intercept_optimization.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_optimized_analysis(df, best_a, best_c, best_r2, fixed_growth_rate):
    """Calculate analysis for the optimized model."""
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    # Calculate model fair value
    fair_value = exponential_func(current_days, best_a, fixed_growth_rate, best_c)
    
    # Calculate deviation
    deviation = ((current_price - fair_value) / fair_value) * 100
    
    return {
        'fair_value': fair_value,
        'current_price': current_price,
        'deviation': deviation,
        'growth_rate': fixed_growth_rate,
        'y_intercept': best_a,
        'offset': best_c,
        'r2': best_r2
    }

def main():
    print("Palladium Y-Intercept Optimization")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fixed growth rate
    fixed_growth_rate = 0.000172
    
    # Optimize y-intercept
    a_values, r2_values, best_a, best_c, best_r2 = optimize_intercept(df, fixed_growth_rate)
    
    print(f"\nOptimization Results:")
    print(f"  Fixed Growth Rate: {fixed_growth_rate}")
    print(f"  Optimal y-intercept (a): {best_a:.4f}")
    print(f"  Optimal offset (c): {best_c:.2f}")
    print(f"  Maximum R²: {best_r2:.4f}")
    
    # Calculate analysis
    analysis = calculate_optimized_analysis(df, best_a, best_c, best_r2, fixed_growth_rate)
    
    print(f"\nOptimized Model Analysis:")
    print(f"  Current Price: ${analysis['current_price']:.2f}")
    print(f"  Model Fair Value: ${analysis['fair_value']:.2f}")
    print(f"  Deviation: {analysis['deviation']:.1f}%")
    print(f"  R² Score: {analysis['r2']:.4f}")
    
    # Create chart
    print("\nCreating optimization chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_optimization_chart(df, a_values, r2_values, best_a, best_c, best_r2, fixed_growth_rate, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_intercept_optimization.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM Y-INTERCEPT OPTIMIZATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Fixed growth rate at 0.000172\n")
        f.write("2. Tested 100 different y-intercept values (a parameter)\n")
        f.write("3. For each a, optimized the offset parameter (c)\n")
        f.write("4. Found the combination that maximizes R²\n\n")
        
        f.write(f"Optimization Results:\n")
        f.write(f"  Fixed Growth Rate: {fixed_growth_rate}\n")
        f.write(f"  Optimal y-intercept (a): {best_a:.4f}\n")
        f.write(f"  Optimal offset (c): {best_c:.2f}\n")
        f.write(f"  Maximum R²: {best_r2:.4f}\n\n")
        
        f.write(f"Optimized Model: y = {best_a:.4f} * e^({fixed_growth_rate:.6f} * x) + {best_c:.2f}\n\n")
        
        f.write("Fair Value Analysis:\n")
        f.write(f"  Current Price: ${analysis['current_price']:.2f}\n")
        f.write(f"  Model Fair Value: ${analysis['fair_value']:.2f}\n")
        f.write(f"  Deviation: {analysis['deviation']:.1f}%\n\n")
        
        f.write("Key Insights:\n")
        f.write("- This shows the optimal y-intercept for the given growth rate\n")
        f.write("- The chart shows how R² changes with different y-intercepts\n")
        f.write("- This helps understand the sensitivity of the model to parameters\n")
        f.write("- The optimized model provides the best fit for the fixed growth rate\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("Y-INTERCEPT OPTIMIZATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
