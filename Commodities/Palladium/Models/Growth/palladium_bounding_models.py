import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.optimize import minimize

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

def find_upper_bound_model(df):
    """Find the tightest exponential model that bounds all data from above."""
    
    x = df['days'].values
    y = df[' value'].values
    
    def objective(params):
        """Minimize the growth rate (b parameter) while keeping all points below the curve."""
        a, b, c = params
        
        # Calculate model predictions
        y_pred = exponential_func(x, a, b, c)
        
        # Check if any point is above the curve
        violations = y_pred - y
        max_violation = np.max(violations)
        
        # If any point is above, add a large penalty
        if max_violation < 0:
            return 1000 + abs(max_violation)  # Large penalty for violations
        
        # If all points are below, minimize the growth rate (b)
        return b
    
    # Initial guess
    x0 = [100, 0.0001, 0]
    
    # Bounds: a > 0, b > 0, c can be negative
    bounds = [(1, 1000), (0.00001, 0.01), (-1000, 1000)]
    
    # Constraints: all points must be below the curve
    def constraint(params):
        a, b, c = params
        y_pred = exponential_func(x, a, b, c)
        return y_pred - y  # Must be >= 0 for all points
    
    constraints = {'type': 'ineq', 'fun': constraint}
    
    try:
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if result.success:
            a, b, c = result.x
            y_pred = exponential_func(x, a, b, c)
            
            # Calculate R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'name': 'Upper Bound Model',
                'params': [a, b, c],
                'r2': r2,
                'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}',
                'growth_rate': b
            }
        else:
            print(f"Upper bound optimization failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"Error in upper bound optimization: {e}")
        return None

def find_lower_bound_model(df):
    """Find the tightest exponential model that bounds all data from below."""
    
    x = df['days'].values
    y = df[' value'].values
    
    def objective(params):
        """Maximize the growth rate (b parameter) while keeping all points above the curve."""
        a, b, c = params
        
        # Calculate model predictions
        y_pred = exponential_func(x, a, b, c)
        
        # Check if any point is below the curve
        violations = y - y_pred
        max_violation = np.max(violations)
        
        # If any point is below, add a large penalty
        if max_violation < 0:
            return 1000 + abs(max_violation)  # Large penalty for violations
        
        # If all points are above, maximize the growth rate (b) by minimizing -b
        return -b
    
    # Initial guess
    x0 = [10, 0.0001, 0]
    
    # Bounds: a > 0, b > 0, c can be negative
    bounds = [(1, 1000), (0.00001, 0.01), (-1000, 1000)]
    
    # Constraints: all points must be above the curve
    def constraint(params):
        a, b, c = params
        y_pred = exponential_func(x, a, b, c)
        return y - y_pred  # Must be >= 0 for all points
    
    constraints = {'type': 'ineq', 'fun': constraint}
    
    try:
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if result.success:
            a, b, c = result.x
            y_pred = exponential_func(x, a, b, c)
            
            # Calculate R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'name': 'Lower Bound Model',
                'params': [a, b, c],
                'r2': r2,
                'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}',
                'growth_rate': b
            }
        else:
            print(f"Lower bound optimization failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"Error in lower bound optimization: {e}")
        return None

def create_bounding_chart(df, upper_model, lower_model, output_dir):
    """Create a chart showing the bounding models."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    x = df['days'].values
    
    if upper_model:
        y_upper = exponential_func(x, *upper_model['params'])
        ax1.plot(df['years'], y_upper, 'r-', linewidth=2, label=f'Upper Bound (R² = {upper_model["r2"]:.3f})')
    
    if lower_model:
        y_lower = exponential_func(x, *lower_model['params'])
        ax1.plot(df['years'], y_lower, 'g-', linewidth=2, label=f'Lower Bound (R² = {lower_model["r2"]:.3f})')
    
    # Fill between the bounds
    if upper_model and lower_model:
        ax1.fill_between(df['years'], y_lower, y_upper, alpha=0.2, color='yellow', label='Bounding Region')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Bounding Models - Tightest Exponential Bounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    if upper_model:
        ax2.plot(df['years'], y_upper, 'r-', linewidth=2, label=f'Upper Bound')
    
    if lower_model:
        ax2.plot(df['years'], y_lower, 'g-', linewidth=2, label=f'Lower Bound')
    
    if upper_model and lower_model:
        ax2.fill_between(df['years'], y_lower, y_upper, alpha=0.2, color='yellow', label='Bounding Region')
    
    ax2.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Bounding Models - Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_bounding_models.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_bounding_analysis(df, upper_model, lower_model):
    """Calculate analysis for the bounding models."""
    
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    results = {}
    
    if upper_model:
        fair_value = exponential_func(current_days, *upper_model['params'])
        undervaluation = ((fair_value - current_price) / current_price) * 100
        results['upper'] = {
            'fair_value': fair_value,
            'undervaluation': undervaluation,
            'growth_rate': upper_model['growth_rate']
        }
    
    if lower_model:
        fair_value = exponential_func(current_days, *lower_model['params'])
        undervaluation = ((fair_value - current_price) / current_price) * 100
        results['lower'] = {
            'fair_value': fair_value,
            'undervaluation': undervaluation,
            'growth_rate': lower_model['growth_rate']
        }
    
    return results

def main():
    print("Palladium Bounding Models Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Find upper bound model
    print("\nFinding upper bound model (lowest growth rate that bounds all data from above)...")
    upper_model = find_upper_bound_model(df)
    
    if upper_model:
        print(f"Upper Bound Model: {upper_model['formula']}")
        print(f"  R² = {upper_model['r2']:.4f}")
        print(f"  Growth Rate = {upper_model['growth_rate']:.6f}")
    else:
        print("Failed to find upper bound model")
    
    # Find lower bound model
    print("\nFinding lower bound model (highest growth rate that bounds all data from below)...")
    lower_model = find_lower_bound_model(df)
    
    if lower_model:
        print(f"Lower Bound Model: {lower_model['formula']}")
        print(f"  R² = {lower_model['r2']:.4f}")
        print(f"  Growth Rate = {lower_model['growth_rate']:.6f}")
    else:
        print("Failed to find lower bound model")
    
    # Calculate analysis
    if upper_model and lower_model:
        print("\nCalculating bounding analysis...")
        results = calculate_bounding_analysis(df, upper_model, lower_model)
        
        current_price = df[' value'].iloc[-1]
        print(f"\nCurrent Price: ${current_price:.2f}")
        
        if 'upper' in results:
            print(f"Upper Bound Fair Value: ${results['upper']['fair_value']:.2f}")
            print(f"  Undervaluation: {results['upper']['undervaluation']:.1f}%")
        
        if 'lower' in results:
            print(f"Lower Bound Fair Value: ${results['lower']['fair_value']:.2f}")
            print(f"  Undervaluation: {results['lower']['undervaluation']:.1f}%")
        
        # Calculate the difference
        if 'upper' in results and 'lower' in results:
            growth_diff = results['upper']['growth_rate'] - results['lower']['growth_rate']
            fair_diff = results['upper']['fair_value'] - results['lower']['fair_value']
            print(f"\nBounding Analysis:")
            print(f"  Growth Rate Difference: {growth_diff:.6f}")
            print(f"  Fair Value Range: ${results['lower']['fair_value']:.2f} to ${results['upper']['fair_value']:.2f}")
            print(f"  Range Width: ${fair_diff:.2f}")
    
    # Create chart
    print("\nCreating bounding models chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_bounding_chart(df, upper_model, lower_model, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_bounding_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM BOUNDING MODELS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Upper Bound: Find lowest growth rate exponential that bounds all data from above\n")
        f.write("2. Lower Bound: Find highest growth rate exponential that bounds all data from below\n")
        f.write("3. This creates the tightest possible exponential bounds\n")
        f.write("4. No data point escapes the bounding region\n\n")
        
        if upper_model:
            f.write(f"Upper Bound Model: {upper_model['formula']}\n")
            f.write(f"  R² = {upper_model['r2']:.6f}\n")
            f.write(f"  Growth Rate = {upper_model['growth_rate']:.6f}\n")
            if 'upper' in results:
                f.write(f"  Fair Value = ${results['upper']['fair_value']:.2f}\n")
                f.write(f"  Undervaluation = {results['upper']['undervaluation']:.1f}%\n")
            f.write("\n")
        
        if lower_model:
            f.write(f"Lower Bound Model: {lower_model['formula']}\n")
            f.write(f"  R² = {lower_model['r2']:.6f}\n")
            f.write(f"  Growth Rate = {lower_model['growth_rate']:.6f}\n")
            if 'lower' in results:
                f.write(f"  Fair Value = ${results['lower']['fair_value']:.2f}\n")
                f.write(f"  Undervaluation = {results['lower']['undervaluation']:.1f}%\n")
            f.write("\n")
        
        f.write("Key Insights:\n")
        f.write("- Upper bound shows minimum growth rate needed to bound all peaks\n")
        f.write("- Lower bound shows maximum growth rate that bounds all valleys\n")
        f.write("- The region between bounds contains all historical data\n")
        f.write("- This provides the tightest possible exponential bounds\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("BOUNDING MODELS ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
