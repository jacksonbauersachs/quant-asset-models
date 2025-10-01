import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

def calculate_r2(y_actual, y_pred):
    """Calculate R² score."""
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def optimize_intercept_for_r2(df, fixed_a=54.36, fixed_b=0.000172):
    """Find the optimal intercept (c) that maximizes R² for fixed a and b."""
    
    x = df['days'].values
    y = df[' value'].values
    
    print(f"\nOptimizing intercept for maximum R²:")
    print(f"  Fixed a (y-intercept): {fixed_a}")
    print(f"  Fixed b (growth rate): {fixed_b}")
    
    def objective(c):
        """Objective function to maximize R² (minimize negative R²)."""
        y_pred = exponential_func(x, fixed_a, fixed_b, c)
        r2 = calculate_r2(y, y_pred)
        return -r2  # Minimize negative R² to maximize R²
    
    # Search for optimal intercept
    try:
        result = minimize_scalar(objective, bounds=(-1000, 1000), method='bounded')
        
        if result.success:
            optimal_c = result.x
            y_pred = exponential_func(x, fixed_a, fixed_b, optimal_c)
            r2 = calculate_r2(y, y_pred)
            
            print(f"\nOptimization Results:")
            print(f"  Optimal intercept (c): {optimal_c:.2f}")
            print(f"  Maximum R²: {r2:.6f}")
            
            return optimal_c, r2
        else:
            print("Optimization failed")
            return None, None
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None, None

def calculate_projections(df, fixed_a, fixed_b, optimal_c, current_price):
    """Calculate projections for various time horizons."""
    
    current_days = df['days'].iloc[-1]
    current_date = df['date'].iloc[-1]
    
    # Time horizons in years
    horizons = [1, 3, 5, 10, 15, 20]
    
    projections = {}
    
    for years in horizons:
        # Calculate future days
        future_days = current_days + (years * 365.25)
        
        # Model-based projection
        model_price = exponential_func(future_days, fixed_a, fixed_b, optimal_c)
        
        # Calculate returns
        model_return = ((model_price - current_price) / current_price) * 100
        model_growth_rate = ((model_price / current_price) ** (1/years) - 1) * 100
        
        # Current price-based projection: Start from current undervalued price
        # The current return should be: model_return + undervaluation_benefit
        model_fair_price = exponential_func(current_days, fixed_a, fixed_b, optimal_c)
        model_growth_rate_from_fair = ((model_price / model_fair_price) ** (1/years) - 1) * 100
        
        # Calculate undervaluation benefit
        undervaluation = ((current_price - model_fair_price) / model_fair_price) * 100
        undervaluation_benefit = abs(undervaluation)
        
        # Current return = model return + undervaluation benefit
        current_return = model_growth_rate_from_fair + undervaluation_benefit
        current_based_price = current_price * (1 + current_return/100)
        
        projections[years] = {
            'model_price': model_price,
            'model_return': model_return,
            'model_growth_rate': model_growth_rate,
            'current_based_price': current_based_price,
            'current_return': current_return,
            'current_growth_rate': model_growth_rate_from_fair
        }
    
    return projections

def save_projections_file(df, fixed_a, fixed_b, optimal_c, current_price, projections, output_dir):
    """Save projections to a text file."""
    
    current_date = df['date'].iloc[-1]
    model_fair_price = exponential_func(df['days'].iloc[-1], fixed_a, fixed_b, optimal_c)
    
    # Calculate current undervaluation
    undervaluation = ((current_price - model_fair_price) / model_fair_price) * 100
    
    results_file = os.path.join(output_dir, "palladium_optimized_projections.txt")
    
    with open(results_file, 'w') as f:
        f.write("PALLADIUM OPTIMIZED MODEL PROJECTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fixed Growth Rate (b): {fixed_b:.6f}\n")
        f.write(f"Fixed Y-Intercept (a): {fixed_a:.2f}\n")
        f.write(f"Optimized Intercept (c): {optimal_c:.2f}\n")
        f.write(f"Maximum R²: {calculate_r2(df[' value'].values, exponential_func(df['days'].values, fixed_a, fixed_b, optimal_c)):.6f}\n\n")
        
        f.write("CURRENT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"As of: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Current Price: ${current_price:.4f}\n")
        f.write(f"Model Fair Price (today): ${model_fair_price:.4f}\n")
        f.write(f"Undervaluation: {undervaluation:.2f}%\n\n")
        
        f.write("MODEL-BASED PROJECTIONS\n")
        f.write("-" * 30 + "\n")
        for years in [1, 3, 5, 10, 15, 20]:
            proj = projections[years]
            f.write(f"{years}-Year Projected Price: ${proj['model_price']:.4f} | Return vs Current: {proj['model_return']:.2f}%\n")
        
        f.write("\nCURRENT PRICE-BASED PROJECTIONS\n")
        f.write("-" * 35 + "\n")
        for years in [1, 3, 5, 10, 15, 20]:
            proj = projections[years]
            f.write(f"{years}-Year Projected Price: ${proj['current_based_price']:.4f} | Return vs Current: {proj['current_return']:.2f}%\n")
        
        f.write("\nMODEL-BASED GROWTH RATES\n")
        f.write("-" * 30 + "\n")
        for years in [1, 3, 5, 10, 15, 20]:
            proj = projections[years]
            f.write(f"  {years}-Year Growth: {proj['model_growth_rate']:.2f}%\n")
        
        f.write("\nCURRENT PRICE-BASED GROWTH RATES\n")
        f.write("-" * 35 + "\n")
        for years in [1, 3, 5, 10, 15, 20]:
            proj = projections[years]
            f.write(f"  {years}-Year Growth: {proj['current_growth_rate']:.2f}%\n")
        
        f.write("\nNOTES:\n")
        f.write("- Model-based projections use the optimized exponential model\n")
        f.write("- Current price-based projections use the model's growth rate from current price\n")
        f.write("- Growth rates are annualized compound rates\n")
        f.write("- These are statistical projections, not financial advice\n")
    
    return results_file

def create_optimization_chart(df, fixed_a, fixed_b, optimal_c, output_dir):
    """Create a chart showing the optimized model."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    x = df['days'].values
    y = df[' value'].values
    
    # Calculate model predictions
    y_pred = exponential_func(x, fixed_a, fixed_b, optimal_c)
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax1.plot(df['years'], y_pred, 'r-', linewidth=2, label=f'Optimized Model (R² = {calculate_r2(y, y_pred):.3f})')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Palladium Optimized Model - Normal Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax2.plot(df['years'], y_pred, 'r-', linewidth=2, label=f'Optimized Model (R² = {calculate_r2(y, y_pred):.3f})')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Price ($) - Log Scale')
    ax2.set_title('Palladium Optimized Model - Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f"""Optimized Model Formula:
y = {fixed_a:.2f} * e^({fixed_b:.6f} * x) + {optimal_c:.2f}

Optimization Results:
R² = {calculate_r2(y, y_pred):.6f}
Intercept (c) = {optimal_c:.2f}

This model maximizes R² for the
fixed growth rate of {fixed_b:.6f}"""
    
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(output_dir, "palladium_optimized_model.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_file

def main():
    print("Palladium Optimized Projections Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fixed parameters
    fixed_a = 54.36
    fixed_b = 0.000172
    
    # Optimize intercept for maximum R²
    optimal_c, r2 = optimize_intercept_for_r2(df, fixed_a, fixed_b)
    
    if optimal_c is None:
        print("Failed to optimize intercept")
        return
    
    # Get current price
    current_price = df[' value'].iloc[-1]
    
    # Calculate projections
    print(f"\nCalculating projections...")
    projections = calculate_projections(df, fixed_a, fixed_b, optimal_c, current_price)
    
    # Create output directory
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save projections file
    print(f"\nSaving projections...")
    results_file = save_projections_file(df, fixed_a, fixed_b, optimal_c, current_price, projections, output_dir)
    print(f"Projections saved: {results_file}")
    
    # Create chart
    print(f"\nCreating optimization chart...")
    chart_file = create_optimization_chart(df, fixed_a, fixed_b, optimal_c, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Print summary
    print(f"\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Fixed Growth Rate: {fixed_b:.6f}")
    print(f"Optimal Intercept: {optimal_c:.2f}")
    print(f"Maximum R²: {r2:.6f}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Model Fair Price: ${exponential_func(df['days'].iloc[-1], fixed_a, fixed_b, optimal_c):.2f}")
    
    print(f"\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
