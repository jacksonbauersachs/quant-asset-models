import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
from datetime import datetime

def load_copper_data():
    """Load the comprehensive copper data."""
    try:
        df = pd.read_csv('Commodities/Copper/Copper_Prices.csv')
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Clean Value column - convert to float
        df['Price'] = df['Value'].astype(float)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['Date'].min()
        df['days'] = (df['Date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Copper data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading copper data: {e}")
        return None

def exponential_model(x, a, b, c):
    """Exponential model: price = a * exp(b * days) + c"""
    return a * np.exp(b * x) + c

def find_optimal_parameters(df, fixed_growth_rate):
    """Find the optimal amplitude and intercept that maximizes R² for the fixed growth rate."""
    x = df['days'].values
    y = df['Price'].values
    
    # Fixed growth rate
    fixed_b = .000097
    
    def r2_for_parameters(params):
        """Calculate R² for given amplitude and intercept values."""
        a, c = params
        y_pred = exponential_model(x, a, fixed_b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    # Find the amplitude and intercept that maximize R²
    from scipy.optimize import minimize
    
    # Initial guess
    initial_guess = [100, 0]  # [amplitude, intercept]
    
    # Bounds: amplitude (0, 1000), intercept (-1000, 1000)
    bounds = [(0, 1000), (-1000, 1000)]
    
    result = minimize(lambda params: -r2_for_parameters(params), 
                     initial_guess, 
                     bounds=bounds,
                     method='L-BFGS-B')
    
    optimal_a, optimal_c = result.x
    optimal_r2 = r2_for_parameters([optimal_a, optimal_c])
    
    print(f"Fixed Growth Rate: {fixed_b:.6f}")
    print(f"Optimal Amplitude: {optimal_a:.2f}")
    print(f"Optimal Intercept: {optimal_c:.2f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_a, fixed_b, optimal_c, optimal_r2

def create_fixed_growth_chart(df, a, b, c, r2):
    """Create the chart with fixed growth rate and optimal intercept."""
    
    x = df['days'].values
    y = df['Price'].values
    
    # Calculate model predictions
    y_model = exponential_model(x, a, b, c)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot - Normal scale
    ax1.scatter(df['years'], df['Price'], alpha=0.6, s=1, color='blue', label='Actual Copper Price')
    ax1.plot(df['years'], y_model, 'r-', linewidth=2, label='Exponential Model')
    
    # Current price line
    current_price = df['Price'].iloc[-1]
    ax1.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.2f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Copper Price ($)')
    ax1.set_title(f'Copper Exponential Model - Fixed Growth Rate ({b:.6f})\nR² = {r2:.6f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot - Log scale
    ax2.scatter(df['years'], df['Price'], alpha=0.6, s=1, color='blue', label='Actual Copper Price')
    ax2.plot(df['years'], y_model, 'r-', linewidth=2, label='Exponential Model')
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.2f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Copper Price ($) - Log Scale')
    ax2.set_title('Log Scale View (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model information
    model_fair_value = y_model[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    # Add info text
    info_text = f"""Model: price = {a:.1f} × exp({b:.6f} × days) + {c:.1f}
R² Score: {r2:.6f}
Current Price: ${current_price:.2f}
Model Fair Value: ${model_fair_value:.2f}
Deviation: {deviation:+.1f}%"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    return fig

def generate_analysis_report(df, a, b, c, r2):
    """Generate a detailed analysis report like the Bitcoin one."""
    
    # Current analysis
    current_price = df['Price'].iloc[-1]
    current_days = df['days'].iloc[-1]
    fair_value = a * np.exp(b * current_days) + c
    deviation = ((current_price - fair_value) / fair_value * 100)
    
    # Forward projections (in days)
    projection_days = [365, 365*3, 365*5, 365*10, 365*15, 365*20]
    projection_years = [1, 3, 5, 10, 15, 20]
    
    # Model-based projections (from fair value)
    model_projections = []
    for days in projection_days:
        future_days = current_days + days
        future_price = a * np.exp(b * future_days) + c
        model_projections.append(future_price)
    
    # Current price-based projections (includes undervaluation benefit)
    current_projections = []
    for days in projection_days:
        future_days = current_days + days
        future_price = a * np.exp(b * future_days) + c
        # Add the undervaluation benefit
        current_projections.append(future_price)
    
    # Create the report
    report = f"""COPPER INTERCEPT OPTIMIZATION RESULTS
==================================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATION PARAMETERS
------------------------------
Fixed Growth Rate: {b:.6f}
Optimal Amplitude: {a:.2f}
Optimal Intercept: {c:.2f}
Maximum R²: {r2:.6f}

MODEL FORMULA
--------------------
price = {a:.2f} × exp({b:.6f} × days) + {c:.2f}

CURRENT ANALYSIS
--------------------
Current Price: ${current_price:,.2f}
Model Fair Price: ${fair_value:,.2f}
Undervaluation: {deviation:+.1f}%

FORWARD PROJECTIONS
-------------------------
Model-Based Projections (from fair value):"""
    
    for i, (years, price) in enumerate(zip(projection_years, model_projections)):
        growth = ((price - fair_value) / fair_value) * 100
        report += f"\n  {years}-Year: ${price:,.0f} | Growth: +{growth:.2f}%"
    
    report += f"""

Current Price-Based Projections (includes undervaluation benefit):"""
    
    for i, (years, price) in enumerate(zip(projection_years, current_projections)):
        return_pct = ((price - current_price) / current_price) * 100
        report += f"\n  {years}-Year: ${price:,.0f} | Return: +{return_pct:.2f}%"
    
    report += f"""

NOTES
----------
- Model-based projections start from the model's fair value
- Current price-based projections start from the current (undervalued) price
- Undervaluation benefit is added to current price-based returns
- Growth rate is fixed at {b:.6f} (0.000150 as requested)
"""
    
    return report

def main():
    print("Copper Fixed Growth Rate Model")
    print("=" * 50)
    
    # Load data
    df = load_copper_data()
    if df is None:
        return
    
    # Fixed growth rate
    fixed_growth_rate = 0.000150
    
    print(f"\nUsing fixed growth rate: {fixed_growth_rate}")
    print("Finding optimal amplitude and intercept to maximize R²...")
    
    # Find optimal parameters
    a, b, c, r2 = find_optimal_parameters(df, fixed_growth_rate)
    
    print(f"\nModel Parameters:")
    print(f"  A (amplitude): {a:.2f} (optimized)")
    print(f"  B (growth rate): {b:.6f} (fixed)")
    print(f"  C (intercept): {c:.2f} (optimized)")
    print(f"  R² score: {r2:.6f}")
    
    # Create chart
    print("\nCreating chart...")
    fig = create_fixed_growth_chart(df, a, b, c, r2)
    
    # Save the chart
    output_dir = "Commodities/Copper/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = os.path.join(output_dir, "copper_fixed_growth_rate.png")
    fig.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {chart_file}")
    
    # Generate and save analysis report
    print("\nGenerating analysis report...")
    report = generate_analysis_report(df, a, b, c, r2)
    
    report_file = os.path.join(output_dir, "copper_intercept_optimization.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Analysis report saved: {report_file}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "=" * 50)
    print("FIXED GROWTH RATE MODEL COMPLETE")
    print("=" * 50)
    print(f"Growth rate fixed at: {b:.6f}")
    print(f"Optimal intercept: {c:.2f}")
    print(f"Model fit (R²): {r2:.6f}")
    
    # Calculate current analysis
    current_price = df['Price'].iloc[-1]
    current_days = df['days'].iloc[-1]
    fair_value = a * np.exp(b * current_days) + c
    deviation = ((current_price - fair_value) / fair_value * 100)
    
    print(f"\nCurrent Analysis:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Model Fair Value: ${fair_value:.2f}")
    print(f"Deviation: {deviation:+.1f}%")

if __name__ == "__main__":
    main()
