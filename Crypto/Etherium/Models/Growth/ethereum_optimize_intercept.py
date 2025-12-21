import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from datetime import datetime

def load_ethereum_data():
    """Load and prepare Ethereum price data for analysis."""
    
    data_file = "Crypto/Etherium/Data/Ethereum Historical Data.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Ethereum data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since Ethereum genesis (July 30, 2015) for rainbow chart model
    ethereum_genesis = pd.Timestamp('2015-07-30')
    df['Days_Since_Genesis'] = (df['Date'] - ethereum_genesis).dt.days
    
    # Start from May 18, 2017 (specific date requested)
    start_date = pd.Timestamp('2017-05-18')
    df_filtered = df[df['Date'] >= start_date].copy()
    
    print(f"Ethereum data loaded: {len(df_filtered)} records from {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df_filtered['Price'].min():.2f} to ${df_filtered['Price'].max():,.2f}")
    
    return df_filtered

def ethereum_rainbow_model(x, slope, intercept):
    """Ethereum rainbow model: price = 10^(slope * ln(days_since_genesis) + intercept)"""
    return 10**(slope * np.log(x) + intercept)

def calculate_r2_log_space(y_actual, y_pred):
    """Calculate R² score in log space (like the original Bitcoin model)."""
    # Convert to log space for R² calculation (matching original model)
    log_actual = np.log10(y_actual)
    log_pred = np.log10(y_pred)
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def objective_function(intercept):
    """Objective function to minimize (negative R² to maximize R²)."""
    global df_filtered, fixed_slope
    
    # Calculate model predictions
    y_pred = ethereum_rainbow_model(df_filtered['Days_Since_Genesis'], fixed_slope, intercept)
    
    # Calculate R² in log space
    r2 = calculate_r2_log_space(df_filtered['Price'], y_pred)
    
    # Return negative R² (minimize negative = maximize positive)
    return -r2

def optimize_intercept_for_r2(df, slope):
    """Find the optimal intercept that maximizes R² for a given fixed slope."""
    global df_filtered, fixed_slope
    df_filtered = df
    fixed_slope = slope
    
    print(f"Optimizing intercept for fixed slope: {slope}")
    print("Searching for optimal intercept...")
    
    # Use minimize_scalar to find optimal intercept
    result = minimize_scalar(objective_function, bounds=(-30, 0), method='bounded')
    
    optimal_intercept = result.x
    optimal_r2 = -result.fun  # Convert back to positive R²
    
    print(f"Optimal intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    
    return optimal_intercept, optimal_r2

def create_optimization_chart(df, slope, optimal_intercept, optimal_r2):
    """Create a chart showing the optimization results."""
    
    # Calculate model predictions with optimal parameters
    df['Model_Price'] = df['Days_Since_Genesis'].apply(lambda x: ethereum_rainbow_model(x, slope, optimal_intercept))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual Ethereum price on both subplots
    ax1.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Ethereum Price')
    ax2.plot(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Ethereum Price')
    
    # Plot optimized model line
    ax1.semilogy(df['Date'], df['Model_Price'], 
                color='red', linestyle='--', linewidth=2, label='Optimized Model')
    ax2.plot(df['Date'], df['Model_Price'], 
            color='red', linestyle='--', linewidth=2, label='Optimized Model')
    
    # Highlight current position
    current_price = df['Price'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    ax1.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    ax2.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    
    # Formatting for both subplots
    for ax, ylabel_suffix in [(ax1, 'Log Scale'), (ax2, 'Normal Scale')]:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'Ethereum Price (USD) - {ylabel_suffix}', fontsize=12)
        ax.set_title(f'Ethereum Optimized Model - {ylabel_suffix}\nSlope: {slope:.6f}, Intercept: {optimal_intercept:.6f}, R²: {optimal_r2:.6f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Custom y-axis formatting
        def price_formatter(x, pos):
            if x >= 1000000:
                return f'${x/1000000:.1f}M'
            elif x >= 1000:
                return f'${x/1000:.0f}K'
            else:
                return f'${x:.0f}'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
        
        # Add legend
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set y-axis limits - reasonable for normal scale, full range for log scale
        if ax == ax2:  # Normal scale - use reasonable limits based on actual data
            max_price = max(df['Price'].max(), df['Model_Price'].max())
            # Cap at 2x the maximum actual price to keep it reasonable
            ax.set_ylim(0, min(max_price * 2, df['Price'].max() * 3))
        else:  # Log scale
            ax.set_ylim(0.01, max(df['Price'].max(), df['Model_Price'].max()) * 2)
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = "Crypto/Etherium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = os.path.join(output_dir, "ethereum_intercept_optimization.png")
    fig.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return chart_file

def calculate_projections(df, slope, intercept):
    """Calculate projections for 1, 3, 5, 10, 15, and 20 years."""
    
    current_price = df['Price'].iloc[-1]
    current_days = df['Days_Since_Genesis'].iloc[-1]
    
    # Calculate current fair value
    current_fair_price = ethereum_rainbow_model(current_days, slope, intercept)
    
    # Calculate undervaluation
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    horizons = [1, 3, 5, 10, 15, 20]
    projections = {}
    
    for years in horizons:
        future_days = current_days + (years * 365.25)
        model_future_price = ethereum_rainbow_model(future_days, slope, intercept)
        
        # Model-based projection (from fair value)
        model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
        
        # Current price-based projection (includes undervaluation benefit)
        if undervaluation < 0:  # If undervalued
            undervaluation_benefit = abs(undervaluation)
            current_return = model_return + undervaluation_benefit
        else:  # If overvalued
            current_return = model_return - abs(undervaluation)
        
        current_based_price = current_price * (1 + current_return / 100)
        
        projections[years] = {
            'model_price': model_future_price,
            'current_based_price': current_based_price,
            'model_return': model_return,
            'current_return': current_return
        }
    
    return {
        'current_price': current_price,
        'current_fair_price': current_fair_price,
        'undervaluation': undervaluation,
        'projections': projections
    }

def save_optimization_results(slope, optimal_intercept, optimal_r2, projections):
    """Save optimization results to a text file."""
    
    output_dir = "Crypto/Etherium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "ethereum_intercept_optimization.txt")
    
    with open(results_file, 'w') as f:
        f.write("ETHEREUM INTERCEPT OPTIMIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OPTIMIZATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fixed Slope: {slope:.6f}\n")
        f.write(f"Optimal Intercept: {optimal_intercept:.6f}\n")
        f.write(f"Maximum R²: {optimal_r2:.6f}\n\n")
        
        f.write("MODEL FORMULA\n")
        f.write("-" * 20 + "\n")
        f.write(f"price = 10^({slope:.6f} * ln(days_since_genesis) + {optimal_intercept:.6f})\n\n")
        
        f.write("CURRENT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Current Price: ${projections['current_price']:,.2f}\n")
        f.write(f"Model Fair Price: ${projections['current_fair_price']:,.2f}\n")
        f.write(f"Undervaluation: {projections['undervaluation']:+.2f}%\n\n")
        
        f.write("FORWARD PROJECTIONS\n")
        f.write("-" * 25 + "\n")
        f.write("Model-Based Projections (from fair value):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['model_price']:,.0f} | Growth: {p['model_return']:+.2f}%\n")
        
        f.write("\nCurrent Price-Based Projections (includes undervaluation benefit):\n")
        for years in [1, 3, 5, 10, 15, 20]:
            p = projections['projections'][years]
            f.write(f"  {years}-Year: ${p['current_based_price']:,.0f} | Return: {p['current_return']:+.2f}%\n")
        
        f.write("\nNOTES\n")
        f.write("-" * 10 + "\n")
        f.write("- Model-based projections start from the model's fair value\n")
        f.write("- Current price-based projections start from the current (undervalued) price\n")
        f.write("- Undervaluation benefit is added to current price-based returns\n")
    
    print(f"Optimization results saved to: {results_file}")
    return results_file

def main():
    """Main function to optimize Ethereum intercept."""
    
    print("Ethereum Intercept Optimization")
    print("=" * 40)
    
    # Load data
    df = load_ethereum_data()
    if df is None:
        return
    
    # Fixed slope (user specified - adjust as needed)
    fixed_slope = .717579
    
    # Calculate intercept to start at first day's price
    print(f"\nUsing fixed slope: {fixed_slope}")
    print("Intercept will be calculated so model starts at first day's price")
    optimal_intercept, optimal_r2 = optimize_intercept_for_r2(df, fixed_slope)
    
    # Create optimization chart
    print("\nCreating optimization chart...")
    chart_file = create_optimization_chart(df, fixed_slope, optimal_intercept, optimal_r2)
    print(f"Chart saved: {chart_file}")
    
    # Calculate projections
    print("\nCalculating forward projections...")
    projections = calculate_projections(df, fixed_slope, optimal_intercept)
    
    # Save results
    results_file = save_optimization_results(fixed_slope, optimal_intercept, optimal_r2, projections)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Fixed Slope: {fixed_slope:.6f}")
    print(f"Optimal Intercept: {optimal_intercept:.6f}")
    print(f"Maximum R²: {optimal_r2:.6f}")
    print(f"Current Price: ${projections['current_price']:,.2f}")
    print(f"Model Fair Price: ${projections['current_fair_price']:,.2f}")
    print(f"Undervaluation: {projections['undervaluation']:+.2f}%")
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
