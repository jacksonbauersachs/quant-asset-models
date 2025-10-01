import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def fit_exponential_model(df):
    """Fit an exponential model to the data."""
    
    x = df['days'].values
    y = df[' value'].values
    
    # Exponential model
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(exponential_func, x, y, p0=[26, 0.0002, 65], maxfev=10000)
        a, b, c = popt
        
        # Calculate R²
        y_pred = exponential_func(x, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        model = {
            'name': 'Exponential',
            'params': popt,
            'r2': r2,
            'formula': f'y = {a:.4f} * e^({b:.6f} * x) + {c:.2f}'
        }
        
        return model
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None

def create_linear_growth_chart(df, model, output_dir):
    """Create a chart where exponential growth appears linear."""
    
    if not model:
        print("No model available for charting")
        return None
    
    # Get model predictions
    x = df['days'].values
    y = df[' value'].values
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    model_predictions = exponential_func(x, *model['params'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale (exponential growth)
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax1.plot(df['years'], model_predictions, 'r-', linewidth=2, label=f'{model["name"]} Model')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Price - Normal Scale (Exponential Growth)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Log scale (linear growth appearance)
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax2.plot(df['years'], model_predictions, 'r-', linewidth=2, label=f'{model["name"]} Model')
    
    # Set log scale to make exponential growth appear linear
    ax2.set_yscale('log')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Palladium Price - Log Scale (Linear Growth Appearance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f'Model: {model["name"]}\nFormula: {model["formula"]}\nR² = {model["r2"]:.4f}'
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add explanation text
    explanation_text = """Log Scale Explanation:
- Exponential growth appears as a straight line
- Each vertical unit represents a 10x price increase
- Makes long-term trends easier to see
- Shows relative growth rates clearly"""
    
    ax1.text(0.02, 0.98, explanation_text, transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(facecolor='lightblue', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_linear_growth_chart.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def main():
    print("Palladium Linear Growth Chart")
    print("=" * 40)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit exponential model
    print("\nFitting exponential model...")
    model = fit_exponential_model(df)
    if model is None:
        print("Failed to fit model")
        return
    
    print(f"Model: {model['name']}")
    print(f"Formula: {model['formula']}")
    print(f"R² Score: {model['r2']:.4f}")
    
    # Create chart
    print("\nCreating linear growth chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_linear_growth_chart(df, model, output_dir)
    if chart_file:
        print(f"Linear growth chart saved to: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_linear_growth_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM LINEAR GROWTH CHART ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Chart Explanation:\n")
        f.write("- Top chart: Normal scale showing exponential growth curve\n")
        f.write("- Bottom chart: Log scale making exponential growth appear linear\n")
        f.write("- Log scale transforms exponential curves into straight lines\n")
        f.write("- Makes it easier to see long-term trends and growth rates\n\n")
        
        f.write("Model Details:\n")
        f.write(f"  Name: {model['name']}\n")
        f.write(f"  Formula: {model['formula']}\n")
        f.write(f"  R² Score: {model['r2']:.6f}\n\n")
        
        f.write("Why Log Scale Works:\n")
        f.write("- Exponential growth: y = a * e^(b * x)\n")
        f.write("- Taking log: log(y) = log(a) + b * x\n")
        f.write("- This becomes a straight line: log(y) = mx + b\n")
        f.write("- So exponential curves appear linear on log scale\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 40)
    print("LINEAR GROWTH CHART COMPLETE")
    print("=" * 40)
    print("The bottom chart shows palladium's exponential growth as a straight line!")
    print("This makes it easier to see the long-term trend and growth rate.")

if __name__ == "__main__":
    main()
