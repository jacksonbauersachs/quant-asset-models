import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load the complete Bitcoin dataset"""
    print("Loading complete Bitcoin dataset...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    return df

def calculate_365d_volatility(df):
    """Calculate 365-day rolling volatility"""
    print("\nCalculating 365-day rolling volatility...")
    
    # Calculate time metrics
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Years'] = df['Days'] / 365.25
    
    # Calculate returns and 365-day volatility
    df['Returns'] = df['Price'].pct_change()
    df['Volatility_365d'] = df['Returns'].rolling(365).std() * np.sqrt(365)
    
    # Remove NaN values
    clean_data = df[['Years', 'Volatility_365d']].dropna()
    
    print(f"Using {len(clean_data):,} days for volatility analysis")
    print(f"Volatility range: {clean_data['Volatility_365d'].min():.4f} to {clean_data['Volatility_365d'].max():.4f}")
    print(f"Mean volatility: {clean_data['Volatility_365d'].mean():.4f}")
    
    return clean_data

def fit_exponential_decay(x, y):
    """Fit exponential decay model: volatility = a * exp(-b * years) + c"""
    def exponential_model(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Initial guess
    p0 = [np.max(y), 0.1, np.min(y)]
    params, _ = curve_fit(exponential_model, x, y, p0=p0, maxfev=10000)
    predictions = exponential_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def create_exponential_analysis(years, volatility, params, r_squared):
    """Create exponential decay analysis graph"""
    print("\n" + "="*60)
    print("EXPONENTIAL DECAY VOLATILITY ANALYSIS")
    print("="*60)
    
    # Set up the plot
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin 365-Day Volatility: Exponential Decay Model', fontsize=16, fontweight='bold')
    
    # Get model function
    a, b, c = params
    model_func = lambda x: a * np.exp(-b * x) + c
    
    # Generate model predictions
    years_range = np.linspace(years.min(), years.max(), 100)
    model_predictions = model_func(years_range)
    
    # Plot 1: Raw volatility data with exponential fit
    ax1.scatter(years, volatility, alpha=0.6, s=10, color='blue', label='Actual Volatility')
    ax1.plot(years_range, model_predictions, 'r-', linewidth=2, label='Exponential Decay Fit')
    ax1.set_xlabel('Years Since Start')
    ax1.set_ylabel('365-Day Annualized Volatility')
    ax1.set_title(f'Exponential Decay Model\nR² = {r_squared:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility distribution
    ax2.hist(volatility, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(volatility.mean(), color='red', linestyle='--', label=f'Mean: {volatility.mean():.3f}')
    ax2.axvline(volatility.median(), color='green', linestyle='--', label=f'Median: {volatility.median():.3f}')
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Volatility Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility over time (log scale)
    ax3.scatter(years, volatility, alpha=0.6, s=10, color='blue')
    ax3.plot(years_range, model_predictions, 'r-', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Years Since Start')
    ax3.set_ylabel('Volatility (Log Scale)')
    ax3.set_title('Volatility vs Time (Log Scale)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Future predictions
    current_years = years.max()
    future_years = np.linspace(current_years, current_years + 10, 100)
    future_predictions = model_func(future_years)
    
    # Ensure predictions are reasonable
    future_predictions = np.maximum(future_predictions, 0.05)
    future_predictions = np.minimum(future_predictions, 2.0)
    
    ax4.scatter(years, volatility, alpha=0.6, s=10, color='blue', label='Historical')
    ax4.plot(years_range, model_predictions, 'r-', linewidth=2, label='Model Fit')
    ax4.plot(future_years, future_predictions, 'g--', linewidth=2, label='Future Predictions')
    ax4.axvline(x=current_years, color='black', linestyle=':', label='Current Time')
    ax4.set_xlabel('Years Since Start')
    ax4.set_ylabel('Volatility')
    ax4.set_title('Volatility with Future Predictions (10 Years)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'Results/Bitcoin/bitcoin_exponential_volatility_analysis_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {filename}")
    
    plt.show()
    
    return future_years, future_predictions

def analyze_exponential_model(params, r_squared, future_years, future_predictions):
    """Analyze the exponential decay model"""
    a, b, c = params
    current_years = future_years[0]
    
    print("\n" + "="*60)
    print("EXPONENTIAL DECAY MODEL ANALYSIS")
    print("="*60)
    print(f"Model: volatility = {a:.6f} * exp(-{b:.6f} * years) + {c:.6f}")
    print(f"R² = {r_squared:.6f}")
    print(f"\nParameters:")
    print(f"  a (amplitude) = {a:.6f}")
    print(f"  b (decay rate) = {b:.6f}")
    print(f"  c (asymptote) = {c:.6f}")
    
    print(f"\nModel Interpretation:")
    print(f"  - Initial volatility: {a + c:.1%}")
    print(f"  - Long-term volatility: {c:.1%}")
    print(f"  - Half-life: {np.log(2) / b:.2f} years")
    print(f"  - Decay rate: {b:.4f} per year")
    
    print(f"\nFuture Volatility Predictions:")
    for i, (year, pred) in enumerate(zip(future_years[::10], future_predictions[::10])):
        years_ahead = year - current_years
        if years_ahead <= 10:
            print(f"  Year {years_ahead:.1f} from now: {pred:.1%}")
    
    # Check if this makes economic sense
    print(f"\nEconomic Sense Check:")
    if c > 0 and b > 0:
        print(f"  ✅ Volatility decreases over time (decay rate = {b:.4f})")
        print(f"  ✅ Approaches a stable long-term level of {c:.1%}")
        if c < 0.5:  # Less than 50% long-term volatility
            print(f"  ✅ Long-term volatility is reasonable ({c:.1%})")
        else:
            print(f"  ⚠️  Long-term volatility seems high ({c:.1%})")
    else:
        print(f"  ❌ Model parameters don't make economic sense")
    
    return future_years, future_predictions

def save_exponential_results(params, r_squared, future_years, future_predictions):
    """Save exponential decay results"""
    filename = f'Models/Volatility Models/bitcoin_exponential_volatility_results_{datetime.now().strftime("%Y%m%d")}.txt'
    
    a, b, c = params
    current_years = future_years[0]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BITCOIN 365-DAY VOLATILITY: EXPONENTIAL DECAY MODEL\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Bitcoin_Final_Complete_Data_20250719.csv\n")
        f.write("\n")
        
        f.write("-"*40 + "\n")
        f.write("MODEL RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Model Type: Exponential Decay\n")
        f.write(f"R² = {r_squared:.6f}\n")
        f.write(f"Formula: volatility = {a:.6f} * exp(-{b:.6f} * years) + {c:.6f}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  a (amplitude) = {a:.6f}\n")
        f.write(f"  b (decay rate) = {b:.6f}\n")
        f.write(f"  c (asymptote) = {c:.6f}\n")
        f.write(f"\nModel Interpretation:\n")
        f.write(f"  Initial volatility: {a + c:.1%}\n")
        f.write(f"  Long-term volatility: {c:.1%}\n")
        f.write(f"  Half-life: {np.log(2) / b:.2f} years\n")
        f.write(f"  Decay rate: {b:.4f} per year\n")
        
        f.write("\n")
        f.write("-"*40 + "\n")
        f.write("FUTURE PREDICTIONS\n")
        f.write("-"*40 + "\n")
        for i, (year, pred) in enumerate(zip(future_years[::10], future_predictions[::10])):
            years_ahead = year - current_years
            if years_ahead <= 10:
                f.write(f"Year {years_ahead:.1f} from now: {pred:.1%}\n")
        
        f.write("\n")
        f.write("="*60 + "\n")
    
    print(f"Results saved to: {filename}")
    return filename

def main():
    """Main function to analyze exponential decay model"""
    print("="*60)
    print("BITCOIN EXPONENTIAL DECAY VOLATILITY ANALYSIS")
    print("="*60)
    print("Analyzing the second-best model (Exponential Decay)")
    print("to see if it makes more economic sense than Polynomial.")
    print("="*60)
    
    # Load data
    df = load_bitcoin_data()
    
    # Calculate 365-day volatility
    clean_data = calculate_365d_volatility(df)
    
    # Fit exponential decay model
    params, r_squared = fit_exponential_decay(
        clean_data['Years'], clean_data['Volatility_365d']
    )
    
    print(f"\nExponential Decay Model:")
    print(f"R² = {r_squared:.4f}")
    print(f"Parameters: {params}")
    
    # Create graph
    future_years, future_predictions = create_exponential_analysis(
        clean_data['Years'], clean_data['Volatility_365d'], 
        params, r_squared
    )
    
    # Analyze model
    analyze_exponential_model(params, r_squared, future_years, future_predictions)
    
    # Save results
    filename = save_exponential_results(params, r_squared, future_years, future_predictions)
    
    print(f"\nResults file: {filename}")
    print("\nThis exponential decay model may be more economically sensible!")

if __name__ == "__main__":
    main() 