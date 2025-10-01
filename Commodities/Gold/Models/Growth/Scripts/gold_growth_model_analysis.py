import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare gold price data for analysis."""
    
    # Load the data
    data_file = "Gold/Data/Gold Price 1_3_75 to 8_6_25.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - remove quotes and commas, convert to float
    df['Price'] = df['Price'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def linear_model(x, a, b):
    """Linear model: y = ax + b"""
    return a * x + b

def exponential_model(x, a, b, c):
    """Exponential model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def polynomial_model(x, a, b, c, d):
    """Cubic polynomial model: y = ax³ + bx² + cx + d"""
    return a * x**3 + b * x**2 + c * x + d

def logarithmic_model(x, a, b, c):
    """Logarithmic model: y = a * ln(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c

def power_model(x, a, b, c):
    """Power model: y = a * x^b + c"""
    return a * np.power(x, b) + c

def fit_models(df):
    """Fit different growth models to the data."""
    
    x = df['Days'].values
    y = df['Price'].values
    
    models = {}
    results = {}
    
    # Linear model
    try:
        popt_linear, _ = curve_fit(linear_model, x, y, p0=[0.1, 100])
        y_pred_linear = linear_model(x, *popt_linear)
        r2_linear = r2_score(y, y_pred_linear)
        rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
        
        models['Linear'] = {
            'function': linear_model,
            'params': popt_linear,
            'predictions': y_pred_linear,
            'r2': r2_linear,
            'rmse': rmse_linear,
            'formula': f"Price = {popt_linear[0]:.6f} * Days + {popt_linear[1]:.2f}"
        }
    except:
        print("Warning: Could not fit linear model")
    
    # Exponential model
    try:
        popt_exp, _ = curve_fit(exponential_model, x, y, p0=[100, 0.001, 100])
        y_pred_exp = exponential_model(x, *popt_exp)
        r2_exp = r2_score(y, y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
        
        models['Exponential'] = {
            'function': exponential_model,
            'params': popt_exp,
            'predictions': y_pred_exp,
            'r2': r2_exp,
            'rmse': rmse_exp,
            'formula': f"Price = {popt_exp[0]:.2f} * exp({popt_exp[1]:.6f} * Days) + {popt_exp[2]:.2f}"
        }
    except:
        print("Warning: Could not fit exponential model")
    
    # Polynomial model (cubic)
    try:
        popt_poly, _ = curve_fit(polynomial_model, x, y, p0=[1e-8, 1e-5, 0.1, 100])
        y_pred_poly = polynomial_model(x, *popt_poly)
        r2_poly = r2_score(y, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
        
        models['Polynomial'] = {
            'function': polynomial_model,
            'params': popt_poly,
            'predictions': y_pred_poly,
            'r2': r2_poly,
            'rmse': rmse_poly,
            'formula': f"Price = {popt_poly[0]:.2e}*Days³ + {popt_poly[1]:.2e}*Days² + {popt_poly[2]:.4f}*Days + {popt_poly[3]:.2f}"
        }
    except:
        print("Warning: Could not fit polynomial model")
    
    # Logarithmic model
    try:
        popt_log, _ = curve_fit(logarithmic_model, x, y, p0=[100, 0.01, 100])
        y_pred_log = logarithmic_model(x, *popt_log)
        r2_log = r2_score(y, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
        
        models['Logarithmic'] = {
            'function': logarithmic_model,
            'params': popt_log,
            'predictions': y_pred_log,
            'r2': r2_log,
            'rmse': rmse_log,
            'formula': f"Price = {popt_log[0]:.2f} * ln({popt_log[1]:.4f} * Days + 1) + {popt_log[2]:.2f}"
        }
    except:
        print("Warning: Could not fit logarithmic model")
    
    # Power model
    try:
        popt_power, _ = curve_fit(power_model, x, y, p0=[1, 0.5, 100])
        y_pred_power = power_model(x, *popt_power)
        r2_power = r2_score(y, y_pred_power)
        rmse_power = np.sqrt(mean_squared_error(y, y_pred_power))
        
        models['Power'] = {
            'function': power_model,
            'params': popt_power,
            'predictions': y_pred_power,
            'r2': r2_power,
            'rmse': rmse_power,
            'formula': f"Price = {popt_power[0]:.2f} * Days^{popt_power[1]:.4f} + {popt_power[2]:.2f}"
        }
    except:
        print("Warning: Could not fit power model")
    
    return models

def create_visualization(df, models):
    """Create comprehensive visualization of all models."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: All models comparison
    ax1.scatter(df['Date'], df['Price'], alpha=0.3, s=1, color='black', label='Actual Data')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, model_data) in enumerate(models.items()):
        ax1.plot(df['Date'], model_data['predictions'], 
                color=colors[i], linewidth=2, label=f'{model_name} (R²={model_data["r2"]:.3f})')
    
    ax1.set_title('Gold Price Growth Models Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Gold Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model performance comparison
    model_names = list(models.keys())
    r2_scores = [models[name]['r2'] for name in model_names]
    rmse_scores = [models[name]['rmse'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars1 = ax2.bar(x_pos - 0.2, r2_scores, 0.4, label='R² Score', color='skyblue')
    bars2 = ax2.bar(x_pos + 0.2, [1/r for r in rmse_scores], 0.4, label='1/RMSE (scaled)', color='lightcoral')
    
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 3: Residuals analysis
    best_model = max(models.items(), key=lambda x: x[1]['r2'])
    residuals = df['Price'] - best_model[1]['predictions']
    
    ax3.scatter(df['Date'], residuals, alpha=0.5, s=1)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_title(f'Residuals for Best Model: {best_model[0]}', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residuals ($)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price vs Days scatter with best fit
    ax4.scatter(df['Days'], df['Price'], alpha=0.3, s=1, color='black', label='Actual Data')
    ax4.plot(df['Days'], best_model[1]['predictions'], 
             color='red', linewidth=2, label=f'{best_model[0]} Fit')
    ax4.set_title(f'Best Model: {best_model[0]} (R²={best_model[1]["r2"]:.3f})', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Days Since Start')
    ax4.set_ylabel('Gold Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"Gold/Models/Growth/Results/gold_growth_models_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_file}")
    
    return plot_file

def save_results(models, df):
    """Save detailed results to text file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"Gold/Models/Growth/Results/gold_growth_models_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("GOLD PRICE GROWTH MODEL ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Data Summary:\n")
        f.write(f"- Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- Total Days: {df['Days'].max()} days\n")
        f.write(f"- Price Range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}\n")
        f.write(f"- Total Records: {len(df):,}\n\n")
        
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("-" * 30 + "\n\n")
        
        # Sort models by R² score
        sorted_models = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        for i, (model_name, model_data) in enumerate(sorted_models, 1):
            f.write(f"{i}. {model_name.upper()} MODEL\n")
            f.write(f"   R² Score: {model_data['r2']:.6f}\n")
            f.write(f"   RMSE: {model_data['rmse']:.2f}\n")
            f.write(f"   Formula: {model_data['formula']}\n")
            f.write(f"   Parameters: {model_data['params']}\n\n")
        
        # Best model analysis
        best_model = sorted_models[0]
        f.write("BEST MODEL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best Model: {best_model[0]}\n")
        f.write(f"R² Score: {best_model[1]['r2']:.6f}\n")
        f.write(f"RMSE: {best_model[1]['rmse']:.2f}\n")
        f.write(f"Formula: {best_model[1]['formula']}\n\n")
        
        # Future predictions (next 10 years)
        future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 3650, 365)
        future_predictions = best_model[1]['function'](future_days, *best_model[1]['params'])
        
        f.write("FUTURE PREDICTIONS (Next 10 Years)\n")
        f.write("-" * 35 + "\n")
        for i, (days, pred) in enumerate(zip(future_days, future_predictions)):
            future_date = df['Date'].min() + pd.Timedelta(days=int(days))
            f.write(f"Year {i+1}: {future_date.strftime('%Y')} - ${pred:.2f}\n")
    
    print(f"Results saved: {results_file}")
    return results_file

def main():
    """Main function to run the complete analysis."""
    
    print("Gold Price Growth Model Analysis")
    print("=" * 40)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Fit models
    print("\nFitting growth models...")
    models = fit_models(df)
    
    if not models:
        print("Error: No models could be fitted successfully.")
        return
    
    # Create visualization
    print("\nCreating visualizations...")
    plot_file = create_visualization(df, models)
    
    # Save results
    print("\nSaving results...")
    results_file = save_results(models, df)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    
    best_model = max(models.items(), key=lambda x: x[1]['r2'])
    print(f"Best Model: {best_model[0]}")
    print(f"R² Score: {best_model[1]['r2']:.6f}")
    print(f"RMSE: {best_model[1]['rmse']:.2f}")
    print(f"Formula: {best_model[1]['formula']}")
    
    print(f"\nFiles created:")
    print(f"- Plot: {plot_file}")
    print(f"- Results: {results_file}")

if __name__ == "__main__":
    main() 