import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_palladium_data():
    """Load and prepare palladium price data for analysis."""
    
    data_file = "Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Palladium data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['days'] = (df['date'] - df['date'].min()).dt.days
    
    # Create years column for x-axis
    start_year = df['date'].min().year
    df['years'] = (df['days'] / 365.25) + start_year
    
    print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
    
    return df

def fit_palladium_growth_models(df):
    """Fit growth models to palladium data."""
    
    x = df['days'].values
    y = df[' value'].values
    
    models = {}
    
    # Power model
    try:
        popt_power, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x, y, p0=[1, 0.5, 100])
        y_pred_power = popt_power[0] * np.power(x, popt_power[1]) + popt_power[2]
        r2_power = r2_score(y, y_pred_power)
        rmse_power = np.sqrt(mean_squared_error(y, y_pred_power))
        
        models['Power'] = {
            'formula': f"y = {popt_power[0]:.4f} * x^{popt_power[1]:.4f} + {popt_power[2]:.2f}",
            'params': popt_power,
            'r2': r2_power,
            'rmse': rmse_power
        }
    except:
        print("Warning: Could not fit Power model")
    
    # Exponential model
    try:
        popt_exp, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x, y, p0=[100, 0.0001, 100])
        y_pred_exp = popt_exp[0] * np.exp(popt_exp[1] * x) + popt_exp[2]
        r2_exp = r2_score(y, y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
        
        models['Exponential'] = {
            'formula': f"y = {popt_exp[0]:.4f} * e^({popt_exp[1]:.6f} * x) + {popt_exp[2]:.2f}",
            'params': popt_exp,
            'r2': r2_exp,
            'rmse': rmse_exp
        }
    except:
        print("Warning: Could not fit Exponential model")
    
    # Select best model - prefer Exponential if available
    if models:
        if 'Exponential' in models:
            best_model_name = 'Exponential'
            best_model = models['Exponential']
        else:
            best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
            best_model = models[best_model_name]
        
        models['Best_Model'] = {
            'name': best_model_name,
            'formula': best_model['formula'],
            'r2': best_model['r2'],
            'rmse': best_model['rmse'],
            'params': best_model['params']
        }
        
        print(f"Best Model: {best_model_name} (R² = {best_model['r2']:.4f})")
    
    return models

def create_detrended_chart(df, models, output_dir):
    """Create detrended charts with model-adjusted y-axis scale."""
    
    if not models or 'Best_Model' not in models:
        print("No models available for charting")
        return None
    
    best_model = models['Best_Model']
    best_model_name = best_model['name']
    best_params = best_model['params']
    
    # Calculate model predictions
    x = df['days'].values
    if best_model_name == 'Power':
        model_predictions = best_params[0] * np.power(x, best_params[1]) + best_params[2]
    elif best_model_name == 'Exponential':
        model_predictions = best_params[0] * np.exp(best_params[1] * x) + best_params[2]
    
    # Create model-adjusted y-axis transformation
    # This makes the model appear as a straight diagonal line (up and to the right)
    if best_model_name == 'Exponential':
        # For exponential: y_adjusted = actual / exp(model_growth_rate * time)
        # This makes the exponential model appear as a straight diagonal line
        growth_rate = best_params[1]  # The exponential growth rate
        y_adjusted = df[' value'].values / np.exp(growth_rate * x)
        y_label = "Price Adjusted for Model Growth Rate"
        y_title_suffix = "Model-Adjusted Exponential Scale"
        
        # The adjusted model should be constant (the base value)
        model_adjusted = np.full_like(y_adjusted, best_params[0])
    else:
        # For power: use a simpler approach - just divide by the model
        y_adjusted = df[' value'].values / model_predictions
        y_label = "Price Ratio to Model"
        y_title_suffix = "Model-Adjusted Ratio Scale"
        
        # The adjusted model should be constant at 1
        model_adjusted = np.ones_like(y_adjusted)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Model-adjusted view (model appears as straight horizontal line)
    ax1.plot(df['years'], y_adjusted, 'b-', linewidth=1, alpha=0.7, label='Actual Data (Model-Adjusted)')
    ax1.plot(df['years'], model_adjusted, 'r--', linewidth=2, label=f'{best_model_name} Model (Adjusted)')
    
    # Highlight current position
    current_y = y_adjusted[-1]
    current_year = df['years'].iloc[-1]
    ax1.scatter(current_year, current_y, color='red', s=100, zorder=10, label=f'Current: {current_y:.1f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel(y_label)
    ax1.set_title(f'Palladium Price - {y_title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f'Best Model: {best_model_name}\nFormula: {best_model["formula"]}\nR² = {best_model["r2"]:.4f}'
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Chart 2: Traditional view for comparison
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax2.plot(df['years'], model_predictions, 'r-', linewidth=2, label=f'{best_model_name} Model')
    
    # Highlight current position
    current_price = df[' value'].iloc[-1]
    current_model_price = model_predictions[-1]
    ax2.scatter(current_year, current_price, color='red', s=100, zorder=10, label=f'Current: ${current_price:.0f}')
    ax2.scatter(current_year, current_model_price, color='orange', s=100, zorder=10, label=f'Model: ${current_model_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($)')
    ax2.set_title('Palladium Price - Traditional View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_detrended_chart.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def analyze_deviations(df, models):
    """Analyze deviations from the model to identify overfitting."""
    
    if not models or 'Best_Model' not in models:
        return None
    
    best_model = models['Best_Model']
    best_params = best_model['params']
    
    # Calculate model predictions
    x = df['days'].values
    if best_model['name'] == 'Power':
        model_predictions = best_params[0] * np.power(x, best_params[1]) + best_params[2]
    elif best_model['name'] == 'Exponential':
        model_predictions = best_params[0] * np.exp(best_params[1] * x) + best_params[2]
    
    # Calculate deviations using the same adjustment as the chart
    if best_model['name'] == 'Exponential':
        growth_rate = best_params[1]
        deviations = df[' value'].values / np.exp(growth_rate * x)
        deviation_type = "Model-Adjusted Price"
    else:
        power = best_params[1]
        deviations = df[' value'].values / (best_params[0] * np.power(x, power))
        deviation_type = "Model-Adjusted Price"
    
    # Analyze by time periods
    df['deviations'] = deviations
    df['decade'] = (df['years'] // 10) * 10
    
    decade_stats = df.groupby('decade')['deviations'].agg(['mean', 'std', 'min', 'max']).round(3)
    
    print(f"\n=== DEVIATION ANALYSIS ({deviation_type}) ===")
    print("Decade-by-decade deviations from model:")
    print(decade_stats)
    
    # Recent vs historical comparison
    recent_cutoff = 2020
    recent_data = df[df['years'] >= recent_cutoff]
    historical_data = df[df['years'] < recent_cutoff]
    
    if len(recent_data) > 0 and len(historical_data) > 0:
        recent_mean_dev = recent_data['deviations'].mean()
        historical_mean_dev = historical_data['deviations'].mean()
        
        print(f"\nRecent period (2020+): Mean deviation = {recent_mean_dev:.3f}")
        print(f"Historical period (<2020): Mean deviation = {historical_mean_dev:.3f}")
        print(f"Difference: {recent_mean_dev - historical_mean_dev:.3f}")
        
        if abs(recent_mean_dev - historical_mean_dev) > 0.1:
            print("WARNING: Significant difference between recent and historical deviations!")
            print("   This suggests the model may be overfitted to recent data.")
        else:
            print("Model appears consistent across time periods.")
    
    return decade_stats

def main():
    """Main function to create detrended palladium growth model analysis."""
    
    print("Palladium Detrended Growth Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit models
    print("\nFitting growth models...")
    models = fit_palladium_growth_models(df)
    
    if not models:
        print("No models could be fitted")
        return
    
    # Create detrended charts
    print("\nCreating detrended charts...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_detrended_chart(df, models, output_dir)
    if chart_file:
        print(f"Detrended chart saved to: {chart_file}")
    
    # Analyze deviations
    print("\nAnalyzing deviations from model...")
    deviation_stats = analyze_deviations(df, models)
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_detrended_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM DETRENDED GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This analysis uses a model-adjusted y-axis scale that makes the growth model\n")
        f.write("appear as a straight diagonal line, making it easy to see deviations.\n\n")
        
        if 'Best_Model' in models:
            f.write(f"Best Model: {models['Best_Model']['name']}\n")
            f.write(f"  Formula: {models['Best_Model']['formula']}\n")
            f.write(f"  R² Score: {models['Best_Model']['r2']:.6f}\n")
            f.write(f"  RMSE: {models['Best_Model']['rmse']:.2f}\n\n")
        
        if deviation_stats is not None:
            f.write("Decade-by-Decade Deviation Analysis:\n")
            f.write(str(deviation_stats))
            f.write("\n\n")
        
        f.write("Chart Interpretation:\n")
        f.write("- Top chart: Model-adjusted view (model appears as straight diagonal line)\n")
        f.write("- Bottom chart: Traditional view for comparison\n")
        f.write("- Deviations above/below the model line show over/under-valuation\n")
        f.write("- Consistent deviations in recent years may indicate overfitting\n")
    
    print(f"Detrended analysis results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("DETRENDED ANALYSIS COMPLETE")
    print("=" * 60)
    
    if 'Best_Model' in models:
        print(f"Best Model: {models['Best_Model']['name']}")
        print(f"R² Score: {models['Best_Model']['r2']:.4f}")
        print(f"Formula: {models['Best_Model']['formula']}")
        print(f"\nChart saved: {chart_file}")
        print("The detrended chart makes it easy to see if the model is overfitted!")

if __name__ == "__main__":
    main()
