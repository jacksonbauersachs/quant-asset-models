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

def fit_pre2020_models(df):
    """Fit growth models using only data up to 2020."""
    
    # Filter data to end of 2015
    cutoff = pd.Timestamp('2015-12-31')
    df_pre2015 = df[df['date'] <= cutoff].copy()
    
    print(f"\nPre-2015 data: {len(df_pre2015)} records from {df_pre2015['date'].min().strftime('%Y-%m-%d')} to {df_pre2015['date'].max().strftime('%Y-%m-%d')}")
    
    x = df_pre2015['days'].values
    y = df_pre2015[' value'].values
    
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
        print(f"Power Model: R² = {r2_power:.4f}")
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
        print(f"Exponential Model: R² = {r2_exp:.4f}")
    except:
        print("Warning: Could not fit Exponential model")
    
    # Select best model
    if models:
        best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
        best_model = models[best_model_name]
        
        models['Best_Model'] = {
            'name': best_model_name,
            'formula': best_model['formula'],
            'r2': best_model['r2'],
            'rmse': best_model['rmse'],
            'params': best_model['params']
        }
        
        print(f"\nBest Pre-2015 Model: {best_model_name} (R² = {best_model['r2']:.4f})")
    
    return models, df_pre2015

def create_pre2015_chart(df, df_pre2015, models, output_dir):
    """Create chart showing pre-2015 model with post-2015 overlay."""
    
    if not models or 'Best_Model' not in models:
        print("No models available for charting")
        return None
    
    best_model = models['Best_Model']
    best_model_name = best_model['name']
    best_params = best_model['params']
    
    # Calculate model predictions for ALL data (including post-2020)
    x_all = df['days'].values
    if best_model_name == 'Power':
        model_predictions_all = best_params[0] * np.power(x_all, best_params[1]) + best_params[2]
    elif best_model_name == 'Exponential':
        model_predictions_all = best_params[0] * np.exp(best_params[1] * x_all) + best_params[2]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Normal scale
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data', zorder=1)
    ax1.plot(df['years'], model_predictions_all, 'r-', linewidth=2, label=f'{best_model_name} Model (Pre-2015 Fit)', zorder=2)
    
    # Highlight the cutoff point
    cutoff_year = 2015
    ax1.axvline(x=cutoff_year, color='green', linestyle='--', alpha=0.7, label='Model Cutoff (2015)')
    
    # Highlight current position
    current_price = df[' value'].iloc[-1]
    current_year = df['years'].iloc[-1]
    current_model_price = model_predictions_all[-1]
    ax1.scatter(current_year, current_price, color='red', s=100, zorder=10, label=f'Current: ${current_price:.0f}')
    ax1.scatter(current_year, current_model_price, color='orange', s=100, zorder=10, label=f'Model: ${current_model_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Price - Pre-2015 Model with Post-2015 Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f'Pre-2015 Model: {best_model_name}\nFormula: {best_model["formula"]}\nR² = {best_model["r2"]:.4f}'
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Chart 2: Log scale for better visibility
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data', zorder=1)
    ax2.plot(df['years'], model_predictions_all, 'r-', linewidth=2, label=f'{best_model_name} Model (Pre-2015 Fit)', zorder=2)
    
    # Highlight the cutoff point
    ax2.axvline(x=cutoff_year, color='green', linestyle='--', alpha=0.7, label='Model Cutoff (2015)')
    
    # Highlight current position
    ax2.scatter(current_year, current_price, color='red', s=100, zorder=10, label=f'Current: ${current_price:.0f}')
    ax2.scatter(current_year, current_model_price, color='orange', s=100, zorder=10, label=f'Model: ${current_model_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_yscale('log')
    ax2.set_title('Palladium Price - Pre-2015 Model (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_pre2015_model.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def calculate_fair_value(df, models):
    """Calculate current fair value using the pre-2020 model."""
    
    if not models or 'Best_Model' not in models:
        return None
    
    best_model = models['Best_Model']
    best_params = best_model['params']
    
    # Get current data point
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    
    # Calculate fair value using the model
    if best_model['name'] == 'Power':
        fair_value = best_params[0] * np.power(current_days, best_params[1]) + best_params[2]
    elif best_model['name'] == 'Exponential':
        fair_value = best_params[0] * np.exp(best_params[1] * current_days) + best_params[2]
    
    # Calculate undervaluation/overvaluation
    if current_price > fair_value:
        deviation_pct = ((current_price - fair_value) / fair_value) * 100
        status = f"Overvalued by {deviation_pct:.1f}%"
    else:
        deviation_pct = ((fair_value - current_price) / current_price) * 100
        status = f"Undervalued by {deviation_pct:.1f}%"
    
    print(f"\n=== FAIR VALUE ANALYSIS ===")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Model Fair Value: ${fair_value:.2f}")
    print(f"Status: {status}")
    
    return {
        'current_price': current_price,
        'fair_value': fair_value,
        'deviation_pct': deviation_pct,
        'status': status
    }

def main():
    """Main function to create pre-2020 palladium growth model analysis."""
    
    print("Palladium Pre-2020 Growth Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit models using only pre-2015 data
    print("\nFitting models using pre-2015 data only...")
    models, df_pre2015 = fit_pre2020_models(df)
    
    if not models:
        print("No models could be fitted")
        return
    
    # Create charts
    print("\nCreating pre-2015 model charts...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_pre2015_chart(df, df_pre2015, models, output_dir)
    if chart_file:
        print(f"Pre-2015 model chart saved to: {chart_file}")
    
    # Calculate fair value
    print("\nCalculating fair value...")
    fair_value_analysis = calculate_fair_value(df, models)
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_pre2015_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM PRE-2015 GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This analysis trains the model on data up to 2015 only, then overlays\n")
        f.write("the fair value line on post-2015 data to see how it compares.\n\n")
        
        if 'Best_Model' in models:
            f.write(f"Best Pre-2015 Model: {models['Best_Model']['name']}\n")
            f.write(f"  Formula: {models['Best_Model']['formula']}\n")
            f.write(f"  R² Score: {models['Best_Model']['r2']:.6f}\n")
            f.write(f"  RMSE: {models['Best_Model']['rmse']:.2f}\n")
            f.write(f"  Training Data: 1977-2015 ({len(df_pre2015)} records)\n\n")
        
        if fair_value_analysis:
            f.write("Fair Value Analysis:\n")
            f.write(f"  Current Price: ${fair_value_analysis['current_price']:.2f}\n")
            f.write(f"  Model Fair Value: ${fair_value_analysis['fair_value']:.2f}\n")
            f.write(f"  Status: {fair_value_analysis['status']}\n\n")
        
        f.write("Chart Interpretation:\n")
        f.write("- Green dashed line: Model training cutoff (2015)\n")
        f.write("- Red line: Model predictions (trained on pre-2015 data)\n")
        f.write("- Blue dots: Actual prices (including post-2015)\n")
        f.write("- Orange dot: Model prediction for current date\n")
        f.write("- Red dot: Current actual price\n")
        f.write("- This shows how the model would have predicted post-2015 prices\n")
    
    print(f"Pre-2015 analysis results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("PRE-2015 ANALYSIS COMPLETE")
    print("=" * 60)
    
    if 'Best_Model' in models:
        print(f"Best Pre-2015 Model: {models['Best_Model']['name']}")
        print(f"R² Score: {models['Best_Model']['r2']:.4f}")
        print(f"Formula: {models['Best_Model']['formula']}")
        print(f"\nChart saved: {chart_file}")
        print("This shows how your model would have predicted the recent spike!")

if __name__ == "__main__":
    main()
