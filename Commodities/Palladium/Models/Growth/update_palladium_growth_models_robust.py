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

def fit_palladium_growth_models_robust(df):
    """Fit growth models with different data filtering approaches to avoid overfitting."""
    
    x = df['days'].values
    y = df[' value'].values
    
    models = {}
    
    # Approach 1: Exclude recent spike (2020-2025)
    print("\n=== APPROACH 1: Exclude Recent Spike (2020-2025) ===")
    recent_cutoff = pd.Timestamp('2020-01-01')
    df_no_recent = df[df['date'] < recent_cutoff].copy()
    
    if len(df_no_recent) > 100:  # Ensure we have enough data
        x_no_recent = df_no_recent['days'].values
        y_no_recent = df_no_recent[' value'].values
        
        # Power model
        try:
            popt_power, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x_no_recent, y_no_recent, p0=[1, 0.5, 100])
            y_pred_power = popt_power[0] * np.power(x_no_recent, popt_power[1]) + popt_power[2]
            r2_power = r2_score(y_no_recent, y_pred_power)
            
            models['Power_NoRecent'] = {
                'formula': f"y = {popt_power[0]:.4f} * x^{popt_power[1]:.4f} + {popt_power[2]:.2f}",
                'params': popt_power,
                'r2': r2_power,
                'data_points': len(df_no_recent),
                'date_range': f"{df_no_recent['date'].min().strftime('%Y-%m-%d')} to {df_no_recent['date'].max().strftime('%Y-%m-%d')}"
            }
            print(f"Power Model (No Recent): R² = {r2_power:.4f}")
        except:
            print("Could not fit Power model without recent data")
        
        # Exponential model
        try:
            popt_exp, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x_no_recent, y_no_recent, p0=[100, 0.0001, 100])
            y_pred_exp = popt_exp[0] * np.exp(popt_exp[1] * x_no_recent) + popt_exp[2]
            r2_exp = r2_score(y_no_recent, y_pred_exp)
            
            models['Exponential_NoRecent'] = {
                'formula': f"y = {popt_exp[0]:.4f} * e^({popt_exp[1]:.6f} * x) + {popt_exp[2]:.2f}",
                'params': popt_exp,
                'r2': r2_exp,
                'data_points': len(df_no_recent),
                'date_range': f"{df_no_recent['date'].min().strftime('%Y-%m-%d')} to {df_no_recent['date'].max().strftime('%Y-%m-%d')}"
            }
            print(f"Exponential Model (No Recent): R² = {r2_exp:.4f}")
        except:
            print("Could not fit Exponential model without recent data")
    
    # Approach 2: Use rolling median to smooth out spikes
    print("\n=== APPROACH 2: Rolling Median Smoothing ===")
    window_size = 365  # 1 year rolling window
    df['value_smoothed'] = df[' value'].rolling(window=window_size, center=True).median()
    df_smooth = df.dropna().copy()
    
    if len(df_smooth) > 100:
        x_smooth = df_smooth['days'].values
        y_smooth = df_smooth['value_smoothed'].values
        
        # Power model on smoothed data
        try:
            popt_power_smooth, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x_smooth, y_smooth, p0=[1, 0.5, 100])
            y_pred_power_smooth = popt_power_smooth[0] * np.power(x_smooth, popt_power_smooth[1]) + popt_power_smooth[2]
            r2_power_smooth = r2_score(y_smooth, y_pred_power_smooth)
            
            models['Power_Smoothed'] = {
                'formula': f"y = {popt_power_smooth[0]:.4f} * x^{popt_power_smooth[1]:.4f} + {popt_power_smooth[2]:.2f}",
                'params': popt_power_smooth,
                'r2': r2_power_smooth,
                'data_points': len(df_smooth),
                'date_range': f"{df_smooth['date'].min().strftime('%Y-%m-%d')} to {df_smooth['date'].max().strftime('%Y-%m-%d')}"
            }
            print(f"Power Model (Smoothed): R² = {r2_power_smooth:.4f}")
        except:
            print("Could not fit Power model on smoothed data")
    
    # Approach 3: Use percentile-based filtering (remove top 5% of values)
    print("\n=== APPROACH 3: Remove Top 5% Outliers ===")
    price_95th = np.percentile(df[' value'], 95)
    df_filtered = df[df[' value'] <= price_95th].copy()
    
    if len(df_filtered) > 100:
        x_filtered = df_filtered['days'].values
        y_filtered = df_filtered[' value'].values
        
        # Power model on filtered data
        try:
            popt_power_filtered, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x_filtered, y_filtered, p0=[1, 0.5, 100])
            y_pred_power_filtered = popt_power_filtered[0] * np.power(x_filtered, popt_power_filtered[1]) + popt_power_filtered[2]
            r2_power_filtered = r2_score(y_filtered, y_pred_power_filtered)
            
            models['Power_Filtered'] = {
                'formula': f"y = {popt_power_filtered[0]:.4f} * x^{popt_power_filtered[1]:.4f} + {popt_power_filtered[2]:.2f}",
                'params': popt_power_filtered,
                'r2': r2_power_filtered,
                'data_points': len(df_filtered),
                'date_range': f"{df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}"
            }
            print(f"Power Model (Filtered): R² = {r2_power_filtered:.4f}")
        except:
            print("Could not fit Power model on filtered data")
    
    # Select best model
    if models:
        best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
        best_model = models[best_model_name]
        
        models['Best_Model'] = {
            'name': best_model_name,
            'formula': best_model['formula'],
            'r2': best_model['r2'],
            'params': best_model['params'],
            'data_points': best_model['data_points'],
            'date_range': best_model['date_range']
        }
        
        print(f"\nBest Model: {best_model_name} (R² = {best_model['r2']:.4f})")
    
    return models

def create_robust_chart(df, models, output_dir):
    """Create charts showing different approaches to avoid overfitting."""
    
    if not models or 'Best_Model' not in models:
        print("No models available for charting")
        return None
    
    best_model = models['Best_Model']
    best_model_name = best_model['name']
    best_params = best_model['params']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Original data with all models
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    # Plot all models
    x_range = np.linspace(df['days'].min(), df['days'].max(), 1000)
    years_range = (x_range / 365.25) + df['date'].min().year
    
    for model_name, model_data in models.items():
        if model_name != 'Best_Model' and 'params' in model_data:
            params = model_data['params']
            if 'Power' in model_name:
                y_pred = params[0] * np.power(x_range, params[1]) + params[2]
                ax1.plot(years_range, y_pred, '--', alpha=0.7, label=f'{model_name} (R² = {model_data["r2"]:.3f})')
            elif 'Exponential' in model_name:
                y_pred = params[0] * np.exp(params[1] * x_range) + params[2]
                ax1.plot(years_range, y_pred, '--', alpha=0.7, label=f'{model_name} (R² = {model_data["r2"]:.3f})')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Growth Models - Overfitting Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Focus on recent period with projections
    recent_start = 2010
    df_recent = df[df['years'] >= recent_start]
    ax2.scatter(df_recent['years'], df_recent[' value'], alpha=0.8, s=2, color='blue', label='Actual Data')
    
    # Plot best model with projections
    x_recent = np.linspace(df_recent['days'].min(), df_recent['days'].max() + 365*5, 1000)
    years_recent = (x_recent / 365.25) + df['date'].min().year
    
    if 'Power' in best_model_name:
        y_pred_recent = best_params[0] * np.power(x_recent, best_params[1]) + best_params[2]
    elif 'Exponential' in best_model_name:
        y_pred_recent = best_params[0] * np.exp(best_params[1] * x_recent) + best_params[2]
    
    ax2.plot(years_recent, y_pred_recent, 'r-', linewidth=2, label=f'Best Model: {best_model_name}')
    
    # Highlight current price
    current_price = df[' value'].iloc[-1]
    current_year = df['years'].iloc[-1]
    ax2.scatter(current_year, current_price, color='red', s=100, zorder=10, label=f'Current: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($)')
    ax2.set_title(f'Best Model: {best_model_name} - Recent Period & Projections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model info
    formula_text = f'Best Model: {best_model_name}\nFormula: {best_model["formula"]}\nR² = {best_model["r2"]:.4f}'
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, "palladium_robust_models.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def main():
    """Main function to analyze robust palladium growth models."""
    
    print("Palladium Robust Growth Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit robust models
    print("\nFitting robust growth models...")
    models = fit_palladium_growth_models_robust(df)
    
    if not models:
        print("No models could be fitted")
        return
    
    # Create charts
    print("\nCreating robust model charts...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_robust_chart(df, models, output_dir)
    if chart_file:
        print(f"Robust model chart saved to: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_robust_models.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM ROBUST GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This analysis addresses overfitting concerns by testing different approaches:\n\n")
        
        for model_name, model_data in models.items():
            if model_name != 'Best_Model':
                f.write(f"{model_name}:\n")
                f.write(f"  Formula: {model_data['formula']}\n")
                f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                f.write(f"  Data Points: {model_data['data_points']:,}\n")
                f.write(f"  Date Range: {model_data['date_range']}\n\n")
        
        if 'Best_Model' in models:
            f.write(f"Best Model: {models['Best_Model']['name']}\n")
            f.write(f"  Formula: {models['Best_Model']['formula']}\n")
            f.write(f"  R² Score: {models['Best_Model']['r2']:.6f}\n")
            f.write(f"  Data Points: {models['Best_Model']['data_points']:,}\n")
            f.write(f"  Date Range: {models['Best_Model']['date_range']}\n\n")
        
        f.write("Notes:\n")
        f.write("- This analysis tests different approaches to avoid overfitting to recent spikes\n")
        f.write("- Models are compared based on R² score and data coverage\n")
        f.write("- The best model provides a more robust long-term trend\n")
    
    print(f"Robust model results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("ROBUST ANALYSIS COMPLETE")
    print("=" * 60)
    
    if 'Best_Model' in models:
        print(f"Best Model: {models['Best_Model']['name']}")
        print(f"R² Score: {models['Best_Model']['r2']:.4f}")
        print(f"Formula: {models['Best_Model']['formula']}")

if __name__ == "__main__":
    main()
