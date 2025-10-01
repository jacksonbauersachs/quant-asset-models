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

def fit_2015_model(df):
    """Fit the 2015 model (same as our previous script)."""
    
    # Filter data to end of 2015
    cutoff = pd.Timestamp('2015-12-31')
    df_pre2015 = df[df['date'] <= cutoff].copy()
    
    x = df_pre2015['days'].values
    y = df_pre2015[' value'].values
    
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
        
        return model, df_pre2015
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None

def calculate_deviations(df, model, cutoff_date):
    """Calculate mean deviations for different time periods."""
    
    # Get model predictions for all data
    x_all = df['days'].values
    y_all = df[' value'].values
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    model_predictions = exponential_func(x_all, *model['params'])
    
    # Calculate deviations
    deviations = np.abs(y_all - model_predictions)
    
    # Split data by cutoff
    cutoff_days = (pd.Timestamp(cutoff_date) - df['date'].min()).days
    
    pre_cutoff = deviations[x_all <= cutoff_days]
    post_cutoff = deviations[x_all > cutoff_days]
    
    # Calculate mean deviations
    pre_mean_dev = np.mean(pre_cutoff)
    post_mean_dev = np.mean(post_cutoff)
    
    # Calculate overfitting score
    overfitting_score = (post_mean_dev - pre_mean_dev) / pre_mean_dev
    
    return {
        'pre_cutoff_deviation': pre_mean_dev,
        'post_cutoff_deviation': post_mean_dev,
        'overfitting_score': overfitting_score,
        'pre_cutoff_count': len(pre_cutoff),
        'post_cutoff_count': len(post_cutoff)
    }

def create_overfitting_chart(df, model, cutoff_date, deviations, output_dir):
    """Create a chart showing the overfitting analysis."""
    
    # Get model predictions
    x_all = df['days'].values
    y_all = df[' value'].values
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    model_predictions = exponential_func(x_all, *model['params'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Chart 1: Price data with model
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    ax1.plot(df['years'], model_predictions, 'r-', linewidth=2, label=f'{model["name"]} Model (Pre-2015 Fit)')
    
    # Highlight cutoff
    cutoff_year = 2015
    ax1.axvline(x=cutoff_year, color='green', linestyle='--', alpha=0.7, label='Model Cutoff (2015)')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Palladium Price - Pre-2015 Model Overfitting Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Deviation analysis
    cutoff_days = (pd.Timestamp(cutoff_date) - df['date'].min()).days
    pre_mask = df['days'] <= cutoff_days
    post_mask = df['days'] > cutoff_days
    
    ax2.scatter(df[pre_mask]['years'], np.abs(df[pre_mask][' value'] - model_predictions[pre_mask]), 
                alpha=0.6, s=1, color='green', label='Pre-2015 Deviations')
    ax2.scatter(df[post_mask]['years'], np.abs(df[post_mask][' value'] - model_predictions[post_mask]), 
                alpha=0.6, s=1, color='red', label='Post-2015 Deviations')
    
    ax2.axvline(x=cutoff_year, color='green', linestyle='--', alpha=0.7, label='Model Cutoff (2015)')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Absolute Deviation ($)')
    ax2.set_title('Model Deviations - Overfitting Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add overfitting analysis text
    analysis_text = f"""Overfitting Analysis:
Pre-2015 Mean Deviation: ${deviations['pre_cutoff_deviation']:.2f}
Post-2015 Mean Deviation: ${deviations['post_cutoff_deviation']:.2f}
Overfitting Score: {deviations['overfitting_score']:.2f}

Interpretation:
- Score < 1.0: Model is robust
- Score 1.0-2.0: Some overfitting
- Score > 2.0: Significant overfitting"""
    
    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save chart
    output_file = os.path.join(output_dir, "palladium_overfitting_test_2015.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def main():
    print("Palladium Pre-2015 Model Overfitting Test")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit 2015 model
    print("\nFitting 2015 model...")
    model, df_pre2015 = fit_2015_model(df)
    if model is None:
        print("Failed to fit model")
        return
    
    print(f"Model: {model['name']}")
    print(f"Formula: {model['formula']}")
    print(f"R² Score: {model['r2']:.4f}")
    
    # Calculate deviations
    print("\nCalculating deviations...")
    deviations = calculate_deviations(df, model, '2015-12-31')
    
    print(f"\n=== OVERFITTING ANALYSIS ===")
    print(f"Pre-2015 Mean Deviation: ${deviations['pre_cutoff_deviation']:.2f}")
    print(f"Post-2015 Mean Deviation: ${deviations['post_cutoff_deviation']:.2f}")
    print(f"Overfitting Score: {deviations['overfitting_score']:.2f}")
    
    # Interpretation
    score = deviations['overfitting_score']
    if score < 1.0:
        print("Model is ROBUST - No significant overfitting")
    elif score < 2.0:
        print("Model has SOME overfitting - Moderate concern")
    else:
        print("Model is OVERFITTED - Significant concern")
    
    # Create chart
    print("\nCreating overfitting analysis chart...")
    output_dir = "Commodities/Palladium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_overfitting_chart(df, model, '2015-12-31', deviations, output_dir)
    print(f"Chart saved: {chart_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "palladium_overfitting_analysis_2015.txt")
    with open(results_file, 'w') as f:
        f.write("PALLADIUM PRE-2015 MODEL OVERFITTING ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Details:\n")
        f.write(f"  Name: {model['name']}\n")
        f.write(f"  Formula: {model['formula']}\n")
        f.write(f"  R² Score: {model['r2']:.6f}\n")
        f.write(f"  Training Data: 1977-2015 ({len(df_pre2015)} records)\n\n")
        
        f.write("Overfitting Analysis:\n")
        f.write(f"  Pre-2015 Mean Deviation: ${deviations['pre_cutoff_deviation']:.2f}\n")
        f.write(f"  Post-2015 Mean Deviation: ${deviations['post_cutoff_deviation']:.2f}\n")
        f.write(f"  Overfitting Score: {deviations['overfitting_score']:.2f}\n\n")
        
        f.write("Interpretation:\n")
        if score < 1.0:
            f.write("  Model is ROBUST - No significant overfitting\n")
        elif score < 2.0:
            f.write("  Model has SOME overfitting - Moderate concern\n")
        else:
            f.write("  Model is OVERFITTED - Significant concern\n")
        
        f.write(f"\n  Score < 1.0: Model is robust\n")
        f.write(f"  Score 1.0-2.0: Some overfitting\n")
        f.write(f"  Score > 2.0: Significant overfitting\n")
    
    print(f"Analysis results saved to: {results_file}")
    
    print("\n" + "=" * 50)
    print("OVERFITTING TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
