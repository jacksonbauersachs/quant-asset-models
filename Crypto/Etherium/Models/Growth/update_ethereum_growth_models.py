import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import linregress
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_ethereum_data():
    """Load and prepare Ethereum price data for analysis."""
    
    data_file = "Crypto/Etherium/Data/Ethereum Historical Data.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Ethereum data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime (M/D/YYYY format)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Clean Price column - convert to float (should already be clean from our cleanup script)
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Ethereum data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    print(f"Days since start: {df['Days'].min()} to {df['Days'].max()}")
    
    return df

def fit_ethereum_growth_models(df):
    """Fit multiple growth models for Ethereum and select the best one."""
    
    x = df['Days'].values
    y = df['Price'].values
    
    models = {}
    
    # Linear model - REMOVED as it's too simple for cryptocurrency growth patterns
    
    # Exponential model
    try:
        popt_exp, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x, y, p0=[100, 0.001, 100])
        y_pred_exp = popt_exp[0] * np.exp(popt_exp[1] * x) + popt_exp[2]
        r2_exp = r2_score(y, y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
        
        models['Exponential'] = {
            'formula': f"Price = {popt_exp[0]:.2f} * exp({popt_exp[1]:.6f} * Days) + {popt_exp[2]:.2f}",
            'params': popt_exp,
            'r2': r2_exp,
            'rmse': rmse_exp
        }
    except:
        print("Warning: Could not fit exponential model for Ethereum")
    
    # Polynomial model (cubic) - REMOVED due to unrealistic negative future predictions
    # Polynomial models tend to overfit and predict negative prices in the future
    
    # Logarithmic model
    try:
        popt_log, _ = curve_fit(lambda x, a, b, c: a * np.log(b * x + 1) + c, x, y, p0=[100, 0.01, 100])
        y_pred_log = popt_log[0] * np.log(popt_log[1] * x + 1) + popt_log[2]
        r2_log = r2_score(y, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
        
        models['Logarithmic'] = {
            'formula': f"Price = {popt_log[0]:.2f} * ln({popt_log[1]:.4f} * Days + 1) + {popt_log[2]:.2f}",
            'params': popt_log,
            'r2': r2_log,
            'rmse': rmse_log
        }
    except:
        print("Warning: Could not fit logarithmic model for Ethereum")
    
    # Power model
    try:
        popt_power, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x, y, p0=[1, 0.5, 100])
        y_pred_power = popt_power[0] * np.power(x, popt_power[1]) + popt_power[2]
        r2_power = r2_score(y, y_pred_power)
        rmse_power = np.sqrt(mean_squared_error(y, y_pred_power))
        
        models['Power'] = {
            'formula': f"Price = {popt_power[0]:.2f} * Days^{popt_power[1]:.4f} + {popt_power[2]:.2f}",
            'params': popt_power,
            'r2': r2_power,
            'rmse': rmse_power
        }
    except:
        print("Warning: Could not fit power model for Ethereum")
    
    # Find best model with smart selection to avoid overfitting
    if models:
        # Score models considering both R² and parameter reasonableness
        model_scores = {}
        
        for model_name, model_data in models.items():
            base_score = model_data['r2']
            
            # Polynomial model removed - no longer needed
            
            # Penalize models with zero coefficients (broken models)
            if model_name == 'Power' and 'params' in model_data:
                params = model_data['params']
                if abs(params[0]) < 1e-10:  # If coefficient is essentially zero
                    base_score *= 0.1  # Heavily penalize broken power models
                    print(f"Warning: Ethereum Power model has zero coefficient - penalizing score")
            
            # Slight preference for exponential model when R² is close
            if model_name == 'Exponential':
                base_score *= 1.01  # 1% bonus for reasonable complexity
            
            model_scores[model_name] = base_score
        
        # Select best model based on adjusted scores
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        best_model = models[best_model_name]
        
        models['Best_Model'] = {
            'name': best_model_name,
            'formula': best_model['formula'],
            'r2': best_model['r2'],
            'rmse': best_model['rmse'],
            'params': best_model['params']  # Include parameters for visualization
        }
        
        print(f"Selected Ethereum model: {best_model_name} (R² = {best_model['r2']:.6f}, Adjusted Score = {model_scores[best_model_name]:.6f})")
    
    return models

def create_ethereum_growth_chart(df, model_results, output_dir):
    """Create a growth chart visualization for Ethereum."""
    
    # Set up the plot
    plt.style.use('default')  # Use default style for better compatibility
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual price
    ax.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Ethereum Price')
    
    # Get the best model parameters
    best_model = model_results['Best_Model']
    
    # Create model formula based on best model type
    if best_model['name'] == 'Exponential':
        model_formula = lambda x: best_model['params'][0] * np.exp(best_model['params'][1] * x) + best_model['params'][2]
    elif best_model['name'] == 'Power':
        model_formula = lambda x: best_model['params'][0] * np.power(x, best_model['params'][1]) + best_model['params'][2]
    else:
        # Fallback to exponential if unknown model type
        model_formula = lambda x: best_model['params'][0] * np.exp(best_model['params'][1] * x) + best_model['params'][2]
    
    # Plot trend line
    df['Model_Price'] = df['Days'].apply(model_formula)
    ax.semilogy(df['Date'], df['Model_Price'], 
                color='red', linestyle='--', linewidth=2, label=f'Ethereum {best_model["name"]} Trend Model')
    
    # Highlight current position
    current_price = df['Price'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    ax.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.2f}')
    
    # Add model information
    formula_text = f'Best Model: {best_model["name"]}\nR² = {best_model["r2"]:.4f}'
    ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ethereum Price (USD) - Log Scale', fontsize=12)
    ax.set_title('Ethereum Growth Model Analysis\nPrice Trend and Model Fit', fontsize=16, fontweight='bold')
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
    
    # Set y-axis limits
    ax.set_ylim(0.01, max(df['Price'].max(), df['Model_Price'].max()) * 2)
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, f"ethereum_growth_chart_latest.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to prevent display
    
    return output_file

def _predict_ethereum_future_price(df, model_results, years_ahead):
    """Predict future Ethereum price using the fitted model for a given horizon (in years)."""
    days_in_year = 365
    days_col = 'Days'
    last_day = int(df[days_col].iloc[-1])
    future_day = last_day + years_ahead * days_in_year
    
    best_model = model_results['Best_Model']
    if best_model['name'] == 'Exponential':
        return float(best_model['params'][0] * np.exp(best_model['params'][1] * future_day) + best_model['params'][2])
    elif best_model['name'] == 'Power':
        return float(best_model['params'][0] * np.power(future_day, best_model['params'][1]) + best_model['params'][2])
    else:
        # Fallback to exponential if unknown model type
        return float(best_model['params'][0] * np.exp(best_model['params'][1] * future_day) + best_model['params'][2])

def compute_ethereum_projections(df, model_results):
    """Compute 1, 5, and 10-year projected returns using the fitted model."""
    if model_results is None:
        return None
    
    current_price = float(df['Price'].iloc[-1])
    
    # Model-implied fair price today
    days_col = 'Days'
    last_day = int(df[days_col].iloc[-1])
    best_model = model_results['Best_Model']
    
    if best_model['name'] == 'Exponential':
        fair_price = float(best_model['params'][0] * np.exp(best_model['params'][1] * last_day) + best_model['params'][2])
    elif best_model['name'] == 'Power':
        fair_price = float(best_model['params'][0] * np.power(last_day, best_model['params'][1]) + best_model['params'][2])
    else:
        # Fallback to exponential if unknown model type
        fair_price = float(best_model['params'][0] * np.exp(best_model['params'][1] * last_day) + best_model['params'][2])
    
    horizons = [1, 5, 10]
    projections = {}
    for y in horizons:
        future_price = _predict_ethereum_future_price(df, model_results, y)
        ret_pct = (future_price / current_price - 1.0) * 100.0
        projections[y] = {
            'future_price': future_price,
            'return_pct': ret_pct,
            'fair_return_pct': (future_price / fair_price - 1.0) * 100.0
        }
    
    return {
        'asset': 'Ethereum',
        'as_of_date': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        'current_price': current_price,
        'fair_price': fair_price,
        'projections': projections
    }

def save_ethereum_projections(ethereum_proj, output_dir):
    """Write Ethereum projections to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "ethereum_projections.txt")
    
    with open(results_file, 'w') as f:
        f.write("ETHEREUM MODEL-BASED FORWARD PROJECTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if ethereum_proj:
            f.write(f"{ethereum_proj['asset'].upper()} PROJECTIONS\n")
            f.write("-" * (len(ethereum_proj['asset']) + 12) + "\n")
            f.write(f"As of: {ethereum_proj['as_of_date']}\n")
            f.write(f"Current Price: ${ethereum_proj['current_price']:,.2f}\n")
            f.write(f"Model Fair Price (today): ${ethereum_proj['fair_price']:,.2f}\n")
            for years in [1, 5, 10]:
                p = ethereum_proj['projections'][years]
                f.write(f"{years}-Year Projected Price: ${p['future_price']:,.2f} | Return vs Current: {p['return_pct']:.2f}%\n")
            f.write("Fair-Based Growth Rates (from model fair price):\n")
            for years in [1, 5, 10]:
                p = ethereum_proj['projections'][years]
                f.write(f"  {years}-Year Growth: {p['fair_return_pct']:.2f}%\n")
            f.write("\n")

        f.write("Notes:\n")
        f.write("- Projections use the latest fitted Ethereum growth model.\n")
        f.write("- Returns are computed versus the latest observed price.\n")
        f.write("- Horizons are 1, 5, and 10 calendar years (365d per year).\n")

    print(f"Ethereum projections saved to: {results_file}")
    return results_file

def save_ethereum_comprehensive_results(ethereum_results, output_dir):
    """Save all Ethereum growth model results to a comprehensive file."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "ethereum_growth_models.txt")
    
    with open(results_file, 'w') as f:
        f.write("ETHEREUM COMPREHENSIVE GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Crypto/Etherium/Data/Ethereum Historical Data.csv\n\n")
        
        # Ethereum Results
        f.write("ETHEREUM GROWTH MODELS\n")
        f.write("-" * 30 + "\n")
        if ethereum_results:
            for model_name, model_data in ethereum_results.items():
                if model_name != 'Best_Model':
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Formula: {model_data['formula']}\n")
                    f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                    f.write(f"  RMSE: {model_data['rmse']:.2f}\n\n")
            
            if 'Best_Model' in ethereum_results:
                f.write(f"Best Ethereum Model: {ethereum_results['Best_Model']['name']}\n")
                f.write(f"  Formula: {ethereum_results['Best_Model']['formula']}\n")
                f.write(f"  R² Score: {ethereum_results['Best_Model']['r2']:.6f}\n")
                f.write(f"  RMSE: {ethereum_results['Best_Model']['rmse']:.2f}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write("Ethereum growth models have been fitted using the latest data from:\n")
        f.write("- Crypto/Etherium/Data/Ethereum Historical Data.csv\n\n")
        f.write("The models can be used for:\n")
        f.write("- Ethereum price trend analysis\n")
        f.write("- Future price predictions\n")
        f.write("- Investment timing decisions\n")
        f.write("- Portfolio allocation strategies\n")
        f.write("- Risk assessment and volatility analysis\n")
    
    print(f"Comprehensive Ethereum results saved to: {results_file}")
    return results_file

def main():
    """Main function to update Ethereum growth models."""
    
    print("Ethereum Growth Model Analysis")
    print("=" * 40)
    
    # Load Ethereum data
    print("\nLoading Ethereum data...")
    ethereum_df = load_ethereum_data()
    
    if ethereum_df is None:
        print("Error: Could not load Ethereum data file.")
        return
    
    # Fit Ethereum growth models
    print("\nFitting Ethereum growth models...")
    ethereum_results = fit_ethereum_growth_models(ethereum_df)
    
    if not ethereum_results:
        print("Error: Could not fit any Ethereum growth models.")
        return
    
    # Create growth chart
    print("\nCreating Ethereum growth chart...")
    output_dir = "Crypto/Etherium/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = create_ethereum_growth_chart(ethereum_df, ethereum_results, output_dir)
    print(f"Ethereum chart saved to: {chart_file}")
    
    # Save comprehensive results
    print("\nSaving comprehensive results...")
    results_file = save_ethereum_comprehensive_results(ethereum_results, output_dir)
    
    # Compute and save projections
    print("\nComputing forward projections (1y, 5y, 10y)...")
    ethereum_proj = compute_ethereum_projections(ethereum_df, ethereum_results)
    projections_file = save_ethereum_projections(ethereum_proj, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ETHEREUM ANALYSIS COMPLETE")
    print("=" * 60)
    
    if ethereum_results and 'Best_Model' in ethereum_results:
        print(f"Best Ethereum Model: {ethereum_results['Best_Model']['name']} (R² = {ethereum_results['Best_Model']['r2']:.6f})")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Projections saved to: {projections_file}")
    print(f"Growth chart created: {chart_file}")
    print(f"\nAll files saved to: {output_dir}")

if __name__ == "__main__":
    main()
