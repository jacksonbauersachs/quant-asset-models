import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime

def load_palladium_data():
    """Load Palladium historical data from CSV file."""
    data_file = "Commodities/Palladium/Data/Palladium Historical Data.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Palladium data file not found: {data_file}")
        return None
    
    try:
        df = pd.read_csv(data_file)
        print(f"Palladium data loaded: {len(df)} records from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"❌ Error loading Palladium data: {e}")
        return None

def fit_palladium_growth_models(df):
    """Fit multiple growth models for Palladium and select the best one."""
    
    x = df['Days'].values
    y = df['Price'].values
    
    models = {}
    
    # Power model
    try:
        popt_power, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x, y, p0=[0.01, 0.5, 50])
        y_pred_power = popt_power[0] * np.power(x, popt_power[1]) + popt_power[2]
        r2_power = r2_score(y, y_pred_power)
        rmse_power = np.sqrt(mean_squared_error(y, y_pred_power))
        
        models['Power'] = {
            'formula': f"Price = {popt_power[0]:.4f} * Days^{popt_power[1]:.4f} + {popt_power[2]:.4f}",
            'params': popt_power,
            'r2': r2_power,
            'rmse': rmse_power
        }
    except:
        print("Warning: Could not fit power model for Palladium")
    
    # Exponential model
    try:
        # Better initial parameters for palladium price range
        popt_exp, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x, y, 
                               p0=[50, 0.0001, 50], maxfev=10000)
        y_pred_exp = popt_exp[0] * np.exp(popt_exp[1] * x) + popt_exp[2]
        r2_exp = r2_score(y, y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
        
        models['Exponential'] = {
            'formula': f"Price = {popt_exp[0]:.4f} * exp({popt_exp[1]:.6f} * Days) + {popt_exp[2]:.4f}",
            'params': popt_exp,
            'r2': r2_exp,
            'rmse': rmse_exp
        }
    except Exception as e:
        print(f"Warning: Could not fit exponential model for Palladium: {e}")
        # Try alternative exponential form
        try:
            popt_exp2, _ = curve_fit(lambda x, a, b: a * np.exp(b * x), x, y, 
                                   p0=[100, 0.0001], maxfev=10000)
            y_pred_exp2 = popt_exp2[0] * np.exp(popt_exp2[1] * x)
            r2_exp2 = r2_score(y, y_pred_exp2)
            rmse_exp2 = np.sqrt(mean_squared_error(y, y_pred_exp2))
            
            models['Exponential'] = {
                'formula': f"Price = {popt_exp2[0]:.4f} * exp({popt_exp2[1]:.6f} * Days)",
                'params': [popt_exp2[0], popt_exp2[1], 0],  # Add zero for consistency
                'r2': r2_exp2,
                'rmse': rmse_exp2
            }
        except Exception as e2:
            print(f"Warning: Alternative exponential model also failed: {e2}")
    
    # Find best model with smart selection to avoid overfitting
    if models:
        # Score models considering both R² and parameter reasonableness
        model_scores = {}
        
        for model_name, model_data in models.items():
            base_score = model_data['r2']
            
            # Penalize models with zero coefficients (broken models)
            if model_name == 'Power' and 'params' in model_data:
                params = model_data['params']
                if abs(params[0]) < 1e-10:  # If coefficient is essentially zero
                    base_score *= 0.1  # Heavily penalize broken power models
                    print(f"Warning: Palladium Power model has zero coefficient - penalizing score")
            
            # Slight preference for reasonable models when R² is close
            if model_name == 'Power':
                base_score *= 1.02  # 2% bonus for power model
            elif model_name == 'Exponential':
                base_score *= 1.01  # 1% bonus for exponential model
            
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
        
        print(f"Selected Palladium model: {best_model_name} (R² = {best_model['r2']:.6f}, Adjusted Score = {model_scores[best_model_name]:.6f})")
    
    return models

def create_palladium_growth_chart(df, model_results):
    """Create comprehensive growth charts for Palladium (both log10 and normal scales)."""
    if model_results is None or 'Best_Model' not in model_results:
        print("No model results available for chart creation")
        return
    
    # Convert dates for plotting
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Get the best model parameters
    best_model = model_results['Best_Model']
    
    # Create model formula based on best model type
    if best_model['name'] == 'Power':
        model_formula = lambda x: best_model['params'][0] * np.power(x, best_model['params'][1]) + best_model['params'][2]
    elif best_model['name'] == 'Exponential':
        model_formula = lambda x: best_model['params'][0] * np.exp(best_model['params'][1] * x) + best_model['params'][2]
    else:
        # Fallback to power if unknown model type
        model_formula = lambda x: best_model['params'][0] * np.power(x, best_model['params'][1]) + best_model['params'][2]
    
    # Calculate model prices
    df['Model_Price'] = df['Days'].apply(model_formula)
    
    # Create both log10 and normal scale charts
    for scale_type, use_log in [("log10", True), ("normal", False)]:
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot actual price data
        if use_log:
            ax.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Palladium Price')
            ax.semilogy(df['Date'], df['Model_Price'], 
                       color='red', linestyle='--', linewidth=2, label=f'Palladium {best_model["name"]} Trend Model')
        else:
            ax.plot(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Palladium Price')
            ax.plot(df['Date'], df['Model_Price'], 
                   color='red', linestyle='--', linewidth=2, label=f'Palladium {best_model["name"]} Trend Model')
        
        # Formatting
        ax.set_title(f'Palladium Price Growth Analysis ({scale_type.title()} Scale)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Palladium Price ($)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show dollar signs
        if use_log:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Add formula and R² at the bottom
        formula_text = f"Formula: {best_model['formula']}"
        r2_text = f"R² = {best_model['r2']:.6f}"
        ax.text(0.5, 0.02, f"{formula_text}\n{r2_text}", 
                transform=ax.transAxes, fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the chart
        chart_path = f"Commodities/Palladium/Models/Growth/palladium_growth_chart_{scale_type}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Palladium {scale_type} chart saved to: {chart_path}")
        plt.close()

def _predict_palladium_future_price(df, model_results, years_ahead):
    """Helper function to predict future Palladium price using the best model."""
    if model_results is None or 'Best_Model' not in model_results:
        return None
    
    days_in_year = 365
    days_col = 'Days'
    last_day = int(df[days_col].iloc[-1])
    future_day = last_day + years_ahead * days_in_year
    
    best_model = model_results['Best_Model']
    if best_model['name'] == 'Power':
        return float(best_model['params'][0] * np.power(future_day, best_model['params'][1]) + best_model['params'][2])
    elif best_model['name'] == 'Exponential':
        return float(best_model['params'][0] * np.exp(best_model['params'][1] * future_day) + best_model['params'][2])
    else:
        # Fallback to power if unknown model type
        return float(best_model['params'][0] * np.power(future_day, best_model['params'][1]) + best_model['params'][2])

def compute_palladium_projections(df, model_results):
    """Compute 1, 5, and 10-year projected returns using the fitted model."""
    if model_results is None:
        return None
    
    # Model-implied fair price today
    days_col = 'Days'
    last_day = int(df[days_col].iloc[-1])
    best_model = model_results['Best_Model']
    
    if best_model['name'] == 'Power':
        fair_price = float(best_model['params'][0] * np.power(last_day, best_model['params'][1]) + best_model['params'][2])
    elif best_model['name'] == 'Exponential':
        fair_price = float(best_model['params'][0] * np.exp(best_model['params'][1] * last_day) + best_model['params'][2])
    else:
        # Fallback to power if unknown model type
        fair_price = float(best_model['params'][0] * np.power(last_day, best_model['params'][1]) + best_model['params'][2])
    
    horizons = [1, 5, 10]
    projections = {}
    for y in horizons:
        future_price = _predict_palladium_future_price(df, model_results, y)
        if future_price is not None:
            projections[f"{y}Y"] = future_price
    
    return projections, fair_price

def save_palladium_results(model_results, projections, fair_price, df):
    """Save comprehensive Palladium analysis results to text files."""
    
    # Save model results
    results_path = "Commodities/Palladium/Models/Growth/palladium_growth_models.txt"
    with open(results_path, 'w') as f:
        f.write("PALLADIUM COMPREHENSIVE GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Commodities/Palladium/Data/Palladium Historical Data.csv\n\n")
        
        f.write("PALLADIUM GROWTH MODELS\n")
        f.write("-" * 25 + "\n")
        
        for model_name, model_data in model_results.items():
            if model_name != 'Best_Model':
                f.write(f"{model_name} Model:\n")
                f.write(f"  Formula: {model_data['formula']}\n")
                f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                f.write(f"  RMSE: {model_data['rmse']:.4f}\n\n")
        
        if 'Best_Model' in model_results:
            best = model_results['Best_Model']
            f.write(f"Best Palladium Model: {best['name']}\n")
            f.write(f"  Formula: {best['formula']}\n")
            f.write(f"  R² Score: {best['r2']:.6f}\n")
            f.write(f"  RMSE: {best['rmse']:.4f}\n\n")
        
        f.write("SUMMARY\n")
        f.write("----------\n")
        f.write(f"Total Data Points: {len(df)}\n")
        f.write(f"Date Range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}\n")
        f.write(f"Price Range: ${df['Price'].min():.4f} to ${df['Price'].max():.4f}\n")
        f.write(f"Days Since Start: {df['Days'].iloc[-1]}\n")
    
    print(f"Comprehensive Palladium results saved to: {results_path}")
    
    # Save projections
    if projections:
        projections_path = "Commodities/Palladium/Models/Growth/palladium_projections.txt"
        with open(projections_path, 'w') as f:
            f.write("PALLADIUM MODEL-BASED FORWARD PROJECTIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PALLADIUM PROJECTIONS\n")
            f.write("-" * 20 + "\n")
            f.write(f"As of: {df['Date'].iloc[-1]}\n")
            f.write(f"Current Price: ${df['Price'].iloc[-1]:.4f}\n")
            f.write(f"Model Fair Price (today): ${fair_price:.4f}\n")
            
            for horizon, price in projections.items():
                years = horizon.replace('Y', '')
                current_price = df['Price'].iloc[-1]
                return_pct = ((price - current_price) / current_price) * 100
                f.write(f"{years}-Year Projected Price: ${price:.4f} | Return vs Current: {return_pct:.2f}%\n")
            
            f.write(f"\nFair-Based Growth Rates (from model fair price):\n")
            for horizon, price in projections.items():
                years = horizon.replace('Y', '')
                growth_pct = ((price - fair_price) / fair_price) * 100
                f.write(f"  {years}-Year Growth: {growth_pct:.2f}%\n")
            
            f.write(f"\nNotes:\n")
            f.write(f"- Projections use the latest fitted Palladium growth model.\n")
            f.write(f"- Model fair price represents the theoretical value based on historical trends.\n")
            f.write(f"- These are statistical projections, not financial advice.\n")
        
        print(f"Palladium projections saved to: {projections_path}")

def main():
    """Main function to run Palladium growth model analysis."""
    
    print("Palladium Growth Model Analysis")
    print("=" * 40)
    
    # Load data
    print("\nLoading Palladium data...")
    df = load_palladium_data()
    if df is None:
        return
    
    # Add days since start column
    df['Days'] = range(len(df))
    
    print(f"Price range: ${df['Price'].min():.4f} to ${df['Price'].max():.4f}")
    print(f"Days since start: {df['Days'].iloc[0]} to {df['Days'].iloc[-1]}")
    
    # Fit models
    print("\nFitting Palladium growth models...")
    model_results = fit_palladium_growth_models(df)
    
    if not model_results or 'Best_Model' not in model_results:
        print("Error: Could not fit any Palladium growth models.")
        return
    
    # Create chart
    print("\nCreating Palladium growth chart...")
    create_palladium_growth_chart(df, model_results)
    
    # Save results
    print("\nSaving comprehensive results...")
    projections, fair_price = compute_palladium_projections(df, model_results)
    save_palladium_results(model_results, projections, fair_price, df)
    
    # Save projections
    if projections:
        print("\nComputing forward projections (1y, 5y, 10y)...")
        print(f"Palladium projections saved to: Commodities/Palladium/Models/Growth/palladium_projections.txt")
    
    print("\n" + "=" * 60)
    print("PALLADIUM ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Best Palladium Model: {model_results['Best_Model']['name']} (R² = {model_results['Best_Model']['r2']:.6f})")
    print(f"\nResults saved to: Commodities/Palladium/Models/Growth/palladium_growth_models.txt")
    print(f"Projections saved to: Commodities/Palladium/Models/Growth/palladium_projections.txt")
    print(f"Growth chart created: Commodities/Palladium/Models/Growth/palladium_growth_chart_latest.png")
    print(f"\nAll files saved to: Commodities/Palladium/Models/Growth")

if __name__ == "__main__":
    main()
