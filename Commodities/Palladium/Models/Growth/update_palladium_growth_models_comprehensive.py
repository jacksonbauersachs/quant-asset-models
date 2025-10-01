import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from datetime import datetime
from sklearn.metrics import r2_score
import os

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        # Load the cleaned comprehensive historical data
        df = pd.read_csv("Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv")
        
        # Convert date column - handle both ISO and M/D/YYYY formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add day number (days since first date)
        df['days'] = (df['date'] - df['date'].iloc[0]).dt.days
        
        # Find the value column (handle spaces in column names)
        value_col = None
        for col in df.columns:
            if 'value' in col.lower():
                value_col = col
                break
        
        if value_col is None:
            print("Error: Could not find value column")
            return None
        
        print(f"Loaded {len(df)} records")
        print(f"Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[value_col].min():.2f} to ${df[value_col].max():.2f}")
        
        # Rename the value column for consistency
        df = df.rename(columns={value_col: 'value'})
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def fit_palladium_growth_models(df):
    """Fit various growth models to palladium data."""
    print("\nFitting growth models...")
    
    models = {}
    x = df['days'].values
    y = df['value'].values
    
    # Power model: y = a * x^b
    try:
        def power_func(x, a, b):
            return a * np.power(x, b)
        
        # Initial parameters for power model
        p0_power = [1.0, 0.5]
        popt_power, _ = curve_fit(power_func, x, y, p0=p0_power, maxfev=5000)
        y_pred_power = power_func(x, *popt_power)
        r2_power = r2_score(y, y_pred_power)
        
        models['Power'] = {
            'params': popt_power,
            'r2': r2_power,
            'func': power_func,
            'formula': f"y = {popt_power[0]:.4f} * x^{popt_power[1]:.4f}"
        }
        print(f"Power model: R² = {r2_power:.4f}")
        
    except Exception as e:
        print(f"Power model failed: {e}")
    
    # Exponential model: y = a * e^(b * x)
    try:
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        # Initial parameters for exponential model
        p0_exp = [50.0, 0.0001]
        popt_exp, _ = curve_fit(exp_func, x, y, p0=p0_exp, maxfev=5000)
        y_pred_exp = exp_func(x, *popt_exp)
        r2_exp = r2_score(y, y_pred_exp)
        
        models['Exponential'] = {
            'params': popt_exp,
            'r2': r2_exp,
            'func': exp_func,
            'formula': f"y = {popt_exp[0]:.4f} * e^({popt_exp[1]:.6f} * x)"
        }
        print(f"Exponential model: R² = {r2_exp:.4f}")
        
    except Exception as e:
        print(f"Exponential model failed: {e}")
    
    # Polynomial model removed as requested
    
    # Logarithmic model: y = a + b * ln(x)
    try:
        def log_func(x, a, b):
            return a + b * np.log(x + 1)  # +1 to avoid log(0)
        
        # Initial parameters for logarithmic model
        p0_log = [50.0, 10.0]
        popt_log, _ = curve_fit(log_func, x, y, p0=p0_log, maxfev=5000)
        y_pred_log = log_func(x, *popt_log)
        r2_log = r2_score(y, y_pred_log)
        
        models['Logarithmic'] = {
            'params': popt_log,
            'r2': r2_log,
            'func': log_func,
            'formula': f"y = {popt_log[0]:.4f} + {popt_log[1]:.4f} * ln(x)"
        }
        print(f"Logarithmic model: R² = {r2_log:.4f}")
        
    except Exception as e:
        print(f"Logarithmic model failed: {e}")
    
    return models

def select_best_model(models):
    """Select the best model based on R² score."""
    if not models:
        return None
    
    best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"R² = {best_model['r2']:.4f}")
    print(f"Formula: {best_model['formula']}")
    
    return best_model_name, best_model

def create_palladium_growth_chart(df, models, best_model_name, best_model):
    """Create growth charts for palladium."""
    print("\nCreating growth charts...")
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    
    # Convert days to years for x-axis
    start_year = df['date'].iloc[0].year
    df['years'] = (df['days'] / 365.25) + start_year
    
    # Create x_plot for smooth curves (in years)
    x_plot_days = np.linspace(df['days'].min(), df['days'].max(), 1000)
    x_plot_years = (x_plot_days / 365.25) + start_year
    
    # Chart 1: Log10 Y-axis
    plt.figure(figsize=(12, 8))
    plt.semilogy(df['years'], df['value'], 'o', markersize=1, alpha=0.6, color='blue', label='Actual Data')
    
    # Plot only the best model
    try:
        y_plot = best_model['func'](x_plot_days, *best_model['params'])
        plt.semilogy(x_plot_years, y_plot, '--', linewidth=3, color='red', 
                    label=f"{best_model_name} (R² = {best_model['r2']:.4f})")
    except:
        pass
    
    plt.xlabel('Year')
    plt.ylabel('Palladium Price ($) - Log Scale')
    plt.title('Palladium Price Growth Model (Log10 Y-axis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add formula and R² at the bottom
    plt.figtext(0.5, 0.02, f"Best Model: {best_model_name} | Formula: {best_model['formula']} | R² = {best_model['r2']:.4f}", 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the log10 chart
    chart_path_log = "Commodities/Palladium/Models/Growth/palladium_growth_chart_log10.png"
    plt.savefig(chart_path_log, dpi=300, bbox_inches='tight')
    print(f"Log10 chart saved: {chart_path_log}")
    
    plt.close()  # Close the figure to free memory
    
    # Chart 2: Normal Y-axis
    plt.figure(figsize=(12, 8))
    plt.plot(df['years'], df['value'], 'o', markersize=1, alpha=0.6, color='blue', label='Actual Data')
    
    # Plot only the best model
    try:
        y_plot = best_model['func'](x_plot_days, *best_model['params'])
        plt.plot(x_plot_years, y_plot, '--', linewidth=3, color='red', 
                label=f"{best_model_name} (R² = {best_model['r2']:.4f})")
    except:
        pass
    
    plt.xlabel('Year')
    plt.ylabel('Palladium Price ($)')
    plt.title('Palladium Price Growth Model (Normal Y-axis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add formula and R² at the bottom
    plt.figtext(0.5, 0.02, f"Best Model: {best_model_name} | Formula: {best_model['formula']} | R² = {best_model['r2']:.4f}", 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the normal chart
    chart_path_normal = "Commodities/Palladium/Models/Growth/palladium_growth_chart_normal.png"
    plt.savefig(chart_path_normal, dpi=300, bbox_inches='tight')
    print(f"Normal chart saved: {chart_path_normal}")
    
    plt.close()  # Close the figure to free memory

def _predict_palladium_future_price(best_model_name, best_model, days_from_start):
    """Predict palladium price for a given number of days from start."""
    if best_model_name == 'Power':
        return best_model['func'](days_from_start, *best_model['params'])
    elif best_model_name == 'Exponential':
        return best_model['func'](days_from_start, *best_model['params'])
    elif best_model_name == 'Logarithmic':
        return best_model['func'](days_from_start, *best_model['params'])
    else:
        return None

def compute_palladium_projections(df, best_model_name, best_model):
    """Compute future price projections."""
    print(f"\nComputing projections using {best_model_name} model...")
    
    # Get the last date and days
    last_date = df['date'].iloc[-1]
    last_days = df['days'].iloc[-1]
    
    # Projection periods (in years from current date)
    projection_years = [1, 2, 3, 5, 10]
    
    projections = []
    
    for years_ahead in projection_years:
        # Calculate days from start for the projection
        projection_days = last_days + (years_ahead * 365.25)
        
        predicted_price = _predict_palladium_future_price(best_model_name, best_model, projection_days)
        if predicted_price and predicted_price > 0:
            projection_date = last_date + pd.Timedelta(days=years_ahead * 365.25)
            
            projections.append({
                'Years_Ahead': years_ahead,
                'Projected_Date': projection_date.strftime('%Y-%m-%d'),
                'Projected_Price': predicted_price
            })
    
    return projections

def save_palladium_model_results(models, best_model_name, best_model, projections, df):
    """Save model results to files."""
    print("\nSaving model results...")
    
    # Save model details
    model_file = "Commodities/Palladium/Models/Growth/palladium_growth_models_comprehensive.txt"
    with open(model_file, 'w') as f:
        f.write("Palladium Growth Model Analysis (Comprehensive Data)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Model Performance:\n")
        for model_name, model in models.items():
            f.write(f"{model_name}: R² = {model['r2']:.4f}\n")
            f.write(f"  Formula: {model['formula']}\n\n")
        
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"R² = {best_model['r2']:.4f}\n")
        f.write(f"Formula: {best_model['formula']}\n")
    
    # Save projections
    if projections:
        projection_file = "Commodities/Palladium/Models/Growth/palladium_projections.txt"
        with open(projection_file, 'w') as f:
            f.write("PALLADIUM MODEL-BASED FORWARD PROJECTIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("PALLADIUM PROJECTIONS\n")
            f.write("-" * 20 + "\n")
            f.write(f"As of: {datetime.now().strftime('%Y-%m-%d 00:00:00')}\n")
            
            # Get current price from the data
            current_price = df['value'].iloc[-1]
            f.write(f"Current Price: ${current_price:.4f}\n")
            
            # Get fair value (model price for today)
            current_days = df['days'].iloc[-1]
            fair_price = _predict_palladium_future_price(best_model_name, best_model, current_days)
            f.write(f"Model Fair Price (today): ${fair_price:.4f}\n")
            
            # Write projections in the same format as the original
            for proj in projections:
                years = proj['Years_Ahead']
                price = proj['Projected_Price']
                return_pct = ((price - current_price) / current_price) * 100
                f.write(f"{years:.0f}-Year Projected Price: ${price:.4f} | Return vs Current: {return_pct:.2f}%\n")
            
            f.write("\nFair-Based Growth Rates (from model fair price):\n")
            for proj in projections:
                years = proj['Years_Ahead']
                price = proj['Projected_Price']
                growth_rate = ((price / fair_price) - 1) * 100
                f.write(f"  {years:.0f}-Year Growth: {growth_rate:.2f}%\n")
            
            f.write("\nNotes:\n")
            f.write("- Projections use the latest fitted Palladium growth model.\n")
            f.write("- Model fair price represents the theoretical value based on historical trends.\n")
            f.write("- These are statistical projections, not financial advice.\n")
    
    print(f"Model details saved: {model_file}")
    if projections:
        print(f"Projections saved: {projection_file}")

def main():
    """Main function to run palladium growth model analysis."""
    print("Palladium Growth Model Analysis (Comprehensive Data)")
    print("=" * 60)
    print("Models: Power, Exponential, Logarithmic")
    print()
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit models
    models = fit_palladium_growth_models(df)
    if not models:
        print("No models could be fitted!")
        return
    
    # Select best model
    best_model_name, best_model = select_best_model(models)
    if best_model_name is None:
        print("No valid model found!")
        return
    
    # Create chart
    create_palladium_growth_chart(df, models, best_model_name, best_model)
    
    # Compute projections
    projections = compute_palladium_projections(df, best_model_name, best_model)
    
    # Save results
    save_palladium_model_results(models, best_model_name, best_model, projections, df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
