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

def load_bitcoin_data():
    """Load and prepare Bitcoin price data for analysis."""
    
    data_file = "Portfolio/Data/Bitcoin_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Bitcoin data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since Bitcoin genesis (January 3, 2009) for rainbow chart model
    bitcoin_genesis = pd.Timestamp('2009-01-03')
    df['Days_Since_Genesis'] = (df['Date'] - bitcoin_genesis).dt.days
    
    # Also create days since start for other models (keeping compatibility)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Bitcoin data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    print(f"Days since Bitcoin genesis: {df['Days_Since_Genesis'].min()} to {df['Days_Since_Genesis'].max()}")
    
    return df

def load_gold_data():
    """Load and prepare gold price data for analysis."""
    
    data_file = "Portfolio/Data/Gold_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Gold data file not found - {data_file}")
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
    
    print(f"Gold data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def load_silver_data():
    """Load and prepare silver price data for analysis."""
    
    data_file = "Portfolio/Data/Silver_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Silver data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Silver data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.3f} to ${df['Price'].max():.3f}")
    
    return df

def fit_bitcoin_rainbow_model(df):
    """Fit Bitcoin rainbow chart model (log10(price) = a * ln(day) + b)."""
    
    # Skip the first year of Bitcoin (start at day 365 since genesis)
    df_filtered = df[df['Days_Since_Genesis'] >= 365].copy()
    
    if len(df_filtered) == 0:
        print("Warning: Not enough data for Bitcoin rainbow model (need at least 365 days since genesis)")
        return None
    
    # Model: log10(price) = a * ln(day) + b
    X = np.log(df_filtered['Days_Since_Genesis'])
    Y = np.log10(df_filtered['Price'])
    
    # Central model fit
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    R2 = r_value ** 2
    
    model_results = {
        'asset': 'Bitcoin',
        'model_type': 'Rainbow Chart (Log-Log)',
        'formula': f"log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}",
        'slope': slope,
        'intercept': intercept,
        'r2': R2,
        'data_points': len(df_filtered),
        'date_range': f"{df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}",
        'current_price': df_filtered['Price'].iloc[-1],
        'all_time_high': df_filtered['Price'].max(),
        'days_since_genesis_range': f"{df_filtered['Days_Since_Genesis'].min()} to {df_filtered['Days_Since_Genesis'].max()}",
        'filtered_df': df_filtered  # Include filtered data for visualization
    }
    
    return model_results

def fit_standard_models(df, asset_name):
    """Fit standard growth models (linear, exponential, polynomial, logarithmic, power)."""
    
    x = df['Days'].values
    y = df['Price'].values
    
    models = {}
    
    # Linear model
    try:
        popt_linear, _ = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[0.1, 100])
        y_pred_linear = popt_linear[0] * x + popt_linear[1]
        r2_linear = r2_score(y, y_pred_linear)
        rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
        
        models['Linear'] = {
            'formula': f"Price = {popt_linear[0]:.6f} * Days + {popt_linear[1]:.2f}",
            'params': popt_linear,
            'r2': r2_linear,
            'rmse': rmse_linear
        }
    except:
        print(f"Warning: Could not fit linear model for {asset_name}")
    
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
        print(f"Warning: Could not fit exponential model for {asset_name}")
    
    # Polynomial model (cubic)
    try:
        popt_poly, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, x, y, p0=[1e-8, 1e-5, 0.1, 100])
        y_pred_poly = popt_poly[0] * x**3 + popt_poly[1] * x**2 + popt_poly[2] * x + popt_poly[3]
        r2_poly = r2_score(y, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
        
        models['Polynomial'] = {
            'formula': f"Price = {popt_poly[0]:.2e}*Days³ + {popt_poly[1]:.2e}*Days² + {popt_poly[2]:.4f}*Days + {popt_poly[3]:.2f}",
            'params': popt_poly,
            'r2': r2_poly,
            'rmse': rmse_poly
        }
    except:
        print(f"Warning: Could not fit polynomial model for {asset_name}")
    
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
        print(f"Warning: Could not fit logarithmic model for {asset_name}")
    
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
        print(f"Warning: Could not fit power model for {asset_name}")
    
    # Find best model with smart selection to avoid overfitting
    if models:
        # Score models considering both R² and parameter reasonableness
        model_scores = {}
        
        for model_name, model_data in models.items():
            base_score = model_data['r2']
            
            # Penalize models with extremely small coefficients (overfitting)
            if model_name == 'Polynomial' and 'params' in model_data:
                params = model_data['params']
                # Check if cubic and quadratic terms are essentially zero
                if abs(params[0]) < 1e-8 and abs(params[1]) < 1e-6:
                    # Heavily penalize overfitted polynomial models
                    base_score *= 0.5
                    print(f"Warning: {asset_name} Polynomial model has tiny coefficients - penalizing score")
            
            # Penalize models with zero coefficients (broken models)
            if model_name == 'Power' and 'params' in model_data:
                params = model_data['params']
                if abs(params[0]) < 1e-10:  # If coefficient is essentially zero
                    base_score *= 0.1  # Heavily penalize broken power models
                    print(f"Warning: {asset_name} Power model has zero coefficient - penalizing score")
            
            # Slight preference for simpler models when R² is close
            if model_name == 'Linear':
                base_score *= 1.02  # 2% bonus for simplicity
            elif model_name == 'Exponential':
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
        
        print(f"Selected {asset_name} model: {best_model_name} (R² = {best_model['r2']:.6f}, Adjusted Score = {model_scores[best_model_name]:.6f})")
    
    return models

def create_rainbow_chart(df, asset_name, model_results, output_dir):
    """Create a rainbow chart visualization for an asset."""
    
    if asset_name == 'Bitcoin':
        # Bitcoin uses days since genesis
        days_col = 'Days_Since_Genesis'
        model_formula = lambda x: 10**(model_results['slope'] * np.log(x) + model_results['intercept'])
        x_label = 'Days Since Bitcoin Genesis'
        title_suffix = 'Bitcoin Rainbow Chart'
    else:
        # Gold and Silver use days since start
        days_col = 'Days'
        # For metals, we'll create a simple trend line (not rainbow bands)
        # Get the best model parameters based on the model type
        best_model = model_results['Best_Model']
        if best_model['name'] == 'Linear':
            model_formula = lambda x: best_model['params'][0] * x + best_model['params'][1]
        elif best_model['name'] == 'Exponential':
            model_formula = lambda x: best_model['params'][0] * np.exp(best_model['params'][1] * x) + best_model['params'][2]
        elif best_model['name'] == 'Polynomial':
            model_formula = lambda x: best_model['params'][0] * x**3 + best_model['params'][1] * x**2 + best_model['params'][2] * x + best_model['params'][3]
        elif best_model['name'] == 'Power':
            model_formula = lambda x: best_model['params'][0] * np.power(x, best_model['params'][1]) + best_model['params'][2]
        else:
            # Fallback to linear if unknown model type
            model_formula = lambda x: best_model['params'][0] * x + best_model['params'][1]
        
        x_label = 'Days Since Start'
        title_suffix = f'{asset_name} Price Trend'
    
    # Set up the plot
    plt.style.use('default')  # Use default style for better compatibility
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual price
    ax.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label=f'Actual {asset_name} Price')
    
    if asset_name == 'Bitcoin':
        # Create rainbow bands for Bitcoin
        df_filtered = model_results['filtered_df']
        
        # Calculate model prices
        df_filtered['Model_Price'] = df_filtered[days_col].apply(model_formula)
        
        # Create rainbow bands by calculating standard deviations from the model
        log_model = np.log10(df_filtered['Model_Price'])
        log_actual = np.log10(df_filtered['Price'])
        log_deviations = log_actual - log_model
        
        # Calculate band boundaries based on standard deviations
        std_dev = np.std(log_deviations)
        mean_dev = np.mean(log_deviations)
        
        # Define band boundaries (in log space)
        band_boundaries = [
            mean_dev - 2.0 * std_dev,  # Firesale! (2 std dev below)
            mean_dev - 1.0 * std_dev,  # Buy (1 std dev below)
            mean_dev + 1.0 * std_dev,  # Sell (1 std dev above)
            mean_dev + 2.0 * std_dev   # Danger! (2 std dev above)
        ]
        
        # Convert band boundaries back to price space
        band_prices = []
        for boundary in band_boundaries:
            band_price = 10**(log_model + boundary)
            band_prices.append(band_price)
        
        # Plot rainbow bands
        band_names = ['Firesale!', 'Buy', 'Hold', 'Sell', 'Danger!']
        band_colors = ['blue', 'green', 'yellow', 'orange', 'red']
        band_alphas = [0.8, 0.7, 0.6, 0.7, 0.8]
        
        for i in range(len(band_names)):
            if i == 0:
                # First band: from 0 to first boundary
                lower_bound = np.zeros(len(df_filtered))
                upper_bound = band_prices[0]
            elif i == len(band_names) - 1:
                # Last band: from last boundary to infinity
                lower_bound = band_prices[-1]
                upper_bound = df_filtered['Price'].max() * 10
            else:
                # Middle bands: between boundaries
                lower_bound = band_prices[i-1]
                upper_bound = band_prices[i]
            
            ax.fill_between(df_filtered['Date'], lower_bound, upper_bound, 
                           color=band_colors[i], alpha=band_alphas[i], 
                           label=f"{band_names[i]}")
        
        # Plot model line
        ax.semilogy(df_filtered['Date'], df_filtered['Model_Price'], 
                    color='black', linestyle='--', linewidth=2, label='Growth Model')
        
        # Highlight current position
        current_price = df_filtered['Price'].iloc[-1]
        current_date = df_filtered['Date'].iloc[-1]
        ax.scatter(current_date, current_price, color='red', s=100, 
                   zorder=10, label=f'Current: ${current_price:,.0f}')
        
        # Add model information
        formula_text = f'Model: log₁₀(price) = {model_results["slope"]:.3f} × ln(day) + {model_results["intercept"]:.3f}\nR² = {model_results["r2"]:.4f}'
        ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=10, 
                color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
    else:
        # For metals, plot trend line
        df['Model_Price'] = df[days_col].apply(model_formula)
        ax.semilogy(df['Date'], df['Model_Price'], 
                    color='red', linestyle='--', linewidth=2, label=f'{asset_name} Trend Model')
        
        # Highlight current position
        current_price = df['Price'].iloc[-1]
        current_date = df['Date'].iloc[-1]
        ax.scatter(current_date, current_price, color='red', s=100, 
                   zorder=10, label=f'Current: ${current_price:,.2f}')
        
        # Add model information
        best_model = model_results['Best_Model']
        formula_text = f'Best Model: {best_model["name"]}\nR² = {best_model["r2"]:.4f}'
        ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=10, 
                color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{asset_name} Price (USD) - Log Scale', fontsize=12)
    ax.set_title(f'{title_suffix}\nGrowth Model Analysis', fontsize=16, fontweight='bold')
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
    if asset_name == 'Bitcoin':
        ax.set_ylim(0.01, max(df_filtered['Price'].max(), df_filtered['Model_Price'].max()) * 2)
    else:
        ax.set_ylim(0.01, max(df['Price'].max(), df['Model_Price'].max()) * 2)
    
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, f"{asset_name.lower()}_rainbow_chart_latest.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to prevent display
    
    return output_file

def _predict_future_price(df, asset_name, model_results, years_ahead):
    """Predict future price using the fitted model for a given horizon (in years)."""
    days_in_year = 365
    if asset_name == 'Bitcoin':
        days_col = 'Days_Since_Genesis'
        last_day = int(df[days_col].iloc[-1])
        future_day = last_day + years_ahead * days_in_year
        price = 10**(model_results['slope'] * np.log(future_day) + model_results['intercept'])
        return float(price)
    else:
        days_col = 'Days'
        last_day = int(df[days_col].iloc[-1])
        future_day = last_day + years_ahead * days_in_year
        best_model = model_results['Best_Model']
        if best_model['name'] == 'Linear':
            return float(best_model['params'][0] * future_day + best_model['params'][1])
        elif best_model['name'] == 'Exponential':
            return float(best_model['params'][0] * np.exp(best_model['params'][1] * future_day) + best_model['params'][2])
        elif best_model['name'] == 'Polynomial':
            return float(best_model['params'][0] * future_day**3 + best_model['params'][1] * future_day**2 + best_model['params'][2] * future_day + best_model['params'][3])
        elif best_model['name'] == 'Power':
            return float(best_model['params'][0] * np.power(future_day, best_model['params'][1]) + best_model['params'][2])
        else:
            return float(best_model['params'][0] * future_day + best_model['params'][1])

def compute_projections(df, asset_name, model_results):
    """Compute 1, 2, 3, 5, and 10-year projected returns using the fitted model."""
    if model_results is None:
        return None
    current_price = float(df['Price'].iloc[-1])
    # Model-implied fair price today
    if asset_name == 'Bitcoin':
        days_col = 'Days_Since_Genesis'
        last_day = int(df[days_col].iloc[-1])
        fair_price = float(10**(model_results['slope'] * np.log(last_day) + model_results['intercept']))
    else:
        days_col = 'Days'
        last_day = int(df[days_col].iloc[-1])
        best_model = model_results['Best_Model']
        if best_model['name'] == 'Linear':
            fair_price = float(best_model['params'][0] * last_day + best_model['params'][1])
        elif best_model['name'] == 'Exponential':
            fair_price = float(best_model['params'][0] * np.exp(best_model['params'][1] * last_day) + best_model['params'][2])
        elif best_model['name'] == 'Polynomial':
            fair_price = float(best_model['params'][0] * last_day**3 + best_model['params'][1] * last_day**2 + best_model['params'][2] * last_day + best_model['params'][3])
        elif best_model['name'] == 'Power':
            fair_price = float(best_model['params'][0] * np.power(last_day, best_model['params'][1]) + best_model['params'][2])
        else:
            fair_price = float(best_model['params'][0] * last_day + best_model['params'][1])
    horizons = [1, 2, 3, 5, 10]
    projections = {}
    for y in horizons:
        future_price = _predict_future_price(df, asset_name, model_results, y)
        ret_pct = (future_price / current_price - 1.0) * 100.0
        projections[y] = {
            'future_price': future_price,
            'return_pct': ret_pct,
            'fair_return_pct': (future_price / fair_price - 1.0) * 100.0
        }
    return {
        'asset': asset_name,
        'as_of_date': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        'current_price': current_price,
        'fair_price': fair_price,
        'projections': projections
    }

def save_projections(bitcoin_proj, gold_proj, silver_proj):
    """Write projections to Portfolio/Models/updated_projections.txt."""
    os.makedirs("Portfolio/Models", exist_ok=True)
    results_file = "Portfolio/Models/updated_projections.txt"
    with open(results_file, 'w') as f:
        f.write("MODEL-BASED FORWARD PROJECTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        def write_block(proj):
            if not proj:
                return
            f.write(f"{proj['asset'].upper()} PROJECTIONS\n")
            f.write("-" * (len(proj['asset']) + 12) + "\n")
            f.write(f"As of: {proj['as_of_date']}\n")
            f.write(f"Current Price: ${proj['current_price']:,.2f}\n")
            f.write(f"Model Fair Price (today): ${proj['fair_price']:,.2f}\n")
            for years in [1, 2, 3, 5, 10]:
                p = proj['projections'][years]
                f.write(f"{years}-Year Projected Price: ${p['future_price']:,.2f} | Return vs Current: {p['return_pct']:.2f}%\n")
            f.write("Fair-Based Growth Rates (from model fair price):\n")
            for years in [1, 2, 3, 5, 10]:
                p = proj['projections'][years]
                f.write(f"  {years}-Year Growth: {p['fair_return_pct']:.2f}%\n")
            f.write("\n")

        write_block(bitcoin_proj)
        write_block(gold_proj)
        write_block(silver_proj)

        f.write("Notes:\n")
        f.write("- Projections use the latest fitted models in this session.\n")
        f.write("- Returns are computed versus the latest observed price.\n")
        f.write("- Horizons are 1, 2, 3, 5, and 10 calendar years (365d per year).\n")

    print(f"Projections saved to: {results_file}")
    return results_file

def save_comprehensive_results(bitcoin_results, gold_results, silver_results):
    """Save all growth model results to a comprehensive file."""
    
    # Create Portfolio/Models directory if it doesn't exist
    os.makedirs("Portfolio/Models", exist_ok=True)
    
    results_file = "Portfolio/Models/updated_models.txt"
    
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Sources: Updated Portfolio/Data/ files\n\n")
        
        # Bitcoin Results
        f.write("BITCOIN GROWTH MODELS\n")
        f.write("-" * 30 + "\n")
        if bitcoin_results:
            f.write(f"Rainbow Chart Model:\n")
            f.write(f"  Formula: {bitcoin_results['formula']}\n")
            f.write(f"  R² Score: {bitcoin_results['r2']:.6f}\n")
            f.write(f"  Data Points: {bitcoin_results['data_points']:,}\n")
            f.write(f"  Date Range: {bitcoin_results['date_range']}\n")
            f.write(f"  Days Since Genesis: {bitcoin_results['days_since_genesis_range']}\n")
            f.write(f"  Current Price: ${bitcoin_results['current_price']:,.2f}\n")
            f.write(f"  All-Time High: ${bitcoin_results['all_time_high']:,.2f}\n\n")
        
        # Gold Results
        f.write("GOLD GROWTH MODELS\n")
        f.write("-" * 25 + "\n")
        if gold_results:
            for model_name, model_data in gold_results.items():
                if model_name != 'Best_Model':
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Formula: {model_data['formula']}\n")
                    f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                    f.write(f"  RMSE: {model_data['rmse']:.2f}\n\n")
            
            if 'Best_Model' in gold_results:
                f.write(f"Best Gold Model: {gold_results['Best_Model']['name']}\n")
                f.write(f"  Formula: {gold_results['Best_Model']['formula']}\n")
                f.write(f"  R² Score: {gold_results['Best_Model']['r2']:.6f}\n")
                f.write(f"  RMSE: {gold_results['Best_Model']['rmse']:.2f}\n\n")
        
        # Silver Results
        f.write("SILVER GROWTH MODELS\n")
        f.write("-" * 27 + "\n")
        if silver_results:
            for model_name, model_data in silver_results.items():
                if model_name != 'Best_Model':
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Formula: {model_data['formula']}\n")
                    f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                    f.write(f"  RMSE: {model_data['rmse']:.2f}\n\n")
            
            if 'Best_Model' in silver_results:
                f.write(f"Best Silver Model: {silver_results['Best_Model']['name']}\n")
                f.write(f"  Formula: {silver_results['Best_Model']['formula']}\n")
                f.write(f"  R² Score: {silver_results['Best_Model']['r2']:.6f}\n")
                f.write(f"  RMSE: {silver_results['Best_Model']['rmse']:.2f}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write("All growth models have been updated using the latest data from:\n")
        f.write("- Portfolio/Data/Bitcoin_all_time_price.csv\n")
        f.write("- Portfolio/Data/Gold_all_time_price.csv\n")
        f.write("- Portfolio/Data/Silver_all_time_price.csv\n\n")
        f.write("Bitcoin Rainbow Chart Model Notes:\n")
        f.write("- Uses days since Bitcoin genesis (January 3, 2009)\n")
        f.write("- Skips first year (days 0-364) for stability\n")
        f.write("- Formula: log10(price) = slope * ln(days_since_genesis) + intercept\n\n")
        f.write("The models can be used for:\n")
        f.write("- Price trend analysis\n")
        f.write("- Future price predictions\n")
        f.write("- Investment timing decisions\n")
        f.write("- Portfolio allocation strategies\n")
    
    print(f"Comprehensive results saved to: {results_file}")
    return results_file

def main():
    """Main function to update all growth models."""
    
    print("Comprehensive Growth Model Update")
    print("=" * 40)
    
    # Load data
    print("\nLoading updated data files...")
    bitcoin_df = load_bitcoin_data()
    gold_df = load_gold_data()
    silver_df = load_silver_data()
    
    if bitcoin_df is None or gold_df is None or silver_df is None:
        print("Error: Could not load all data files.")
        return
    
    # Fit models
    print("\nFitting growth models...")
    
    # Bitcoin rainbow model
    print("Fitting Bitcoin rainbow chart model...")
    bitcoin_results = fit_bitcoin_rainbow_model(bitcoin_df)
    
    # Gold standard models
    print("Fitting Gold growth models...")
    gold_results = fit_standard_models(gold_df, "Gold")
    
    # Silver standard models
    print("Fitting Silver growth models...")
    silver_results = fit_standard_models(silver_df, "Silver")
    
    # Create rainbow charts
    print("\nCreating rainbow chart visualizations...")
    output_dir = "Portfolio/Models"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_files = []
    
    if bitcoin_results:
        print("Creating Bitcoin rainbow chart...")
        bitcoin_chart = create_rainbow_chart(bitcoin_df, 'Bitcoin', bitcoin_results, output_dir)
        chart_files.append(bitcoin_chart)
        print(f"  Bitcoin chart saved to: {bitcoin_chart}")
    
    if gold_results:
        print("Creating Gold trend chart...")
        gold_chart = create_rainbow_chart(gold_df, 'Gold', gold_results, output_dir)
        chart_files.append(gold_chart)
        print(f"  Gold chart saved to: {gold_chart}")
    
    if silver_results:
        print("Creating Silver trend chart...")
        silver_chart = create_rainbow_chart(silver_df, 'Silver', silver_results, output_dir)
        chart_files.append(silver_chart)
        print(f"  Silver chart saved to: {silver_chart}")
    
    # Save results
    print("\nSaving comprehensive results...")
    results_file = save_comprehensive_results(bitcoin_results, gold_results, silver_results)
    
    # Compute and save projections
    print("\nComputing forward projections (1y, 2y, 3y, 5y, 10y)...")
    btc_proj = compute_projections(bitcoin_df if bitcoin_results else None and bitcoin_df, 'Bitcoin', bitcoin_results)
    gold_proj = compute_projections(gold_df if gold_results else None and gold_df, 'Gold', gold_results)
    silver_proj = compute_projections(silver_df if silver_results else None and silver_df, 'Silver', silver_results)
    projections_file = save_projections(btc_proj, gold_proj, silver_proj)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if bitcoin_results:
        print(f"Bitcoin Rainbow Model: R² = {bitcoin_results['r2']:.6f}")
    
    if gold_results and 'Best_Model' in gold_results:
        print(f"Best Gold Model: {gold_results['Best_Model']['name']} (R² = {gold_results['Best_Model']['r2']:.6f})")
    
    if silver_results and 'Best_Model' in silver_results:
        print(f"Best Silver Model: {silver_results['Best_Model']['name']} (R² = {silver_results['Best_Model']['r2']:.6f})")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Projections saved to: {projections_file}")
    
    if chart_files:
        print(f"\nRainbow charts created:")
        for chart_file in chart_files:
            print(f"  - {chart_file}")
        print(f"\nAll charts saved to: {output_dir}")

if __name__ == "__main__":
    main() 