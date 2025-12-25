import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit
from scipy.misc import derivative

def load_bitcoin_data():
    """Load Bitcoin historical data"""
    print("Loading Bitcoin historical data...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    return df

def calculate_formula_fair_values(df):
    """Calculate formula fair values for each date"""
    print("Calculating formula fair values...")
    
    # Load growth formula parameters
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    # Calculate fair values
    fair_values = []
    for _, row in df.iterrows():
        # Calculate days since Bitcoin genesis (January 3, 2009)
        genesis_date = pd.to_datetime('2009-01-03')
        days_since_genesis = (row['Date'] - genesis_date).days
        
        if days_since_genesis > 0:
            fair_value = 10**(a * np.log(days_since_genesis) + b)
        else:
            fair_value = 0.01  # Very early Bitcoin price
        
        fair_values.append(fair_value)
    
    df['Fair_Value'] = fair_values
    df['Price_Ratio'] = df['Price'] / df['Fair_Value']
    df['Log_Price_Ratio'] = np.log(df['Price_Ratio'])
    
    return df

def analyze_mean_reversion_by_period(df, window_years=0.5):
    """Analyze mean reversion in smaller time windows for better resolution"""
    print(f"Analyzing mean reversion in {window_years}-year windows...")
    
    # Calculate Bitcoin age for each date
    genesis_date = pd.to_datetime('2009-01-03')
    df['Bitcoin_Age'] = (df['Date'] - genesis_date).dt.days / 365.25
    
    # Group by time periods
    periods = []
    
    for start_age in np.arange(0, 15, window_years):
        end_age = start_age + window_years
        
        # Get data for this period
        period_data = df[(df['Bitcoin_Age'] >= start_age) & (df['Bitcoin_Age'] < end_age)].copy()
        
        if len(period_data) < 30:  # Need minimum data points
            continue
        
        # Calculate mean reversion statistics
        log_ratios = period_data['Log_Price_Ratio'].values
        
        # Calculate autocorrelation (mean reversion indicator)
        if len(log_ratios) > 1:
            autocorr_1 = np.corrcoef(log_ratios[:-1], log_ratios[1:])[0, 1]
            
            # Calculate half-life (how long it takes to revert halfway)
            if autocorr_1 < 1 and autocorr_1 > 0:
                half_life = -np.log(2) / np.log(autocorr_1)
            else:
                half_life = np.inf
            
            # Calculate mean reversion speed (λ)
            if autocorr_1 < 1:
                lambda_speed = -np.log(autocorr_1)  # Daily mean reversion speed
                lambda_annual = lambda_speed * 365.25  # Annualized
            else:
                lambda_annual = 0
            
            # Calculate volatility of deviations
            deviation_vol = np.std(log_ratios)
            
            periods.append({
                'start_age': start_age,
                'end_age': end_age,
                'mid_age': (start_age + end_age) / 2,
                'autocorr_1': autocorr_1,
                'half_life_days': half_life,
                'lambda_annual': lambda_annual,
                'deviation_vol': deviation_vol,
                'data_points': len(period_data)
            })
    
    return pd.DataFrame(periods)

def calculate_rate_of_change(periods_df):
    """Calculate the rate of change of mean reversion speed"""
    print("Calculating rate of change of mean reversion...")
    
    # Sort by age
    periods_df = periods_df.sort_values('mid_age').reset_index(drop=True)
    
    # Calculate rate of change (derivative)
    rates_of_change = []
    
    for i in range(1, len(periods_df)):
        current_lambda = periods_df.iloc[i]['lambda_annual']
        prev_lambda = periods_df.iloc[i-1]['lambda_annual']
        current_age = periods_df.iloc[i]['mid_age']
        prev_age = periods_df.iloc[i-1]['mid_age']
        
        # Rate of change = change in lambda / change in age
        rate_of_change = (current_lambda - prev_lambda) / (current_age - prev_age)
        
        rates_of_change.append({
            'age': current_age,
            'lambda': current_lambda,
            'rate_of_change': rate_of_change,
            'period': f"{prev_age:.1f}-{current_age:.1f}"
        })
    
    return pd.DataFrame(rates_of_change)

def fit_rate_of_change_model(rates_df):
    """Fit a model to the rate of change of mean reversion"""
    print("Fitting rate of change model...")
    
    # Filter out extreme values
    valid_data = rates_df[np.abs(rates_df['rate_of_change']) < 50].copy()
    
    if len(valid_data) < 5:
        print("❌ Not enough valid data for fitting")
        return None
    
    # Try different models for rate of change
    models = {
        'linear': lambda x, a, b: a * x + b,
        'exponential': lambda x, a, b, c: a * np.exp(-b * x) + c,
        'polynomial': lambda x, a, b, c: a * x**2 + b * x + c
    }
    
    best_model = None
    best_r2 = -np.inf
    
    for model_name, model_func in models.items():
        try:
            popt, _ = curve_fit(model_func, valid_data['age'], valid_data['rate_of_change'])
            y_pred = model_func(valid_data['age'], *popt)
            r2 = 1 - np.sum((valid_data['rate_of_change'] - y_pred)**2) / np.sum((valid_data['rate_of_change'] - valid_data['rate_of_change'].mean())**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = (model_name, model_func, popt)
                
        except Exception as e:
            continue
    
    if best_model:
        model_name, model_func, popt = best_model
        print(f"Best model: {model_name} (R² = {best_r2:.3f})")
        print(f"Parameters: {popt}")
        return model_name, model_func, popt, best_r2
    
    return None

def create_rate_of_change_visualization(periods_df, rates_df, model_fit=None):
    """Create comprehensive visualization of mean reversion rate of change"""
    print("Creating rate of change visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Mean Reversion Rate of Change Analysis', fontsize=16, fontweight='bold')
    
    # 1. Mean reversion speed over time
    valid_periods = periods_df[periods_df['lambda_annual'] > 0]
    ax1.scatter(valid_periods['mid_age'], valid_periods['lambda_annual'], 
               alpha=0.7, s=50, color='blue', label='Observed')
    ax1.set_title('Mean Reversion Speed Evolution')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Mean Reversion Speed (λ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rate of change of mean reversion speed
    ax2.scatter(rates_df['age'], rates_df['rate_of_change'], 
               alpha=0.7, s=50, color='red', label='Observed Rate of Change')
    
    if model_fit:
        model_name, model_func, popt, r2 = model_fit
        ages = np.linspace(rates_df['age'].min(), rates_df['age'].max(), 100)
        predicted_rates = model_func(ages, *popt)
        ax2.plot(ages, predicted_rates, 'g-', linewidth=2, 
                label=f'{model_name.capitalize()} Fit (R²={r2:.3f})')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Rate of Change of Mean Reversion Speed')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Rate of Change (dλ/dt)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative change in mean reversion speed
    cumulative_change = rates_df['rate_of_change'].cumsum()
    ax3.plot(rates_df['age'], cumulative_change, 'purple', linewidth=2, marker='o')
    ax3.set_title('Cumulative Change in Mean Reversion Speed')
    ax3.set_xlabel('Bitcoin Age (years)')
    ax3.set_ylabel('Cumulative Change')
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean reversion speed vs rate of change
    ax4.scatter(rates_df['lambda'], rates_df['rate_of_change'], 
               alpha=0.7, s=50, color='orange')
    ax4.set_title('Mean Reversion Speed vs Rate of Change')
    ax4.set_xlabel('Mean Reversion Speed (λ)')
    ax4.set_ylabel('Rate of Change (dλ/dt)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_mean_reversion_rate_of_change_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Rate of change visualization saved to: {filename}")
    
    return filename

def create_detailed_analysis_plot(periods_df, rates_df, model_fit=None):
    """Create a detailed analysis plot with formulas and interpretations"""
    print("Creating detailed analysis plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Bitcoin Mean Reversion: Speed and Rate of Change Analysis', fontsize=16, fontweight='bold')
    
    # Top plot: Mean reversion speed with fitted model
    valid_periods = periods_df[periods_df['lambda_annual'] > 0]
    ax1.scatter(valid_periods['mid_age'], valid_periods['lambda_annual'], 
               alpha=0.7, s=60, color='blue', label='Observed λ(t)')
    
    # Fit the original mean reversion model
    try:
        def lambda_model(age, lambda_0, alpha, lambda_inf):
            return lambda_0 * np.exp(-alpha * age) + lambda_inf
        
        popt, _ = curve_fit(lambda_model, valid_periods['mid_age'], valid_periods['lambda_annual'])
        lambda_0, alpha, lambda_inf = popt
        
        ages = np.linspace(0, 15, 100)
        model_lambda = lambda_model(ages, *popt)
        ax1.plot(ages, model_lambda, 'r-', linewidth=2, 
                label=f'λ(t) = {lambda_0:.1f}*exp(-{alpha:.3f}*t) + {lambda_inf:.1f}')
        
        # Calculate and plot rate of change
        def rate_of_change(age):
            return -lambda_0 * alpha * np.exp(-alpha * age)
        
        model_rates = rate_of_change(ages)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(ages, model_rates, 'g--', linewidth=2, alpha=0.7, label='dλ/dt (theoretical)')
        ax1_twin.set_ylabel('Rate of Change (dλ/dt)', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        
    except Exception as e:
        print(f"Could not fit model: {e}")
    
    ax1.set_title('Mean Reversion Speed Evolution with Rate of Change')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Mean Reversion Speed (λ)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Rate of change analysis
    ax2.scatter(rates_df['age'], rates_df['rate_of_change'], 
               alpha=0.7, s=50, color='red', label='Observed dλ/dt')
    
    if model_fit:
        model_name, model_func, popt, r2 = model_fit
        ages = np.linspace(rates_df['age'].min(), rates_df['age'].max(), 100)
        predicted_rates = model_func(ages, *popt)
        ax2.plot(ages, predicted_rates, 'g-', linewidth=2, 
                label=f'{model_name.capitalize()} Fit (R²={r2:.3f})')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Rate of Change of Mean Reversion Speed')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Rate of Change (dλ/dt)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_mean_reversion_detailed_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {filename}")
    
    return filename

def save_rate_of_change_results(periods_df, rates_df, model_fit=None):
    """Save rate of change analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save rate of change data
    rates_file = f'Results/Bitcoin/bitcoin_mean_reversion_rate_of_change_{timestamp}.csv'
    rates_df.to_csv(rates_file, index=False)
    print(f"Rate of change data saved to: {rates_file}")
    
    # Save model parameters
    if model_fit:
        model_name, model_func, popt, r2 = model_fit
        model_file = f'Results/Bitcoin/bitcoin_mean_reversion_rate_model_{timestamp}.txt'
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(f"Mean Reversion Rate of Change Model\n")
            f.write(f"==================================\n\n")
            f.write(f"Model Type: {model_name.capitalize()}\n")
            f.write(f"R-squared: {r2:.6f}\n\n")
            f.write(f"Parameters: {popt}\n\n")
            f.write(f"Interpretation:\n")
            f.write(f"- Model explains {r2*100:.1f}% of rate of change variation\n")
            f.write(f"- Shows how mean reversion speed changes over time\n")
            f.write(f"- Can be used to predict future mean reversion behavior\n")
        
        print(f"Rate of change model saved to: {model_file}")
    
    return rates_file, model_file if model_fit else None

def main():
    """Main analysis function"""
    print("Bitcoin Mean Reversion Rate of Change Analysis")
    print("="*50)
    
    # Load and prepare data
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Analyze mean reversion by periods (smaller windows for better resolution)
    periods_df = analyze_mean_reversion_by_period(df, window_years=0.5)
    
    # Calculate rate of change
    rates_df = calculate_rate_of_change(periods_df)
    
    # Fit rate of change model
    model_fit = fit_rate_of_change_model(rates_df)
    
    # Create visualizations
    viz_file1 = create_rate_of_change_visualization(periods_df, rates_df, model_fit)
    viz_file2 = create_detailed_analysis_plot(periods_df, rates_df, model_fit)
    
    # Save results
    rates_file, model_file = save_rate_of_change_results(periods_df, rates_df, model_fit)
    
    # Summary
    print(f"\n" + "="*50)
    print("RATE OF CHANGE ANALYSIS SUMMARY")
    print("="*50)
    
    if model_fit:
        model_name, model_func, popt, r2 = model_fit
        print(f"✅ Rate of change model fitted successfully!")
        print(f"   Model type: {model_name}")
        print(f"   R-squared: {r2:.3f}")
        print(f"   Parameters: {popt}")
        
        # Calculate average rate of change
        avg_rate = rates_df['rate_of_change'].mean()
        print(f"   Average rate of change: {avg_rate:.3f}")
        
        print(f"\nThis shows how Bitcoin's mean reversion speed changes over time:")
        print(f"   - Rate of change varies by {avg_rate:.3f} per year")
        print(f"   - Model explains {r2*100:.1f}% of the variation")
        print(f"   - Can predict future mean reversion behavior")
    else:
        print("❌ Could not fit rate of change model")
    
    print(f"\nFiles created:")
    print(f"  Rate of change data: {rates_file}")
    if model_file:
        print(f"  Rate of change model: {model_file}")
    print(f"  Visualizations: {viz_file1}, {viz_file2}")

if __name__ == "__main__":
    main() 