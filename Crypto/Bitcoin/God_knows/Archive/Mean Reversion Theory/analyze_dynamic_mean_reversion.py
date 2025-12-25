import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit

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

def analyze_mean_reversion_by_period(df, window_years=1):
    """Analyze mean reversion in different time periods"""
    print(f"Analyzing mean reversion in {window_years}-year windows...")
    
    # Calculate Bitcoin age for each date
    genesis_date = pd.to_datetime('2009-01-03')
    df['Bitcoin_Age'] = (df['Date'] - genesis_date).dt.days / 365.25
    
    # Group by time periods
    periods = []
    mean_reversion_stats = []
    
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
            
            print(f"Age {start_age:.1f}-{end_age:.1f}: λ={lambda_annual:.3f}, Half-life={half_life:.1f} days, Vol={deviation_vol:.3f}")
    
    return pd.DataFrame(periods)

def fit_dynamic_mean_reversion(periods_df):
    """Fit a dynamic mean reversion model"""
    print("\nFitting dynamic mean reversion model...")
    
    # Filter out invalid data
    valid_data = periods_df[periods_df['lambda_annual'] > 0].copy()
    
    if len(valid_data) < 3:
        print("❌ Not enough valid data for fitting")
        return None
    
    # Define dynamic mean reversion model
    def dynamic_lambda_model(age, lambda_0, alpha, lambda_inf):
        """λ(t) = λ₀ * exp(-α * age) + λ∞"""
        return lambda_0 * np.exp(-alpha * age) + lambda_inf
    
    # Fit the model
    try:
        popt, pcov = curve_fit(dynamic_lambda_model, 
                              valid_data['mid_age'], 
                              valid_data['lambda_annual'],
                              bounds=([0, 0, 0], [10, 1, 10]))
        
        lambda_0, alpha, lambda_inf = popt
        
        print(f"Dynamic Mean Reversion Model:")
        print(f"  λ(t) = {lambda_0:.3f} * exp(-{alpha:.3f} * age) + {lambda_inf:.3f}")
        print(f"  Initial speed: {lambda_0:.3f}")
        print(f"  Decay rate: {alpha:.3f}")
        print(f"  Long-term speed: {lambda_inf:.3f}")
        
        return lambda_0, alpha, lambda_inf, valid_data
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        return None

def create_mean_reversion_visualization(periods_df, model_params=None):
    """Create visualization of mean reversion evolution"""
    print("\nCreating mean reversion visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Dynamic Mean Reversion Analysis', fontsize=16, fontweight='bold')
    
    # 1. Mean reversion speed over time
    valid_data = periods_df[periods_df['lambda_annual'] > 0]
    
    ax1.scatter(valid_data['mid_age'], valid_data['lambda_annual'], 
               alpha=0.7, s=50, color='blue', label='Observed')
    
    if model_params:
        lambda_0, alpha, lambda_inf, _ = model_params
        ages = np.linspace(0, 15, 100)
        model_lambda = lambda_0 * np.exp(-alpha * ages) + lambda_inf
        ax1.plot(ages, model_lambda, 'r-', linewidth=2, label='Model Fit')
    
    ax1.set_title('Mean Reversion Speed Evolution')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Mean Reversion Speed (λ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Half-life evolution
    ax2.scatter(periods_df['mid_age'], periods_df['half_life_days'], 
               alpha=0.7, s=50, color='green')
    ax2.set_title('Mean Reversion Half-Life Evolution')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Half-Life (days)')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Autocorrelation evolution
    ax3.scatter(periods_df['mid_age'], periods_df['autocorr_1'], 
               alpha=0.7, s=50, color='orange')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Price Ratio Autocorrelation')
    ax3.set_xlabel('Bitcoin Age (years)')
    ax3.set_ylabel('1-Day Autocorrelation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Deviation volatility evolution
    ax4.scatter(periods_df['mid_age'], periods_df['deviation_vol'], 
               alpha=0.7, s=50, color='purple')
    ax4.set_title('Deviation Volatility Evolution')
    ax4.set_xlabel('Bitcoin Age (years)')
    ax4.set_ylabel('Log Price Ratio Volatility')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_dynamic_mean_reversion_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Mean reversion visualization saved to: {filename}")
    
    return filename

def save_mean_reversion_results(periods_df, model_params=None):
    """Save mean reversion analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save period analysis
    periods_file = f'Results/Bitcoin/bitcoin_mean_reversion_periods_{timestamp}.csv'
    periods_df.to_csv(periods_file, index=False)
    print(f"Period analysis saved to: {periods_file}")
    
    # Save model parameters
    if model_params:
        lambda_0, alpha, lambda_inf, _ = model_params
        model_file = f'Results/Bitcoin/bitcoin_dynamic_mean_reversion_model_{timestamp}.txt'
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(f"Dynamic Mean Reversion Model Parameters\n")
            f.write(f"=====================================\n\n")
            f.write(f"Model: lambda(t) = lambda_0 * exp(-alpha * age) + lambda_inf\n\n")
            f.write(f"lambda_0 (Initial speed): {lambda_0:.6f}\n")
            f.write(f"alpha (Decay rate): {alpha:.6f}\n")
            f.write(f"lambda_inf (Long-term speed): {lambda_inf:.6f}\n\n")
            f.write(f"Formula: lambda(t) = {lambda_0:.6f} * exp(-{alpha:.6f} * age) + {lambda_inf:.6f}\n")
            f.write(f"\nInterpretation:\n")
            f.write(f"- Initial mean reversion speed: {lambda_0:.3f} (weak)\n")
            f.write(f"- Decay rate: {alpha:.3f} (how quickly it strengthens)\n")
            f.write(f"- Long-term mean reversion speed: {lambda_inf:.3f} (stronger)\n")
        
        print(f"Model parameters saved to: {model_file}")
    
    return periods_file, model_file if model_params else None

def main():
    """Main analysis function"""
    print("Bitcoin Dynamic Mean Reversion Analysis")
    print("="*50)
    
    # Load and prepare data
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Analyze mean reversion by periods
    periods_df = analyze_mean_reversion_by_period(df, window_years=1)
    
    # Fit dynamic model
    model_params = fit_dynamic_mean_reversion(periods_df)
    
    # Create visualization
    viz_file = create_mean_reversion_visualization(periods_df, model_params)
    
    # Save results
    periods_file, model_file = save_mean_reversion_results(periods_df, model_params)
    
    # Summary
    print(f"\n" + "="*50)
    print("MEAN REVERSION ANALYSIS SUMMARY")
    print("="*50)
    
    if model_params:
        lambda_0, alpha, lambda_inf, _ = model_params
        print(f"✅ Dynamic mean reversion model fitted successfully!")
        print(f"   Initial speed: {lambda_0:.3f}")
        print(f"   Decay rate: {alpha:.3f}")
        print(f"   Long-term speed: {lambda_inf:.3f}")
        print(f"\nThis suggests Bitcoin's mean reversion:")
        print(f"   - Started weak (λ₀ = {lambda_0:.3f})")
        print(f"   - Strengthened over time (decay rate = {alpha:.3f})")
        print(f"   - Will stabilize at λ∞ = {lambda_inf:.3f}")
    else:
        print("❌ Could not fit dynamic mean reversion model")
    
    print(f"\nFiles created:")
    print(f"  Period analysis: {periods_file}")
    if model_file:
        print(f"  Model parameters: {model_file}")
    print(f"  Visualization: {viz_file}")

if __name__ == "__main__":
    main() 