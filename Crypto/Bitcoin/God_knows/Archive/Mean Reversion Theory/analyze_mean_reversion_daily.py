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

def analyze_mean_reversion_rolling(df, window_days=30):
    """Analyze mean reversion using rolling windows for high resolution"""
    print(f"Analyzing mean reversion with {window_days}-day rolling windows...")
    
    # Calculate Bitcoin age for each date
    genesis_date = pd.to_datetime('2009-01-03')
    df['Bitcoin_Age'] = (df['Date'] - genesis_date).dt.days / 365.25
    
    # Calculate rolling autocorrelation
    log_ratios = df['Log_Price_Ratio'].values
    dates = df['Date'].values
    ages = df['Bitcoin_Age'].values
    
    rolling_stats = []
    
    for i in range(window_days, len(log_ratios)):
        # Get window of data
        window_data = log_ratios[i-window_days:i+1]
        
        if len(window_data) > 1:
            # Calculate autocorrelation
            autocorr_1 = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
            
            # Calculate mean reversion speed (λ)
            if autocorr_1 < 1 and not np.isnan(autocorr_1):
                lambda_speed = -np.log(autocorr_1)  # Daily mean reversion speed
                lambda_annual = lambda_speed * 365.25  # Annualized
                
                # Calculate half-life
                half_life = -np.log(2) / np.log(autocorr_1)
                
                # Calculate deviation volatility
                deviation_vol = np.std(window_data)
                
                rolling_stats.append({
                    'date': dates[i],
                    'age': ages[i],
                    'autocorr_1': autocorr_1,
                    'lambda_annual': lambda_annual,
                    'half_life_days': half_life,
                    'deviation_vol': deviation_vol,
                    'price_ratio': df.iloc[i]['Price_Ratio']
                })
    
    return pd.DataFrame(rolling_stats)

def analyze_mean_reversion_weekly(df):
    """Analyze mean reversion using weekly windows"""
    print("Analyzing mean reversion with weekly windows...")
    
    # Calculate Bitcoin age for each date
    genesis_date = pd.to_datetime('2009-01-03')
    df['Bitcoin_Age'] = (df['Date'] - genesis_date).dt.days / 365.25
    
    # Resample to weekly data
    df_weekly = df.set_index('Date').resample('W').agg({
        'Price': 'last',
        'Fair_Value': 'last',
        'Bitcoin_Age': 'last'
    }).dropna()
    
    df_weekly['Price_Ratio'] = df_weekly['Price'] / df_weekly['Fair_Value']
    df_weekly['Log_Price_Ratio'] = np.log(df_weekly['Price_Ratio'])
    
    # Calculate rolling autocorrelation with 8-week window
    log_ratios = df_weekly['Log_Price_Ratio'].values
    dates = df_weekly.index.values
    ages = df_weekly['Bitcoin_Age'].values
    
    weekly_stats = []
    
    for i in range(8, len(log_ratios)):
        # Get 8-week window
        window_data = log_ratios[i-8:i+1]
        
        if len(window_data) > 1:
            # Calculate autocorrelation
            autocorr_1 = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
            
            # Calculate mean reversion speed (λ)
            if autocorr_1 < 1 and not np.isnan(autocorr_1):
                lambda_speed = -np.log(autocorr_1)  # Weekly mean reversion speed
                lambda_annual = lambda_speed * 52  # Annualized (52 weeks)
                
                # Calculate half-life
                half_life = -np.log(2) / np.log(autocorr_1)
                
                # Calculate deviation volatility
                deviation_vol = np.std(window_data)
                
                weekly_stats.append({
                    'date': dates[i],
                    'age': ages[i],
                    'autocorr_1': autocorr_1,
                    'lambda_annual': lambda_annual,
                    'half_life_weeks': half_life,
                    'deviation_vol': deviation_vol,
                    'price_ratio': df_weekly.iloc[i]['Price_Ratio']
                })
    
    return pd.DataFrame(weekly_stats)

def calculate_rate_of_change_fine(df_stats):
    """Calculate rate of change for fine-resolution data"""
    print("Calculating rate of change for fine-resolution data...")
    
    # Sort by date
    df_stats = df_stats.sort_values('date').reset_index(drop=True)
    
    # Calculate rate of change
    rates_of_change = []
    
    for i in range(1, len(df_stats)):
        current_lambda = df_stats.iloc[i]['lambda_annual']
        prev_lambda = df_stats.iloc[i-1]['lambda_annual']
        current_age = df_stats.iloc[i]['age']
        prev_age = df_stats.iloc[i-1]['age']
        
        # Rate of change = change in lambda / change in age
        rate_of_change = (current_lambda - prev_lambda) / (current_age - prev_age)
        
        rates_of_change.append({
            'date': df_stats.iloc[i]['date'],
            'age': current_age,
            'lambda': current_lambda,
            'rate_of_change': rate_of_change,
            'price_ratio': df_stats.iloc[i]['price_ratio']
        })
    
    return pd.DataFrame(rates_of_change)

def create_fine_resolution_visualization(df_daily, df_weekly, rates_daily, rates_weekly):
    """Create visualization of fine-resolution mean reversion analysis"""
    print("Creating fine-resolution visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Mean Reversion: Fine-Resolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Daily mean reversion speed
    ax1.plot(df_daily['age'], df_daily['lambda_annual'], 
            alpha=0.7, linewidth=1, color='blue', label='Daily (30-day window)')
    ax1.plot(df_weekly['age'], df_weekly['lambda_annual'], 
            alpha=0.8, linewidth=2, color='red', label='Weekly (8-week window)')
    ax1.set_title('Mean Reversion Speed: Daily vs Weekly Resolution')
    ax1.set_xlabel('Bitcoin Age (years)')
    ax1.set_ylabel('Mean Reversion Speed (λ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rate of change comparison
    ax2.plot(rates_daily['age'], rates_daily['rate_of_change'], 
            alpha=0.7, linewidth=1, color='blue', label='Daily')
    ax2.plot(rates_weekly['age'], rates_weekly['rate_of_change'], 
            alpha=0.8, linewidth=2, color='red', label='Weekly')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Rate of Change: Daily vs Weekly Resolution')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Rate of Change (dλ/dt)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Price ratio vs mean reversion speed
    ax3.scatter(df_daily['price_ratio'], df_daily['lambda_annual'], 
               alpha=0.5, s=10, color='blue', label='Daily')
    ax3.scatter(df_weekly['price_ratio'], df_weekly['lambda_annual'], 
               alpha=0.7, s=20, color='red', label='Weekly')
    ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Fair Value')
    ax3.set_title('Price Ratio vs Mean Reversion Speed')
    ax3.set_xlabel('Price / Fair Value Ratio')
    ax3.set_ylabel('Mean Reversion Speed (λ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of mean reversion speed
    ax4.plot(df_daily['date'], df_daily['lambda_annual'], 
            alpha=0.7, linewidth=1, color='blue', label='Daily')
    ax4.plot(df_weekly['date'], df_weekly['lambda_annual'], 
            alpha=0.8, linewidth=2, color='red', label='Weekly')
    ax4.set_title('Mean Reversion Speed Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Mean Reversion Speed (λ)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_mean_reversion_fine_resolution_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Fine-resolution visualization saved to: {filename}")
    
    return filename

def create_statistical_comparison(df_daily, df_weekly):
    """Create statistical comparison of different resolutions"""
    print("Creating statistical comparison...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mean Reversion: Statistical Comparison by Resolution', fontsize=16, fontweight='bold')
    
    # 1. Distribution of mean reversion speeds
    ax1.hist(df_daily['lambda_annual'].dropna(), bins=50, alpha=0.6, 
            color='blue', label='Daily', density=True)
    ax1.hist(df_weekly['lambda_annual'].dropna(), bins=30, alpha=0.6, 
            color='red', label='Weekly', density=True)
    ax1.set_title('Distribution of Mean Reversion Speeds')
    ax1.set_xlabel('Mean Reversion Speed (λ)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling statistics comparison
    # Calculate rolling mean and std for daily data
    daily_rolling_mean = df_daily['lambda_annual'].rolling(window=100).mean()
    daily_rolling_std = df_daily['lambda_annual'].rolling(window=100).std()
    
    ax2.plot(df_daily['age'], daily_rolling_mean, 'b-', linewidth=2, label='Daily Rolling Mean')
    ax2.fill_between(df_daily['age'], 
                     daily_rolling_mean - daily_rolling_std,
                     daily_rolling_mean + daily_rolling_std,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax2.set_title('Daily Mean Reversion: Rolling Statistics')
    ax2.set_xlabel('Bitcoin Age (years)')
    ax2.set_ylabel('Mean Reversion Speed (λ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Autocorrelation comparison
    ax3.scatter(df_daily['age'], df_daily['autocorr_1'], 
               alpha=0.5, s=10, color='blue', label='Daily')
    ax3.scatter(df_weekly['age'], df_weekly['autocorr_1'], 
               alpha=0.7, s=20, color='red', label='Weekly')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Autocorrelation by Resolution')
    ax3.set_xlabel('Bitcoin Age (years)')
    ax3.set_ylabel('1-Period Autocorrelation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Half-life comparison
    ax4.scatter(df_daily['age'], df_daily['half_life_days'], 
               alpha=0.5, s=10, color='blue', label='Daily')
    ax4.scatter(df_weekly['age'], df_weekly['half_life_weeks'] * 7, 
               alpha=0.7, s=20, color='red', label='Weekly')
    ax4.set_yscale('log')
    ax4.set_title('Mean Reversion Half-Life by Resolution')
    ax4.set_xlabel('Bitcoin Age (years)')
    ax4.set_ylabel('Half-Life (days)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Results/Bitcoin/bitcoin_mean_reversion_statistical_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Statistical comparison saved to: {filename}")
    
    return filename

def save_fine_resolution_results(df_daily, df_weekly, rates_daily, rates_weekly):
    """Save fine-resolution analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save daily data
    daily_file = f'Results/Bitcoin/bitcoin_mean_reversion_daily_{timestamp}.csv'
    df_daily.to_csv(daily_file, index=False)
    print(f"Daily mean reversion data saved to: {daily_file}")
    
    # Save weekly data
    weekly_file = f'Results/Bitcoin/bitcoin_mean_reversion_weekly_{timestamp}.csv'
    df_weekly.to_csv(weekly_file, index=False)
    print(f"Weekly mean reversion data saved to: {weekly_file}")
    
    # Save rate of change data
    rates_daily_file = f'Results/Bitcoin/bitcoin_mean_reversion_rates_daily_{timestamp}.csv'
    rates_daily.to_csv(rates_daily_file, index=False)
    print(f"Daily rate of change data saved to: {rates_daily_file}")
    
    rates_weekly_file = f'Results/Bitcoin/bitcoin_mean_reversion_rates_weekly_{timestamp}.csv'
    rates_weekly.to_csv(rates_weekly_file, index=False)
    print(f"Weekly rate of change data saved to: {rates_weekly_file}")
    
    return daily_file, weekly_file, rates_daily_file, rates_weekly_file

def main():
    """Main analysis function"""
    print("Bitcoin Mean Reversion: Fine-Resolution Analysis")
    print("="*50)
    
    # Load and prepare data
    df = load_bitcoin_data()
    df = calculate_formula_fair_values(df)
    
    # Analyze mean reversion with different resolutions
    print("\nAnalyzing daily resolution...")
    df_daily = analyze_mean_reversion_rolling(df, window_days=30)
    
    print("\nAnalyzing weekly resolution...")
    df_weekly = analyze_mean_reversion_weekly(df)
    
    # Calculate rate of change for both resolutions
    print("\nCalculating rate of change...")
    rates_daily = calculate_rate_of_change_fine(df_daily)
    rates_weekly = calculate_rate_of_change_fine(df_weekly)
    
    # Create visualizations
    viz_file1 = create_fine_resolution_visualization(df_daily, df_weekly, rates_daily, rates_weekly)
    viz_file2 = create_statistical_comparison(df_daily, df_weekly)
    
    # Save results
    daily_file, weekly_file, rates_daily_file, rates_weekly_file = save_fine_resolution_results(
        df_daily, df_weekly, rates_daily, rates_weekly
    )
    
    # Summary
    print(f"\n" + "="*50)
    print("FINE-RESOLUTION ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Daily Analysis:")
    print(f"  Data points: {len(df_daily):,}")
    print(f"  Mean λ: {df_daily['lambda_annual'].mean():.3f}")
    print(f"  Std λ: {df_daily['lambda_annual'].std():.3f}")
    print(f"  Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    
    print(f"\nWeekly Analysis:")
    print(f"  Data points: {len(df_weekly):,}")
    print(f"  Mean λ: {df_weekly['lambda_annual'].mean():.3f}")
    print(f"  Std λ: {df_weekly['lambda_annual'].std():.3f}")
    print(f"  Date range: {df_weekly['date'].min()} to {df_weekly['date'].max()}")
    
    print(f"\nResolution Comparison:")
    print(f"  Daily resolution: {len(df_daily):,} observations")
    print(f"  Weekly resolution: {len(df_weekly):,} observations")
    print(f"  Daily/Weekly ratio: {len(df_daily)/len(df_weekly):.1f}x more data points")
    
    print(f"\nFiles created:")
    print(f"  Daily data: {daily_file}")
    print(f"  Weekly data: {weekly_file}")
    print(f"  Daily rates: {rates_daily_file}")
    print(f"  Weekly rates: {rates_weekly_file}")
    print(f"  Visualizations: {viz_file1}, {viz_file2}")

if __name__ == "__main__":
    main() 