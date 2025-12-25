"""
BITCOIN RAINBOW CHART CALCULATOR
================================

This script calculates Bitcoin's current overvaluation using the latest growth model
and creates a comprehensive rainbow chart visualization.

CURRENT MODEL (as of 2025-08-25):
- Formula: log10(price) = 2.464300 * ln(day) + (-16.448768)
- R² = 0.9600 (96.00% of variance explained)
- Data range: 2010-07-18 to 2025-08-25
- Start day: 365 (skips first year of Bitcoin)

RAINBOW BANDS:
- Blue (Firesale!): Maximum undervaluation
- Green (Buy): Undervaluation
- Yellow (Hold): Fair value range
- Orange (Sell): Overvaluation
- Red (Danger!): Maximum overvaluation

USAGE:
- Run script to get current Bitcoin overvaluation
- View rainbow chart visualization
- Results consistent with asset allocation calculator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import os

# =============================================================================
# CONFIGURATION - CURRENT BITCOIN GROWTH MODEL COEFFICIENTS
# =============================================================================

# Current Bitcoin growth model coefficients (updated 2025-08-25)
# These coefficients are fitted on the current Portfolio/Data/Bitcoin_all_time_price.csv dataset
BITCOIN_GROWTH_COEFFICIENTS = {
    'slope': 2.464300,
    'intercept': -16.448768,
    'r2': 0.959960,
    'start_day': 365,  # Skip first year of Bitcoin
    'data_range': '2010-07-18 to 2025-08-25'
}

# Rainbow band configuration
RAINBOW_BANDS = {
    'Firesale!': {'color': 'blue', 'alpha': 0.8, 'description': 'Maximum undervaluation'},
    'Buy': {'color': 'green', 'alpha': 0.7, 'description': 'Undervaluation'},
    'Hold': {'color': 'yellow', 'alpha': 0.6, 'description': 'Fair value range'},
    'Sell': {'color': 'orange', 'alpha': 0.7, 'description': 'Overvaluation'},
    'Danger!': {'color': 'red', 'alpha': 0.8, 'description': 'Maximum overvaluation'}
}

# Bitcoin genesis date (January 3, 2009)
BITCOIN_GENESIS = datetime(2009, 1, 3)

# =============================================================================

def calculate_bitcoin_days(date):
    """Calculate days since Bitcoin genesis for a given date"""
    return (date - BITCOIN_GENESIS).days

def calculate_model_price(days):
    """Calculate Bitcoin price using the growth model formula"""
    if days < BITCOIN_GROWTH_COEFFICIENTS['start_day']:
        return None
    
    slope = BITCOIN_GROWTH_COEFFICIENTS['slope']
    intercept = BITCOIN_GROWTH_COEFFICIENTS['intercept']
    
    # Formula: log10(price) = slope * ln(day) + intercept
    log_price = slope * np.log(days) + intercept
    price = 10**log_price
    
    return price

def calculate_overvaluation(current_price, model_price):
    """Calculate overvaluation percentage"""
    if model_price <= 0:
        return None
    
    overvaluation = ((current_price - model_price) / model_price) * 100
    return overvaluation

def get_rainbow_band(overvaluation):
    """Determine which rainbow band the current overvaluation falls into"""
    if overvaluation is None:
        return 'Unknown'
    
    if overvaluation <= -50:
        return 'Firesale!'
    elif overvaluation <= -20:
        return 'Buy'
    elif overvaluation <= 20:
        return 'Hold'
    elif overvaluation <= 50:
        return 'Sell'
    else:
        return 'Danger!'

def create_rainbow_chart(df, current_price, current_date, model_price, overvaluation):
    """Create the rainbow chart visualization"""
    
    # Set up the plot with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Calculate days for the entire dataset
    df['Days'] = df['Date'].apply(calculate_bitcoin_days)
    
    # Filter to only include days >= start_day
    df_filtered = df[df['Days'] >= BITCOIN_GROWTH_COEFFICIENTS['start_day']].copy()
    
    if len(df_filtered) == 0:
        print("Error: No data available for the specified date range")
        return
    
    # Calculate model prices for the filtered data
    df_filtered['Model_Price'] = df_filtered['Days'].apply(calculate_model_price)
    
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
    band_names = list(RAINBOW_BANDS.keys())
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
        
        band_config = RAINBOW_BANDS[band_names[i]]
        ax.fill_between(df_filtered['Date'], lower_bound, upper_bound, 
                       color=band_config['color'], alpha=band_config['alpha'], 
                       label=f"{band_names[i]}: {band_config['description']}")
    
    # Plot actual Bitcoin price
    ax.semilogy(df_filtered['Date'], df_filtered['Price'], 
                color='white', linewidth=2, label='Actual Bitcoin Price')
    
    # Plot model line
    ax.semilogy(df_filtered['Date'], df_filtered['Model_Price'], 
                color='black', linestyle='--', linewidth=2, label='Growth Model')
    
    # Highlight current position
    ax.scatter(current_date, current_price, color='white', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Bitcoin Price (USD) - Log Scale', fontsize=14)
    ax.set_title('Bitcoin Rainbow Chart\nCurrent Growth Model Analysis', 
                 fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Custom y-axis formatting
    def price_formatter(x, pos):
        if x >= 1000000:
            return f'${x/1000000:.1f}M'
        elif x >= 1000:
            return f'${x/1000:.0f}K'
        else:
            return f'${x:.0f}'
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))
    
    # Add legend
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add model information
    formula_text = f'Model: log₁₀(price) = {BITCOIN_GROWTH_COEFFICIENTS["slope"]:.3f} × ln(day) + {BITCOIN_GROWTH_COEFFICIENTS["intercept"]:.3f}\nR² = {BITCOIN_GROWTH_COEFFICIENTS["r2"]:.4f}'
    ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=12, 
            color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add current status
    current_band = get_rainbow_band(overvaluation)
    status_text = f'Current Status: {current_band}\nOvervaluation: {overvaluation:+.1f}%\nModel Price: ${model_price:,.0f}'
    ax.text(0.98, 0.02, status_text, transform=ax.transAxes, fontsize=12, 
            color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'),
            horizontalalignment='right', verticalalignment='bottom')
    
    # Set y-axis limits
    ax.set_ylim(0.01, max(df_filtered['Price'].max(), df_filtered['Model_Price'].max()) * 2)
    
    plt.tight_layout()
    return fig

def load_bitcoin_data():
    """Load Bitcoin price data"""
    
    # Try multiple possible file locations
    possible_files = [
        "Portfolio/Data/Bitcoin_all_time_price.csv",
        "Crypto/Bitcoin/Data/Bitcoin_all_time_price.csv",
        "Crypto/Bitcoin/Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded data from: {file_path}")
                return df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    print("Error: Could not find Bitcoin data file")
    return None

def main():
    """Main function to run the Bitcoin rainbow chart calculator"""
    
    print("=" * 60)
    print("BITCOIN RAINBOW CHART CALCULATOR")
    print("=" * 60)
    print(f"Model Formula: log₁₀(price) = {BITCOIN_GROWTH_COEFFICIENTS['slope']:.3f} × ln(day) + {BITCOIN_GROWTH_COEFFICIENTS['intercept']:.3f}")
    print(f"R² = {BITCOIN_GROWTH_COEFFICIENTS['r2']:.4f}")
    print(f"Data Range: {BITCOIN_GROWTH_COEFFICIENTS['data_range']}")
    print(f"Start Day: {BITCOIN_GROWTH_COEFFICIENTS['start_day']}")
    print()
    
    # Load Bitcoin data
    df = load_bitcoin_data()
    if df is None:
        return
    
    # Clean and prepare data
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle Price column - check if it's already numeric or needs conversion
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    else:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    
    # Get current data
    current_price = df['Price'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    current_days = calculate_bitcoin_days(current_date)
    
    print(f"\nCURRENT BITCOIN STATUS:")
    print(f"Date: {current_date.strftime('%Y-%m-%d')}")
    print(f"Price: ${current_price:,.2f}")
    print(f"Days since genesis: {current_days:,}")
    
    # Calculate model price and overvaluation
    model_price = calculate_model_price(current_days)
    if model_price is None:
        print(f"Error: Current day {current_days} is below start day {BITCOIN_GROWTH_COEFFICIENTS['start_day']}")
        return
    
    overvaluation = calculate_overvaluation(current_price, model_price)
    current_band = get_rainbow_band(overvaluation)
    
    print(f"Model Price: ${model_price:,.2f}")
    print(f"Overvaluation: {overvaluation:+.1f}%")
    print(f"Rainbow Band: {current_band}")
    
    # Create rainbow chart
    print(f"\nCreating rainbow chart visualization...")
    fig = create_rainbow_chart(df, current_price, current_date, model_price, overvaluation)
    
    if fig:
        # Save the chart
        output_file = "Portfolio/bitcoin_rainbow_chart_latest.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Rainbow chart saved to: {output_file}")
        
        # Show the chart
        plt.show()
    
    # Save analysis results
    results_file = "Portfolio/bitcoin_rainbow_analysis_latest.txt"
    with open(results_file, 'w') as f:
        f.write("BITCOIN RAINBOW CHART ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Formula: log10(price) = {BITCOIN_GROWTH_COEFFICIENTS['slope']:.6f} * ln(day) + {BITCOIN_GROWTH_COEFFICIENTS['intercept']:.6f}\n")
        f.write(f"R² = {BITCOIN_GROWTH_COEFFICIENTS['r2']:.6f}\n")
        f.write(f"Data Range: {BITCOIN_GROWTH_COEFFICIENTS['data_range']}\n")
        f.write(f"Start Day: {BITCOIN_GROWTH_COEFFICIENTS['start_day']}\n\n")
        
        f.write("CURRENT STATUS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Date: {current_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Current Price: ${current_price:,.2f}\n")
        f.write(f"Days Since Genesis: {current_days:,}\n")
        f.write(f"Model Price: ${model_price:,.2f}\n")
        f.write(f"Overvaluation: {overvaluation:+.1f}%\n")
        f.write(f"Rainbow Band: {current_band}\n\n")
        
        f.write("RAINBOW BANDS:\n")
        f.write("-" * 15 + "\n")
        for band_name, config in RAINBOW_BANDS.items():
            f.write(f"{band_name}: {config['description']}\n")
        
        f.write("\nINTERPRETATION:\n")
        f.write("-" * 15 + "\n")
        if overvaluation > 50:
            f.write("DANGER: Bitcoin is significantly overvalued\n")
            f.write("   Consider reducing allocation or taking profits\n")
        elif overvaluation > 20:
            f.write("SELL: Bitcoin is overvalued\n")
            f.write("   Consider reducing allocation\n")
        elif overvaluation > -20:
            f.write("HOLD: Bitcoin is near fair value\n")
            f.write("   Maintain current allocation\n")
        elif overvaluation > -50:
            f.write("BUY: Bitcoin is undervalued\n")
            f.write("   Consider increasing allocation\n")
        else:
            f.write("FIRESALE: Bitcoin is significantly undervalued\n")
            f.write("   Strong buy signal - consider increasing allocation\n")
        
        f.write(f"\nThis analysis is consistent with the asset allocation calculator\n")
        f.write(f"and can be used to inform portfolio rebalancing decisions.\n")
    
    print(f"\nAnalysis results saved to: {results_file}")
    print(f"\n✅ Bitcoin rainbow chart analysis completed!")
    print(f"📊 Current overvaluation: {overvaluation:+.1f}%")
    print(f"🎯 Rainbow band: {current_band}")
    print(f"📈 Model price: ${model_price:,.2f}")
    print(f"💰 Current price: ${current_price:,.2f}")

if __name__ == "__main__":
    main() 