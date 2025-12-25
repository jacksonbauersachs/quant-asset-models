"""
BITCOIN OVERVALUATION QUICK CALCULATOR
======================================

Quick script to calculate Bitcoin's current overvaluation using the latest growth model.
Useful for quick checks without generating the full rainbow chart.

CURRENT MODEL (as of 2025-08-25):
- Formula: log10(price) = 2.464300 * ln(day) + (-16.448768)
- R² = 0.9600 (96.00% of variance explained)
- Data range: 2010-07-18 to 2025-08-25
- Start day: 365 (skips first year of Bitcoin)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Current Bitcoin growth model coefficients (updated 2025-08-25)
# These coefficients are fitted on the current Portfolio/Data/Bitcoin_all_time_price.csv dataset
BITCOIN_GROWTH_COEFFICIENTS = {
    'slope': 2.464300,
    'intercept': -16.448768,
    'r2': 0.959960,
    'start_day': 365,  # Skip first year of Bitcoin
    'data_range': '2010-07-18 to 2025-08-25'
}

# Bitcoin genesis date (January 3, 2009)
BITCOIN_GENESIS = datetime(2009, 1, 3)

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
                return df, file_path
            except Exception as e:
                continue
    
    return None, None

def main():
    """Main function to run the quick Bitcoin overvaluation calculator"""
    
    print("=" * 60)
    print("BITCOIN OVERVALUATION QUICK CALCULATOR")
    print("=" * 60)
    print(f"Model Formula: log₁₀(price) = {BITCOIN_GROWTH_COEFFICIENTS['slope']:.3f} × ln(day) + {BITCOIN_GROWTH_COEFFICIENTS['intercept']:.3f}")
    print(f"R² = {BITCOIN_GROWTH_COEFFICIENTS['r2']:.4f}")
    print(f"Data Range: {BITCOIN_GROWTH_COEFFICIENTS['data_range']}")
    print(f"Start Day: {BITCOIN_GROWTH_COEFFICIENTS['start_day']}")
    print()
    
    # Load Bitcoin data
    df, file_path = load_bitcoin_data()
    if df is None:
        print("Error: Could not find Bitcoin data file")
        return
    
    print(f"Loaded data from: {file_path}")
    
    # Clean and prepare data
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle Price column - check if it's already numeric or needs conversion
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    else:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Data points: {len(df):,}")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
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
    
    print(f"\nANALYSIS RESULTS:")
    print(f"Model Price: ${model_price:,.2f}")
    print(f"Overvaluation: {overvaluation:+.1f}%")
    print(f"Rainbow Band: {current_band}")
    
    # Calculate price ratio
    price_ratio = current_price / model_price
    print(f"Price Ratio: {price_ratio:.2f}x model price")
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    if overvaluation > 50:
        print("🔴 Bitcoin is in the DANGER zone - significantly overvalued")
        print("   Consider reducing allocation or taking profits")
    elif overvaluation > 20:
        print("🟡 Bitcoin is overvalued - in the SELL zone")
        print("   Consider reducing allocation")
    elif overvaluation > -20:
        print("🟢 Bitcoin is in the HOLD zone - near fair value")
        print("   Maintain current allocation")
    elif overvaluation > -50:
        print("🟢 Bitcoin is undervalued - in the BUY zone")
        print("   Consider increasing allocation")
    else:
        print("🔵 Bitcoin is in the FIRESALE zone - significantly undervalued")
        print("   Strong buy signal - consider increasing allocation")
    
    # Save quick results
    results_file = "Portfolio/bitcoin_overvaluation_quick_latest.txt"
    with open(results_file, 'w') as f:
        f.write("BITCOIN OVERVALUATION QUICK ANALYSIS\n")
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
        f.write(f"Rainbow Band: {current_band}\n")
        f.write(f"Price Ratio: {price_ratio:.2f}x model price\n\n")
        
        f.write("INTERPRETATION:\n")
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
    
    print(f"\nQuick analysis saved to: {results_file}")
    print(f"\n✅ Bitcoin overvaluation analysis completed!")
    print(f"📊 Current overvaluation: {overvaluation:+.1f}%")
    print(f"🎯 Rainbow band: {current_band}")
    print(f"📈 Model price: ${model_price:,.2f}")
    print(f"💰 Current price: ${current_price:,.2f}")

if __name__ == "__main__":
    main() 