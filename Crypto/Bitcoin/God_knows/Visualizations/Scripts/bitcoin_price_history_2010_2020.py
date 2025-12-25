import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_bitcoin_data():
    """Load Bitcoin data filtered to 2014-2020"""
    print("Loading Bitcoin dataset (2014-2020)...")
    df = pd.read_csv('Bitcoin/Data/2010_2025_Daily_Data_(BTC).csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to 2014-2020
    start_date = '2014-01-01'
    end_date = '2020-10-20'
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def create_bitcoin_chart(df):
    """Create Bitcoin price history chart (2014-2020)"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot Bitcoin price with enhanced styling
    ax.plot(df['Date'], df['Price'], color='#F7931A', linewidth=2.5, label='Bitcoin (BTC)', alpha=0.9)
    
    # Customize the plot with professional styling
    ax.set_xlabel('Date', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title('Bitcoin Price History: 2014-2020', fontsize=20, fontweight='bold', color='#333333', pad=20)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data
    
    # Legend styling
    ax.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add subtle grid lines for key price levels (no labels)
    key_prices = [100, 1000, 10000, 20000, 50000, 100000]
    for price in key_prices:
        if price <= df['Price'].max():
            ax.axhline(y=price, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Remove data info box for cleaner look
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'Bitcoin/Visualizations/Images/bitcoin_price_history_2014_2020.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {save_path}")
    plt.show()

def main():
    """Main function"""
    print("Bitcoin Price History Chart (2014-2020)")
    print("="*50)
    
    # Load data
    df = load_bitcoin_data()
    
    # Create chart
    create_bitcoin_chart(df)
    
    print("\n✅ Chart creation completed!")

if __name__ == "__main__":
    main() 