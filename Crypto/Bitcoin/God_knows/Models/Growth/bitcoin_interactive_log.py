import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from datetime import datetime

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
    
    # Skip the first year of Bitcoin (start at day 365 since genesis)
    df_filtered = df[df['Days_Since_Genesis'] >= 365].copy()
    
    print(f"Bitcoin data loaded: {len(df_filtered)} records from {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df_filtered['Price'].min():.2f} to ${df_filtered['Price'].max():,.2f}")
    
    return df_filtered

def bitcoin_rainbow_model(x, slope, intercept):
    """Bitcoin rainbow model: price = 10^(slope * ln(days_since_genesis) + intercept)"""
    return 10**(slope * np.log(x) + intercept)

def calculate_r2(y_actual, y_pred):
    """Calculate R² score."""
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def update_model(val):
    """Update the model when slider changes."""
    slope = slider_slope.val
    intercept = 16.451  # Hardcoded intercept
    
    # Calculate model predictions using Bitcoin rainbow formula
    y_pred = bitcoin_rainbow_model(x, slope, intercept)
    r2 = calculate_r2(y, y_pred)
    
    # Update the model line
    line.set_ydata(y_pred)
    
    # Update the title with current R²
    ax_main.set_title(f'Bitcoin Interactive Rainbow Model\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    
    # Update info text
    info_text = f"""Current Model: price = 10^({slope:.6f} * ln(days_since_genesis) + {intercept:.6f})
R² = {r2:.6f}
Current Price: ${current_price:,.0f}
Model Fair Value: ${y_pred[-1]:,.0f}
Deviation: {((current_price - y_pred[-1]) / y_pred[-1] * 100):+.1f}%"""
    
    ax_info.texts[0].set_text(info_text)
    
    # Redraw
    fig.canvas.draw_idle()

def main():
    """Create interactive Bitcoin logarithmic model."""
    
    print("Bitcoin Interactive Rainbow Model")
    print("=" * 40)
    
    # Load data
    global df, x, y, current_price
    df = load_bitcoin_data()
    if df is None:
        return
    
    x = df['Days_Since_Genesis'].values
    y = df['Price'].values
    current_price = y[-1]
    
    # Initial parameters (hardcoded values)
    initial_slope = 2.465
    initial_intercept = 16.451
    
    # Create figure and subplots
    global fig, ax_main, ax_info, line, slider_slope
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main chart (log scale)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
    
    # Info panel
    ax_info = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    ax_info.axis('off')
    
    # Plot actual Bitcoin price
    ax_main.semilogy(df['Date'], df['Price'], color='blue', linewidth=2, label='Actual Bitcoin Price', alpha=0.7)
    
    # Plot initial model
    y_pred_initial = bitcoin_rainbow_model(x, initial_slope, initial_intercept)
    line, = ax_main.semilogy(df['Date'], y_pred_initial, color='red', linewidth=2, label='Bitcoin Rainbow Model')
    
    # Highlight current position
    current_date = df['Date'].iloc[-1]
    ax_main.scatter(current_date, current_price, color='red', s=100, 
                   zorder=10, label=f'Current: ${current_price:,.0f}')
    
    # Formatting
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel('Bitcoin Price (USD) - Log Scale', fontsize=12)
    ax_main.set_title(f'Bitcoin Interactive Rainbow Model\nR² = {calculate_r2(y, y_pred_initial):.6f}', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.legend(fontsize=10, loc='upper left')
    
    # Custom y-axis formatting
    def price_formatter(x, pos):
        if x >= 1000000:
            return f'${x/1000000:.1f}M'
        elif x >= 1000:
            return f'${x/1000:.0f}K'
        else:
            return f'${x:.0f}'
    
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
    
    # Create slider (only slope)
    ax_slope = plt.axes([0.1, 0.02, 0.8, 0.03])
    
    slider_slope = Slider(ax_slope, 'Slope', 0, 10, valinit=initial_slope, valfmt='%.3f')
    
    # Connect slider to update function
    slider_slope.on_changed(update_model)
    
    # Initial info text
    initial_r2 = calculate_r2(y, y_pred_initial)
    initial_fair_value = y_pred_initial[-1]
    initial_deviation = ((current_price - initial_fair_value) / initial_fair_value * 100)
    
    info_text = f"""Current Model: price = 10^({initial_slope:.6f} * ln(days_since_genesis) + {initial_intercept:.6f})
R² = {initial_r2:.6f}
Current Price: ${current_price:,.0f}
Model Fair Value: ${initial_fair_value:,.0f}
Deviation: {initial_deviation:+.1f}%"""
    
    ax_info.text(0.02, 0.5, info_text, transform=ax_info.transAxes, fontsize=12, 
                verticalalignment='center', bbox=dict(facecolor='lightblue', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Instructions
    instructions = """Instructions:
• Adjust the Slope slider to change the growth rate
• Intercept is fixed at 16.451
• Watch the R² score and deviation change in real-time
• Find the best fit by maximizing R² score"""
    
    ax_info.text(0.52, 0.5, instructions, transform=ax_info.transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the interactive chart
    output_dir = "Crypto/Bitcoin/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = os.path.join(output_dir, "bitcoin_interactive_log.png")
    fig.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Interactive chart saved: {chart_file}")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE BITCOIN RAINBOW MODEL")
    print("=" * 60)
    print("Use the slider to adjust the model parameters:")
    print("• Slope: Controls the growth rate (higher = steeper)")
    print("• Intercept: Fixed at 16.451")
    print("• Watch R² score to find the best fit")
    print("• Current deviation shows if Bitcoin is over/undervalued")
    
    plt.show()

if __name__ == "__main__":
    main()
