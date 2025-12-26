import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from datetime import datetime

def load_bitcoin_data():
    """Load and prepare Bitcoin price data for analysis."""
    
    data_file = "Crypto\Bitcoin\We_both_know\BTC_historical_data.csv"
    
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
    """Calculate R² score in log space (like the original Bitcoin model)."""
    # Convert to log space for R² calculation (matching original model)
    log_actual = np.log10(y_actual)
    log_pred = np.log10(y_pred)
    
    ss_res = np.sum((log_actual - log_pred) ** 2)
    ss_tot = np.sum((log_actual - np.mean(log_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def update_model(val):
    """Update the model when slider changes."""
    slope = slider_slope.val
    intercept = -16.451114  # Hardcoded intercept (actual fitted value)
    
    # Calculate model predictions
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(lambda x: bitcoin_rainbow_model(x, slope, intercept))
    
    # Update the model line on both subplots
    line1.set_ydata(df_filtered['Model_Price'])
    line2.set_ydata(df_filtered['Model_Price'])
    
    # Calculate R²
    r2 = calculate_r2(df_filtered['Price'], df_filtered['Model_Price'])
    
    # Update titles
    ax1.set_title(f'Bitcoin Rainbow Chart - Log Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    ax2.set_title(f'Bitcoin Rainbow Chart - Normal Scale\nR² = {r2:.6f}', fontsize=14, fontweight='bold')
    
    # Update info text (simplified)
    current_price = df_filtered['Price'].iloc[-1]
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    info_text = f"Slope: {slope:.6f} | R²: {r2:.6f} | Current: ${current_price:,.0f} | Model: ${model_fair_value:,.0f} | Deviation: {deviation:+.1f}%"
    
    ax_info.texts[0].set_text(info_text)
    
    # Redraw
    fig.canvas.draw_idle()

def main():
    """Create interactive Bitcoin rainbow chart."""
    
    print("Bitcoin Interactive Rainbow Chart")
    print("=" * 40)
    
    # Load data
    global df_filtered, ax1, ax2, line1, line2, ax_info, slider_slope, fig
    df_filtered = load_bitcoin_data()
    if df_filtered is None:
        return
    
    # Initial parameters (actual fitted values from the model)
    initial_slope = 2.464614
    initial_intercept = -16.451114
    
    # Create figure with two subplots (exactly like the original)
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual Bitcoin price on both subplots
    ax1.semilogy(df_filtered['Date'], df_filtered['Price'], color='blue', linewidth=2, label='Actual Bitcoin Price')
    ax2.plot(df_filtered['Date'], df_filtered['Price'], color='blue', linewidth=2, label='Actual Bitcoin Price')
    
    # Calculate initial model prices
    df_filtered['Model_Price'] = df_filtered['Days_Since_Genesis'].apply(lambda x: bitcoin_rainbow_model(x, initial_slope, initial_intercept))
    
    # Skip the rainbow bands - just show the model line
    
    # Plot model line
    line1, = ax1.semilogy(df_filtered['Date'], df_filtered['Model_Price'], 
                color='black', linestyle='--', linewidth=2, label='Growth Model')
    line2, = ax2.plot(df_filtered['Date'], df_filtered['Model_Price'], 
            color='black', linestyle='--', linewidth=2, label='Growth Model')
    
    # Highlight current position
    current_price = df_filtered['Price'].iloc[-1]
    current_date = df_filtered['Date'].iloc[-1]
    ax1.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    ax2.scatter(current_date, current_price, color='red', s=100, 
               zorder=10, label=f'Current: ${current_price:,.0f}')
    
    # Add model information
    initial_r2 = calculate_r2(df_filtered['Price'], df_filtered['Model_Price'])
    formula_text = f'Model: log₁₀(price) = {initial_slope:.3f} × ln(day) + {initial_intercept:.3f}\nR² = {initial_r2:.4f}'
    ax1.text(0.02, 0.02, formula_text, transform=ax1.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    ax2.text(0.02, 0.02, formula_text, transform=ax2.transAxes, fontsize=10, 
            color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Formatting for both subplots (exactly like the original)
    for ax, ylabel_suffix in [(ax1, 'Log Scale'), (ax2, 'Normal Scale')]:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'Bitcoin Price (USD) - {ylabel_suffix}', fontsize=12)
        ax.set_title(f'Bitcoin Rainbow Chart - {ylabel_suffix}\nR² = {initial_r2:.6f}', fontsize=14, fontweight='bold')
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
        ax.set_ylim(0.01, max(df_filtered['Price'].max(), df_filtered['Model_Price'].max()) * 2)
    
    # Create slider (less sensitive)
    ax_slope = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider_slope = Slider(ax_slope, 'Slope', 2.0, 3.0, valinit=initial_slope, valfmt='%.6f')
    slider_slope.on_changed(update_model)
    
    # Add info panel at the top
    ax_info = plt.axes([0.02, 0.88, 0.96, 0.08])
    ax_info.axis('off')
    
    # Initial info text (simplified)
    model_fair_value = df_filtered['Model_Price'].iloc[-1]
    deviation = ((current_price - model_fair_value) / model_fair_value * 100)
    
    info_text = f"Slope: {initial_slope:.6f} | R²: {initial_r2:.6f} | Current: ${current_price:,.0f} | Model: ${model_fair_value:,.0f} | Deviation: {deviation:+.1f}%"
    
    ax_info.text(0.02, 0.5, info_text, transform=ax_info.transAxes, fontsize=11, 
                verticalalignment='center', bbox=dict(facecolor='lightgray', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Instructions (simplified)
    instructions = "Adjust slider to change slope. Intercept fixed at -16.451114"
    
    ax_info.text(0.52, 0.5, instructions, transform=ax_info.transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the interactive chart
    output_dir = "Crypto/Bitcoin/Models/Growth"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_file = os.path.join(output_dir, "bitcoin_interactive_rainbow.png")
    fig.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Interactive chart saved: {chart_file}")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE BITCOIN RAINBOW CHART")
    print("=" * 60)
    print("Use the slider to adjust the model parameters:")
    print("• Slope: Controls the growth rate (higher = steeper)")
    print("• Intercept: Fixed at -16.451114")
    print("• Watch R² score to find the best fit")
    print("• Current deviation shows if Bitcoin is over/undervalued")
    
    plt.show()

if __name__ == "__main__":
    main()
