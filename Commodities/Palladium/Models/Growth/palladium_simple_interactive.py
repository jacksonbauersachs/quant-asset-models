import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['date'].min()
        df['days'] = (df['date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def create_simple_interactive_chart(df):
    """Create a simple interactive chart with growth rate and intercept sliders."""
    
    # Get initial parameters
    x = df['days'].values
    y = df[' value'].values
    
    # Initial model parameters
    initial_a = 54.36
    fixed_b = 0.000172  # Fixed growth rate
    initial_c = -11.07  # This is the intercept we'll adjust
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual data
    ax1.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    # Initial model line
    y_initial = initial_a * np.exp(fixed_b * x) + initial_c
    line1, = ax1.plot(df['years'], y_initial, 'r-', linewidth=2, label='Exponential Model')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    ax1.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Palladium Price ($)')
    ax1.set_title('Interactive Palladium Growth Model - Fixed Growth Rate (0.000172), Adjust Intercept')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    y_initial_log = initial_a * np.exp(fixed_b * x) + initial_c
    line2, = ax2.plot(df['years'], y_initial_log, 'r-', linewidth=2, label='Exponential Model')
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Palladium Price ($) - Log Scale')
    ax2.set_title('Log Scale View (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Create slider (only intercept now)
    ax_slider_c = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider_c = Slider(ax_slider_c, 'Intercept (c) - Move Curve Up/Down', -200, 200, valinit=initial_c, valfmt='%.1f')
    
    # Info text
    info_text = fig.text(0.02, 0.20, '', fontsize=11, verticalalignment='top')
    
    def update_model(val):
        """Update the model when slider changes."""
        c = slider_c.val
        
        # Calculate new model predictions (using fixed growth rate)
        y_new = initial_a * np.exp(fixed_b * x) + c
        
        # Update both lines
        line1.set_ydata(y_new)
        line2.set_ydata(y_new)
        
        # Calculate R²
        ss_res = np.sum((y - y_new) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate fair value (current date)
        current_days = df['days'].iloc[-1]
        fair_value = initial_a * np.exp(fixed_b * current_days) + c
        
        # Calculate undervaluation
        undervaluation = ((fair_value - current_price) / current_price) * 100
        
        # Update info text
        info_text.set_text(f"""Fixed Growth Rate (b): {fixed_b:.6f}
Intercept (c): {c:.1f}
R² Score: {r2:.4f}
Current Price: ${current_price:.2f}
Model Fair Value: ${fair_value:.2f}
Undervaluation: {undervaluation:.1f}%

Instructions:
- Drag the slider to move the whole curve up or down
- Growth rate is fixed at 0.000172
- Watch how the model line changes in real-time
- See how fair value and undervaluation change""")
        
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider_c.on_changed(update_model)
    
    # Initial update
    update_model(None)
    
    plt.tight_layout()
    return fig

def main():
    print("Simple Interactive Palladium Growth Model")
    print("=" * 50)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    print("\nCreating interactive chart...")
    print("Use the slider to adjust the intercept (move curve up/down)!")
    print("Growth rate is fixed at 0.000172!")
    print("Watch how the model line and fair value change in real-time!")
    
    fig = create_simple_interactive_chart(df)
    
    # Show the interactive plot
    plt.show()
    
    print("\n" + "=" * 50)
    print("INTERACTIVE CHART COMPLETE")
    print("=" * 50)
    print("Adjust the slider to see how it affects:")
    print("- Intercept slider: Move the whole curve up or down")
    print("- Growth rate is fixed at 0.000172")
    print("- Model fit (R² score)")
    print("- Fair value predictions")
    print("- Undervaluation percentage")
    print("- Overall model behavior")

if __name__ == "__main__":
    main()
