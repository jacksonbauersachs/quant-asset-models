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

def fit_initial_model(df):
    """Fit the initial exponential model."""
    
    x = df['days'].values
    y = df[' value'].values
    
    # Exponential model
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(exponential_func, x, y, p0=[54, 0.0002, -11], maxfev=10000)
        a, b, c = popt
        
        # Calculate R²
        y_pred = exponential_func(x, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return a, b, c, r2
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return 54, 0.0002, -11, 0.7

def create_interactive_chart(df, initial_a, initial_b, initial_c, initial_r2):
    """Create an interactive chart with sliders."""
    
    # Create figure and subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main chart
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    
    # Slider area
    ax_slider_a = plt.subplot2grid((4, 3), (2, 0))
    ax_slider_b = plt.subplot2grid((4, 3), (2, 1))
    ax_slider_c = plt.subplot2grid((4, 3), (2, 2))
    
    # Info panel
    ax_info = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    ax_info.axis('off')
    
    # Initial plot
    ax_main.scatter(df['years'], df[' value'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    # Initial model line
    x = df['days'].values
    y_initial = initial_a * np.exp(initial_b * x) + initial_c
    line, = ax_main.plot(df['years'], y_initial, 'r-', linewidth=2, label='Exponential Model')
    
    # Current price line
    current_price = df[' value'].iloc[-1]
    current_year = df['years'].iloc[-1]
    ax_main.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
    
    ax_main.set_xlabel('Year')
    ax_main.set_ylabel('Palladium Price ($)')
    ax_main.set_title('Interactive Palladium Growth Model - Adjust Parameters with Sliders')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Create sliders
    slider_a = Slider(ax_slider_a, 'Parameter a', 0, 200, valinit=initial_a, valfmt='%.1f')
    slider_b = Slider(ax_slider_b, 'Growth Rate b', 0, 0.001, valinit=initial_b, valfmt='%.6f')
    slider_c = Slider(ax_slider_c, 'Parameter c', -100, 100, valinit=initial_c, valfmt='%.1f')
    
    # Info text
    info_text = ax_info.text(0.02, 0.5, '', fontsize=12, verticalalignment='center')
    
    def update_model(val):
        """Update the model when sliders change."""
        a = slider_a.val
        b = slider_b.val
        c = slider_c.val
        
        # Calculate new model predictions
        y_new = a * np.exp(b * x) + c
        
        # Update the line
        line.set_ydata(y_new)
        
        # Calculate R²
        ss_res = np.sum((df[' value'].values - y_new) ** 2)
        ss_tot = np.sum((df[' value'].values - np.mean(df[' value'].values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate fair value (current date)
        current_days = df['days'].iloc[-1]
        fair_value = a * np.exp(b * current_days) + c
        
        # Calculate undervaluation
        undervaluation = ((fair_value - current_price) / current_price) * 100
        
        # Update info text
        info_text.set_text(f"""Model Parameters:
a = {a:.2f}, b = {b:.6f}, c = {c:.2f}

Model Performance:
R² = {r2:.4f}

Fair Value Analysis:
Current Price: ${current_price:.2f}
Model Fair Value: ${fair_value:.2f}
Undervaluation: {undervaluation:.1f}%""")
        
        # Update current price line (remove old one first)
        for line in ax_main.lines[:]:
            if line.get_label().startswith('Current Price'):
                line.remove()
        ax_main.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.0f}')
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_a.on_changed(update_model)
    slider_b.on_changed(update_model)
    slider_c.on_changed(update_model)
    
    # Initial update
    update_model(None)
    
    plt.tight_layout()
    return fig

def main():
    print("Interactive Palladium Growth Model")
    print("=" * 40)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Fit initial model
    print("\nFitting initial model...")
    a, b, c, r2 = fit_initial_model(df)
    
    print(f"Initial Model Parameters:")
    print(f"  a = {a:.2f}")
    print(f"  b = {b:.6f} (growth rate)")
    print(f"  c = {c:.2f}")
    print(f"  R² = {r2:.4f}")
    
    # Create interactive chart
    print("\nCreating interactive chart...")
    print("Use the sliders to adjust the model parameters!")
    print("Watch how the model line and fair value change in real-time!")
    
    fig = create_interactive_chart(df, a, b, c, r2)
    
    # Show the interactive plot
    plt.show()
    
    print("\n" + "=" * 40)
    print("INTERACTIVE CHART COMPLETE")
    print("=" * 40)
    print("Adjust the sliders to see how different parameters affect:")
    print("- Model fit (R² score)")
    print("- Fair value predictions")
    print("- Undervaluation percentage")
    print("- Overall model behavior")

if __name__ == "__main__":
    main()
