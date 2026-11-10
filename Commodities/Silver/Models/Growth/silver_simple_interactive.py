import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
import os

def load_silver_data():
    """Load the comprehensive silver data."""
    try:
        df = pd.read_csv('Commodities/Silver/exponential/Silver_historical_data.csv')
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Clean Price column - convert to float
        df['Price'] = df['Price'].astype(float)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['Date'].min()
        df['days'] = (df['Date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Silver data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading silver data: {e}")
        return None

def fit_optimal_parameters(df):
    """Find optimal parameters that maximize R² for the silver data."""
    x = df['days'].values
    y = df['Price'].values
    
    def exponential_model(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        # Fit the model with reasonable bounds
        popt, _ = curve_fit(exponential_model, x, y, 
                           p0=[100, 0.0001, 200],  # Initial guess
                           bounds=([0, 0, 0], [1000, 0.01, 1000]),  # Reasonable bounds
                           maxfev=2000)
        a_opt, b_opt, c_opt = popt
        
        # Calculate R² for the optimal parameters
        y_pred = exponential_model(x, a_opt, b_opt, c_opt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_opt = 1 - (ss_res / ss_tot)
        
        print(f"Optimal parameters found:")
        print(f"  A (amplitude): {a_opt:.2f}")
        print(f"  B (growth rate): {b_opt:.6f}")
        print(f"  C (intercept): {c_opt:.2f}")
        print(f"  R² score: {r2_opt:.6f}")
        
        return a_opt, b_opt, c_opt, r2_opt
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        print("Using default parameters...")
        return 54.36, 0.000172, -11.07, 0.0

def create_simple_interactive_chart(df):
    """Create a simple interactive chart with growth rate and intercept sliders."""
    
    # Get initial parameters
    x = df['days'].values
    y = df['Price'].values
    
    # Find optimal parameters for this data
    print("\nFinding optimal parameters for silver data...")
    initial_a, initial_b, fixed_c, r2_optimal = fit_optimal_parameters(df)
    
    # Adjust intercept so model starts at first data point
    first_price = df['Price'].iloc[0]
    first_days = df['days'].iloc[0]
    # Calculate what C should be so that model starts at first price
    adjusted_c = first_price - initial_a * np.exp(initial_b * first_days)
    
    print(f"\nUsing optimized parameters:")
    print(f"  A: {initial_a:.2f}")
    print(f"  B: {initial_b:.6f} (adjustable)")
    print(f"  C: {adjusted_c:.2f} (fixed - adjusted to start at first data point)")
    print(f"  Optimal R²: {r2_optimal:.6f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual data
    ax1.scatter(df['years'], df['Price'], alpha=0.6, s=1, color='blue', label='Actual Data')
    
    # Initial model line
    y_initial = initial_a * np.exp(initial_b * x) + adjusted_c
    line1, = ax1.plot(df['years'], y_initial, 'r-', linewidth=2, label='Exponential Model')
    
    # Current price line
    current_price = df['Price'].iloc[-1]
    ax1.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.2f}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Silver Price ($)')
    ax1.set_title(f'Interactive Silver Growth Model - Fixed Intercept ({adjusted_c:.2f}), Adjust Growth Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.scatter(df['years'], df['Price'], alpha=0.6, s=1, color='blue', label='Actual Data')
    y_initial_log = initial_a * np.exp(initial_b * x) + adjusted_c
    line2, = ax2.plot(df['years'], y_initial_log, 'r-', linewidth=2, label='Exponential Model')
    ax2.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, label=f'Current Price: ${current_price:.2f}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Silver Price ($) - Log Scale')
    ax2.set_title('Log Scale View (Linear Growth Appearance)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Create slider (only growth rate now) - less sensitive range
    ax_slider_b = plt.axes([0.2, 0.02, 0.6, 0.03])
    # Use a much smaller range around the optimal value for less sensitivity
    slider_range = max(0.0001, initial_b * 0.5)  # 50% of optimal value as range
    slider_b = Slider(ax_slider_b, 'Growth Rate (b) - Adjust Curve Steepness', 
                     initial_b - slider_range, initial_b + slider_range, 
                     valinit=initial_b, valfmt='%.6f')
    
    # Info text
    info_text = fig.text(0.02, 0.20, '', fontsize=11, verticalalignment='top')
    
    def update_model(val):
        """Update the model when slider changes."""
        b = slider_b.val
        
        # Calculate new model predictions (using fixed intercept)
        y_new = initial_a * np.exp(b * x) + adjusted_c
        
        # Update both lines
        line1.set_ydata(y_new)
        line2.set_ydata(y_new)
        
        # Calculate R²
        ss_res = np.sum((y - y_new) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate fair value (current date)
        current_days = df['days'].iloc[-1]
        fair_value = initial_a * np.exp(b * current_days) + adjusted_c
        
        # Calculate undervaluation
        undervaluation = ((fair_value - current_price) / current_price) * 100
        
        # Update info text
        info_text.set_text(f"""Growth Rate (b): {b:.6f}
Fixed Intercept (c): {adjusted_c:.1f}
R² Score: {r2:.4f}
Current Price: ${current_price:.2f}
Model Fair Value: ${fair_value:.2f}
Undervaluation: {undervaluation:.1f}%

Instructions:
- Drag the slider to adjust the growth rate (curve steepness)
- Intercept is fixed at {adjusted_c:.1f}
- Watch how the model line changes in real-time
- See how fair value and undervaluation change""")
        
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider_b.on_changed(update_model)
    
    # Initial update
    update_model(None)
    
    plt.tight_layout()
    return fig

def main():
    print("Simple Interactive Silver Growth Model")
    print("=" * 50)
    
    # Load data
    df = load_silver_data()
    if df is None:
        return
    
    print("\nCreating interactive chart...")
    print("Use the slider to adjust the growth rate (curve steepness)!")
    print("Intercept is optimized for your silver data!")
    print("Watch how the model line and fair value change in real-time!")
    
    fig = create_simple_interactive_chart(df)
    
    # Show the interactive plot
    plt.show()
    
    print("\n" + "=" * 50)
    print("INTERACTIVE CHART COMPLETE")
    print("=" * 50)
    print("Adjust the slider to see how it affects:")
    print("- Growth rate slider: Adjust the curve steepness")
    print("- Intercept is optimized for your silver data")
    print("- Model fit (R² score)")
    print("- Fair value predictions")
    print("- Undervaluation percentage")
    print("- Overall model behavior")

if __name__ == "__main__":
    main()
