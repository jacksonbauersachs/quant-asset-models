import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker

# Load the 2010-2020 Bitcoin data
print("Loading 2010-2020 Bitcoin dataset...")
df = pd.read_csv('../../../Data/2010_2020_Daily_Data_(BTC).csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Price'])
df = df.sort_values('Date')

print(f"Loaded {len(df):,} days of Bitcoin data")
print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")

# Calculate days from start (start at 1)
df['Days'] = (df['Date'] - df['Date'].min()).dt.days + 1

# Skip the first 364 days (start at day 365)
df = df[df['Days'] >= 365].copy()

print(f"Using {len(df):,} days starting from day 365")

# Model: log10(price) = a * ln(day) + b
X = np.log(df['Days'])
Y = np.log10(df['Price'])

# Central model fit
slope, intercept, r_value, p_value, std_err = linregress(X, Y)
R2 = r_value ** 2

print('='*60)
print('BITCOIN RAINBOW CHART 2010-2020 (AUTOMATIC FLATTENING BANDS)')
print('='*60)
print(f"Model Formula: log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}")
print(f"R² = {R2:.6f}")
print(f"Data points: {len(df)}")

# Save coefficients with update information
with open('../Results/bitcoin_growth_model_coefficients_2010_2020.txt', 'w') as f:
    f.write(f'a = {slope}\n')
    f.write(f'b = {intercept}\n')
    f.write(f'R2 = {R2}\n')
    f.write('# Formula: log10(price) = a * ln(day) + b (day >= 365)\n')
    f.write(f'# Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'# Data range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}\n')
    f.write(f'# Total data points: {len(df):,}\n')
    f.write(f'# Current price: ${df["Price"].iloc[-1]:,.2f}\n')
    f.write(f'# All-time high: ${df["Price"].max():,.2f}\n')

# Model prediction
model_price = 10**(slope * np.log(df['Days']) + intercept)
log_model = np.log10(model_price)
log_price = np.log10(df['Price'])
log_dev = log_price - log_model

df['log_dev'] = log_dev

# Assign each row to a year
df['Year'] = df['Date'].dt.year

# For each year, find the max and min log deviation
annual = df.groupby('Year').agg({'Days': 'median', 'log_dev': ['max', 'min']})
annual.columns = ['Days', 'log_dev_max', 'log_dev_min']
annual = annual.reset_index()

# Expand envelopes by 10% of the band range
band_range = annual['log_dev_max'].max() - annual['log_dev_min'].min()
expand = 0.10 * band_range
annual['log_dev_max'] += expand
annual['log_dev_min'] -= expand

# Fit log-linear models to the annual max and min log deviations
X_env = np.log(annual['Days'])
log_model_annual = slope * np.log(annual['Days']) + intercept
Y_max = log_model_annual + annual['log_dev_max']
Y_min = log_model_annual + annual['log_dev_min']

# Fit upper envelope
slope_upper, intercept_upper, *_ = linregress(X_env, Y_max)
# Fit lower envelope
slope_lower, intercept_lower, *_ = linregress(X_env, Y_min)

print(f"Upper envelope: log10(price) = {slope_upper:.6f} * ln(day) + {intercept_upper:.6f}")
print(f"Lower envelope: log10(price) = {slope_lower:.6f} * ln(day) + {intercept_lower:.6f}")

# Prepare band colors and custom names for 5 fills (6 boundaries)
band_colors = ['blue', 'green', 'yellow', 'orange', 'red']
band_names = ['Firesale!', 'Buy', 'Hold', 'Sell', 'Danger!']
num_fills = len(band_colors)
num_bounds = num_fills + 1

# Interpolate slopes and intercepts for each boundary
slopes = np.linspace(slope_lower, slope_upper, num_bounds)
intercepts = np.linspace(intercept_lower, intercept_upper, num_bounds)

# Save band formulas with custom names (boundaries)
with open('../Results/bitcoin_rainbow_band_formulas_2010_2020.txt', 'w') as f:
    f.write(f'# Bitcoin Rainbow Band Formulas (2010-2020)\n')
    f.write(f'# Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'# Data range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}\n')
    f.write(f'# R² = {R2:.6f}\n\n')
    for i in range(num_bounds):
        if i == 0:
            name = 'Lower Envelope'
        elif i == num_bounds - 1:
            name = 'Upper Envelope'
        else:
            name = band_names[i-1]
        f.write(f'{name}: log10(price) = {slopes[i]:.6f} * ln(day) + {intercepts[i]:.6f}\n')

# Set up black background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 10))

# Plot rainbow bands as log-linear curves (fill between boundaries)
bound_curves = [10**(slopes[i] * np.log(df['Days']) + intercepts[i]) for i in range(num_bounds)]
for i in range(num_fills):
    ax.fill_between(df['Date'], bound_curves[i], bound_curves[i+1], color=band_colors[i], alpha=0.7, label=None)

# Plot actual price as white line
ax.semilogy(df['Date'], df['Price'], color='white', linewidth=2, label='Actual Price')

# Plot model as dashed line
ax.semilogy(df['Date'], model_price, color='black', linestyle='--', linewidth=2, label='Model')

# Formatting
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Bitcoin Price (USD)', fontsize=14)
ax.set_title('Bitcoin Rainbow Chart 2010-2020 (Auto-Flattening Bands, Start Day 365)', fontsize=18, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Custom y-axis formatting to show actual prices
def price_formatter(x, pos):
    if x >= 1000:
        return f'${x/1000:.0f}K'
    else:
        return f'${x:.0f}'

ax.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))

# Custom legend for bands with names (reverse order for legend)
from matplotlib.patches import Patch
legend_patches = [Patch(color=band_colors[i], label=band_names[i]) for i in reversed(range(num_fills))]
legend_patches.append(Patch(color='white', label='Actual Price'))
legend_patches.append(Patch(color='black', label='Model'))
ax.legend(handles=legend_patches, fontsize=12, loc='upper left')

# Annotate the chart with the main model formula and R²
formula_text = f'Model: log₁₀(price) = {slope:.3f} × ln(day) + {intercept:.3f}\nR² = {R2:.4f}'
ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=13, color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

# Add date range information
date_range_text = f'Data: {df["Date"].min().strftime("%b %Y")} - {df["Date"].max().strftime("%b %Y")}\nCurrent: ${df["Price"].iloc[-1]:,.0f}'
ax.text(0.98, 0.02, date_range_text, transform=ax.transAxes, fontsize=11, color='white', 
        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'), 
        horizontalalignment='right', verticalalignment='bottom')

plt.tight_layout()
ax.set_ylim(0.01, max(df['Price'].max(), model_price.max()) * 2)
plt.savefig('../../../Visualizations/Images/bitcoin_rainbow_chart_2010_2020.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nRainbow chart saved to: ../../../Visualizations/Images/bitcoin_rainbow_chart_2010_2020.png")
print(f"Model coefficients saved to: ../Results/bitcoin_growth_model_coefficients_2010_2020.txt")
print(f"Band formulas saved to: ../Results/bitcoin_rainbow_band_formulas_2010_2020.txt")
print(f"\nModel Summary:")
print(f"  Formula: log₁₀(price) = {slope:.3f} × ln(day) + {intercept:.3f}")
print(f"  R² = {R2:.4f}")
print(f"  Data range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Current price: ${df['Price'].iloc[-1]:,.2f}")
print(f"  All-time high: ${df['Price'].max():,.2f}") 