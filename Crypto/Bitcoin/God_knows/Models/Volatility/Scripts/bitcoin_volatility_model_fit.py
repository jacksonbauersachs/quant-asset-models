import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load Bitcoin data
file_path = 'Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv'
df = pd.read_csv(file_path)
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df = df.dropna(subset=['Price'])

# Calculate returns and rolling volatility (30d window)
df['Returns'] = df['Price'].pct_change()
df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)
df['Years'] = df['Days'] / 365.25

# Remove early NaNs and outliers
vol = df['Volatility_30d'].dropna()
years = df['Years'][vol.index]
# Remove extreme outliers (vol > 10)
mask = (vol > 0) & (vol < 10)
vol = vol[mask]
years = years[mask]

results = []

# 1. log(vol) = a * years + b (exponential decay)
log_vol = np.log(vol)
res = linregress(years, log_vol)
results.append((res.rvalue**2, 'log(vol) = a * years + b', res.slope, res.intercept))

# 2. log10(vol) = a * years + b
log10_vol = np.log10(vol)
res = linregress(years, log10_vol)
results.append((res.rvalue**2, 'log10(vol) = a * years + b', res.slope, res.intercept))

# 3. log10(vol) = a * log10(years) + b (power law)
log10_years = np.log10(years + 1e-6)
res = linregress(log10_years, log10_vol)
results.append((res.rvalue**2, 'log10(vol) = a * log10(years) + b', res.slope, res.intercept))

# 4. log(vol) = a * log(years) + b (power law)
log_years = np.log(years + 1e-6)
res = linregress(log_years, log_vol)
results.append((res.rvalue**2, 'log(vol) = a * log(years) + b', res.slope, res.intercept))

# 5. 1/vol = a * years + b (inverse decay)
inv_vol = 1 / vol
res = linregress(years, inv_vol)
results.append((res.rvalue**2, '1/vol = a * years + b', res.slope, res.intercept))

# Find best
results.sort(reverse=True)
best = results[0]

print('Best Bitcoin volatility decay fit:')
print(f'R² = {best[0]:.6f}')
print(f'Formula: {best[1]}')
print(f'a = {best[2]}')
print(f'b = {best[3]}')

# Save to file
with open('bitcoin_volatility_best_fit.txt', 'w') as f:
    f.write(f'R2 = {best[0]:.6f}\n')
    f.write(f'Formula: {best[1]}\n')
    f.write(f'a = {best[2]}\n')
    f.write(f'b = {best[3]}\n') 