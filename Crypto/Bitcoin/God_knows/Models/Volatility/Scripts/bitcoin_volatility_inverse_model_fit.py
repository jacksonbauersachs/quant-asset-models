import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

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
mask = (vol > 0) & (vol < 10)
vol = vol[mask]
years = years[mask]

# Inverse decay model
def inverse_decay(years, a, b, c):
    return a / (1 + b * years) + c

# Fit the model
params, _ = curve_fit(inverse_decay, years, vol, p0=[30, 20, 0.3], maxfev=10000)
vol_pred = inverse_decay(years, *params)

# Calculate R²
ss_res = np.sum((vol - vol_pred) ** 2)
ss_tot = np.sum((vol - np.mean(vol)) ** 2)
r2 = 1 - ss_res / ss_tot

print('Best inverse decay fit for Bitcoin volatility:')
print(f'R² = {r2:.6f}')
print(f'Formula: volatility = a / (1 + b * years) + c')
print(f'a = {params[0]}')
print(f'b = {params[1]}')
print(f'c = {params[2]}')

# Save to file
with open('bitcoin_volatility_inverse_fit.txt', 'w') as f:
    f.write(f'R2 = {r2:.6f}\n')
    f.write('Formula: volatility = a / (1 + b * years) + c\n')
    f.write(f'a = {params[0]}\n')
    f.write(f'b = {params[1]}\n')
    f.write(f'c = {params[2]}\n') 