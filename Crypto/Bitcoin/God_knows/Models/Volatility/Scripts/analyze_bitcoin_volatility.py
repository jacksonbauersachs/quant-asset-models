import pandas as pd
import numpy as np

# Load Bitcoin data
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Price'])
df = df.sort_values('Date')

# Calculate daily returns
df['Returns'] = np.log(df['Price'] / df['Price'].shift(1))

# Calculate rolling 90-day volatility (annualized)
df['Vol_90d'] = df['Returns'].rolling(90).std() * np.sqrt(365)

# Get recent volatility (last data point)
recent_vol = df['Vol_90d'].iloc[-1]
recent_date = df['Date'].iloc[-1]

# Get volatility from 5 years ago
five_years_ago = recent_date - pd.DateOffset(years=5)
five_year_mask = df['Date'] >= five_years_ago
if five_year_mask.sum() > 0:
    five_year_idx = df[five_year_mask].index[0]
    five_year_vol = df['Vol_90d'].iloc[five_year_idx]
    five_year_date = df['Date'].iloc[five_year_idx]
else:
    five_year_idx = len(df)//2
    five_year_vol = df['Vol_90d'].iloc[five_year_idx]
    five_year_date = df['Date'].iloc[five_year_idx]

print('BITCOIN VOLATILITY ANALYSIS')
print('='*40)
print(f'Recent volatility ({recent_date.strftime("%Y-%m-%d")}): {recent_vol:.1%}')
print(f'5 years ago volatility ({five_year_date.strftime("%Y-%m-%d")}): {five_year_vol:.1%}')
print()

# Calculate some statistics
vol_clean = df['Vol_90d'].dropna()
print('HISTORICAL VOLATILITY STATISTICS:')
print(f'Minimum 90-day volatility: {vol_clean.min():.1%}')
print(f'Maximum 90-day volatility: {vol_clean.max():.1%}')
print(f'Mean 90-day volatility: {vol_clean.mean():.1%}')
print(f'Median 90-day volatility: {vol_clean.median():.1%}')
print()

# Look at volatility over time
print('VOLATILITY BY YEAR:')
df['Year'] = df['Date'].dt.year
yearly_vol = df.groupby('Year')['Vol_90d'].mean()
for year in sorted(yearly_vol.index)[-10:]:  # Last 10 years
    if not pd.isna(yearly_vol[year]):
        print(f'{year}: {yearly_vol[year]:.1%}')

print()
print('VOLATILITY DECAY ANALYSIS:')
print('='*30)

# Calculate years since Bitcoin start
df['Years'] = (df['Date'] - df['Date'].min()).dt.days / 365.25

# Look at volatility vs years for recent period
recent_data = df[df['Years'] >= 5].copy()  # Last 10+ years
if len(recent_data) > 0:
    print('Recent volatility trends:')
    for year_mark in [5, 7, 10, 12, 14]:
        year_data = recent_data[abs(recent_data['Years'] - year_mark) < 0.5]
        if len(year_data) > 0:
            avg_vol = year_data['Vol_90d'].mean()
            print(f'Year {year_mark}: {avg_vol:.1%}')

# Your model predictions
print()
print('YOUR MODEL PREDICTIONS:')
print('='*25)
a, b, c = 31.78, 22.19, 0.31
for years in [1, 2, 5, 10, 15]:
    model_vol = (a / (1 + b * years) + c) / 100
    print(f'Year {years}: {model_vol:.1%} (model prediction)') 