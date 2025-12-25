import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('2010_2025_Daily_Data_(BTC).csv')
df['Date'] = pd.to_datetime(df['Date'])

# Find the row for 2020-10-20
target_date = '2020-10-20'
target_row = df[df['Date'] == target_date].iloc[0]
target_index = df[df['Date'] == target_date].index[0]

print(f"Target date: {target_date}")
print(f"Target price: ${target_row['Price']:,.2f}")

# Get the 365 days leading up to 2020-10-20
start_index = target_index - 364  # 365 days total (including the target date)
end_index = target_index

yearly_data = df.iloc[start_index:end_index + 1].copy()
yearly_data = yearly_data.sort_values('Date')

print(f"Data range: {yearly_data['Date'].min()} to {yearly_data['Date'].max()}")
print(f"Number of days: {len(yearly_data)}")

# Calculate daily returns
yearly_data['Daily_Return'] = yearly_data['Price'].pct_change()

# Remove the first row (NaN return)
yearly_data = yearly_data.dropna(subset=['Daily_Return'])

# Calculate volatility (standard deviation of daily returns)
daily_volatility = yearly_data['Daily_Return'].std()
annualized_volatility = daily_volatility * np.sqrt(365)

print(f"\nVolatility Analysis:")
print(f"Daily volatility: {daily_volatility:.4f} ({daily_volatility*100:.2f}%)")
print(f"Annualized volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")

# Additional statistics
print(f"\nPrice Statistics (365 days):")
print(f"Starting price: ${yearly_data['Price'].iloc[0]:,.2f}")
print(f"Ending price: ${yearly_data['Price'].iloc[-1]:,.2f}")
print(f"Lowest price: ${yearly_data['Price'].min():,.2f}")
print(f"Highest price: ${yearly_data['Price'].max():,.2f}")
print(f"Total return: {((yearly_data['Price'].iloc[-1] / yearly_data['Price'].iloc[0]) - 1)*100:.2f}%")

# Calculate rolling 30-day volatility
yearly_data['Rolling_30d_Vol'] = yearly_data['Daily_Return'].rolling(window=30).std() * np.sqrt(365)
recent_30d_vol = yearly_data['Rolling_30d_Vol'].iloc[-1]

print(f"\nRecent 30-day annualized volatility: {recent_30d_vol:.4f} ({recent_30d_vol*100:.2f}%)") 