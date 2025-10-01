import pandas as pd
import numpy as np

# Load the data files to get current values
bitcoin_df = pd.read_csv('Portfolio/Data/Bitcoin_all_time_price.csv')
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
bitcoin_df = bitcoin_df.sort_values('Date').reset_index(drop=True)

gold_df = pd.read_csv('Portfolio/Data/Gold_all_time_price.csv')
gold_df['Date'] = pd.to_datetime(gold_df['Date'])
gold_df['Price'] = gold_df['Price'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
gold_df = gold_df.sort_values('Date').reset_index(drop=True)

silver_df = pd.read_csv('Portfolio/Data/Silver_all_time_price.csv')
silver_df['Date'] = pd.to_datetime(silver_df['Date'])
silver_df = silver_df.sort_values('Date').reset_index(drop=True)

# Calculate days
bitcoin_genesis = pd.Timestamp('2009-01-03')
bitcoin_df['Days_Since_Genesis'] = (bitcoin_df['Date'] - bitcoin_genesis).dt.days
bitcoin_df['Days'] = (bitcoin_df['Date'] - bitcoin_df['Date'].min()).dt.days

gold_df['Days'] = (gold_df['Date'] - gold_df['Date'].min()).dt.days
silver_df['Days'] = (silver_df['Date'] - silver_df['Date'].min()).dt.days

print('=== BITCOIN MANUAL VERIFICATION ===')
print(f'Current price: ${bitcoin_df["Price"].iloc[-1]:,.2f}')
print(f'Days since genesis: {bitcoin_df["Days_Since_Genesis"].iloc[-1]}')

# Bitcoin formula: log10(price) = 2.464614 * ln(day) + -16.451114
current_day = bitcoin_df['Days_Since_Genesis'].iloc[-1]
log_price = 2.464614 * np.log(current_day) + (-16.451114)
fair_price = 10**log_price
print(f'Fair price calculation: 10^({2.464614} * ln({current_day}) + {-16.451114})')
print(f'Fair price = 10^{log_price:.6f} = ${fair_price:.2f}')

# 1-year projection
future_day = current_day + 365
log_price_1y = 2.464614 * np.log(future_day) + (-16.451114)
price_1y = 10**log_price_1y
return_1y = ((price_1y - bitcoin_df['Price'].iloc[-1]) / bitcoin_df['Price'].iloc[-1]) * 100
print(f'1-year projection: 10^({2.464614} * ln({future_day}) + {-16.451114}) = ${price_1y:.2f}')
print(f'1-year return: {return_1y:.2f}%')

print('\n=== GOLD MANUAL VERIFICATION ===')
print(f'Current price: ${gold_df["Price"].iloc[-1]:,.2f}')
print(f'Days since start: {gold_df["Days"].iloc[-1]}')

# Gold formula: Price = 28.89 * exp(0.000239 * Days) + 239.95
current_days = gold_df['Days'].iloc[-1]
fair_price_gold = 28.89 * np.exp(0.000239 * current_days) + 239.95
print(f'Fair price calculation: 28.89 * exp(0.000239 * {current_days}) + 239.95')
print(f'Fair price = ${fair_price_gold:.2f}')

# 1-year projection
future_days = current_days + 365
price_1y_gold = 28.89 * np.exp(0.000239 * future_days) + 239.95
return_1y_gold = ((price_1y_gold - gold_df['Price'].iloc[-1]) / gold_df['Price'].iloc[-1]) * 100
print(f'1-year projection: 28.89 * exp(0.000239 * {future_days}) + 239.95 = ${price_1y_gold:.2f}')
print(f'1-year return: {return_1y_gold:.2f}%')

print('\n=== SILVER MANUAL VERIFICATION ===')
print(f'Current price: ${silver_df["Price"].iloc[-1]:.2f}')
print(f'Days since start: {silver_df["Days"].iloc[-1]}')

# Silver formula: Price = 0.96 * exp(0.000164 * Days) + 3.56
current_days_silver = silver_df['Days'].iloc[-1]
fair_price_silver = 0.96 * np.exp(0.000164 * current_days_silver) + 3.56
print(f'Fair price calculation: 0.96 * exp(0.000164 * {current_days_silver}) + 3.56')
print(f'Fair price = ${fair_price_silver:.2f}')

# 1-year projection
future_days_silver = current_days_silver + 365
price_1y_silver = 0.96 * np.exp(0.000164 * future_days_silver) + 3.56
return_1y_silver = ((price_1y_silver - silver_df['Price'].iloc[-1]) / silver_df['Price'].iloc[-1]) * 100
print(f'1-year projection: 0.96 * exp(0.000164 * {future_days_silver}) + 3.56 = ${price_1y_silver:.2f}')
print(f'1-year return: {return_1y_silver:.2f}%')
