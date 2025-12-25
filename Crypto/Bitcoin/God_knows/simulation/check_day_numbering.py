import pandas as pd
import numpy as np

# Load the data to understand day numbering
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Price'])
df = df.sort_values('Date')

print("="*60)
print("UNDERSTANDING DAY NUMBERING")
print("="*60)

print(f"Data range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total days in dataset: {len(df)}")

# Bitcoin started in 2009, so let's calculate what day we're at
bitcoin_start = pd.to_datetime('2009-01-03')  # Bitcoin genesis block
today = df['Date'].max()

days_since_genesis = (today - bitcoin_start).days
print(f"Days since Bitcoin genesis: {days_since_genesis}")

# The formula uses day >= 365, so day 365 corresponds to 2010-01-03
formula_day_365_date = bitcoin_start + pd.Timedelta(days=365)
print(f"Formula day 365 corresponds to: {formula_day_365_date.strftime('%Y-%m-%d')}")

# Calculate what formula day number today should be
formula_day_today = days_since_genesis - 365 + 365  # Adjust for formula's day >= 365
print(f"Today's formula day number should be: {formula_day_today}")

# Check what the formula predicts
a = 1.827743
b = -10.880943

formula_prediction = 10**(a * np.log(formula_day_today) + b)
actual_price = df['Price'].iloc[-1]

print(f"Formula prediction for today: ${formula_prediction:,.2f}")
print(f"Actual price today: ${actual_price:,.2f}")
print(f"Difference: {((actual_price/formula_prediction - 1) * 100):.1f}%")

print()
print("="*60)
print("CHECKING DIFFERENT DAY NUMBERING APPROACHES")
print("="*60)

# Approach 1: Use the dataset length as day number
dataset_day = len(df)
prediction_1 = 10**(a * np.log(dataset_day) + b)
print(f"Dataset length ({dataset_day}) as day number: ${prediction_1:,.2f}")

# Approach 2: Use days since genesis
genesis_day = days_since_genesis
prediction_2 = 10**(a * np.log(genesis_day) + b)
print(f"Days since genesis ({genesis_day}) as day number: ${prediction_2:,.2f}")

# Approach 3: Use formula day (genesis - 365 + 365)
formula_day = formula_day_today
prediction_3 = 10**(a * np.log(formula_day) + b)
print(f"Formula day ({formula_day}) as day number: ${prediction_3:,.2f}")

print()
print("Which approach gives the best prediction?")
differences = [
    abs(actual_price - prediction_1),
    abs(actual_price - prediction_2), 
    abs(actual_price - prediction_3)
]

best_approach = differences.index(min(differences)) + 1
print(f"Approach {best_approach} gives the best prediction!") 