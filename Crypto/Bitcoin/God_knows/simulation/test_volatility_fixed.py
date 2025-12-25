import numpy as np

def calculate_volatility_at_time(years, vol_params):
    """Calculate volatility at a given time using exponential decay model"""
    a, b, c = vol_params['a'], vol_params['b'], vol_params['c']
    
    # Bitcoin's current age (years since 2010-07-18)
    bitcoin_current_age = 15.0  # 2025-07-19 minus 2010-07-18
    
    # Calculate volatility at Bitcoin's age + future years
    total_years = bitcoin_current_age + years
    volatility = a * np.exp(-b * total_years) + c
    
    # Ensure volatility stays within reasonable bounds
    volatility = np.maximum(volatility, 0.05)  # Minimum 5%
    volatility = np.minimum(volatility, 1.0)   # Maximum 100%
    
    return volatility

# Test with the parameters from the model
a_vol = 2.310879
b_vol = 0.124138
c_vol = 0.077392

vol_params = {'a': a_vol, 'b': b_vol, 'c': c_vol}

print("="*60)
print("TESTING CORRECTED VOLATILITY CALCULATION")
print("="*60)
print(f"Parameters: a={a_vol}, b={b_vol}, c={c_vol}")
print(f"Bitcoin current age: 15.0 years")
print()

print("Expected values from model file:")
print("Year 0.0 from now: 43.6%")
print("Year 1.0 from now: 39.4%")
print("Year 2.0 from now: 35.7%")
print()

print("Calculated values (with Bitcoin age adjustment):")
for year in range(11):
    vol = calculate_volatility_at_time(year, vol_params)
    total_years = 15.0 + year
    print(f"Year {year} (total age {total_years:.1f}): {vol*100:.1f}%")

print()
print("Raw calculation (before bounds):")
for year in range(11):
    total_years = 15.0 + year
    raw_vol = a_vol * np.exp(-b_vol * total_years) + c_vol
    print(f"Year {year} (total age {total_years:.1f}): {raw_vol*100:.1f}%") 