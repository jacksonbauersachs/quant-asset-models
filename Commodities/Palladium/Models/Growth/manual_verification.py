import pandas as pd
import numpy as np

def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def manual_verification():
    """Manually verify the projections."""
    
    # Load data
    df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    start_date = df['date'].min()
    df['days'] = (df['date'] - start_date).dt.days
    
    # Model parameters
    fixed_a = 54.36
    fixed_b = 0.000172
    optimal_c = 143.98
    
    # Current values
    current_days = df['days'].iloc[-1]
    current_price = df[' value'].iloc[-1]
    current_date = df['date'].iloc[-1]
    
    print("MANUAL VERIFICATION")
    print("=" * 50)
    print(f"Current Date: {current_date}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Current Days: {current_days}")
    
    # Calculate current fair value
    current_fair_price = exponential_func(current_days, fixed_a, fixed_b, optimal_c)
    undervaluation = ((current_price - current_fair_price) / current_fair_price) * 100
    
    print(f"Current Fair Price: ${current_fair_price:.2f}")
    print(f"Undervaluation: {undervaluation:.2f}%")
    print()
    
    # Test 1-year projection
    years = 1
    future_days = current_days + (years * 365.25)
    
    print(f"1-YEAR PROJECTION VERIFICATION")
    print("-" * 40)
    
    # Model-based projection (from fair value)
    model_future_price = exponential_func(future_days, fixed_a, fixed_b, optimal_c)
    model_return = ((model_future_price - current_fair_price) / current_fair_price) * 100
    
    print(f"Model Future Price: ${model_future_price:.2f}")
    print(f"Model Return: {model_return:.2f}%")
    
    # Current price-based projection
    # If palladium is undervalued, we should get BETTER returns
    # The current return should be: model_return + undervaluation_benefit
    model_growth_rate = ((model_future_price / current_fair_price) ** (1/years) - 1) * 100
    current_based_price = current_price * ((1 + model_growth_rate/100) ** years)
    current_return = ((current_based_price - current_price) / current_price) * 100
    
    # The current return should be higher because we start from undervalued price
    # It should be: model_return + the benefit of starting from undervalued price
    expected_current_return = model_return + abs(undervaluation)  # Add the undervaluation benefit
    
    print(f"Model Growth Rate: {model_growth_rate:.2f}%")
    print(f"Current-Based Price: ${current_based_price:.2f}")
    print(f"Current Return: {current_return:.2f}%")
    print(f"Expected Current Return: {expected_current_return:.2f}% (model_return + undervaluation_benefit)")
    
    print()
    print("VERIFICATION LOGIC:")
    print(f"1. Current price (${current_price:.2f}) is {undervaluation:.2f}% below fair value (${current_fair_price:.2f})")
    print(f"2. Model grows from ${current_fair_price:.2f} to ${model_future_price:.2f} in {years} year")
    print(f"3. Growth rate: {model_growth_rate:.2f}%")
    print(f"4. Apply same growth rate to current price: ${current_price:.2f} * (1 + {model_growth_rate/100:.4f}) = ${current_based_price:.2f}")
    print(f"5. Current return should be HIGHER because we start from undervalued price")
    print(f"6. Expected: {model_return:.2f}% + {abs(undervaluation):.2f}% = {expected_current_return:.2f}%")
    
    # Check if current return is higher
    if current_return > model_return:
        print(f"CORRECT: Current return ({current_return:.2f}%) > Model return ({model_return:.2f}%)")
    else:
        print(f"WRONG: Current return ({current_return:.2f}%) < Model return ({model_return:.2f}%)")
        print(f"Expected current return should be ~{expected_current_return:.2f}%")
    
    print()
    print("EXPECTED BEHAVIOR:")
    print("- If palladium is undervalued, buying at current price should give BETTER returns")
    print("- The current price-based projection should be HIGHER than model-based")
    print("- This is because you get the benefit of the undervaluation upfront")

if __name__ == "__main__":
    manual_verification()
