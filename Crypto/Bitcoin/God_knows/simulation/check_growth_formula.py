import numpy as np
import matplotlib.pyplot as plt

# Load the growth model parameters
a = 1.827743
b = -10.880943

# Calculate what day we're at today (based on our data)
# Our growth formula is offset by 365 days, so we need to adjust
actual_today_day = 5476
today_day = actual_today_day - 365  # Adjust for the 365-day offset

print("="*60)
print("BITCOIN GROWTH FORMULA ANALYSIS")
print("="*60)
print(f"Growth Formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
print(f"Actual today's day number: {actual_today_day}")
print(f"Adjusted for growth formula (day {today_day} in formula terms)")
print()

# Calculate expected prices for the next 10 years
print("Expected Bitcoin Prices (from growth formula):")
print("-" * 50)
for year in range(11):
    future_day = today_day + int(year * 365.25)
    expected_price = 10**(a * np.log(future_day) + b)
    actual_future_day = future_day + 365  # Convert back to actual day numbers
    print(f"Year {year}: Day {actual_future_day} (formula day {future_day}) -> ${expected_price:,.2f}")

print()
print("="*60)
print("ANALYSIS")
print("="*60)

# Check if the formula is predicting growth or decline
year_1_day = today_day + 365
year_1_price = 10**(a * np.log(year_1_day) + b)
current_price = 10**(a * np.log(today_day) + b)

print(f"Current expected price (formula day {today_day}): ${current_price:,.2f}")
print(f"1 year expected price (formula day {year_1_day}): ${year_1_price:,.2f}")
print(f"Growth rate: {((year_1_price/current_price - 1) * 100):.2f}%")

if year_1_price < current_price:
    print("⚠️  WARNING: The growth formula is predicting DECLINE!")
    print("This suggests the formula may not be suitable for future predictions.")
else:
    print("✅ The growth formula is predicting growth.")

# Let's also check what the actual current price is vs what the formula predicts
actual_current_price = 118075
print(f"\nActual current price: ${actual_current_price:,.2f}")
print(f"Formula predicted price: ${current_price:,.2f}")
print(f"Difference: {((actual_current_price/current_price - 1) * 100):.2f}%")

# Create a visualization
days_range = np.arange(today_day - 365, today_day + 365*10)
prices = 10**(a * np.log(days_range) + b)

plt.figure(figsize=(12, 8))
plt.plot(days_range, prices, 'b-', linewidth=2, label='Growth Formula Prediction')
plt.axvline(x=today_day, color='r', linestyle='--', label=f'Today (Formula Day {today_day})')
plt.axhline(y=actual_current_price, color='g', linestyle='--', label=f'Actual Price: ${actual_current_price:,.0f}')

plt.xlabel('Formula Day Number (offset by 365)')
plt.ylabel('Bitcoin Price ($)')
plt.title('Bitcoin Growth Formula: Past and Future Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Add annotations
plt.annotate(f'Today: ${actual_current_price:,.0f}', 
             xy=(today_day, actual_current_price), 
             xytext=(today_day + 200, actual_current_price * 2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10)

plt.tight_layout()
plt.savefig('Results/Bitcoin/growth_formula_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to: Results/Bitcoin/growth_formula_analysis.png") 