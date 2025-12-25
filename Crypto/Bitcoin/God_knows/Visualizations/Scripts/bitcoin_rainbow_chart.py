import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load Bitcoin data
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Price'])
df = df.sort_values('Date')

# Calculate days from start
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Your 94% R² growth model coefficients
a = 1.6329135221917355
b = -9.328646304661454

# Calculate model predictions for all days >= 90
df_model = df[df['Days'] >= 90].copy()
df_model['Model_Price'] = 10**(a * np.log(df_model['Days']) + b)

# Create the plot with log y-axis (rainbow chart style)
plt.figure(figsize=(15, 10))

# Plot actual Bitcoin prices
plt.semilogy(df['Date'], df['Price'], 'b-', linewidth=2, label='Actual Bitcoin Price', alpha=0.8)

# Plot your growth model (only for days >= 90)
plt.semilogy(df_model['Date'], df_model['Model_Price'], 'r--', linewidth=3, 
             label='94% R² Growth Model\nlog₁₀(price) = 1.633×ln(day) - 9.329', alpha=0.9)

# Add some style to make it look like the rainbow chart
plt.grid(True, alpha=0.3, which='both')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Bitcoin Price (USD) - Log Scale', fontsize=14)
plt.title('Bitcoin Price vs 94% R² Growth Model\n(Rainbow Chart Style)', fontsize=16, fontweight='bold')

# Add legend
plt.legend(fontsize=12, loc='upper left')

# Format the plot
plt.tight_layout()

# Add some annotations for key points
current_price = df['Price'].iloc[-1]
current_date = df['Date'].iloc[-1]
current_model = df_model['Model_Price'].iloc[-1]

# Annotate current prices
plt.annotate(f'Current Actual: ${current_price:,.0f}', 
             xy=(current_date, current_price), 
             xytext=(10, 20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
             fontsize=11, color='white', fontweight='bold')

plt.annotate(f'Model Prediction: ${current_model:,.0f}', 
             xy=(current_date, current_model), 
             xytext=(10, -30), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
             fontsize=11, color='white', fontweight='bold')

# Add difference annotation
diff_pct = (current_price - current_model) / current_model * 100
plt.text(0.02, 0.98, f'Current Premium: +{diff_pct:.1f}%\n(Bitcoin trading {current_price/current_model:.1f}x above model)', 
         transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
         verticalalignment='top')

# Set y-axis limits to show the full range nicely
plt.ylim(0.01, max(df['Price'].max(), df_model['Model_Price'].max()) * 2)

# Show the plot
plt.show()

# Print some statistics
print("="*60)
print("BITCOIN GROWTH MODEL ANALYSIS")
print("="*60)
print(f"Model Formula: log₁₀(price) = {a:.3f} × ln(day) + {b:.3f}")
print(f"R² = 0.936 (93.6% of variance explained)")
print(f"Model applies to days >= 90 (starting {df['Date'].min() + pd.Timedelta(days=90)})")
print()
print("CURRENT STATUS:")
print(f"Date: {current_date.strftime('%Y-%m-%d')}")
print(f"Actual Price: ${current_price:,.2f}")
print(f"Model Prediction: ${current_model:,.2f}")
print(f"Difference: ${current_price - current_model:,.2f}")
print(f"Premium: +{diff_pct:.1f}%")
print(f"Multiple: {current_price/current_model:.2f}x above model")
print()
print("INTERPRETATION:")
if diff_pct > 50:
    print("🔴 Bitcoin is trading SIGNIFICANTLY above the long-term growth model")
    print("   This suggests either:")
    print("   1. Bitcoin is in a major bull market phase")
    print("   2. The model needs updating for recent growth acceleration")
    print("   3. Current prices may be unsustainable vs historical trend")
elif diff_pct > 20:
    print("🟡 Bitcoin is trading moderately above the growth model")
elif diff_pct > -20:
    print("🟢 Bitcoin is trading close to the growth model")
else:
    print("🔵 Bitcoin is trading below the growth model") 