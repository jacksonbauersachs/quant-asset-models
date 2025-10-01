import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.optimize import minimize

def load_palladium_data():
    """Load the comprehensive palladium data."""
    try:
        df = pd.read_csv('Commodities/Palladium/Data/palladium-prices-historical-chart-data-cleaned.csv')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days since start
        start_date = df['date'].min()
        df['days'] = (df['date'] - start_date).dt.days
        
        # Calculate years for plotting
        start_year = start_date.year
        df['years'] = (df['days'] / 365.25) + start_year
        
        print(f"Palladium data loaded: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df[' value'].min():.2f} to ${df[' value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading palladium data: {e}")
        return None

def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def calculate_above_below_ratio(x, y, a, b, c):
    """Calculate the ratio of points above vs below the model line."""
    
    # Get model predictions
    y_pred = exponential_func(x, a, b, c)
    
    # Count points above and below
    above = np.sum(y > y_pred)
    below = np.sum(y < y_pred)
    total = len(y)
    
    # Calculate percentages
    above_pct = (above / total) * 100
    below_pct = (below / total) * 100
    
    # Calculate deviation from 50/50
    deviation = abs(above_pct - 50.0)
    
    return above, below, above_pct, below_pct, deviation

def test_multiple_starting_points(df, fixed_a=54.36):
    """Test multiple starting points to see if we get different results."""
    
    x = df['days'].values
    y = df[' value'].values
    
    print(f"\nTesting multiple starting points for 50/50 optimization:")
    print(f"  Fixed a (y-intercept): {fixed_a}")
    
    # Test different starting points
    starting_points = [
        [0.0001, 0],      # Low growth, zero intercept
        [0.0002, 50],     # Medium growth, positive intercept
        [0.0003, -50],    # High growth, negative intercept
        [0.00005, 100],   # Very low growth, high intercept
        [0.0005, -100],   # Very high growth, low intercept
        [0.000172, 20.66] # Previous "optimal" values
    ]
    
    results = []
    
    for i, (start_b, start_c) in enumerate(starting_points):
        print(f"\n  Starting point {i+1}: b={start_b:.6f}, c={start_c:.2f}")
        
        def objective(params):
            """Objective function to minimize deviation from 50/50."""
            b, c = params
            above, below, above_pct, below_pct, deviation = calculate_above_below_ratio(x, y, fixed_a, b, c)
            return deviation
        
        try:
            # Bounds: growth rate between 0.00001 and 0.002, intercept between -1000 and 1000
            bounds = [(0.00001, 0.002), (-1000, 1000)]
            
            result = minimize(objective, [start_b, start_c], bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimal_b, optimal_c = result.x
                above, below, above_pct, below_pct, deviation = calculate_above_below_ratio(x, y, fixed_a, optimal_b, optimal_c)
                
                print(f"    Result: b={optimal_b:.6f}, c={optimal_c:.2f}")
                print(f"    Above: {above} ({above_pct:.1f}%), Below: {below} ({below_pct:.1f}%)")
                print(f"    Deviation: {deviation:.1f}%")
                print(f"    Function calls: {result.nfev}")
                
                results.append({
                    'start_b': start_b,
                    'start_c': start_c,
                    'optimal_b': optimal_b,
                    'optimal_c': optimal_c,
                    'above_pct': above_pct,
                    'below_pct': below_pct,
                    'deviation': deviation,
                    'nfev': result.nfev
                })
            else:
                print(f"    FAILED: {result.message}")
                
        except Exception as e:
            print(f"    ERROR: {e}")
    
    return results

def analyze_results(results):
    """Analyze the optimization results."""
    
    print(f"\n" + "="*80)
    print("DEEP DIVE ANALYSIS")
    print("="*80)
    
    if not results:
        print("No successful optimizations!")
        return
    
    # Check if all results are identical
    first_result = results[0]
    all_same = all(
        abs(r['optimal_b'] - first_result['optimal_b']) < 1e-10 and
        abs(r['optimal_c'] - first_result['optimal_c']) < 1e-10
        for r in results
    )
    
    print(f"\nAll results identical: {all_same}")
    
    if all_same:
        print("SUSPICIOUS: All starting points converged to identical values!")
        print("This suggests either:")
        print("1. The optimization is working correctly and there's only one global optimum")
        print("2. The optimization is getting stuck in a local minimum")
        print("3. There's some hardcoding or constraint we're missing")
    else:
        print("Different starting points gave different results")
        print("This suggests multiple local minima or the optimization is working correctly")
    
    # Show all results
    print(f"\nDetailed Results:")
    print(f"{'Start':<15} {'Optimal b':<12} {'Optimal c':<10} {'Above%':<8} {'Below%':<8} {'Dev%':<8} {'Calls':<6}")
    print("-" * 80)
    
    for r in results:
        print(f"b={r['start_b']:.6f}  {r['optimal_b']:.6f}  {r['optimal_c']:8.2f}  {r['above_pct']:6.1f}%  {r['below_pct']:6.1f}%  {r['deviation']:6.1f}%  {r['nfev']:4d}")
    
    # Check for patterns
    b_values = [r['optimal_b'] for r in results]
    c_values = [r['optimal_c'] for r in results]
    
    print(f"\nParameter Ranges:")
    print(f"  Growth rate (b): {min(b_values):.6f} to {max(b_values):.6f}")
    print(f"  Intercept (c): {min(c_values):.2f} to {max(c_values):.2f}")
    
    # Check if any achieved perfect 50/50
    perfect_results = [r for r in results if r['deviation'] < 0.1]
    print(f"\nPerfect 50/50 results: {len(perfect_results)} out of {len(results)}")
    
    if len(perfect_results) > 1:
        print("Multiple solutions achieve perfect 50/50 split!")
        for r in perfect_results:
            print(f"  b={r['optimal_b']:.6f}, c={r['optimal_c']:.2f}")

def test_brute_force_search(df, fixed_a=54.36):
    """Do a brute force search to verify the optimization results."""
    
    x = df['days'].values
    y = df[' value'].values
    
    print(f"\n" + "="*60)
    print("BRUTE FORCE VERIFICATION")
    print("="*60)
    
    # Test a grid of values around the "optimal" point
    b_range = np.linspace(0.00015, 0.00020, 11)  # Around 0.000172
    c_range = np.linspace(15, 25, 11)            # Around 20.66
    
    best_deviation = float('inf')
    best_b = None
    best_c = None
    
    print(f"Testing {len(b_range)} x {len(c_range)} = {len(b_range)*len(c_range)} combinations...")
    
    for b in b_range:
        for c in c_range:
            above, below, above_pct, below_pct, deviation = calculate_above_below_ratio(x, y, fixed_a, b, c)
            
            if deviation < best_deviation:
                best_deviation = deviation
                best_b = b
                best_c = c
    
    print(f"\nBrute force best result:")
    print(f"  b = {best_b:.6f}")
    print(f"  c = {best_c:.2f}")
    print(f"  Deviation = {best_deviation:.1f}%")
    
    # Check if this matches the optimization result
    if abs(best_b - 0.000172) < 1e-6 and abs(best_c - 20.66) < 1e-2:
        print("Brute force confirms optimization result!")
    else:
        print("Brute force gives DIFFERENT result than optimization!")
        print("This suggests the optimization may have issues.")

def main():
    print("Palladium 50/50 Model Deep Dive Analysis")
    print("=" * 60)
    
    # Load data
    df = load_palladium_data()
    if df is None:
        return
    
    # Test multiple starting points
    results = test_multiple_starting_points(df)
    
    # Analyze results
    analyze_results(results)
    
    # Brute force verification
    test_brute_force_search(df)
    
    print(f"\n" + "="*60)
    print("DEEP DIVE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
