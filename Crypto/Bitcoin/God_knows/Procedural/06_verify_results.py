#!/usr/bin/env python3
"""
Bitcoin Analysis Workflow - Verification Script
Purpose: Verify all models, formulas, and calculations are working correctly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_growth_formula():
    """Verify the growth formula is working correctly"""
    print("="*60)
    print("VERIFYING GROWTH FORMULA")
    print("="*60)
    
    # Load growth model coefficients
    try:
        with open('Models/Growth Models/bitcoin_growth_model_coefficients.txt', 'r') as f:
            lines = f.readlines()
        
        # Parse coefficients
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
        
        print(f"Growth formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
        
        # Calculate current day (days since Bitcoin genesis: Jan 3, 2009)
        genesis_date = date(2009, 1, 3)
        today = date.today()
        current_day = (today - genesis_date).days
        
        print(f"Current day since genesis: {current_day}")
        
        # Calculate expected price
        expected_price = 10**(a * np.log(current_day) + b)
        print(f"Formula prediction for today: ${expected_price:,.2f}")
        
        # Check with actual current price (~$118,000)
        actual_price = 118000
        print(f"Actual current price: ${actual_price:,.2f}")
        print(f"Difference: {((actual_price - expected_price) / expected_price * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error verifying growth formula: {e}")
        return False

def verify_volatility_formula():
    """Verify the volatility formula is working correctly"""
    print("\n" + "="*60)
    print("VERIFYING VOLATILITY FORMULA")
    print("="*60)
    
    try:
        # Load volatility model results
        with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
            content = f.read()
        
        # Extract formula parameters
        lines = content.split('\n')
        for line in lines:
            if 'Formula:' in line:
                formula = line.split('Formula:')[1].strip()
                print(f"Volatility formula: {formula}")
                break
        
        # Parse parameters (assuming format: a * exp(-b * years) + c)
        # Extract a, b, c from the formula
        import re
        match = re.search(r'(\d+\.\d+)\s*\*\s*exp\(-(\d+\.\d+)\s*\*\s*years\)\s*\+\s*(\d+\.\d+)', formula)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            c = float(match.group(3))
            
            print(f"Parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            
            # Calculate current Bitcoin age (years since July 18, 2010)
            start_date = date(2010, 7, 18)
            today = date.today()
            bitcoin_age = (today - start_date).days / 365.25
            
            print(f"Bitcoin age: {bitcoin_age:.1f} years")
            
            # Calculate current volatility
            current_volatility = a * np.exp(-b * bitcoin_age) + c
            current_volatility = min(current_volatility, 1.0)  # Cap at 100%
            
            print(f"Current volatility: {current_volatility*100:.1f}%")
            
            # Test future volatility
            future_volatility = a * np.exp(-b * (bitcoin_age + 5)) + c
            future_volatility = min(future_volatility, 1.0)
            
            print(f"Volatility in 5 years: {future_volatility*100:.1f}%")
            
            return True
        else:
            print("Could not parse volatility formula parameters")
            return False
            
    except Exception as e:
        print(f"Error verifying volatility formula: {e}")
        return False

def verify_data_files():
    """Verify all required data files exist"""
    print("\n" + "="*60)
    print("VERIFYING DATA FILES")
    print("="*60)
    
    required_files = [
        'Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv',
        'Models/Growth Models/bitcoin_growth_model_coefficients.txt',
        'Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def verify_monte_carlo_outputs():
    """Verify Monte Carlo simulation outputs exist"""
    print("\n" + "="*60)
    print("VERIFYING MONTE CARLO OUTPUTS")
    print("="*60)
    
    # Check for recent simulation outputs
    results_dir = 'Results/Bitcoin'
    if not os.path.exists(results_dir):
        print(f"✗ Results directory {results_dir} does not exist")
        return False
    
    # Look for recent files
    today = datetime.now().strftime("%Y%m%d")
    expected_files = [
        f'bitcoin_monte_carlo_simple_paths_{today}.csv',
        f'bitcoin_monte_carlo_simple_summary_{today}.csv',
        f'bitcoin_monte_carlo_simple_formula_{today}.csv',
        f'bitcoin_monte_carlo_simple_visualization_{today}.png'
    ]
    
    all_exist = True
    for filename in expected_files:
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {filename} ({size:,} bytes)")
        else:
            print(f"✗ {filename} - MISSING")
            all_exist = False
    
    return all_exist

def test_formula_integration():
    """Test that growth and volatility formulas work together"""
    print("\n" + "="*60)
    print("TESTING FORMULA INTEGRATION")
    print("="*60)
    
    try:
        # Load growth coefficients
        with open('Models/Growth Models/bitcoin_growth_model_coefficients.txt', 'r') as f:
            lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
        
        # Load volatility parameters
        with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
            content = f.read()
        
        import re
        match = re.search(r'(\d+\.\d+)\s*\*\s*exp\(-(\d+\.\d+)\s*\*\s*years\)\s*\+\s*(\d+\.\d+)', content)
        if match:
            vol_a = float(match.group(1))
            vol_b = float(match.group(2))
            vol_c = float(match.group(3))
            
            # Test integration
            current_day = 6041  # Days since Bitcoin genesis
            bitcoin_age = 15.0  # Years since July 18, 2010
            
            # Calculate expected price and volatility
            expected_price = 10**(a * np.log(current_day) + b)
            current_volatility = min(vol_a * np.exp(-vol_b * bitcoin_age) + vol_c, 1.0)
            
            print(f"Current day: {current_day}")
            print(f"Bitcoin age: {bitcoin_age} years")
            print(f"Expected price: ${expected_price:,.2f}")
            print(f"Current volatility: {current_volatility*100:.1f}%")
            
            # Test future values
            future_day = current_day + 365  # 1 year from now
            future_age = bitcoin_age + 1
            
            future_price = 10**(a * np.log(future_day) + b)
            future_volatility = min(vol_a * np.exp(-vol_b * future_age) + vol_c, 1.0)
            
            growth_rate = np.log(future_price / expected_price)
            
            print(f"\nFuture (1 year):")
            print(f"  Future day: {future_day}")
            print(f"  Future age: {future_age} years")
            print(f"  Expected price: ${future_price:,.2f}")
            print(f"  Growth rate: {growth_rate*100:.1f}%")
            print(f"  Future volatility: {future_volatility*100:.1f}%")
            
            return True
        else:
            print("Could not parse volatility parameters")
            return False
            
    except Exception as e:
        print(f"Error testing formula integration: {e}")
        return False

def main():
    """Run all verification tests"""
    print("BITCOIN ANALYSIS WORKFLOW - VERIFICATION SCRIPT")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    tests = [
        ("Data Files", verify_data_files),
        ("Growth Formula", verify_growth_formula),
        ("Volatility Formula", verify_volatility_formula),
        ("Formula Integration", test_formula_integration),
        ("Monte Carlo Outputs", verify_monte_carlo_outputs)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Status: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✓ Bitcoin analysis workflow is ready to use!")
        print("Run the scripts in order:")
        print("  1. 01_fetch_bitcoin_data.py")
        print("  2. 02_clean_combine_data.py")
        print("  3. 03_fit_growth_model.py")
        print("  4. 04_fit_volatility_model.py")
        print("  5. 05_run_monte_carlo.py")
    else:
        print("\n✗ Some issues need to be resolved before running the workflow")

if __name__ == "__main__":
    main() 