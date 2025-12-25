#!/usr/bin/env python3
"""
Bitcoin Analysis Workflow - Master Script
Purpose: Run the complete Bitcoin analysis workflow from start to finish
"""

import subprocess
import sys
import os
from datetime import datetime
import time

def run_script(script_name, description):
    """Run a script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"DESCRIPTION: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if successful
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✓ {script_name} completed successfully in {elapsed:.1f} seconds")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    """Run the complete Bitcoin analysis workflow"""
    print("BITCOIN ANALYSIS WORKFLOW - MASTER SCRIPT")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Define the workflow steps
    workflow_steps = [
        ("01_fetch_bitcoin_data.py", "Fetch latest Bitcoin price data from API"),
        ("02_clean_combine_data.py", "Clean and combine historical Bitcoin data"),
        ("03_fit_growth_model.py", "Fit logarithmic growth model to Bitcoin data"),
        ("04_fit_volatility_model.py", "Fit exponential decay volatility model"),
        ("05_run_monte_carlo.py", "Run Monte Carlo simulation with growth and volatility models"),
        ("06_verify_results.py", "Verify all models and calculations are working correctly")
    ]
    
    # Track results
    results = {}
    failed_steps = []
    
    # Run each step
    for script_name, description in workflow_steps:
        success = run_script(script_name, description)
        results[script_name] = success
        
        if not success:
            failed_steps.append(script_name)
            print(f"\n⚠️  Workflow stopped due to failure in {script_name}")
            print("You can fix the issue and restart from this step.")
            break
        
        # Small delay between steps
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_steps = len(workflow_steps)
    successful_steps = sum(results.values())
    
    print(f"Steps completed: {successful_steps}/{total_steps}")
    
    if failed_steps:
        print(f"Failed steps: {', '.join(failed_steps)}")
        print("\n✗ Workflow completed with errors")
        print("Check the output above for details on what went wrong.")
        return False
    else:
        print("\n✓ Workflow completed successfully!")
        print("\nGenerated files:")
        print("- Growth model coefficients: Models/Growth Models/bitcoin_growth_model_coefficients.txt")
        print("- Volatility model results: Models/Volatility Models/bitcoin_exponential_volatility_results_*.txt")
        print("- Monte Carlo results: Results/Bitcoin/bitcoin_monte_carlo_simple_*.csv")
        print("- Visualization: Results/Bitcoin/bitcoin_monte_carlo_simple_visualization_*.png")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 