"""
MASTER SCRIPT - RUN ALL REGIME ANALYSES
=======================================
This script runs both the rolling regression analysis and trend+cycle model,
then generates a comprehensive findings report.

Usage:
    python run_regime_analysis.py
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture output."""
    print(f"\n{'='*70}")
    print(f"Running {script_name}...")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"\n[OK] {script_name} completed successfully")
            return True
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False

def main():
    """Run all regime analysis scripts."""
    print("="*70)
    print("BITCOIN REGIME SHIFT ANALYSIS - MASTER SCRIPT")
    print("="*70)
    print(f"\nStarting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run rolling regression analysis
    success1 = run_script("regime_analysis.py")
    
    # Run trend + cycle model
    success2 = run_script("trend_cycle_model.py")
    
    print("\n" + "="*70)
    if success1 and success2:
        print("ALL ANALYSES COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • bitcoin_regime_analysis.png (rolling regression visualization)")
        print("  • bitcoin_regime_findings.txt (rolling regression report)")
        print("  • bitcoin_trend_cycle_model.png (trend + cycle visualization)")
        print("  • bitcoin_valuation_report.txt (current valuation report)")
        print("\nYou can now share these findings with GPT for interpretation.")
    else:
        print("ANALYSIS COMPLETED WITH ERRORS")
        print("="*70)
        print("Check the output above for details.")

if __name__ == "__main__":
    main()

