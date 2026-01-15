"""
Script to run optimize_intercept.py in all configured folders.
Reads folder paths from config.py and executes optimize_intercept.py if it exists.
Generates a summary file with Model Fair Prices after completion.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
CRYPTO_DIR = SCRIPT_DIR.parent

# Import config
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import FOLDERS
except ImportError:
    print("Error: Could not import config.py")
    sys.exit(1)


def run_optimize_intercept(folder_path):
    """Run optimize_intercept.py in the specified folder if it exists."""
    full_path = CRYPTO_DIR / folder_path
    optimize_script = full_path / "optimize_intercept.py"
    
    if not full_path.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return False
    
    if not optimize_script.exists():
        print(f"Info: No optimize_intercept.py found in {folder_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running optimize_intercept.py in: {folder_path}")
    print(f"{'='*60}")
    
    try:
        # Change to the folder directory and run the script
        result = subprocess.run(
            [sys.executable, str(optimize_script)],
            cwd=str(full_path),
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] Successfully completed: {folder_path}")
            return True
        else:
            print(f"[FAILED] Failed with exit code {result.returncode}: {folder_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running optimize_intercept.py in {folder_path}: {e}")
        return False


def find_optimization_file(folder_path):
    """Find the intercept_optimization.txt file, even if nested."""
    full_path = CRYPTO_DIR / folder_path
    pattern = "*intercept_optimization.txt"
    matches = list(full_path.rglob(pattern))
    if matches:
        return matches[0]
    return None


def extract_fair_price(file_path):
    """Extract Model Fair Price from the optimization file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for "Model Fair Price: $X.XX" pattern
        match = re.search(r'Model Fair Price:\s*\$?([\d,]+\.?\d*)', content)
        if match:
            price_str = match.group(1).replace(',', '')
            return float(price_str)
        
        match = re.search(r'Model Fair Price:\s*([\d,]+\.?\d*)', content)
        if match:
            price_str = match.group(1).replace(',', '')
            return float(price_str)
            
    except Exception as e:
        pass
    
    return None


def get_currency_name(folder_path):
    """Get currency name from folder path."""
    return folder_path.split('/')[-1].split('\\')[-1]


def generate_summary(results):
    """Generate summary file with all Model Fair Prices."""
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY")
    print(f"{'='*60}")
    
    fair_prices = []
    
    for folder, success in results:
        if success:
            opt_file = find_optimization_file(folder)
            if opt_file:
                fair_price = extract_fair_price(opt_file)
                currency_name = get_currency_name(folder)
                
                if fair_price is not None:
                    fair_prices.append({
                        'currency': currency_name,
                        'folder': folder,
                        'fair_price': fair_price
                    })
    
    # Sort by fair price (descending)
    fair_prices.sort(key=lambda x: x['fair_price'], reverse=True)
    
    # Generate summary file
    summary_file = SCRIPT_DIR / "model_fair_prices_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MODEL FAIR PRICES SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Currencies: {len(fair_prices)}\n")
        f.write("\n")
        f.write(f"{'Currency':<20} {'Model Fair Price':>20}\n")
        f.write("-" * 60 + "\n")
        
        for item in fair_prices:
            f.write(f"{item['currency']:<20} ${item['fair_price']:>18,.2f}\n")
        
        f.write("\n")
        f.write("DETAILS\n")
        f.write("-" * 60 + "\n")
        for item in fair_prices:
            f.write(f"\n{item['currency']} ({item['folder']}):\n")
            f.write(f"  Model Fair Price: ${item['fair_price']:,.2f}\n")
    
    print(f"Summary saved to: {summary_file}")
    print(f"Found {len(fair_prices)} currencies with fair prices")


def main():
    """Main function to process all folders from config."""
    if not FOLDERS:
        print("Warning: No folders configured in config.py")
        return
    
    print(f"Found {len(FOLDERS)} folder(s) to check")
    print(f"Base directory: {CRYPTO_DIR}")
    
    results = []
    for folder in FOLDERS:
        success = run_optimize_intercept(folder)
        results.append((folder, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Successfully optimized: {successful}/{total}")
    
    if successful < total:
        print("\nFailed or skipped folders:")
        for folder, success in results:
            if not success:
                print(f"  - {folder}")
    
    # Generate fair prices summary
    if successful > 0:
        generate_summary(results)


if __name__ == "__main__":
    main()

