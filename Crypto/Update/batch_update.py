"""
Batch update script for multiple currencies.
Runs update_prices.py and optimize_intercept.py for each currency specified in currencies_config.py.
Creates a consolidated summary file with currency name, growth rate, and current fair price.
"""

import os
import sys
import subprocess
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from currencies_config import CURRENCIES_TO_UPDATE, OUTPUT_FOLDER, CONSOLIDATED_OUTPUT_FILE
except ImportError:
    print("Error: currencies_config.py not found!")
    sys.exit(1)

# Get the project root (parent of Crypto/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CRYPTO_DIR = os.path.join(PROJECT_ROOT, "Crypto")

def find_script_in_folder(currency_folder, script_name):
    """Find update_prices.py or optimize_intercept.py in a currency folder.
    Checks common locations: root of folder, log_model/, log_model_v2/, etc."""
    # First check direct in folder
    direct_path = os.path.join(currency_folder, script_name)
    if os.path.exists(direct_path):
        return direct_path
    
    # Then check common subfolders
    common_subfolders = [
        "log_model",
        "log_model_v2",
        "exponential_v2",
        "point_selecting_method",
        "data_update",
    ]
    
    for subfolder in common_subfolders:
        path = os.path.join(currency_folder, subfolder, script_name)
        if os.path.exists(path):
            return path
    
    # If not found, do a recursive search (but limit depth to 2 levels)
    for root, dirs, files in os.walk(currency_folder):
        depth = root[len(currency_folder):].count(os.sep)
        if depth <= 2 and script_name in files:
            return os.path.join(root, script_name)
    
    return None

def get_currency_name_from_config(currency_folder):
    """Try to get CURRENCY_NAME_LOWER from config.py in the currency folder."""
    config_path = os.path.join(currency_folder, "config.py")
    if os.path.exists(config_path):
        try:
            # Read config file and extract CURRENCY_NAME_LOWER
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Try to find CURRENCY_NAME_LOWER = "..."
            match = re.search(r'CURRENCY_NAME_LOWER\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        except Exception:
            pass
    return None

def find_output_file(currency_folder, currency_name_lower):
    """Find the intercept_optimization.txt file for a currency.
    First tries to get the actual currency name from config.py, then searches for the file."""
    # Try to get the actual currency name from config.py
    actual_currency_name = get_currency_name_from_config(currency_folder)
    if actual_currency_name:
        currency_name_lower = actual_currency_name
    
    # Try direct in folder first
    direct_path = os.path.join(currency_folder, f"{currency_name_lower}_intercept_optimization.txt")
    if os.path.exists(direct_path):
        return direct_path
    
    # Check common subfolders
    common_subfolders = [
        "log_model",
        "log_model_v2",
        "exponential_v2",
        "point_selecting_method",
        "data_update",
    ]
    
    for subfolder in common_subfolders:
        path = os.path.join(currency_folder, subfolder, f"{currency_name_lower}_intercept_optimization.txt")
        if os.path.exists(path):
            return path
    
    # Recursive search for any *_intercept_optimization.txt file (limit depth to 2 levels)
    for root, dirs, files in os.walk(currency_folder):
        depth = root[len(currency_folder):].count(os.sep)
        if depth <= 2:
            for filename in files:
                if filename.endswith("_intercept_optimization.txt"):
                    return os.path.join(root, filename)
    
    return None

def extract_data_from_output(output_file):
    """Extract growth rate and current fair price from optimize_intercept output file.
    Handles different model types (logarithmic, exponential, parametric, power law)."""
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract Annual Growth Rate (try multiple patterns)
        annual_growth_rate = None
        growth_patterns = [
            r'Annual Growth Rate:\s*([+-]?\d+\.?\d*)%',
            r'Growth Rate:\s*([+-]?\d+\.?\d*)%',
            r'Annual Growth:\s*([+-]?\d+\.?\d*)%',
        ]
        for pattern in growth_patterns:
            growth_match = re.search(pattern, content)
            if growth_match:
                annual_growth_rate = float(growth_match.group(1))
                break
        
        # Extract Model Fair Price (try multiple patterns)
        current_fair_price = None
        price_patterns = [
            r'Model Fair Price:\s*\$?([\d,]+\.?\d*)',
            r'Fair Price:\s*\$?([\d,]+\.?\d*)',
            r'Current Fair Price:\s*\$?([\d,]+\.?\d*)',
        ]
        for pattern in price_patterns:
            fair_price_match = re.search(pattern, content)
            if fair_price_match:
                fair_price_str = fair_price_match.group(1).replace(',', '')
                try:
                    current_fair_price = float(fair_price_str)
                    break
                except ValueError:
                    continue
        
        return annual_growth_rate, current_fair_price
    except Exception as e:
        print(f"  Error extracting data: {e}")
        return None, None

def run_script(script_path, working_dir):
    """Run a Python script in a specific directory.
    Handles Unicode encoding issues on Windows."""
    try:
        # Set environment to use UTF-8 encoding to avoid Windows cp1252 issues
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=working_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out after 5 minutes"
    except Exception as e:
        return False, "", str(e)

def process_currency(currency_name):
    """Process a single currency: update prices and optimize intercept.
    Handles nested paths like 'Ethereum/exponential_v2'."""
    print(f"\n{'='*60}")
    print(f"Processing: {currency_name}")
    print(f"{'='*60}")
    
    # Handle nested paths (e.g., "Ethereum/exponential_v2")
    # os.path.join handles forward slashes correctly on all platforms
    currency_folder = os.path.join(CRYPTO_DIR, currency_name)
    # Normalize the path (converts forward slashes to backslashes on Windows, etc.)
    currency_folder = os.path.normpath(currency_folder)
    
    if not os.path.exists(currency_folder):
        print(f"  [FAIL] Folder not found: {currency_folder}")
        # Try to suggest correct folder name
        if "/" in currency_name or "\\" in currency_name:
            base_name = currency_name.split("/")[0].split("\\")[0]
            # Check for similar folder names
            if os.path.exists(CRYPTO_DIR):
                similar_folders = [f for f in os.listdir(CRYPTO_DIR) 
                                 if os.path.isdir(os.path.join(CRYPTO_DIR, f)) 
                                 and base_name.lower() in f.lower()]
                if similar_folders:
                    print(f"  [INFO] Did you mean one of these? {similar_folders[:3]}")
        return None
    
    # Find update_prices.py
    update_script = find_script_in_folder(currency_folder, "update_prices.py")
    if not update_script:
        print(f"  [FAIL] update_prices.py not found in {currency_folder}")
        return None
    
    # Find optimize_intercept.py
    optimize_script = find_script_in_folder(currency_folder, "optimize_intercept.py")
    if not optimize_script:
        print(f"  [FAIL] optimize_intercept.py not found in {currency_folder}")
        return None
    
    # Run update_prices.py
    print(f"  [1/2] Running update_prices.py...")
    script_dir = os.path.dirname(update_script)
    success, stdout, stderr = run_script(update_script, script_dir)
    if not success:
        # Clean up stderr to remove encoding issues
        stderr_clean = stderr.encode('ascii', 'replace').decode('ascii') if stderr else ""
        print(f"  [WARN] update_prices.py failed (continuing anyway)")
        if stderr_clean and len(stderr_clean) < 500:  # Only show if reasonable length
            print(f"         Error: {stderr_clean[:200]}...")
        # Continue anyway - maybe data is already up to date
    
    # Run optimize_intercept.py
    print(f"  [2/2] Running optimize_intercept.py...")
    script_dir = os.path.dirname(optimize_script)
    success, stdout, stderr = run_script(optimize_script, script_dir)
    if not success:
        # Clean up stderr to remove encoding issues
        stderr_clean = stderr.encode('ascii', 'replace').decode('ascii') if stderr else ""
        print(f"  [FAIL] optimize_intercept.py failed")
        if stderr_clean and len(stderr_clean) < 500:
            print(f"         Error: {stderr_clean[:200]}...")
        # Try to find existing output file anyway
        print(f"  [INFO] Attempting to use existing output file if available...")
    
    # Find and parse output file
    currency_name_lower = currency_name.lower()
    output_file = find_output_file(currency_folder, currency_name_lower)
    
    if not output_file:
        print(f"  [FAIL] Could not find output file for {currency_name}")
        return None
    
    # Extract data
    annual_growth_rate, current_fair_price = extract_data_from_output(output_file)
    
    if annual_growth_rate is None or current_fair_price is None:
        print(f"  [WARN] Could not extract all data from output file")
        return {
            'Currency': currency_name,
            'Annual Growth Rate (%)': annual_growth_rate if annual_growth_rate is not None else 'N/A',
            'Current Fair Price ($)': current_fair_price if current_fair_price is not None else 'N/A'
        }
    
    print(f"  [OK] Growth Rate: {annual_growth_rate:+.2f}%, Fair Price: ${current_fair_price:,.2f}")
    
    return {
        'Currency': currency_name,
        'Annual Growth Rate (%)': annual_growth_rate,
        'Current Fair Price ($)': current_fair_price
    }

def main():
    """Main function to process all currencies."""
    print("="*60)
    print("BATCH CURRENCY UPDATE")
    print("="*60)
    print(f"Processing {len(CURRENCIES_TO_UPDATE)} currencies...")
    
    if not CURRENCIES_TO_UPDATE:
        print("\n[WARN] No currencies specified in currencies_config.py!")
        print("Please add currency folder names to CURRENCIES_TO_UPDATE list.")
        return
    
    results = []
    
    for currency_name in CURRENCIES_TO_UPDATE:
        result = process_currency(currency_name)
        if result:
            results.append(result)
    
    # Create consolidated summary
    if results:
        print(f"\n{'='*60}")
        print("CREATING CONSOLIDATED SUMMARY")
        print(f"{'='*60}")
        
        df = pd.DataFrame(results)
        
        # Sort by growth rate (descending)
        if 'Annual Growth Rate (%)' in df.columns:
            # Handle N/A values
            df_sorted = df.copy()
            df_sorted['Sort_Key'] = df_sorted['Annual Growth Rate (%)'].apply(
                lambda x: float('-inf') if x == 'N/A' else x
            )
            df_sorted = df_sorted.sort_values('Sort_Key', ascending=False)
            df_sorted = df_sorted.drop('Sort_Key', axis=1)
            df = df_sorted
        
        # Save to CSV
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_path = os.path.join(OUTPUT_FOLDER, CONSOLIDATED_OUTPUT_FILE)
        df.to_csv(output_path, index=False)
        
        print(f"\n[OK] Consolidated summary saved to: {output_path}")
        print(f"  Total currencies processed: {len(results)}")
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("\n[FAIL] No currencies were successfully processed!")
    
    print(f"\n{'='*60}")
    print("BATCH UPDATE COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

