"""
Generate a CSV summary file with Model Fair Prices for all configured currencies.
Searches recursively for intercept_optimization.txt files (handles nested paths).
"""

import re
import sys
import csv
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


def find_optimization_file(folder_path):
    """Find the intercept_optimization.txt file, even if nested (e.g., Crypto/Avalanche/Crypto/Avalanche/)."""
    full_path = CRYPTO_DIR / folder_path
    
    # Search recursively for the file
    pattern = "*intercept_optimization.txt"
    matches = list(full_path.rglob(pattern))
    
    if matches:
        # Return the first match (or most recent if multiple)
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
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
        
        # Alternative pattern if format is different
        match = re.search(r'Model Fair Price:\s*([\d,]+\.?\d*)', content)
        if match:
            price_str = match.group(1).replace(',', '')
            return float(price_str)
            
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
    
    return None


def extract_current_price(file_path):
    """Extract Current Price from the optimization file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for "Current Price: $X.XX" pattern
        match = re.search(r'Current Price:\s*\$?([\d,]+\.?\d*)', content)
        if match:
            price_str = match.group(1).replace(',', '')
            return float(price_str)
            
    except Exception:
        pass
    
    return None


def extract_undervaluation(file_path):
    """Extract Undervaluation percentage from the optimization file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for "Undervaluation: +/-X.XX%" pattern
        match = re.search(r'Undervaluation:\s*([+-]?[\d,]+\.?\d*)%', content)
        if match:
            val_str = match.group(1).replace(',', '')
            return float(val_str)
            
    except Exception:
        pass
    
    return None


def get_currency_name(folder_path):
    """Get currency name from folder path."""
    return folder_path.split('/')[-1].split('\\')[-1]


def format_price(price):
    """Format price with appropriate significant digits based on magnitude."""
    if price is None or price == '':
        return ''
    
    if price == 0:
        return '$0.00'
    
    # For very small numbers (< 0.0001), use scientific notation or many decimals
    if price < 0.0001:
        # Use up to 8 significant digits
        return f"${price:.8f}".rstrip('0').rstrip('.')
    # For small numbers (< 1), use 6 decimal places
    elif price < 1:
        return f"${price:.6f}".rstrip('0').rstrip('.')
    # For numbers >= 1, use 2 decimal places with commas
    else:
        return f"${price:,.2f}"


def generate_summary():
    """Generate CSV summary file with all Model Fair Prices."""
    
    if not FOLDERS:
        print("Warning: No folders configured in config.py")
        return
    
    print("Generating summary of Model Fair Prices...")
    print()
    
    results = []
    
    for folder in FOLDERS:
        opt_file = find_optimization_file(folder)
        if opt_file:
            fair_price = extract_fair_price(opt_file)
            current_price = extract_current_price(opt_file)
            undervaluation = extract_undervaluation(opt_file)
            currency_name = get_currency_name(folder)
            
            if fair_price is not None:
                results.append({
                    'currency': currency_name,
                    'folder': folder,
                    'current_price': current_price if current_price else '',
                    'model_fair_price': fair_price,
                    'undervaluation_pct': undervaluation if undervaluation else '',
                    'file_path': str(opt_file)
                })
                print(f"[OK] {currency_name}: Current={format_price(current_price) if current_price else 'N/A'}, Fair={format_price(fair_price)}, Undervalued={undervaluation or 'N/A':+.2f}%")
            else:
                print(f"[WARN] {currency_name}: Could not extract fair price")
        else:
            currency_name = get_currency_name(folder)
            print(f"[SKIP] {currency_name}: No optimization file found")
    
    # Sort by fair price (descending)
    results.sort(key=lambda x: x['model_fair_price'], reverse=True)
    
    # Generate CSV file
    summary_file = SCRIPT_DIR / "model_fair_prices_summary.csv"
    
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Currency',
            'Current Price',
            'Model Fair Price',
            'Undervaluation %',
            'Folder',
            'File Path'
        ])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result['currency'],
                format_price(result['current_price']) if result['current_price'] else '',
                format_price(result['model_fair_price']),
                f"{result['undervaluation_pct']:+.2f}%" if result['undervaluation_pct'] else '',
                result['folder'],
                result['file_path']
            ])
    
    print()
    print(f"[DONE] Summary saved to: {summary_file}")
    print(f"       Found {len(results)} currencies with fair prices")


if __name__ == "__main__":
    generate_summary()
