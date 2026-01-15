"""
Script to run update_prices.py in all configured folders.
Reads folder paths from config.py and executes update_prices.py if it exists.
"""

import os
import sys
import subprocess
from pathlib import Path

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


def run_update_prices(folder_path):
    """Run update_prices.py in the specified folder if it exists."""
    full_path = CRYPTO_DIR / folder_path
    update_script = full_path / "update_prices.py"
    
    if not full_path.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return False
    
    if not update_script.exists():
        print(f"Info: No update_prices.py found in {folder_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running update_prices.py in: {folder_path}")
    print(f"{'='*60}")
    
    try:
        # Change to the folder directory and run the script
        result = subprocess.run(
            [sys.executable, str(update_script)],
            cwd=str(full_path),
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully completed: {folder_path}")
            return True
        else:
            print(f"✗ Failed with exit code {result.returncode}: {folder_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error running update_prices.py in {folder_path}: {e}")
        return False


def main():
    """Main function to process all folders from config."""
    if not FOLDERS:
        print("Warning: No folders configured in config.py")
        return
    
    print(f"Found {len(FOLDERS)} folder(s) to check")
    print(f"Base directory: {CRYPTO_DIR}")
    
    results = []
    for folder in FOLDERS:
        success = run_update_prices(folder)
        results.append((folder, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Successfully updated: {successful}/{total}")
    
    if successful < total:
        print("\nFailed or skipped folders:")
        for folder, success in results:
            if not success:
                print(f"  - {folder}")


if __name__ == "__main__":
    main()

