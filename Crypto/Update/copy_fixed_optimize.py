"""
Copy the fixed optimize_intercept.py from Doge to all other currency folders.
Excludes Etherium and Bitcoin folders.
"""

import shutil
from pathlib import Path

# Get paths
script_dir = Path(__file__).parent
crypto_dir = script_dir.parent
source_file = crypto_dir / "Doge" / "optimize_intercept.py"

# Folders to exclude
exclude = {"Doge", "Etherium", "Bitcoin", "update"}

# Get all currency folders
currency_folders = [f for f in crypto_dir.iterdir() 
                    if f.is_dir() and f.name not in exclude]

print(f"Copying {source_file} to {len(currency_folders)} folders...")
print()

for folder in currency_folders:
    dest_file = folder / "optimize_intercept.py"
    if dest_file.exists():
        shutil.copy2(source_file, dest_file)
        print(f"[OK] Copied to {folder.name}/")
    else:
        print(f"[SKIP] Skipped {folder.name}/ (no optimize_intercept.py found)")

print()
print(f"[DONE] Updated {len(currency_folders)} files.")

