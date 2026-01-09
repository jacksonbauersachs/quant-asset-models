# Batch Currency Update System

This folder contains scripts to batch update multiple currencies and generate a consolidated summary.

## Files

- `currencies_config.py` - Configuration file where you list currencies to update
- `batch_update.py` - Main script that processes all currencies
- `currency_summary.csv` - Output file with consolidated results (generated after running)

## Usage

1. **Edit `currencies_config.py`**:
   - Add currency folder names to the `CURRENCIES_TO_UPDATE` list
   - Use the exact folder names from `Crypto/` directory (e.g., "Bitcoin", "AAVE", "Chainlink")

2. **Run the batch update**:
   ```bash
   python batch_update.py
   ```

3. **Check the results**:
   - Each currency's `update_prices.py` and `optimize_intercept.py` will run
   - A consolidated `currency_summary.csv` will be created in this folder
   - The summary contains: Currency Name, Annual Growth Rate (%), Current Fair Price ($)

## Example Configuration

```python
CURRENCIES_TO_UPDATE = [
    "Bitcoin",
    "AAVE",
    "Chainlink",
    "Doge",
]
```

## Notes

- The script automatically finds `update_prices.py` and `optimize_intercept.py` in common locations:
  - Directly in the currency folder
  - In `log_model/`, `log_model_v2/`, `exponential_v2/`, `point_selecting_method/`, etc.
- If a currency fails, it will be skipped and the script continues with others
- The consolidated summary is sorted by growth rate (highest first)

