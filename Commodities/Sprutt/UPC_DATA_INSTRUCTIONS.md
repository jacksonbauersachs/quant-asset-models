# UPC Historical Data Instructions

## Problem
UPC (Uranium Participation Corporation) ticker `U.TO` is delisted and no longer available via Yahoo Finance.

## Solution
You need to manually source UPC historical data and save it as a CSV file.

## Steps

1. **Find UPC Historical Data Sources:**
   - TSX (Toronto Stock Exchange) historical data
   - Financial data providers (Bloomberg, Refinitiv, etc.)
   - Historical financial databases
   - Company investor relations archives

2. **Format Requirements:**
   - Save as CSV file named: `UPC_historical_data.csv`
   - Place it in the `Commodities/Sprutt/` folder
   - Required columns: `Date`, `Price`
   - Date format: `YYYY-MM-DD` (e.g., `2005-06-15`)
   - Price: Closing price in USD (or convert if needed)

3. **Date Range:**
   - Start: UPC inception date (around 2005)
   - End: `2021-07-16` (last trading day before reorganization)

4. **Example CSV Format:**
   ```csv
   Date,Price
   2005-06-15,5.25
   2005-06-16,5.30
   ...
   2021-07-16,8.50
   ```

5. **After Adding UPC Data:**
   - Run `fetch_combined_history.py` again
   - It will automatically combine UPC + SPUT data
   - Output: `SPUT_historical_data.csv` with complete history

## Alternative: Use SPUT Data Only
If you can't find UPC data, the script will work with SPUT data only (starting July 2021).

