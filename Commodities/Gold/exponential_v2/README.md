# Currency Analysis Template System

This is a streamlined template system for creating cryptocurrency/currency analysis workflows. It includes scripts for:
1. **Updating price data** from Yahoo Finance (using yfinance library)
2. **Interactive rainbow chart** with adjustable parameters
3. **Intercept optimization** with fixed slope

## 📁 Files in This Template

- `config.py` - **THE ONLY FILE YOU NEED TO EDIT** - Contains all configuration
- `update_prices.py` - Script to update historical price data
- `interactive_rainbow.py` - Interactive rainbow chart visualization
- `optimize_intercept.py` - Optimize intercept for fixed slope
- `README.md` - This file

## 🚀 Quick Start Guide

### Step 1: Copy the Template

Copy the entire `clone/` folder to your currency's location. For example:
- For Dogecoin: Copy to `Crypto/Dogecoin/` (or wherever you want)
- For Bitcoin: Copy to `Crypto/Bitcoin/` (if not already set up)

### Step 2: Edit `config.py`

Open `config.py` and change **ALL** the values marked with `# CHANGE THIS`:

```python
# Currency Information
CURRENCY_NAME = "Dogecoin"  # Change this
CURRENCY_NAME_LOWER = "dogecoin"  # Change this
YAHOO_TICKER = "DOGE-USD"  # Change this (Yahoo Finance ticker format)

# Dates
# GENESIS_DATE is OPTIONAL - leave as None to auto-detect from data
GENESIS_DATE = None  # Or set to a specific date like "2013-12-06" (YYYY-MM-DD format)
START_DATE = "2014-01-01"  # Change this (when to start analysis)

# File Paths
DATA_FOLDER = "Crypto/Dogecoin/Data"  # Change this
HISTORICAL_DATA_FILE = "Dogecoin Historical Data.csv"  # Change this
OUTPUT_FOLDER = "Crypto/Dogecoin/Models/Growth"  # Change this

# Model Parameters
FIXED_SLOPE = 0.717579  # Change this AFTER running interactive_rainbow.py
```

### Step 3: Run the Update Script (No Initial File Needed!)

**Good news!** You don't need to create an initial CSV file. The script will automatically fetch all available historical data from Yahoo Finance on the first run.

Just make sure the `DATA_FOLDER` path exists (or the script will create it automatically).

### Step 4: Install Required Package

Make sure you have the `yfinance` package installed:
```bash
pip install yfinance
```

**Note:** Yahoo Finance is free and doesn't require an API key!

### Step 5: Run the Scripts

Run the scripts in this order:

1. **Update Price Data:**
   ```bash
   python update_prices.py
   ```
   This will fetch the latest price data and append it to your CSV file.

2. **Create Interactive Rainbow Chart:**
   ```bash
   python interactive_rainbow.py
   ```
   - This opens an interactive chart with a slider
   - **Note the optimal slope value** shown in the console
   - Use the slider to find the best fit
   - Close the chart when done

3. **Update FIXED_SLOPE in config.py:**
   - Open `config.py`
   - Set `FIXED_SLOPE` to the optimal slope value from step 2

4. **Optimize Intercept:**
   ```bash
   python optimize_intercept.py
   ```
   - This creates the optimized model
   - Generates a chart and text file with projections

## 📋 Configuration Reference

### Required Configuration Values

| Variable | Description | Example |
|----------|-------------|---------|
| `CURRENCY_NAME` | Display name (capitalized) | `"Dogecoin"` |
| `CURRENCY_NAME_LOWER` | Lowercase for file names | `"dogecoin"` |
| `YAHOO_TICKER` | Yahoo Finance ticker (CURRENCY-USD) | `"DOGE-USD"` |
| `GENESIS_DATE` | Launch date (YYYY-MM-DD) - **OPTIONAL** | `None` (auto-detect) or `"2013-12-06"` |
| `START_DATE` | Analysis start date (YYYY-MM-DD) | `"2014-01-01"` |
| `DATA_FOLDER` | Path to data folder | `"Crypto/Dogecoin/Data"` |
| `HISTORICAL_DATA_FILE` | CSV filename | `"Dogecoin Historical Data.csv"` |
| `OUTPUT_FOLDER` | Path to output folder | `"Crypto/Dogecoin/Models/Growth"` |
| `FIXED_SLOPE` | Slope for optimization | `0.717579` (set after step 2) |

### Optional Configuration

- `SLOPE_MIN` / `SLOPE_MAX` - Bounds for slope optimization (default: 0.5 to 5.0)
- `INTERCEPT_MIN` / `INTERCEPT_MAX` - Bounds for intercept optimization (default: -30 to 0)

## 📊 Output Files

After running the scripts, you'll get:

1. **Updated CSV file** - In your `DATA_FOLDER` with latest prices
2. **Interactive Rainbow Chart** - `{currency}_interactive_rainbow.png` in `OUTPUT_FOLDER`
3. **Optimization Chart** - `{currency}_intercept_optimization.png` in `OUTPUT_FOLDER`
4. **Optimization Results** - `{currency}_intercept_optimization.txt` in `OUTPUT_FOLDER`

## 🔍 Example: Setting Up Dogecoin

Here's a complete example for Dogecoin:

1. **Copy template:**
   ```bash
   cp -r clone/ Crypto/Dogecoin/
   cd Crypto/Dogecoin/
   ```

2. **Edit `config.py`:**
   ```python
   CURRENCY_NAME = "Dogecoin"
   CURRENCY_NAME_LOWER = "dogecoin"
   YAHOO_TICKER = "DOGE-USD"  # Yahoo Finance ticker
   GENESIS_DATE = None  # Auto-detect from data (or set to "2013-12-06" for specific date)
   START_DATE = "2014-01-01"  # Start analysis from here
   DATA_FOLDER = "Crypto/Dogecoin/Data"
   HISTORICAL_DATA_FILE = "Dogecoin Historical Data.csv"
   OUTPUT_FOLDER = "Crypto/Dogecoin/Models/Growth"
   FIXED_SLOPE = 0.717579  # Will update after running interactive_rainbow.py
   ```

3. **Create initial CSV:**
   - Create `Crypto/Dogecoin/Data/Dogecoin Historical Data.csv`
   - Add at least one row with proper format

4. **Run scripts:**
   ```bash
   python update_prices.py
   python interactive_rainbow.py  # Note the optimal slope
   # Edit config.py with optimal slope
   python optimize_intercept.py
   ```

## ⚠️ Important Notes

1. **Date Format:** The CSV must use `M/D/YYYY` format (e.g., `03/10/2016`)
2. **Rate Limits:** Yahoo Finance may have rate limits. If you encounter issues, wait a moment and try again
3. **First Run:** You need at least one row of data in your CSV before running `update_prices.py`
4. **Slope Value:** Always run `interactive_rainbow.py` first to find the optimal slope before running `optimize_intercept.py`

## 🐛 Troubleshooting

### "config.py not found"
- Make sure you're running the script from the folder containing `config.py`
- Or adjust your Python path

### "Data file not found"
- Check that `DATA_FOLDER` and `HISTORICAL_DATA_FILE` in `config.py` are correct
- Make sure the CSV file exists

### "No data found for TICKER"
- Check that `YAHOO_TICKER` is correct (format: "CURRENCY-USD")
- Common tickers: "BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD"
- Verify the ticker exists on Yahoo Finance website
- Make sure `yfinance` package is installed: `pip install yfinance`

## 📝 Customization

All scripts read from `config.py`, so you can:
- Change optimization bounds
- Adjust date ranges
- Modify output file names (via `get_output_files()` function)
- Add custom parameters as needed

## 🎯 Workflow Summary

```
1. Copy clone/ folder → Your currency folder
2. Edit config.py with your currency info
3. Create initial CSV file with proper format
4. Run: python update_prices.py
5. Run: python interactive_rainbow.py (note optimal slope)
6. Update FIXED_SLOPE in config.py
7. Run: python optimize_intercept.py
8. Done! Check OUTPUT_FOLDER for results
```

## 💡 Tips

- **Start with a small date range** in your initial CSV to test
- **Use the interactive chart** to understand the model behavior
- **Save your optimal slope** - you'll need it for future runs
- **Keep your CSV updated** - run `update_prices.py` regularly

---

**Happy analyzing! 🚀**

