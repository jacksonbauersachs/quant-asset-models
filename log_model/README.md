# Logarithmic Model Template System

This is a streamlined template system for creating cryptocurrency/currency analysis workflows using a **LOGARITHMIC GROWTH MODEL**. This model assumes growth slows over time (unlike exponential models where growth accelerates).

## 📊 Model Type: Logarithmic Growth

**Formula:** `price = 10^(slope × ln(days_since_genesis) + intercept)`

**Key Characteristics:**
- Growth rate **slows over time** (decelerating growth)
- Better for assets that mature and stabilize
- Uses **slope** and **intercept** parameters (not growth rate)
- R² calculated in **log space** (standard for logarithmic models)

## 📁 Files in This Template

- `config.py` - **THE ONLY FILE YOU NEED TO EDIT** - Contains all configuration
- `update_prices.py` - Script to update historical price data from Yahoo Finance
- `interactive_rainbow.py` - Interactive rainbow chart visualization with logarithmic model
- `optimize_intercept.py` - Optimize intercept for fixed slope
- `README.md` - This file

## 🚀 Quick Start Guide

### Step 1: Copy the Template

Copy the entire `log_model/` folder to your currency's location. For example:
- For Chainlink: Copy to `Crypto/Chainlink/` (or wherever you want)
- For another crypto: Copy to `Crypto/YourCoin/`

### Step 2: Edit `config.py`

Open `config.py` and change **ALL** the values:

```python
# Currency Information
CURRENCY_NAME = "Chainlink"  # Change this
CURRENCY_NAME_LOWER = "chainlink"  # Change this
YAHOO_TICKER = "LINK-USD"  # Change this (Yahoo Finance ticker format)

# Dates
GENESIS_DATE = None  # Leave as None to auto-detect, or set like "2017-09-20"
START_DATE = ""  # Leave as "" to auto-detect, or set like "2022-05-12"

# File Paths
DATA_FOLDER = "Crypto/Chainlink"  # Change this
HISTORICAL_DATA_FILE = "Chainlink_historical_data.csv"  # Change this
OUTPUT_FOLDER = "Crypto/Chainlink"  # Change this

# Model Parameters (Logarithmic Model)
FIXED_GROWTH_RATE = None  # This is used as FIXED_SLOPE - set after running interactive_rainbow.py
# Example: 2.5 for a slope of 2.5
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
   - This opens an interactive chart with a slider to adjust the **slope**
   - **Note the optimal slope value** shown in the console
   - The intercept is automatically calculated to anchor the model at the first data point
   - Use the slider to find the best fit
   - Close the chart when done

3. **Update FIXED_GROWTH_RATE in config.py:**
   - Open `config.py`
   - Set `FIXED_GROWTH_RATE` to the optimal slope value from step 2
   - Example: If optimal slope is `2.536`, set `FIXED_GROWTH_RATE = 2.536`

4. **Optimize Intercept:**
   ```bash
   python optimize_intercept.py
   ```
   - This creates the optimized model with the fixed slope
   - Optimizes the intercept for best overall fit (not anchored to first point)
   - Generates a chart and text file with projections

## 📋 Configuration Reference

### Required Configuration Values

| Variable | Description | Example |
|----------|-------------|---------|
| `CURRENCY_NAME` | Display name (capitalized) | `"Chainlink"` |
| `CURRENCY_NAME_LOWER` | Lowercase for file names | `"chainlink"` |
| `YAHOO_TICKER` | Yahoo Finance ticker (CURRENCY-USD) | `"LINK-USD"` |
| `GENESIS_DATE` | Launch date (YYYY-MM-DD) - **OPTIONAL** | `None` (auto-detect) or `"2017-09-20"` |
| `START_DATE` | Analysis start date (YYYY-MM-DD) | `""` (auto-detect) or `"2022-05-12"` |
| `DATA_FOLDER` | Path to data folder | `"Crypto/Chainlink"` |
| `HISTORICAL_DATA_FILE` | CSV filename | `"Chainlink_historical_data.csv"` |
| `OUTPUT_FOLDER` | Path to output folder | `"Crypto/Chainlink"` |
| `FIXED_GROWTH_RATE` | Slope for optimization (set after step 2) | `2.536` (set after running interactive_rainbow.py) |

## 📊 Output Files

After running the scripts, you'll get:

1. **Updated CSV file** - In your `DATA_FOLDER` with latest prices
2. **Interactive Rainbow Chart** - `{currency}_interactive_rainbow.png` in `OUTPUT_FOLDER`
3. **Optimization Chart** - `{currency}_intercept_optimization.png` in `OUTPUT_FOLDER`
4. **Optimization Results** - `{currency}_intercept_optimization.txt` in `OUTPUT_FOLDER`

## 🔍 Example: Setting Up Chainlink

Here's a complete example for Chainlink:

1. **Copy template:**
   ```bash
   cp -r log_model/ Crypto/Chainlink/
   cd Crypto/Chainlink/
   ```

2. **Edit `config.py`:**
   ```python
   CURRENCY_NAME = "Chainlink"
   CURRENCY_NAME_LOWER = "chainlink"
   YAHOO_TICKER = "LINK-USD"
   GENESIS_DATE = None  # Auto-detect from data
   START_DATE = ""  # Auto-detect from data
   DATA_FOLDER = "Crypto/Chainlink"
   HISTORICAL_DATA_FILE = "Chainlink_historical_data.csv"
   OUTPUT_FOLDER = "Crypto/Chainlink"
   FIXED_GROWTH_RATE = None  # Will update after running interactive_rainbow.py
   ```

3. **Run scripts:**
   ```bash
   python update_prices.py
   python interactive_rainbow.py  # Note the optimal slope (e.g., 2.536)
   # Edit config.py: FIXED_GROWTH_RATE = 2.536
   python optimize_intercept.py
   ```

## 🔄 Difference from Exponential Model (`clone/` folder)

| Feature | Exponential Model (`clone/`) | Logarithmic Model (`log_model/`) |
|---------|------------------------------|----------------------------------|
| **Formula** | `price = first_price × e^(growth_rate × (days - first_days) + intercept)` | `price = 10^(slope × ln(days) + intercept)` |
| **Growth Behavior** | Growth accelerates over time | Growth slows over time |
| **Parameters** | `growth_rate` (per day) and `intercept` | `slope` and `intercept` |
| **R² Calculation** | Normal space | Log space |
| **Best For** | Assets with accelerating growth | Assets that mature/stabilize |
| **Config Parameter** | `FIXED_GROWTH_RATE` (daily growth rate) | `FIXED_GROWTH_RATE` (used as slope) |

## ⚠️ Important Notes

1. **Date Format:** The CSV can use `M/D/YYYY` or `YYYY-MM-DD` format - both are supported
2. **Rate Limits:** Yahoo Finance may have rate limits. If you encounter issues, wait a moment and try again
3. **First Run:** The script automatically fetches all historical data if no CSV exists
4. **Slope Value:** Always run `interactive_rainbow.py` first to find the optimal slope before running `optimize_intercept.py`
5. **Model Anchoring:** In the interactive chart, the intercept is automatically calculated to anchor the model at the first data point. In optimization, the intercept is optimized independently for best overall fit.

## 🐛 Troubleshooting

### "config.py not found"
- Make sure you're running the script from the folder containing `config.py`
- Or adjust your Python path

### "Data file not found"
- Check that `DATA_FOLDER` and `HISTORICAL_DATA_FILE` in `config.py` are correct
- The script will create the file automatically on first run if it doesn't exist

### "No data found for TICKER"
- Check that `YAHOO_TICKER` is correct (format: "CURRENCY-USD")
- Common tickers: "BTC-USD", "ETH-USD", "DOGE-USD", "LINK-USD"
- Verify the ticker exists on Yahoo Finance website
- Make sure `yfinance` package is installed: `pip install yfinance`

## 📝 Customization

All scripts read from `config.py`, so you can:
- Change optimization bounds (in the code if needed)
- Adjust date ranges
- Modify output file names (via `get_output_files()` function)
- Add custom parameters as needed

## 🎯 Workflow Summary

```
1. Copy log_model/ folder → Your currency folder
2. Edit config.py with your currency info
3. Run: python update_prices.py (creates CSV automatically)
4. Run: python interactive_rainbow.py (note optimal slope)
5. Update FIXED_GROWTH_RATE in config.py with optimal slope
6. Run: python optimize_intercept.py
7. Done! Check OUTPUT_FOLDER for results
```

## 💡 Tips

- **Start with a small date range** in your initial CSV to test
- **Use the interactive chart** to understand the model behavior
- **Save your optimal slope** - you'll need it for future runs
- **Keep your CSV updated** - run `update_prices.py` regularly
- **Logarithmic models** are better for assets that show decelerating growth patterns

---

**Happy analyzing! 🚀**

