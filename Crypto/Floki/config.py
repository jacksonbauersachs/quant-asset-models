"""
===========================================
CURRENCY CONFIGURATION FILE
===========================================
Edit ALL values in this file to match your currency.
This is the ONLY file you need to modify when cloning the template.
"""

# ============================================
# CURRENCY INFORMATION
# ============================================
CURRENCY_NAME = "Floki"  # Display name (capitalized, e.g., "Bitcoin", "Dogecoin")
CURRENCY_NAME_LOWER = "floki"  # Lowercase name for file naming (e.g., "bitcoin", "dogecoin")
YAHOO_TICKER = "FLOKI-USD"  # Yahoo Finance ticker symbol (e.g., "BTC-USD", "ETH-USD", "DOGE-USD")

# ============================================
# DATES
# ============================================
# GENESIS_DATE is OPTIONAL - if not set, will auto-detect from earliest date in data file
# Only set this if you want to use a specific launch date (e.g., actual crypto launch date)
# GENESIS_DATE = "2015-07-30"  # Launch/genesis date (YYYY-MM-DD format) - OPTIONAL
GENESIS_DATE = None  # Set to None to auto-detect from data, or specify a date string

START_DATE = ""  # Analysis start date - when to begin the rainbow chart analysis (YYYY-MM-DD format)
# Leave as empty string "" to auto-detect from first date in data file
# Or specify a date like "2017-05-18" to start analysis from that date

# ============================================
# FILE PATHS (relative to project root)
# ============================================
# Example: "Crypto/Etherium/Data"
DATA_FOLDER = "Crypto/Floki"

# Example: "Ethereum Historical Data.csv"
HISTORICAL_DATA_FILE = "Floki_historical_data.csv"

# Example: "Crypto/Etherium/Models/Growth"
OUTPUT_FOLDER = "Crypto/Floki"

# ============================================
# MODEL PARAMETERS
# ============================================
# Fixed growth rate for optimization (OPTIONAL - set to None to auto-optimize)
# This is typically found by running the interactive rainbow chart first
# Growth rate is in "per day" units (e.g., 0.001 = 0.1% per day ≈ 36.5% per year)
# Set to None to let optimize_intercept.py find the optimal growth rate automatically
FIXED_GROWTH_RATE = .001107  # Example: 0.001 for 0.1% daily growth

# ============================================
# DATA SOURCE
# ============================================
# Using Yahoo Finance (yfinance library) - no API key required!
# Yahoo Finance ticker format: "CURRENCY-USD" (e.g., "BTC-USD", "DOGE-USD")

# ============================================
# OUTPUT FILE NAMES (auto-generated from CURRENCY_NAME_LOWER)
# ============================================
def get_output_files():
    """Generate output file names based on currency name."""
    base = CURRENCY_NAME_LOWER
    return {
        'interactive_rainbow_png': f"{base}_interactive_rainbow.png",
        'intercept_optimization_png': f"{base}_intercept_optimization.png",
        'intercept_optimization_txt': f"{base}_intercept_optimization.txt"
    }

# ============================================
# VALIDATION
# ============================================
def validate_config():
    """Validate that all required config values are set."""
    required = [
        CURRENCY_NAME, CURRENCY_NAME_LOWER, YAHOO_TICKER,
        DATA_FOLDER, HISTORICAL_DATA_FILE, OUTPUT_FOLDER
    ]
    
    if None in required or "" in [r for r in required if r != START_DATE]:
        raise ValueError("All required configuration values must be set!")
    
    # Validate date formats (GENESIS_DATE and START_DATE are optional)
    from datetime import datetime
    try:
        if START_DATE and START_DATE != "":
            # Try multiple date formats
            try:
                datetime.strptime(START_DATE, '%Y-%m-%d')
            except ValueError:
                try:
                    datetime.strptime(START_DATE, '%m-%d-%Y')
                except ValueError:
                    try:
                        datetime.strptime(START_DATE, '%m/%d/%Y')
                    except ValueError:
                        raise ValueError(f"START_DATE '{START_DATE}' must be in YYYY-MM-DD, M-D-YYYY, or M/D/YYYY format!")
        if GENESIS_DATE is not None and GENESIS_DATE != "":
            # Try multiple date formats
            try:
                datetime.strptime(GENESIS_DATE, '%Y-%m-%d')
            except ValueError:
                try:
                    datetime.strptime(GENESIS_DATE, '%m-%d-%Y')
                except ValueError:
                    try:
                        datetime.strptime(GENESIS_DATE, '%m/%d/%Y')
                    except ValueError:
                        raise ValueError(f"GENESIS_DATE '{GENESIS_DATE}' must be in YYYY-MM-DD, M-D-YYYY, or M/D/YYYY format!")
    except ValueError as e:
        raise ValueError(str(e))
    
    print(f"✅ Configuration validated for {CURRENCY_NAME}")
    return True

