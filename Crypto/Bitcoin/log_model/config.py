"""
===========================================
CURRENCY CONFIGURATION FILE - LOGARITHMIC MODEL
===========================================
Edit ALL values in this file to match your currency.
This is the ONLY file you need to modify when cloning the template.

This template uses a LOGARITHMIC growth model:
  price = 10^(slope × ln(days_since_genesis) + intercept)

Growth slows over time (unlike exponential which accelerates).
"""

# ============================================
# CURRENCY INFORMATION
# ============================================
CURRENCY_NAME = "Bitcoin"  # Display name (capitalized, e.g., "Bitcoin", "Chainlink")
CURRENCY_NAME_LOWER = "bitcoin"  # Lowercase name for file naming (e.g., "bitcoin", "chainlink")
YAHOO_TICKER = "BTC-USD"  # Yahoo Finance ticker symbol (e.g., "BTC-USD", "ETH-USD", "LINK-USD")

# ============================================
# DATES
# ============================================
# GENESIS_DATE is OPTIONAL - if not set, will auto-detect from earliest date in data file
# Only set this if you want to use a specific launch date (e.g., actual crypto launch date)
# GENESIS_DATE = "2015-07-30"  # Launch/genesis date (YYYY-MM-DD format) - OPTIONAL
GENESIS_DATE = None  # Set to None to auto-detect from data, or specify a date string

START_DATE = "2011-06-08"  # Analysis start date - when to begin the rainbow chart analysis (YYYY-MM-DD format)
# Leave as empty string "" to auto-detect from first date in data file
# Or specify a date like "2017-05-18" to start analysis from that date

# ============================================
# FILE PATHS (relative to project root)
# ============================================
# Example: "Crypto/Chainlink"
DATA_FOLDER = "Crypto/Bitcoin/data_update"

# Example: "Chainlink_historical_data.csv"
HISTORICAL_DATA_FILE = "BTC_historical_data.csv"

# Example: "Crypto/Chainlink"
OUTPUT_FOLDER = "Crypto/Bitcoin/log_model"

# ============================================
# MODEL PARAMETERS (LOGARITHMIC MODEL)
# ============================================
# Fixed slope for optimization (OPTIONAL - set to None to auto-optimize)
# This is typically found by running the interactive rainbow chart first
# For logarithmic model: price = 10^(slope × ln(day) + intercept)
# Slope typically ranges from 0.5 to 5.0 for cryptocurrencies
# Set to None to let optimize_intercept.py find the optimal slope automatically
FIXED_GROWTH_RATE = 1.3005  # This is used as FIXED_SLOPE for logarithmic models
# Example: 2.5 for a slope of 2.5

# Power for the growth model (OPTIONAL - default: 1.0)
# Controls acceleration pattern:
#   power < 1: growth slows over time (like logarithmic)
#   power = 1: linear growth (base 10)
#   power > 1: growth accelerates (starts slow, speeds up)
#   power < 0: inverse/decay patterns
# Set this after finding optimal value in interactive_rainbow_v2.py
POWER = 1.0  # Default: 1.0 (linear, base 10)

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
    
    print(f"Configuration validated for {CURRENCY_NAME}")
    return True

