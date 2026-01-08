"""
===========================================
CURRENCY CONFIGURATION FILE - MULTI-MODEL SUPPORT
===========================================
Edit ALL values in this file to match your currency.
This is the ONLY file you need to modify when cloning the template.

This template supports THREE model types:
  1. PARAMETRIC: price = 10^(slope × (days^power) + intercept)
     - Growth pattern controlled by power parameter
     - Good for forecasting with flexible growth curves
  
  2. POWER LAW: price = a × (days - first_day_offset + 1)^b
     - Concave shape (0 < b < 1): steep early, flattens over time
     - Good for modeling diminishing returns
  
  3. EXPONENTIAL: price = a × e^(b × (days - first_day_offset))
     - Accelerating growth (b > 0) or decay (b < 0)
     - Good for modeling compound growth patterns

Set MODEL_TYPE below to choose which model to use.
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

START_DATE = ""  # Analysis start date - when to begin the rainbow chart analysis (YYYY-MM-DD format)
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
OUTPUT_FOLDER = "Crypto/Bitcoin/point_selecting_method"

# ============================================
# MODEL PARAMETERS
# ============================================
# MODEL_TYPE: 'parametric', 'powerlaw', or 'exponential'
#   'parametric': price = 10^(slope × (days^power) + intercept)
#   'powerlaw': price = a × (days - first_day_offset + 1)^b
#   'exponential': price = a × e^(b × (days - first_day_offset))
MODEL_TYPE = ''  # Options: 'parametric', 'powerlaw', or 'exponential'

# ============================================
# PARAMETRIC MODEL PARAMETERS
# ============================================
# Fixed slope for optimization (OPTIONAL - set to None to auto-optimize)
# This is typically found by running the interactive rainbow chart first
# For parametric model: price = 10^(slope × (days^power) + intercept)
# Slope typically ranges from 0.5 to 5.0 for cryptocurrencies
# Set to None to let optimize_intercept.py find the optimal slope automatically
FIXED_GROWTH_RATE = None # This is used as FIXED_SLOPE for parametric models (set to None to auto-optimize)
# Example: 2.5 for a slope of 2.5

# Power for the parametric growth model (OPTIONAL - default: 1.0)
# Controls acceleration pattern:
#   power < 1: growth slows over time (like logarithmic)
#   power = 1: linear growth
#   power > 1: growth accelerates (starts slow, speeds up)
#   power < 0: inverse/decay patterns
# Set this after finding optimal value in interactive_rainbow_v2.py
POWER = None # Default: 1.0 (linear)

# ============================================
# POWER LAW MODEL PARAMETERS
# ============================================
# Power law formula: price = a × (days - first_day_offset + 1)^b
# These are typically found by running interactive_point_selector.py and clicking "Power Law"
# POWER_LAW_A: Coefficient (typically found from power law fit)
POWER_LAW_A = None  # Example value - update after running power law fit (will auto-optimize if None)

# POWER_LAW_B: Exponent (constrained to 0 < b < 1 for concave shape)
POWER_LAW_B = None  # Example value - update after running power law fit (REQUIRED for powerlaw model)

# POWER_LAW_FIRST_DAY_OFFSET: Days since genesis of first selected point (for normalization)
# Set to None to auto-detect from data, or specify the days since genesis of your first reference point
POWER_LAW_FIRST_DAY_OFFSET = None  # Will auto-detect if None

# ============================================
# EXPONENTIAL MODEL PARAMETERS
# ============================================
# Exponential formula: price = a × e^(b × (days - first_day_offset))
# These are typically found by running interactive_point_selector.py and clicking "Exponential (Growth)"
# EXPONENTIAL_A: Coefficient (typically found from exponential fit)
# Set to None to auto-optimize, or set to a value from exponential fit
EXPONENTIAL_A = None  # Will optimize if None, or set to a value from exponential fit

# EXPONENTIAL_B: Growth rate (positive = growth, negative = decay)
# This is the shape parameter from selected points - MUST be set for exponential model
EXPONENTIAL_B = None  # REQUIRED: Set this after running exponential fit in interactive_point_selector.py

# EXPONENTIAL_FIRST_DAY_OFFSET: Days since genesis of first selected point (for normalization)
# Set to None to auto-detect from data, or specify the days since genesis of your first reference point
EXPONENTIAL_FIRST_DAY_OFFSET = None  # Will auto-detect if None

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

