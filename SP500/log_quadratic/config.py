"""
Configuration for the log-quadratic ("accelerating growth") model.

Model: log(price) = a + b * t + q * t^2, with t in years since START_DATE.
The growth rate is b + 2 * q * t, so q > 0 means growth speeds up over time
and q < 0 means it slows down. q = 0 is the plain exponential model.
"""
from pathlib import Path

CURRENCY_NAME = "S&P 500 (dividends reinvested)"
CURRENCY_NAME_LOWER = "sp500"

# First date included in the fit (when the index expanded to 500 stocks).
START_DATE = "1957-03-04"

# None = fit the curvature q automatically.
# Or set q yourself (log growth per year, per year); the line is refitted for it.
FIXED_CURVATURE = None

# Spike check: also refit without the most recent N years.
SPIKE_CHECK_YEARS = 2

# Paths (resolved from this file, so scripts work from any working directory)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "SP500" / "Data" / "SP500_total_return.csv"
OUTPUT_FOLDER = Path(__file__).resolve().parent
