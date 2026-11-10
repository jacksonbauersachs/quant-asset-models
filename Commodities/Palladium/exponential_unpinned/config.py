"""
Configuration for the unpinned exponential model.

Unlike the pinned exponential model, the curve is NOT pinned to the price on the start date:
growth rate and level are both fitted to the whole history on a log scale
(ordinary least squares on log price). Edit the values below as needed.
"""
from pathlib import Path

CURRENCY_NAME = "Palladium"
CURRENCY_NAME_LOWER = "palladium"

# First date included in the fit. "" = use all available data.
# Same start as Commodities/Palladium/config.py.
START_DATE = "2009-10-28"

# None = fit the growth rate automatically.
# Or set a per-day rate (e.g. 0.000233) to use your own slope; the level is
# still fitted automatically for that slope.
FIXED_GROWTH_RATE = None

# Spike check: also refit without the most recent N years, to show how much
# a recent run-up is tilting the line.
SPIKE_CHECK_YEARS = 2

# Paths (resolved from this file, so scripts work from any working directory)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPO_ROOT / "Commodities" / "Palladium" / "Palladium_historical_data.csv"
OUTPUT_FOLDER = Path(__file__).resolve().parent
