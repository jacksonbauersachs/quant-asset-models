"""
Configuration for the unpinned power-law model.

Model: price = A * days^k, where days = days since GENESIS_DATE.
On a log scale this is a straight line against log(days), so the exponent k
and level A are fitted together by least squares on log price (log-scale R²).
The curve is not pinned to any single day's price.
"""
from pathlib import Path

CURRENCY_NAME = "Bitcoin"
CURRENCY_NAME_LOWER = "bitcoin"

# Day 0 of the power law. A power law depends on this choice, so keep it fixed.
# 2010-07-18 is the first day of the Bitcoin price history (same as log_model_v2).
GENESIS_DATE = "2010-07-18"

# First date included in the fit. Prices before ~2011 are rounded to $0.1
# steps in the source data, so the fit starts later (same as log_model_v2).
# "" = use all data from GENESIS_DATE.
START_DATE = "2011-04-10"

# None = fit the exponent k automatically.
# Or set k yourself; the level is still fitted for that k.
# To match log_model_v2's FIXED_GROWTH_RATE (a slope in log10 price per ln day),
# multiply it by 2.302585: e.g. 1.7679 -> k = 4.0707.
FIXED_EXPONENT = None

# Spike check: also refit without the most recent N years, to show how much
# a recent run-up is tilting the line.
SPIKE_CHECK_YEARS = 2

# Paths (resolved from this file, so scripts work from any working directory)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPO_ROOT / "Crypto" / "Bitcoin" / "data_update" / "BTC_historical_data.csv"
OUTPUT_FOLDER = Path(__file__).resolve().parent
