"""
Shared math for the unpinned power-law model.

Model (log scale):   log(price) = c + k * log(days since genesis)
Fair price:          fair = e^c * days^k

- k and c are fitted together by least squares on log price, so every
  period counts by percentage error and no single day pins the curve.
- If k is fixed, c is still fitted: the best level for a given k is the mean
  of (log price - k * log days).
- Unlike an exponential, growth slows as the asset ages:
  annual growth at day d = (1 + 365.25 / d)^k - 1.
- sigma is the standard deviation of log residuals; the bands
  fair * e^(+/- n * sigma) mark typical swings around the trend.
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, GENESIS_DATE, START_DATE


def _parse_dates(dates):
    """Parse dates written as MM/DD/YYYY, YYYY-MM-DD, or anything pandas recognises."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(dates, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(dates, format="mixed")


def load_data(end_date=None):
    """Load prices, add Days since GENESIS_DATE, filter to START_DATE / end_date."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    df["Date"] = _parse_dates(df["Date"])
    price_col = next((c for c in ["Price", "Close", "Adj Close", "Value"] if c in df.columns), None)
    if price_col is None:
        raise ValueError(f"No price column found in {DATA_FILE}: {list(df.columns)}")
    prices = df[price_col]
    if prices.dtype == object:
        prices = prices.str.replace(",", "", regex=False)
    df["Price"] = pd.to_numeric(prices, errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    genesis = pd.Timestamp(GENESIS_DATE) if GENESIS_DATE else df["Date"].min()
    df = df[df["Date"] >= pd.Timestamp(START_DATE)] if START_DATE else df[df["Date"] >= genesis]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    df = df[df["Price"] > 0].reset_index(drop=True)
    # Day 0 = genesis; clip to 1 so log(days) is defined on the genesis day itself
    df["Days"] = (df["Date"] - genesis).dt.days.clip(lower=1).astype(float)
    df.attrs["genesis"] = genesis
    return df


def fit(df, exponent=None):
    """Fit the model. Returns dict with k, c, r2 (log), sigma."""
    log_days = np.log(df["Days"].to_numpy())
    log_price = np.log(df["Price"].to_numpy())
    if exponent is None:
        k, c = np.polyfit(log_days, log_price, 1)
    else:
        k = exponent
        c = np.mean(log_price - k * log_days)
    residuals = log_price - (c + k * log_days)
    ss_tot = np.sum((log_price - log_price.mean()) ** 2)
    r2 = 1 - np.sum(residuals ** 2) / ss_tot if ss_tot > 0 else 0.0
    return {"k": float(k), "c": float(c), "r2": float(r2), "sigma": float(residuals.std())}


def fair_price(model, days):
    """Fair price at the given day number(s) since genesis."""
    return np.exp(model["c"] + model["k"] * np.log(np.asarray(days, dtype=float)))


def annual_growth_at(model, day):
    """Model's compounded growth (%) over the year starting at the given day."""
    return ((1 + 365.25 / day) ** model["k"] - 1) * 100


def valuation(df, model):
    """Current price vs fair value: percent and position in sigma units."""
    current = df["Price"].iloc[-1]
    fair = float(fair_price(model, df["Days"].iloc[-1]))
    pct = (current - fair) / fair * 100
    z = (np.log(current) - np.log(fair)) / model["sigma"] if model["sigma"] > 0 else 0.0
    return {"current": float(current), "fair": fair, "pct": float(pct), "z": float(z)}


def valuation_text(v):
    word = "Overvalued" if v["pct"] > 0 else "Undervalued"
    return f"{word} by {abs(v['pct']):.2f}% vs fair value ({v['z']:+.2f} sigma from trend)"


def format_price(p):
    if p < 1:
        return f"${p:.4f}"
    if p < 1000:
        return f"${p:,.2f}"
    return f"${p:,.0f}"
