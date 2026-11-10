"""
Shared math for the unpinned exponential model.

Model (log scale):   log(price) = c + g * days
Fair price:          fair(t) = e^(c + g * days)

- g (per day) and c are fitted together by least squares on log price, so
  every period of history counts equally and the curve is not tied to the
  price on any single day.
- If a growth rate is fixed, c is still fitted: for a given slope the best
  level is the mean of (log price - g * days).
- sigma is the standard deviation of log residuals; the bands
  fair * e^(+/- k * sigma) mark typical swings around the trend.
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, START_DATE


def load_data(end_date=None):
    """Load prices from DATA_FILE, filtered to START_DATE (and optionally end_date)."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    if df["Price"].dtype == object:
        df["Price"] = df["Price"].str.replace(",", "", regex=False)
    df["Price"] = df["Price"].astype(float)
    df = df.sort_values("Date")
    if START_DATE:
        df = df[df["Date"] >= pd.Timestamp(START_DATE)]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    df = df[df["Price"] > 0].reset_index(drop=True)
    df["Days"] = (df["Date"] - df["Date"].iloc[0]).dt.days.astype(float)
    return df


def fit(df, growth_rate=None):
    """Fit the model. Returns dict with g (per day), c, r2 (log), sigma."""
    days = df["Days"].to_numpy()
    log_price = np.log(df["Price"].to_numpy())
    if growth_rate is None:
        g, c = np.polyfit(days, log_price, 1)
    else:
        g = growth_rate
        c = np.mean(log_price - g * days)
    residuals = log_price - (c + g * days)
    ss_tot = np.sum((log_price - log_price.mean()) ** 2)
    r2 = 1 - np.sum(residuals ** 2) / ss_tot if ss_tot > 0 else 0.0
    return {"g": float(g), "c": float(c), "r2": float(r2), "sigma": float(residuals.std())}


def fair_price(model, days):
    """Fair price at the given day number(s) since the start date."""
    return np.exp(model["c"] + model["g"] * np.asarray(days, dtype=float))


def annual_pct(g):
    """Compounded annual growth (%) for a per-day log growth rate."""
    return (np.exp(g * 365.25) - 1) * 100


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
