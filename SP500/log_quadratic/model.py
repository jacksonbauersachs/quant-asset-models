"""
Shared math for the log-quadratic model.

Model (log scale):  log(price) = a + b*t + q*t^2,   t = years since start
Growth rate at t:   g(t) = b + 2*q*t   (continuous, per year)
Annual growth (%):  e^g(t) - 1

All terms are fitted together by least squares on log price (log-scale R²).
For comparison, the plain exponential (q = 0) is also fitted.
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, START_DATE


def _parse_dates(dates):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(dates, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(dates, format="mixed")


def load_data(end_date=None):
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    df["Date"] = _parse_dates(df["Date"])
    prices = df["Price"]
    if prices.dtype == object:
        prices = prices.str.replace(",", "", regex=False)
    df["Price"] = pd.to_numeric(prices, errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    if START_DATE:
        df = df[df["Date"] >= pd.Timestamp(START_DATE)]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    df = df[df["Price"] > 0].reset_index(drop=True)
    df["Years"] = (df["Date"] - df["Date"].iloc[0]).dt.days / 365.25
    return df


def _stats(log_price, fitted):
    res = log_price - fitted
    ss_tot = np.sum((log_price - log_price.mean()) ** 2)
    return float(1 - np.sum(res ** 2) / ss_tot), float(res.std())


def fit(df, curvature=None):
    """Fit log-quadratic. Returns dict with a, b, q, r2, sigma (and adjusted R²)."""
    t = df["Years"].to_numpy()
    lp = np.log(df["Price"].to_numpy())
    if curvature is None:
        q, b, a = np.polyfit(t, lp, 2)
    else:
        q = curvature
        b, a = np.polyfit(t, lp - q * t ** 2, 1)
    r2, sigma = _stats(lp, a + b * t + q * t ** 2)
    n, k = len(lp), 3
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return {"a": float(a), "b": float(b), "q": float(q), "r2": r2, "adj_r2": float(adj), "sigma": sigma}


def fit_exponential(df):
    """Plain exponential (q = 0) for comparison."""
    t = df["Years"].to_numpy()
    lp = np.log(df["Price"].to_numpy())
    b, a = np.polyfit(t, lp, 1)
    r2, sigma = _stats(lp, a + b * t)
    n, k = len(lp), 2
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return {"a": float(a), "b": float(b), "q": 0.0, "r2": r2, "adj_r2": float(adj), "sigma": sigma}


def fair_price(model, years):
    t = np.asarray(years, dtype=float)
    return np.exp(model["a"] + model["b"] * t + model["q"] * t ** 2)


def annual_growth_at(model, years):
    """Annual growth (%) implied at time t (years since start)."""
    return (np.exp(model["b"] + 2 * model["q"] * years) - 1) * 100


def valuation(df, model):
    current = df["Price"].iloc[-1]
    fair = float(fair_price(model, df["Years"].iloc[-1]))
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
