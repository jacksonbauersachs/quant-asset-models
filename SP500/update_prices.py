"""
Refresh both S&P 500 data files from Yahoo Finance.

- Data/SP500_all_time_price.csv : price index (^GSPC), 1927-present, no dividends.
- Data/SP500_total_return.csv   : dividends reinvested, 1957-present.
    Rows before 1988-01-04 are a one-time reconstruction (daily ^GSPC prices +
    Robert Shiller's monthly dividend data) and are kept as-is. From 1988-01-04
    on, the official total-return index (^SP500TR) is re-downloaded and chained
    onto the saved 1988-01-04 value, so the two pieces join seamlessly.

Usage: python update_prices.py   (from any folder)
"""
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "Data"
PRICE_FILE = DATA / "SP500_all_time_price.csv"
TR_FILE = DATA / "SP500_total_return.csv"
SPLICE = pd.Timestamp("1988-01-04")


def fetch(ticker):
    h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    h.index = h.index.tz_localize(None).normalize()
    return h[h["Close"] > 0]


def to_csv(dates, close, path, ohlcv=None):
    out = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),
        "Price": close.round(4).values,
        "Open": "" if ohlcv is None else ohlcv["Open"].round(2).values,
        "High": "" if ohlcv is None else ohlcv["High"].round(2).values,
        "Low": "" if ohlcv is None else ohlcv["Low"].round(2).values,
        "Vol.": "" if ohlcv is None else (ohlcv["Volume"] / 1e3).map(lambda v: f"{v:.2f}K").values,
        "Change %": (close.pct_change() * 100).fillna(0).map(lambda v: f"{v:.2f}%").values,
    })
    out.to_csv(path, index=False)
    return out


def main():
    gspc = fetch("^GSPC")
    to_csv(gspc.index, gspc["Close"], PRICE_FILE, gspc)
    print(f"Price index: {gspc.index[0]:%Y-%m-%d} -> {gspc.index[-1]:%Y-%m-%d} ({len(gspc)} rows)")

    if not TR_FILE.exists():
        raise FileNotFoundError(f"{TR_FILE} is missing; the pre-1988 reconstruction must be rebuilt first.")
    old = pd.read_csv(TR_FILE)
    old["Date"] = pd.to_datetime(old["Date"], format="%m/%d/%Y")
    old = old.set_index("Date")["Price"].astype(float)
    anchor = old.get(SPLICE)
    if anchor is None:
        raise ValueError(f"{TR_FILE} has no {SPLICE:%Y-%m-%d} row to chain onto.")

    tr = fetch("^SP500TR")["Close"][SPLICE:]
    tr = tr / tr.iloc[0] * anchor
    series = pd.concat([old[old.index < SPLICE], tr])
    to_csv(series.index, series, TR_FILE)
    print(f"Total return: {series.index[0]:%Y-%m-%d} -> {series.index[-1]:%Y-%m-%d} ({len(series)} rows)")


if __name__ == "__main__":
    main()
