"""
Out-of-sample backtest: Kelly-weighted vs equal-weight rebalancing (no Bitcoin).

Each month, using ONLY data available up to that month:
  - expected return per asset  = long-run trend growth (log-linear fit on all history so far)
                                 + 0.5 * variance  (converts compound growth to average return)
  - covariance matrix          = Ledoit-Wolf shrunk covariance of the last 10 years of monthly returns
  - Kelly weights              = maximise  w·(mu - r) - 1/(2·fraction) · wᵀΣw
                                 subject to w >= 0, sum(w) <= 1 (rest in T-bills, no leverage)
Then the portfolio is held for the next month. Costs: fund expense ratios, 0.1% of every dollar
traded, and cash earns the 3-month Treasury bill rate (^IRX).

Usage: python kelly_backtest.py   (from any folder; needs the asset CSVs locally)
"""
import os
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "Results"

# name: (data file, annual expense ratio of the fund you'd hold)
ASSETS = {
    "S&P 500": ("SP500/Data/SP500_total_return.csv", 0.0010),
    "Gold": ("Commodities/Gold/Data/Gold_all_time_price.csv", 0.0025),
    "Silver": ("Commodities/Silver/Data/Silver_all_time_price.csv", 0.0030),
    "Platinum": ("Commodities/Platinum/Data/Platinum_all_time_price.csv", 0.0060),
    "Palladium": ("Commodities/Palladium/Data/Palladium_all_time_price.csv", 0.0060),
    "Copper": ("Commodities/Copper/Copper_historical_data.csv", 0.0130),
    "Uranium": ("Commodities/Uranium/Data/Uranium_all_time_price.csv", 0.0070),
}
# Annual probability each asset goes to ~zero (failure risk, separate from volatility).
FAILURE_PROB = {name: 0.0 for name in ASSETS}

TEST_START = "2006-01-31"
COV_WINDOW_MONTHS = 120
TRADE_COST = 0.001
BAND = 0.05  # drift band for band rebalancing (absolute weight)


def load_monthly(path):
    d = pd.read_csv(ROOT / path)
    d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
    s = d.set_index("Date")["Price"].astype(float)
    m = s.resample("ME").last().dropna()
    return m[m > 0]


def load_data():
    full = {n: load_monthly(p) for n, (p, _) in ASSETS.items()}
    prices = pd.concat(full, axis=1).dropna()  # months where every asset has a price
    irx = yf.Ticker("^IRX").history(period="max")["Close"]
    irx.index = irx.index.tz_localize(None)
    rf_annual = (irx.resample("ME").mean() / 100).reindex(prices.index).ffill().fillna(0.0)
    rf_month = (1 + rf_annual) ** (1 / 12) - 1
    return full, prices, rf_annual, rf_month


def estimate(full, prices, date):
    """Expected annual arithmetic returns and annual covariance using data up to `date`."""
    names = list(ASSETS)
    mu = np.zeros(len(names))
    for j, n in enumerate(names):
        s = full[n][full[n].index <= date]
        t = ((s.index - s.index[0]).days / 365.25).values
        g = np.polyfit(t, np.log(s.values), 1)[0]           # trend: log growth per year
        var = np.log(s).diff().dropna().var() * 12           # annual variance of log returns
        mu[j] = np.expm1(g + 0.5 * var)                      # arithmetic expected return
    hist = prices[prices.index <= date].dropna().pct_change().dropna().iloc[-COV_WINDOW_MONTHS:]
    cov = LedoitWolf().fit(hist.values).covariance_ * 12
    return mu, cov


def kelly(mu, cov, rf, fraction):
    """Fractional Kelly, long-only, no leverage, with failure-risk penalty."""
    p = np.array([FAILURE_PROB[n] for n in ASSETS])
    n = len(mu)

    def neg_growth(w):
        g = w @ (mu - rf) - (0.5 / fraction) * (w @ cov @ w)
        g += np.sum(p * np.log(np.clip(1 - w, 1e-9, None)))
        return -g

    res = minimize(neg_growth, np.full(n, 1 / n) * 0.5, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "ineq", "fun": lambda w: 1 - w.sum()}])
    return np.clip(res.x, 0, 1)


def simulate(prices, rf_month, targets, rule):
    """Run a strategy. targets[i] = asset weights (rest cash) decided at month i."""
    exp = np.array([e for _, e in ASSETS.values()]) / 12
    idx = prices.index
    test = np.where(idx >= TEST_START)[0]
    w = np.zeros(len(ASSETS))
    value, values, turnover, dates = 1.0, [], 0.0, []
    for k, i in enumerate(test[:-1]):
        tgt = targets[i]
        drift = np.abs(w - tgt).max()
        if (k == 0 or rule == "monthly" or (rule == "yearly" and idx[i].month == 12)
                or (rule == "band" and drift > BAND)):
            traded = np.abs(tgt - w).sum()
            value *= 1 - TRADE_COST * traded
            turnover += traded
            w = tgt.copy()
        r = (prices.iloc[i + 1] / prices.iloc[i] - 1).values - exp
        cash = 1 - w.sum()
        gross = w @ (1 + r) + cash * (1 + rf_month.iloc[i])
        value *= gross
        w = w * (1 + r) / gross
        values.append(value)
        dates.append(idx[i + 1])
    s = pd.Series(values, index=dates)
    years = (s.index[-1] - idx[test[0]]).days / 365.25
    monthly = s.pct_change().fillna(s.iloc[0] - 1)
    return {
        "series": s,
        "cagr": s.iloc[-1] ** (1 / years) - 1,
        "vol": monthly.std() * np.sqrt(12),
        "maxdd": (s / s.cummax() - 1).min(),
        "worst12": (s / s.shift(12) - 1).min(),
        "turnover": turnover / years,
    }


def main():
    full, prices, rf_annual, rf_month = load_data()
    idx = prices.index
    n = len(ASSETS)
    test = np.where(idx >= TEST_START)[0]
    print(f"Backtest {idx[test[0]]:%Y-%m} -> {idx[test[-1]]:%Y-%m}, {n} assets, monthly")

    # Three ways to set expected returns (mu) for Kelly:
    #   trend  - each asset's own fitted trend (noisy: short or peak-started histories mislead)
    #   shrunk - halfway between each asset's trend and the average of all assets
    #   equal  - every asset gets the average: no view on which grows fastest, so Kelly
    #            sizes purely on volatility and correlation (pure volatility harvesting)
    full_k, half_k, half_shrunk, half_equal = {}, {}, {}, {}
    for i in test:
        mu, cov = estimate(full, prices, idx[i])
        rf = rf_annual.iloc[i]
        avg = np.full(n, mu.mean())
        full_k[i] = kelly(mu, cov, rf, 1.0)
        half_k[i] = kelly(mu, cov, rf, 0.5)
        half_shrunk[i] = kelly(0.5 * mu + 0.5 * avg, cov, rf, 0.5)
        half_equal[i] = kelly(avg, cov, rf, 0.5)
    equal = {i: np.full(n, 1 / n) for i in test}
    sp_only = {i: np.eye(n)[0] for i in test}

    runs = {
        "S&P 500 only": simulate(prices, rf_month, sp_only, "monthly"),
        "Equal weight, monthly": simulate(prices, rf_month, equal, "monthly"),
        "Equal weight, yearly": simulate(prices, rf_month, equal, "yearly"),
        "Equal weight, 5% bands": simulate(prices, rf_month, equal, "band"),
        "Full Kelly, monthly": simulate(prices, rf_month, full_k, "monthly"),
        "Half Kelly, monthly": simulate(prices, rf_month, half_k, "monthly"),
        "Half Kelly, 5% bands": simulate(prices, rf_month, half_k, "band"),
        "Half Kelly shrunk, bands": simulate(prices, rf_month, half_shrunk, "band"),
        "Half Kelly equal-mu, bands": simulate(prices, rf_month, half_equal, "band"),
    }

    OUT.mkdir(exist_ok=True)
    names = list(ASSETS)
    hk = pd.DataFrame([half_k[i] for i in test], index=idx[test], columns=names)
    fk = pd.DataFrame([full_k[i] for i in test], index=idx[test], columns=names)
    hs = pd.DataFrame([half_shrunk[i] for i in test], index=idx[test], columns=names)
    he = pd.DataFrame([half_equal[i] for i in test], index=idx[test], columns=names)
    lines = [
        "KELLY vs EQUAL-WEIGHT REBALANCING BACKTEST (out of sample, no Bitcoin)",
        "=" * 72,
        f"Period: {idx[test[0]]:%Y-%m} to {idx[test[-1]]:%Y-%m}  |  assets: {', '.join(names)}",
        "Costs: fund expense ratios, 0.1% per dollar traded; cash earns 3-month T-bills.",
        "",
        f"{'Strategy':26}{'Growth/yr':>11}{'Volatility':>12}{'Worst drop':>12}{'Worst 12m':>11}{'Traded/yr':>11}{'$10k ->':>12}",
    ]
    for name, r in runs.items():
        lines.append(f"{name:26}{r['cagr'] * 100:>10.2f}%{r['vol'] * 100:>11.1f}%{r['maxdd'] * 100:>11.0f}%"
                     f"{r['worst12'] * 100:>10.0f}%{r['turnover'] * 100:>10.0f}%{10000 * r['series'].iloc[-1]:>12,.0f}")
    lines += ["", "AVERAGE KELLY WEIGHTS OVER THE TEST (rest = T-bills)",
              f"{'':12}" + "".join(f"{n[:9]:>10}" for n in names) + f"{'Cash':>10}"]
    variants = [("Full Kelly", fk), ("Half Kelly", hk), ("Half shrunk", hs), ("Half eq-mu", he)]
    for label, df in variants:
        lines.append(f"{label:12}" + "".join(f"{v * 100:>9.0f}%" for v in df.mean())
                     + f"{(1 - df.sum(axis=1)).mean() * 100:>9.0f}%")
    lines += ["", f"KELLY WEIGHTS AT THE LATEST MONTH ({idx[test[-1]]:%Y-%m}) — model output, not advice",
              f"{'':12}" + "".join(f"{n[:9]:>10}" for n in names) + f"{'Cash':>10}"]
    for label, df in variants:
        lines.append(f"{label:12}" + "".join(f"{v * 100:>9.0f}%" for v in df.iloc[-1])
                     + f"{(1 - df.iloc[-1].sum()) * 100:>9.0f}%")
    mu, cov = estimate(full, prices, idx[test[-1]])
    sd = np.sqrt(np.diag(cov))
    lines += ["", "LATEST INPUTS (annual)", f"{'':12}{'Exp. return':>12}{'Volatility':>12}"]
    for j, nme in enumerate(names):
        lines.append(f"{nme:12}{mu[j] * 100:>11.1f}%{sd[j] * 100:>11.1f}%")
    lines.append(f"T-bill rate now: {rf_annual.iloc[test[-1]] * 100:.2f}%")
    text = "\n".join(lines) + "\n"
    (OUT / "kelly_backtest_results.txt").write_text(text, encoding="utf-8")
    print(text)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]})
    for name in ["S&P 500 only", "Equal weight, 5% bands", "Half Kelly, 5% bands",
                 "Half Kelly shrunk, bands", "Half Kelly equal-mu, bands"]:
        ax1.plot(runs[name]["series"] * 10000, label=f"{name} ({runs[name]['cagr'] * 100:.1f}%/yr)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_title("Growth of $10,000 — out-of-sample backtest (no Bitcoin)", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="upper left")
    ax2.stackplot(hk.index, hk.T.values, labels=names, alpha=0.85)
    ax2.set_ylim(0, 1)
    ax2.set_title("Half-Kelly target weights over time (white space = T-bills)")
    ax2.legend(loc="upper left", fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    chart = OUT / "kelly_backtest.png"
    fig.savefig(chart, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Chart: {chart}")
    if platform.system() == "Windows" and not os.environ.get("NO_OPEN_CHART"):
        try:
            os.startfile(chart)
        except OSError:
            pass


if __name__ == "__main__":
    main()
