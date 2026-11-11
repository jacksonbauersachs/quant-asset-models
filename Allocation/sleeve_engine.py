"""
Kelly sizing for a sleeve of assets, with an explicit, probability-weighted
mean-reversion belief and a hard per-asset cap. Generic engine: no personal data.
Personal inputs live in a config dict (see private/sleeve_config.py).

Expected return per asset (all in % per year):
    log growth  = ln(1 + model growth) - expense ratio
    average     = log growth + 0.5 * volatility^2          (compound -> average return)
    reversion   = sum over scenarios of  p * ln(fair/price) / years_to_close
                  + p_never * never_drift                   (trend broke: drifts instead)
    uncertainty = variance of the reversion term across scenarios, added to the asset's risk

Weights maximise   w·mu - 1/(2·kelly_fraction) · wᵀΣw
subject to 0 <= w_i <= cap and sum(w) = 1 (fully invested, no cash, no leverage).

"Value of the next dollar" = mu_i - (Σw)_i  is the slope of that growth curve: at the
optimum every held asset has (about) the same value; higher = underweight, lower = overweight.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]


def _monthly(path):
    d = pd.read_csv(ROOT / path, encoding="utf-8-sig")
    d["Date"] = pd.to_datetime(d["Date"], format="mixed")
    p = d["Price"]
    if p.dtype == object:
        p = p.str.replace(",", "", regex=False)
    s = pd.Series(pd.to_numeric(p, errors="coerce").values, index=d["Date"]).dropna()
    return s[s > 0].sort_index()


def run(config):
    assets = config["assets"]
    names = list(assets)
    series = {n: _monthly(a["data"]) for n, a in assets.items()}

    # Volatility and correlation: shrunk covariance of recent monthly returns
    monthly = pd.concat({n: s.resample("ME").last() for n, s in series.items()}, axis=1).dropna()
    rets = monthly.pct_change().dropna().iloc[-config["cov_window_months"]:]
    cov = LedoitWolf().fit(rets.values).covariance_ * 12
    vol = np.sqrt(np.diag(cov))

    price = np.array([assets[n]["price"] if assets[n].get("price") else series[n].iloc[-1] for n in names])
    fair = np.array([assets[n]["fair_value"] for n in names])
    gap = np.log(fair / price)  # > 0 means below fair value
    log_growth = np.array([np.log1p(assets[n]["growth"]) - assets[n]["expense_ratio"] for n in names])

    rev_mean, rev_var = np.zeros(len(names)), np.zeros(len(names))
    for j, n in enumerate(names):
        b = assets[n].get("reversion", config["default_reversion"])
        outcomes = [gap[j] / s["years"] for s in b["scenarios"]] + [b["never_drift"]]
        probs = [s["prob"] for s in b["scenarios"]] + [b["never_prob"]]
        if abs(sum(probs) - 1) > 1e-6:
            raise ValueError(f"{n}: reversion probabilities sum to {sum(probs):.3f}, must be 1")
        outcomes, probs = np.array(outcomes), np.array(probs)
        rev_mean[j] = probs @ outcomes
        rev_var[j] = probs @ (outcomes - rev_mean[j]) ** 2

    mu = log_growth + 0.5 * vol ** 2 + rev_mean
    risk = cov + np.diag(rev_var)
    frac, cap = config["kelly_fraction"], config["max_weight"]

    def neg_growth(w):
        return -(w @ mu - 0.5 / frac * (w @ risk @ w))

    n = len(names)
    best = None
    for start in [np.full(n, 1 / n), np.eye(n)[int(np.argmax(mu))] * 0.5 + 0.5 / n]:
        res = minimize(neg_growth, start, method="SLSQP", bounds=[(0, cap)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        if best is None or res.fun < best.fun:
            best = res
    target = np.clip(best.x, 0, cap)
    target = target / target.sum()

    holdings = np.array([assets[n].get("holding", 0.0) for n in names], dtype=float)
    total = config.get("sleeve_dollars") or holdings.sum()
    current = holdings / holdings.sum() if holdings.sum() > 0 else np.zeros(n)

    def growth(w):
        return float(w @ mu - 0.5 * (w @ risk @ w))

    def bonus(w):
        return float(0.5 * (w @ np.diag(cov) - w @ cov @ w))

    compare = {}
    for label, weights in config.get("compare_mixes", {}).items():
        w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
        w = w / w.sum()
        compare[label] = {"w": w, "growth": growth(w), "vol": float(np.sqrt(w @ cov @ w)), "bonus": bonus(w)}

    return {
        "compare": compare,
        "names": names, "price": price, "fair": fair, "gap": gap, "vol": vol,
        "log_growth": log_growth, "vol_bonus": 0.5 * vol ** 2, "rev_mean": rev_mean, "rev_sd": np.sqrt(rev_var),
        "mu": mu, "target": target, "current": current, "holdings": holdings, "total": total,
        "value_now": mu - risk @ current, "value_target": mu - risk @ target,
        "growth_now": growth(current), "growth_target": growth(target),
        "vol_now": float(np.sqrt(current @ cov @ current)), "vol_target": float(np.sqrt(target @ cov @ target)),
        "bonus_now": bonus(current), "bonus_target": bonus(target),
        "window": (rets.index[0], rets.index[-1]), "config": config,
    }


def report(r):
    pct = lambda x: f"{x * 100:+.1f}%"
    names, cfg = r["names"], r["config"]
    lines = [
        "SLEEVE KELLY CALCULATOR — model output from your assumptions, not advice",
        "=" * 78,
        f"Kelly fraction {cfg['kelly_fraction']:g} | max {cfg['max_weight'] * 100:.0f}% per asset | fully invested, no cash"
        f" | sleeve ${r['total']:,.0f}",
        f"Volatility/correlation from monthly returns {r['window'][0]:%Y-%m} to {r['window'][1]:%Y-%m}",
        "",
        "EXPECTED RETURN BUILD-UP (% per year)",
        f"{'Asset':11}{'Growth-fee':>11}{'+Vol farm':>10}{'Price vs fair':>15}{'+Reversion':>12}{'(± spread)':>11}{'=Expected':>11}",
    ]
    for j, n in enumerate(names):
        vs = np.exp(-r["gap"][j]) - 1
        where = f"{abs(vs) * 100:.0f}% {'above' if vs > 0 else 'below'}"
        lines.append(f"{n:11}{pct(r['log_growth'][j]):>11}{pct(r['vol_bonus'][j]):>10}{where:>15}"
                     f"{pct(r['rev_mean'][j]):>12}{'±' + format(r['rev_sd'][j] * 100, '.1f') + '%':>11}"
                     f"{pct(r['mu'][j]):>11}")
    lines += ["", "TARGET vs CURRENT",
              f"{'Asset':11}{'Current':>9}{'Target':>9}{'Target $':>11}{'Trade $':>11}"
              f"{'Value now':>11}{'Value @target':>15}"]
    for j in np.argsort(-r["target"]):
        n = names[j]
        tgt_d = r["target"][j] * r["total"]
        trade = tgt_d - r["holdings"][j]
        lines.append(f"{n:11}{r['current'][j] * 100:>8.0f}%{r['target'][j] * 100:>8.0f}%{tgt_d:>11,.0f}"
                     f"{('+' if trade >= 0 else '-') + '$' + format(abs(trade), ',.0f'):>11}"
                     f"{pct(r['value_now'][j]):>11}{pct(r['value_target'][j]):>15}")
    big = int(np.argmax(r["target"]))
    mixes = [("Current", r["current"], r["growth_now"], r["vol_now"], r["bonus_now"])]
    mixes += [(k, v["w"], v["growth"], v["vol"], v["bonus"]) for k, v in r["compare"].items()]
    mixes += [("Kelly target", r["target"], r["growth_target"], r["vol_target"], r["bonus_target"])]
    lines += ["", "MIXES COMPARED",
              f"{'Mix':16}" + "".join(f"{n[:9]:>10}" for n in names)
              + f"{'Growth':>9}{'Vol':>6}{'Rebal bonus':>13}"]
    for label, w, g, v, b in mixes:
        lines.append(f"{label[:16]:16}" + "".join(f"{x * 100:>9.0f}%" for x in w)
                     + f"{pct(g):>9}{v * 100:>5.0f}%{pct(b):>13}")
    lines += [
        "",
        f"ONE-BAD-BET CHECK: if {names[big]} (largest target, {r['target'][big] * 100:.0f}%) fell 50% overnight,"
        f" the sleeve would lose {r['target'][big] * 50:.0f}%.",
        "",
        "Value = expected return of the next dollar given the holdings. At the target, held assets",
        "have about equal value; assets at 0% have lower value (not worth adding); capped assets higher.",
    ]
    return "\n".join(lines) + "\n"
