"""
Fit the log-quadratic (accelerating growth) model, compare it with the plain
exponential, save a chart, and write results to a text file.

Usage: python optimize_intercept.py   (from any folder)
"""
import os
import platform
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (CURRENCY_NAME, CURRENCY_NAME_LOWER, FIXED_CURVATURE,
                    OUTPUT_FOLDER, SPIKE_CHECK_YEARS)
from model import (annual_growth_at, fair_price, fit, fit_exponential, format_price,
                   load_data, valuation, valuation_text)

HORIZONS = [1, 3, 5, 10, 15, 20]
BANDS = [(2, "#f4cccc"), (1, "#fce5cd")]


def spike_check(df):
    if not SPIKE_CHECK_YEARS:
        return None
    cutoff = df["Date"].iloc[-1] - pd.DateOffset(years=SPIKE_CHECK_YEARS)
    old = df[df["Date"] <= cutoff]
    if len(old) < 24 or (old["Date"].iloc[-1] - old["Date"].iloc[0]).days < 5 * 365:
        return None
    m = fit(old, FIXED_CURVATURE)
    return {"cutoff": cutoff, "model": m, "fair_today": float(fair_price(m, df["Years"].iloc[-1])),
            "growth_now": float(annual_growth_at(m, df["Years"].iloc[-1]))}


def make_chart(df, model, expo, v, chart_file):
    t = df["Years"].to_numpy()
    fair = fair_price(model, t)
    fair_exp = fair_price(expo, t)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, log_scale in zip(axes, [True, False]):
        for n, colour in BANDS:
            ax.fill_between(df["Date"], fair * np.exp(-n * model["sigma"]),
                            fair * np.exp(n * model["sigma"]), color=colour, label=f"±{n}σ band")
        ax.plot(df["Date"], df["Price"], color="blue", linewidth=1.2, label=f"{CURRENCY_NAME}")
        ax.plot(df["Date"], fair_exp, color="grey", linewidth=1.2,
                label=f"Plain exponential ({annual_growth_at(expo, 0):.2f}%/yr constant)")
        ax.plot(df["Date"], fair, color="black", linestyle="--", linewidth=2,
                label="Log-quadratic fair value")
        ax.scatter(df["Date"].iloc[-1], v["current"], color="red", s=80, zorder=10,
                   label=f"Current: {format_price(v['current'])}")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Level (log scale)")
        else:
            ax.set_ylim(0, max(df["Price"].max(), fair[-1] * np.exp(model["sigma"])) * 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
            ax.set_ylabel("Level (normal scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_title(
        f"{CURRENCY_NAME} — log-quadratic: growth {annual_growth_at(model, 0):.2f}%/yr at start → "
        f"{annual_growth_at(model, t[-1]):.2f}%/yr now, R² (log) {model['r2']:.4f}\n{valuation_text(v)}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(chart_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results(df, model, expo, v, spike, results_file):
    start = df["Date"].iloc[0]
    now = df["Years"].iloc[-1]
    v_exp = valuation(df, expo)
    lines = [
        f"{CURRENCY_NAME.upper()} LOG-QUADRATIC (ACCELERATING GROWTH) MODEL",
        "=" * 60,
        f"Analysis Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data: {start:%Y-%m-%d} to {df['Date'].iloc[-1]:%Y-%m-%d} ({len(df)} data points)",
        "",
        "MODEL FORMULA",
        "-" * 20,
        f"log(level) = {model['a']:.4f} + {model['b']:.6f}·t + {model['q']:.8f}·t²   (t = years since {start:%Y-%m-%d})",
        f"Curvature q: {model['q']:.8f} " + ("(fixed in config)" if FIXED_CURVATURE is not None else "(fitted)")
        + ("  -> growth speeds up" if model["q"] > 0 else "  -> growth slows down"),
        "",
        "GROWTH RATE OVER TIME",
        "-" * 30,
        f"  At start ({start:%Y}): {annual_growth_at(model, 0):.2f}%/yr",
        f"  Now:              {annual_growth_at(model, now):.2f}%/yr",
        f"  In 10 years:      {annual_growth_at(model, now + 10):.2f}%/yr",
        f"  In 20 years:      {annual_growth_at(model, now + 20):.2f}%/yr",
        "",
        "COMPARISON WITH PLAIN EXPONENTIAL (constant growth)",
        "-" * 30,
        f"  {'':22}{'Exponential':>14}{'Log-quadratic':>16}",
        f"  {'Growth now':22}{annual_growth_at(expo, now):>13.2f}%{annual_growth_at(model, now):>15.2f}%",
        f"  {'R² (log)':22}{expo['r2']:>14.6f}{model['r2']:>16.6f}",
        f"  {'Adjusted R² (log)':22}{expo['adj_r2']:>14.6f}{model['adj_r2']:>16.6f}",
        f"  {'Fair value today':22}{format_price(v_exp['fair']):>14}{format_price(v['fair']):>16}",
        f"  {'Valuation':22}{v_exp['pct']:>+13.2f}%{v['pct']:>+15.2f}%",
        "  (Adjusted R² penalises the extra parameter; similar values mean the curve adds little.)",
        "",
        "CURRENT ANALYSIS",
        "-" * 20,
        f"Current level: {format_price(v['current'])}",
        f"Model fair value: {format_price(v['fair'])}",
        valuation_text(v),
        "",
    ]
    if spike:
        lines += [
            f"SPIKE CHECK (refit without the last {SPIKE_CHECK_YEARS} years)",
            "-" * 30,
            f"Fit through {spike['cutoff']:%Y-%m-%d}: growth now {spike['growth_now']:.2f}%/yr, "
            f"fair value today {format_price(spike['fair_today'])}",
            "",
        ]
    lines += ["FORWARD PROJECTIONS (fair value path)", "-" * 25,
              f"  {'':8}{'Log-quadratic':>16}{'Exponential':>16}"]
    for years in HORIZONS:
        lq = float(fair_price(model, now + years))
        ex = float(fair_price(expo, now + years))
        lines.append(f"  {years:>2}-Year {format_price(lq):>16}{format_price(ex):>16}"
                     f"   (log-quadratic from fair value: {(lq / v['fair'] - 1) * 100:+.1f}%)")
    lines += [
        "",
        "NOTES",
        "-" * 10,
        "- Accelerating curves compound fast: long projections are very sensitive to q.",
        "- Levels are a dividends-reinvested index, not the quoted S&P 500 price.",
    ]
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"{CURRENCY_NAME} log-quadratic model")
    print("=" * 40)
    df = load_data()
    model = fit(df, FIXED_CURVATURE)
    expo = fit_exponential(df)
    v = valuation(df, model)
    spike = spike_check(df)
    now = df["Years"].iloc[-1]

    chart_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_log_quadratic_fit.png"
    results_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_log_quadratic_results.txt"
    make_chart(df, model, expo, v, chart_file)
    write_results(df, model, expo, v, spike, results_file)

    print(f"Growth: {annual_growth_at(model, 0):.2f}%/yr at start -> {annual_growth_at(model, now):.2f}%/yr now "
          f"-> {annual_growth_at(model, now + 20):.2f}%/yr in 20 years")
    print(f"R² (log): {model['r2']:.6f} vs exponential {expo['r2']:.6f}  "
          f"(adjusted: {model['adj_r2']:.6f} vs {expo['adj_r2']:.6f})")
    print(f"Fair value: {format_price(v['fair'])}  |  {valuation_text(v)}")
    if spike:
        print(f"Spike check (fit through {spike['cutoff']:%Y-%m-%d}): growth now {spike['growth_now']:.2f}%/yr")
    print(f"Chart: {chart_file}")
    print(f"Results: {results_file}")

    if platform.system() == "Windows" and not os.environ.get("NO_OPEN_CHART"):
        try:
            os.startfile(chart_file)
        except OSError as e:
            print(f"Could not auto-open chart: {e}")


if __name__ == "__main__":
    main()
