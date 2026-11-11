"""
Fit the unpinned exponential model, save a chart with sigma bands, and write
results (fair value, valuation, spike check, projections) to a text file.

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

from config import (CURRENCY_NAME, CURRENCY_NAME_LOWER, FIXED_GROWTH_RATE,
                    OUTPUT_FOLDER, SPIKE_CHECK_YEARS, START_DATE)
from model import (annual_pct, fair_price, fit, format_price, load_data,
                   valuation, valuation_text)

HORIZONS = [1, 3, 5, 10, 15, 20]
BANDS = [(2, "#f4cccc"), (1, "#fce5cd")]  # (sigma multiple, fill colour)


def spike_check(df, model):
    """Refit without the last SPIKE_CHECK_YEARS years and compare today's fair value."""
    if not SPIKE_CHECK_YEARS:
        return None
    cutoff = df["Date"].iloc[-1] - pd.DateOffset(years=SPIKE_CHECK_YEARS)
    df_old = df[df["Date"] <= cutoff]
    if len(df_old) < 24 or (df_old["Date"].iloc[-1] - df_old["Date"].iloc[0]).days < 5 * 365:
        return None
    m_old = fit(df_old, FIXED_GROWTH_RATE)
    fair_today = float(fair_price(m_old, df["Days"].iloc[-1]))
    return {"cutoff": cutoff, "model": m_old, "fair_today": fair_today}


def make_chart(df, model, v, chart_file):
    days = df["Days"].to_numpy()
    fair = fair_price(model, days)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, log_scale in zip(axes, [True, False]):
        for k, colour in BANDS:
            ax.fill_between(df["Date"], fair * np.exp(-k * model["sigma"]),
                            fair * np.exp(k * model["sigma"]), color=colour,
                            label=f"±{k}σ band")
        ax.plot(df["Date"], df["Price"], color="blue", linewidth=1.5,
                label=f"{CURRENCY_NAME} price")
        ax.plot(df["Date"], fair, color="black", linestyle="--", linewidth=2,
                label="Fair value (unpinned fit)")
        ax.scatter(df["Date"].iloc[-1], v["current"], color="red", s=80, zorder=10,
                   label=f"Current: {format_price(v['current'])}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(f"Price (USD) - {'log' if log_scale else 'normal'} scale")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_title(
        f"{CURRENCY_NAME} unpinned exponential fit — {annual_pct(model['g']):.2f}%/yr, "
        f"R² (log) {model['r2']:.4f}\n{valuation_text(v)}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(chart_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results(df, model, v, spike, results_file):
    start = df["Date"].iloc[0]
    last_day = df["Days"].iloc[-1]
    a0 = float(fair_price(model, 0))
    lines = [
        f"{CURRENCY_NAME.upper()} UNPINNED EXPONENTIAL MODEL",
        "=" * 50,
        f"Analysis Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data: {start:%Y-%m-%d} to {df['Date'].iloc[-1]:%Y-%m-%d} ({len(df)} data points)",
        "",
        "FIT (log scale, growth and level fitted together)",
        "-" * 30,
        f"Growth rate: {model['g']:.6f} per day "
        + ("(fixed in config)" if FIXED_GROWTH_RATE is not None else "(fitted)"),
        f"Annual growth (compounded): {annual_pct(model['g']):.2f}% per year",
        f"R² (log): {model['r2']:.6f}",
        f"Sigma (typical log deviation): {model['sigma']:.4f} "
        f"(1σ band = x{np.exp(-model['sigma']):.2f} to x{np.exp(model['sigma']):.2f} of fair value)",
        "",
        "MODEL FORMULA",
        "-" * 20,
        f"fair price = {a0:,.2f} × e^({model['g']:.6f} × days since {start:%Y-%m-%d})",
        "",
        "CURRENT ANALYSIS",
        "-" * 20,
        f"Current Price: {format_price(v['current'])}",
        f"Model Fair Price: {format_price(v['fair'])}",
        valuation_text(v),
        f"Bands today: -2σ {format_price(v['fair'] * np.exp(-2 * model['sigma']))} | "
        f"-1σ {format_price(v['fair'] * np.exp(-model['sigma']))} | "
        f"+1σ {format_price(v['fair'] * np.exp(model['sigma']))} | "
        f"+2σ {format_price(v['fair'] * np.exp(2 * model['sigma']))}",
        "",
    ]
    if spike:
        m = spike["model"]
        diff = (v["fair"] - spike["fair_today"]) / spike["fair_today"] * 100
        lines += [
            f"SPIKE CHECK (refit without the last {SPIKE_CHECK_YEARS} years)",
            "-" * 30,
            f"Fit through {spike['cutoff']:%Y-%m-%d}: {annual_pct(m['g']):.2f}%/yr, "
            f"fair value today {format_price(spike['fair_today'])}",
            f"Full fit: {annual_pct(model['g']):.2f}%/yr, fair value today {format_price(v['fair'])} "
            f"({diff:+.2f}% from recent data)",
            "",
        ]
    lines += ["FORWARD PROJECTIONS (fair value path)", "-" * 25]
    for years in HORIZONS:
        future = float(fair_price(model, last_day + years * 365.25))
        lines.append(
            f"  {years:>2}-Year: {format_price(future)} | from fair value: "
            f"{(future / v['fair'] - 1) * 100:+.2f}% | from current price: "
            f"{(future / v['current'] - 1) * 100:+.2f}%")
    lines += [
        "",
        "NOTES",
        "-" * 10,
        "- Unlike the pinned exponential model, the curve is not pinned to the start-date price.",
        "- R² (log) is not comparable with dollar-scale R² from older results.",
        "- σ bands show typical swings around trend; they are not price targets.",
    ]
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"{CURRENCY_NAME} unpinned exponential model")
    print("=" * 40)
    df = load_data()
    print(f"Data: {df['Date'].iloc[0]:%Y-%m-%d} to {df['Date'].iloc[-1]:%Y-%m-%d} "
          f"({len(df)} data points, start date {'from config' if START_DATE else 'auto'})")
    model = fit(df, FIXED_GROWTH_RATE)
    v = valuation(df, model)
    spike = spike_check(df, model)

    chart_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_unpinned_fit.png"
    results_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_unpinned_results.txt"
    make_chart(df, model, v, chart_file)
    write_results(df, model, v, spike, results_file)

    print(f"Growth rate: {model['g']:.6f} per day ({annual_pct(model['g']):.2f}%/yr compounded)")
    print(f"R² (log): {model['r2']:.6f}   sigma: {model['sigma']:.4f}")
    print(f"Current Price: {format_price(v['current'])}")
    print(f"Model Fair Price: {format_price(v['fair'])}")
    print(valuation_text(v))
    if spike:
        print(f"Spike check (fit through {spike['cutoff']:%Y-%m-%d}): "
              f"{annual_pct(spike['model']['g']):.2f}%/yr, fair today {format_price(spike['fair_today'])}")
    print(f"Chart: {chart_file}")
    print(f"Results: {results_file}")

    if platform.system() == "Windows" and not os.environ.get("NO_OPEN_CHART"):
        try:
            os.startfile(chart_file)
        except OSError as e:
            print(f"Could not auto-open chart: {e}")


if __name__ == "__main__":
    main()
