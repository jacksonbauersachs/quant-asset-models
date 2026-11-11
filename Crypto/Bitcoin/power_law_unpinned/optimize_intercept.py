"""
Fit the unpinned power-law model, save a chart with sigma bands, and write
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

from config import (CURRENCY_NAME, CURRENCY_NAME_LOWER, FIXED_EXPONENT,
                    OUTPUT_FOLDER, SPIKE_CHECK_YEARS)
from model import (annual_growth_at, fair_price, fit, format_price, load_data,
                   valuation, valuation_text)

HORIZONS = [1, 3, 5, 10, 15, 20]
BANDS = [(2, "#f4cccc"), (1, "#fce5cd")]  # (sigma multiple, fill colour)


def spike_check(df):
    """Refit without the last SPIKE_CHECK_YEARS years and compare today's fair value."""
    if not SPIKE_CHECK_YEARS:
        return None
    cutoff = df["Date"].iloc[-1] - pd.DateOffset(years=SPIKE_CHECK_YEARS)
    df_old = df[df["Date"] <= cutoff]
    if len(df_old) < 24 or (df_old["Date"].iloc[-1] - df_old["Date"].iloc[0]).days < 5 * 365:
        return None
    m_old = fit(df_old, FIXED_EXPONENT)
    return {"cutoff": cutoff, "model": m_old,
            "fair_today": float(fair_price(m_old, df["Days"].iloc[-1]))}


def make_chart(df, model, v, chart_file):
    days = df["Days"].to_numpy()
    fair = fair_price(model, days)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = df["Date"]
    for ax, log_scale in zip(axes, [True, False]):
        for n, colour in BANDS:
            ax.fill_between(x, fair * np.exp(-n * model["sigma"]),
                            fair * np.exp(n * model["sigma"]), color=colour, label=f"±{n}σ band")
        ax.plot(x, df["Price"], color="blue", linewidth=1.2, label=f"{CURRENCY_NAME} price")
        ax.plot(x, fair, color="black", linestyle="--", linewidth=2, label="Fair value (power law)")
        ax.scatter(x.iloc[-1], v["current"], color="red", s=80,
                   zorder=10, label=f"Current: {format_price(v['current'])}")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Price (USD, log scale)")
        else:
            # Dollar scale: cap the view so the +2σ band doesn't squash the price line
            ax.set_ylim(0, max(df["Price"].max(), fair[-1] * np.exp(model["sigma"])) * 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"${val:,.0f}"))
            ax.set_ylabel("Price (USD, dollar scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_title(
        f"{CURRENCY_NAME} unpinned power law — price ∝ days^{model['k']:.3f}, R² (log) {model['r2']:.4f}\n"
        f"{valuation_text(v)}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(chart_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results(df, model, v, spike, results_file):
    genesis = df.attrs["genesis"]
    last_day = df["Days"].iloc[-1]
    lines = [
        f"{CURRENCY_NAME.upper()} UNPINNED POWER-LAW MODEL",
        "=" * 50,
        f"Analysis Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Day 0 (genesis): {genesis:%Y-%m-%d}",
        f"Fit data: {df['Date'].iloc[0]:%Y-%m-%d} to {df['Date'].iloc[-1]:%Y-%m-%d} ({len(df)} data points)",
        "",
        "FIT (log scale, exponent and level fitted together)",
        "-" * 30,
        f"Exponent k: {model['k']:.4f} " + ("(fixed in config)" if FIXED_EXPONENT is not None else "(fitted)"),
        f"  = log_model_v2-style slope {model['k'] / np.log(10):.4f} (log10 price per ln day)",
        f"R² (log): {model['r2']:.6f}",
        f"Sigma (typical log deviation): {model['sigma']:.4f} "
        f"(1σ band = x{np.exp(-model['sigma']):.2f} to x{np.exp(model['sigma']):.2f} of fair value)",
        "",
        "MODEL FORMULA",
        "-" * 20,
        f"fair price = {np.exp(model['c']):.4e} × days^{model['k']:.4f}   (days since {genesis:%Y-%m-%d})",
        "",
        "GROWTH (slows as the asset ages)",
        "-" * 30,
    ]
    for years in [0, 5, 10, 20]:
        lines.append(f"  Annual growth {'now' if years == 0 else f'in {years} years':>12}: "
                     f"{annual_growth_at(model, last_day + years * 365.25):.1f}%")
    lines += [
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
            f"Fit through {spike['cutoff']:%Y-%m-%d}: k={m['k']:.4f}, fair value today {format_price(spike['fair_today'])}",
            f"Full fit: k={model['k']:.4f}, fair value today {format_price(v['fair'])} ({diff:+.2f}% from recent data)",
            "",
        ]
    lines += ["FORWARD PROJECTIONS (fair value path)", "-" * 25]
    for years in HORIZONS:
        future = float(fair_price(model, last_day + years * 365.25))
        lines.append(f"  {years:>2}-Year: {format_price(future)} | from fair value: "
                     f"{(future / v['fair'] - 1) * 100:+.2f}% | from current price: "
                     f"{(future / v['current'] - 1) * 100:+.2f}%")
    lines += [
        "",
        "NOTES",
        "-" * 10,
        "- A power law depends on day 0 (GENESIS_DATE); keep it fixed between runs.",
        "- R² (log) is comparable with log_model_v2's R², which is also log-scale.",
        "- σ bands show typical swings around trend; they are not price targets.",
    ]
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"{CURRENCY_NAME} unpinned power-law model")
    print("=" * 40)
    df = load_data()
    print(f"Day 0: {df.attrs['genesis']:%Y-%m-%d} | fit data {df['Date'].iloc[0]:%Y-%m-%d} "
          f"to {df['Date'].iloc[-1]:%Y-%m-%d} ({len(df)} data points)")
    model = fit(df, FIXED_EXPONENT)
    v = valuation(df, model)
    spike = spike_check(df)

    chart_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_power_law_fit.png"
    results_file = OUTPUT_FOLDER / f"{CURRENCY_NAME_LOWER}_power_law_results.txt"
    make_chart(df, model, v, chart_file)
    write_results(df, model, v, spike, results_file)

    print(f"Exponent k: {model['k']:.4f} (log_model_v2-style slope {model['k'] / np.log(10):.4f})")
    print(f"R² (log): {model['r2']:.6f}   sigma: {model['sigma']:.4f}")
    print(f"Annual growth now: {annual_growth_at(model, df['Days'].iloc[-1]):.1f}% (slowing over time)")
    print(f"Current Price: {format_price(v['current'])}")
    print(f"Model Fair Price: {format_price(v['fair'])}")
    print(valuation_text(v))
    if spike:
        print(f"Spike check (fit through {spike['cutoff']:%Y-%m-%d}): k={spike['model']['k']:.4f}, "
              f"fair today {format_price(spike['fair_today'])}")
    print(f"Chart: {chart_file}")
    print(f"Results: {results_file}")

    if platform.system() == "Windows" and not os.environ.get("NO_OPEN_CHART"):
        try:
            os.startfile(chart_file)
        except OSError as e:
            print(f"Could not auto-open chart: {e}")


if __name__ == "__main__":
    main()
