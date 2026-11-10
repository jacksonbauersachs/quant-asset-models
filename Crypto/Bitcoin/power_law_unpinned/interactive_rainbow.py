"""
Interactive unpinned power-law chart.

The slider sets the exponent k; for every k the level is fitted
automatically on a log scale, so the curve is never pinned to one day's
price. The slider starts at the automatic best fit, which stays on the chart
as a grey reference line.

Usage: python interactive_rainbow.py   (from any folder)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from config import CURRENCY_NAME, FIXED_EXPONENT
from model import (annual_growth_at, fair_price, fit, format_price, load_data,
                   valuation, valuation_text)


def main():
    df = load_data()
    days = df["Days"].to_numpy()
    auto = fit(df)
    start_k = FIXED_EXPONENT if FIXED_EXPONENT is not None else auto["k"]
    print(f"Automatic fit: k = {auto['k']:.4f}, R² (log) {auto['r2']:.4f}")

    fig, (ax_time, ax_loglog) = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.12, top=0.88)
    lines, fills = [], []
    for ax, x in [(ax_time, df["Date"]), (ax_loglog, days)]:
        ax.plot(x, df["Price"], color="blue", linewidth=1.2, label=f"{CURRENCY_NAME} price")
        ax.plot(x, fair_price(auto, days), color="grey", linewidth=1,
                label=f"Automatic fit (k = {auto['k']:.3f})")
        (line,) = ax.plot(x, fair_price(auto, days), color="black", linestyle="--",
                          linewidth=2, label="Slider model")
        lines.append((ax, x, line))
        ax.set_yscale("log")
        ax.set_ylabel("Price (USD, log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    ax_loglog.set_xscale("log")
    ax_loglog.set_xlabel(f"Days since {df.attrs['genesis']:%Y-%m-%d} (log scale)")
    info = fig.text(0.5, 0.94, "", ha="center", fontsize=11,
                    bbox=dict(facecolor="lightgray", alpha=0.8, boxstyle="round,pad=0.3"))

    span = max(abs(auto["k"]) * 0.5, 0.5)
    slider = Slider(plt.axes([0.12, 0.03, 0.76, 0.03]), "Exponent k",
                    auto["k"] - span, auto["k"] + span, valinit=start_k,
                    valfmt="%.4f", valstep=span / 5000)

    def update(k):
        model = fit(df, k)
        fair = fair_price(model, days)
        for fill in fills:
            fill.remove()
        fills.clear()
        for ax, x, line in lines:
            line.set_ydata(fair)
            fills.append(ax.fill_between(x, fair * np.exp(-model["sigma"]),
                                         fair * np.exp(model["sigma"]), color="#fce5cd",
                                         alpha=0.7, zorder=0))
        v = valuation(df, model)
        info.set_text(f"k {k:.4f} | R² (log) {model['r2']:.4f} | growth now "
                      f"{annual_growth_at(model, days[-1]):.1f}%/yr | Fair {format_price(v['fair'])} | "
                      f"{valuation_text(v)}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(start_k)
    ax_time.set_title(f"{CURRENCY_NAME} — unpinned power law (level auto-fitted for each k, ±1σ shaded)",
                      fontsize=12, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
