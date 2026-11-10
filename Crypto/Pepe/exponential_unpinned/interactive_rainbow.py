"""
Interactive unpinned exponential chart.

The slider sets the growth rate; for every slope the level (intercept) is
fitted automatically on a log scale, so the curve is never pinned to one
day's price. The slider starts at the automatic best fit, which stays on the
chart as a grey reference line.

Usage: python interactive_rainbow.py   (from any folder)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from config import CURRENCY_NAME, FIXED_GROWTH_RATE
from model import (annual_pct, fair_price, fit, format_price, load_data,
                   valuation, valuation_text)


def main():
    df = load_data()
    days = df["Days"].to_numpy()
    auto = fit(df)
    start_g = FIXED_GROWTH_RATE if FIXED_GROWTH_RATE is not None else auto["g"]
    print(f"Automatic fit: {auto['g']:.6f} per day ({annual_pct(auto['g']):.2f}%/yr), "
          f"R² (log) {auto['r2']:.4f}")

    fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.12, top=0.88)
    lines, fills = [], []
    for ax, log_scale in [(ax_log, True), (ax_lin, False)]:
        ax.plot(df["Date"], df["Price"], color="blue", linewidth=1.5, label=f"{CURRENCY_NAME} price")
        ax.plot(df["Date"], fair_price(auto, days), color="grey", linewidth=1,
                label=f"Automatic fit ({annual_pct(auto['g']):.2f}%/yr)")
        (line,) = ax.plot(df["Date"], fair_price(auto, days), color="black",
                          linestyle="--", linewidth=2, label="Slider model")
        lines.append(line)
        ax.scatter(df["Date"].iloc[-1], df["Price"].iloc[-1], color="red", s=80, zorder=10,
                   label=f"Current: {format_price(df['Price'].iloc[-1])}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(f"Price (USD) - {'log' if log_scale else 'normal'} scale")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    info = fig.text(0.5, 0.94, "", ha="center", fontsize=11,
                    bbox=dict(facecolor="lightgray", alpha=0.8, boxstyle="round,pad=0.3"))

    span = max(abs(auto["g"]) * 2, 0.0002)
    slider = Slider(plt.axes([0.12, 0.03, 0.76, 0.03]), "Growth rate (per day)",
                    auto["g"] - span, auto["g"] + span, valinit=start_g,
                    valfmt="%.6f", valstep=span / 5000)

    def update(g):
        model = fit(df, g)
        fair = fair_price(model, days)
        for fill in fills:
            fill.remove()
        fills.clear()
        for ax, line in zip([ax_log, ax_lin], lines):
            line.set_ydata(fair)
            fills.append(ax.fill_between(df["Date"], fair * np.exp(-model["sigma"]),
                                         fair * np.exp(model["sigma"]), color="#fce5cd",
                                         alpha=0.7, zorder=0))
        v = valuation(df, model)
        info.set_text(f"Growth {g:.6f}/day ({annual_pct(g):.2f}%/yr) | R² (log) {model['r2']:.4f} | "
                      f"Fair {format_price(v['fair'])} | {valuation_text(v)}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(start_g)
    ax_log.set_title(f"{CURRENCY_NAME} — unpinned exponential (level auto-fitted for each slope, ±1σ shaded)",
                     fontsize=12, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
