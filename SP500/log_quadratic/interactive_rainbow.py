"""
Interactive log-quadratic chart.

The slider sets the curvature q (how fast growth speeds up or slows down);
for every q the rest of the curve is refitted automatically on a log scale.
q = 0 is the plain exponential. The automatic best fit stays on the chart as
a grey reference line.

Usage: python interactive_rainbow.py   (from any folder)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from config import CURRENCY_NAME, FIXED_CURVATURE
from model import (annual_growth_at, fair_price, fit, format_price, load_data,
                   valuation, valuation_text)


def main():
    df = load_data()
    t = df["Years"].to_numpy()
    auto = fit(df)
    start_q = FIXED_CURVATURE if FIXED_CURVATURE is not None else auto["q"]
    print(f"Automatic fit: q = {auto['q']:.8f}, growth now {annual_growth_at(auto, t[-1]):.2f}%/yr")

    fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.12, top=0.88)
    lines, fills = [], []
    for ax in (ax_log, ax_lin):
        ax.plot(df["Date"], df["Price"], color="blue", linewidth=1.2, label=CURRENCY_NAME)
        ax.plot(df["Date"], fair_price(auto, t), color="grey", linewidth=1, label="Automatic fit")
        (line,) = ax.plot(df["Date"], fair_price(auto, t), color="black", linestyle="--",
                          linewidth=2, label="Slider model")
        lines.append((ax, line))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("Level (log scale)")
    ax_lin.set_ylabel("Level (normal scale)")
    ax_lin.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    info = fig.text(0.5, 0.94, "", ha="center", fontsize=11,
                    bbox=dict(facecolor="lightgray", alpha=0.8, boxstyle="round,pad=0.3"))

    span = max(abs(auto["q"]) * 4, 0.0005)
    slider = Slider(plt.axes([0.12, 0.03, 0.76, 0.03]), "Curvature q",
                    auto["q"] - span, auto["q"] + span, valinit=start_q,
                    valfmt="%.6f", valstep=span / 5000)

    def update(q):
        model = fit(df, q)
        fair = fair_price(model, t)
        for fill in fills:
            fill.remove()
        fills.clear()
        for ax, line in lines:
            line.set_ydata(fair)
            fills.append(ax.fill_between(df["Date"], fair * np.exp(-model["sigma"]),
                                         fair * np.exp(model["sigma"]), color="#fce5cd",
                                         alpha=0.7, zorder=0))
        ax_lin.set_ylim(0, max(df["Price"].max(), fair[-1] * np.exp(model["sigma"])) * 1.1)
        v = valuation(df, model)
        info.set_text(f"q {q:.6f} | growth {annual_growth_at(model, 0):.2f}% → {annual_growth_at(model, t[-1]):.2f}%/yr | "
                      f"R² (log) {model['r2']:.4f} | Fair {format_price(v['fair'])} | {valuation_text(v)}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(start_q)
    ax_log.set_title(f"{CURRENCY_NAME} — log-quadratic (q > 0: growth speeds up, q < 0: slows down, ±1σ shaded)",
                     fontsize=12, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
