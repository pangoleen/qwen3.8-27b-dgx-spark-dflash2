#!/usr/bin/env python3
"""
plotconc — the concurrency chart from data/concbench.csv: aggregate tok/s,
per-stream tok/s and median time to first token against concurrent streams,
one line per profile.

    python3 bench/plotconc.py data/concbench.csv --out charts/concurrency.png
"""
import argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(colors=INK2, labelsize=13, length=0)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=13)
    ax.set_xlabel("concurrent streams", color=INK2, fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out", default="charts/concurrency.png")
    ap.add_argument("--profiles", default="16-seat,32-seat", help="comma-separated profile names to draw")
    ap.add_argument("--title", default="Qwen3.8-27B on one DGX Spark: 1 to 32 concurrent requests")
    ap.add_argument("--sub", default="SGLang · NVFP4 weights · 4-bit DFlash2 drafter, budget 16 · distinct 2k-token code prefixes · 512 output tokens · temperature 0 · mean of 2 runs · one boot per profile")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    profiles = a.profiles.split(",")
    series = []
    for p in profiles:
        pr = sorted((r for r in rows if r["profile"] == p), key=lambda r: int(r["streams"]))
        series.append((p, pr))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), dpi=150, facecolor=SURF)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.13, wspace=0.28)
    fig.text(0.05, 0.93, a.title, fontsize=23, fontweight="bold", color=INK, ha="left")
    fig.text(0.05, 0.875, a.sub, fontsize=12.5, color=INK2, ha="left")
    panels = ((axes[0], "agg_tok_s", "Aggregate generation", "tokens / s, all streams", "{:,.0f}"),
              (axes[1], "per_stream_tok_s", "Per stream", "tokens / s, median stream", "{:,.0f}"),
              (axes[2], "ttft_med_s", "Time to first token", "seconds, median", "{:.2f}"))
    for ax, key, title, ylabel, fmt in panels:
        style(ax, title, ylabel); top = 0
        for si, (name, pr) in enumerate(series):
            xs = [int(r["streams"]) for r in pr]; ys = [float(r[key]) for r in pr]; top = max(top, max(ys))
            col = COLORS[si % len(COLORS)]
            label = f"{name} profile (MAX_RUNNING {pr[0]['max_running']}, mem {pr[0]['mem_fraction']})"
            ax.plot(xs, ys, color=col, linewidth=2.2, solid_joinstyle="round", zorder=3, label=label)
            ax.scatter(xs, ys, s=52, color=col, edgecolors=SURF, linewidths=2, zorder=4)
            marks = range(len(xs)) if (key == "agg_tok_s" and si == 0) else ((0, len(xs) - 1) if si == 0 else (len(xs) - 1,))
            for i in marks:
                ax.annotate(fmt.format(ys[i]), (xs[i], ys[i]), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.set_ylim(0, top * 1.22)
        ax.legend(frameon=False, fontsize=11, loc="upper left" if key != "per_stream_tok_s" else "upper right", labelcolor=INK2)
    fig.savefig(a.out, facecolor=SURF); print("wrote", a.out)


if __name__ == "__main__":
    main()
