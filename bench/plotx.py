#!/usr/bin/env python3
"""
plotx — regenerate every chart in charts/ from a CSV in data/.

Three modes:

  --panels all       nine panels from one sweep, temperature 1 overlaid on the
                     generation panel when the file carries those rows
  --panels three     the same sweep as three panels, sized for a feed
  --panels compare   one panel per measure, one line per `config` value

Input is a CSV from data/ (default) or a raw ctxsweep jsonl (--jsonl). The CSV
is the source of truth; if a chart cannot be rebuilt from data/, it does not
ship.

Usage:
    python3 bench/plotx.py data/ctxsweep-recommended.csv \\
        --panels all --out charts/ctxsweep-27b.png
    python3 bench/plotx.py data/ctxsweep-three-recipes.csv \\
        --panels compare --out charts/three-recipes.png
    python3 bench/plotx.py results/ctxsweep-mine-*.jsonl --jsonl --out mine.png
"""
import argparse, csv, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#3f8f6b"      # categorical slots 1, 2, 3

NUMERIC = ("temperature", "ctx_target", "prompt_tokens", "prefill_tok_s",
           "gen_tok_s", "accepted_per_pass", "saturation", "cold_ttft_s",
           "warm_ttft_s", "prefill_bytes_s", "gen_bytes_s", "out_tokens",
           "reps", "boot")


def ktok(v, _=None):
    return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def load_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            for k in NUMERIC:
                if k in r:
                    r[k] = float(r[k]) if r[k] not in ("", None) else None
            r["_i"] = i                 # file order, so `compare` keeps it
            rows.append(r)
    rows.sort(key=lambda r: r["prompt_tokens"])
    return rows


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:                      # jsonl uses the engine's own field name
        r.setdefault("accepted_per_pass", r.get("tok_per_pass"))
        r.setdefault("config", r.get("label", "run"))
        r.setdefault("temperature", 0.0)
    rows.sort(key=lambda r: r["prompt_tokens"])
    return rows


def style(ax, title, ylabel, xlabel="prompt tokens"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(ktok))
    ax.tick_params(colors=INK2, labelsize=13, length=0)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=13)
    ax.set_xlabel(xlabel, color=INK2, fontsize=13)


def line(ax, xs, ys, color, label=None, mark=()):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=2.2, solid_joinstyle="round", label=label, zorder=3)
    ax.scatter(xs, ys, s=52, color=color, edgecolors=SURF, linewidths=2, zorder=4)
    for i in mark:
        if i >= len(ys):
            continue
        ax.annotate(f"{ys[i]:,.0f}" if ys[i] >= 10 else f"{ys[i]:.1f}", (xs[i], ys[i]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=13, color=INK, fontweight="bold")


def series(rows, field):
    return [r["prompt_tokens"] for r in rows], [r.get(field) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default="charts/sweep.png")
    ap.add_argument("--panels", choices=("three", "all", "compare"), default="all")
    ap.add_argument("--jsonl", action="store_true", help="input is a ctxsweep jsonl, not a CSV")
    ap.add_argument("--csv", action="store_true", help="input is a CSV (the default)")
    ap.add_argument("--t1", default=None,
                    help="second file for the temperature-1 overlay; by default the "
                         "temperature-1 rows of the input file are used")
    ap.add_argument("--title", default="Qwen3.8-27B on one DGX Spark, 320 tokens to 260k of context")
    ap.add_argument("--sub", default="SGLang · NVFP4 weights · NVFP4 speculative drafter, budget 16 · "
                                     "512 output tokens · generation = median of 4 runs per rung, "
                                     "prefill and cold TTFT = the one cold pass · temperature 0 · one boot · 262,144-token window; the top rung is a 257k prompt plus 512 output tokens")
    a = ap.parse_args()

    f = sorted(glob.glob(a.file))[-1] if glob.glob(a.file) else a.file
    load = load_jsonl if (a.jsonl or (f.endswith(".jsonl") and not a.csv)) else load_csv
    allrows = load(f)

    if a.panels == "compare":
        return compare(a, allrows)

    rows = [r for r in allrows if (r.get("temperature") or 0) == 0] or allrows
    # one boot per line: the first boot in the file (boot 2 rows stay in the CSV for the diff)
    boots = sorted({int(r.get("boot") or 1) for r in rows})
    if len(boots) > 1:
        rows = [r for r in rows if int(r.get("boot") or 1) == boots[0]]
    t1 = None
    if a.t1:
        t1rows = []
        for tf in sorted(glob.glob(a.t1)):
            t1rows += load(tf)
        t1 = ([r["prompt_tokens"] for r in t1rows], [r["gen_tok_s"] for r in t1rows])
    else:
        t1rows = [r for r in allrows if (r.get("temperature") or 0) == 1]
        if t1rows:
            t1 = ([r["prompt_tokens"] for r in t1rows], [r["gen_tok_s"] for r in t1rows])

    xs = [r["prompt_tokens"] for r in rows]
    gen = [r["gen_tok_s"] for r in rows]; pre = [r["prefill_tok_s"] for r in rows]
    cold = [r["cold_ttft_s"] for r in rows]; warm = [r["warm_ttft_s"] for r in rows]
    n = len(xs); last = n - 1; i65 = min(range(n), key=lambda i: abs(xs[i] - 65536))

    pk = max(range(n), key=lambda i: pre[i])
    tpot = [1000.0 / g for g in gen]
    tpp = [r["accepted_per_pass"] for r in rows]
    sat = [100.0 * (r.get("saturation") or 0) for r in rows]
    ratio = [c / w for c, w in zip(cold, warm)]
    pkb = [r["prefill_bytes_s"] / 1000.0 for r in rows]
    gbs = [r["gen_bytes_s"] for r in rows]

    if a.panels == "all":
        fig, axes = plt.subplots(3, 3, figsize=(18, 15.5), dpi=150, facecolor=SURF)
        fig.subplots_adjust(left=0.05, right=0.985, top=0.885, bottom=0.05, wspace=0.28, hspace=0.42)
        fig.text(0.05, 0.965, a.title, fontsize=24, fontweight="bold", color=INK, ha="left")
        parts = a.sub.split(" · ")
        fig.text(0.05, 0.943, " · ".join(parts[:4]), fontsize=13, color=INK2, ha="left")
        fig.text(0.05, 0.925, " · ".join(parts[4:]), fontsize=13, color=INK2, ha="left")
        A = axes.flat
        ax = A[0]; style(ax, "Generation", "tokens / s")
        line(ax, xs, gen, S1, label="temperature 0", mark=(0, i65, last)); ax.set_ylim(0, max(gen) * 1.18)
        if t1:
            line(ax, t1[0], t1[1], S2, label="temperature 1")
            ax.legend(frameon=False, fontsize=12, loc="lower left", labelcolor=INK2)
        ax = A[1]; style(ax, "Prompt processing", "tokens / s"); line(ax, xs, pre, S1, mark=(pk, last)); ax.set_ylim(0, max(pre) * 1.18)
        ax = A[2]; style(ax, "Time to first token", "seconds, log scale"); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        line(ax, xs, cold, S1, label="cold prompt", mark=(last,))
        line(ax, xs, warm, S2, label="same prompt again (prefix cache)", mark=(last,))
        ax.legend(frameon=False, fontsize=12, loc="upper left", labelcolor=INK2)
        ax = A[3]; style(ax, "Time per output token", "milliseconds"); line(ax, xs, tpot, S1, mark=(0, last)); ax.set_ylim(0, max(tpot) * 1.18)
        ax = A[4]; style(ax, "Accepted tokens per verify pass", "tokens (budget 16)"); line(ax, xs, tpp, S1, mark=(0, last)); ax.set_ylim(0, 16)
        ax = A[5]; style(ax, "Draft budget used", "% of 16"); line(ax, xs, sat, S1, mark=(0, last)); ax.set_ylim(0, 100)
        ax = A[6]; style(ax, "Prompt throughput", "KB / s"); line(ax, xs, pkb, S1, mark=(pk, last)); ax.set_ylim(0, max(pkb) * 1.18)
        ax = A[7]; style(ax, "Generation throughput", "bytes / s"); line(ax, xs, gbs, S1, mark=(0, last)); ax.set_ylim(0, max(gbs) * 1.18)
        ax = A[8]; style(ax, "Cold ÷ warm time to first token", "×, log scale"); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); line(ax, xs, ratio, S1, mark=(i65, last))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), dpi=150, facecolor=SURF)
        fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.13, wspace=0.28)
        fig.text(0.05, 0.93, a.title, fontsize=23, fontweight="bold", color=INK, ha="left")
        fig.text(0.05, 0.875, a.sub, fontsize=12.5, color=INK2, ha="left")
        ax = axes[0]; style(ax, "Generation", "tokens / s")
        line(ax, xs, gen, S1, label="temperature 0", mark=(0, i65, last)); ax.set_ylim(0, max(gen) * 1.18)
        if t1:
            line(ax, t1[0], t1[1], S2, label="temperature 1")
            ax.legend(frameon=False, fontsize=12, loc="lower left", labelcolor=INK2)
        ax = axes[1]; style(ax, "Prompt processing", "tokens / s"); line(ax, xs, pre, S1, mark=(pk, last)); ax.set_ylim(0, max(pre) * 1.18)
        ax = axes[2]; style(ax, "Time to first token", "seconds, log scale"); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        line(ax, xs, cold, S1, label="cold prompt", mark=(last,))
        line(ax, xs, warm, S2, label="same prompt again (prefix cache)", mark=(last,))
        ax.legend(frameon=False, fontsize=12, loc="upper left", labelcolor=INK2)

    fig.savefig(a.out, facecolor=SURF)
    print("wrote", a.out)


def compare(a, allrows):
    """One line per `config` value: generation, prompt processing, acceptance."""
    groups = {}
    for r in allrows:
        groups.setdefault(r.get("config", "run"), []).append(r)
    order = sorted(groups, key=lambda c: min(r.get("_i", 0) for r in groups[c]))
    colors = [S1, S2, S3, INK2]
    title = a.title if a.title != ap_default_title() else "Three recipes on one DGX Spark, same box, same prompts"
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6), dpi=150, facecolor=SURF)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.14, wspace=0.26)
    fig.text(0.05, 0.93, title, fontsize=23, fontweight="bold", color=INK, ha="left")
    fig.text(0.05, 0.873, a.sub, fontsize=12.5, color=INK2, ha="left")
    panels = [("Generation", "gen_tok_s", "tokens / s"),
              ("Prompt processing", "prefill_tok_s", "tokens / s"),
              ("Accepted tokens per verify pass", "accepted_per_pass", "tokens (budget 16)")]
    for ax, (t, field, ylab) in zip(axes, panels):
        style(ax, t, ylab)
        top = 0
        for cfg, color in zip(order, colors):
            xs, ys = series(groups[cfg], field)
            line(ax, xs, ys, color, label=cfg)
            top = max(top, max(y for y in ys if y is not None))
        ax.set_ylim(0, 16 if field == "accepted_per_pass" else top * 1.18)
        ax.legend(frameon=False, fontsize=12, loc="lower left", labelcolor=INK2)
    fig.savefig(a.out, facecolor=SURF)
    print("wrote", a.out)


def ap_default_title():
    return "Qwen3.8-27B on one DGX Spark, 320 tokens to 260k of context"


if __name__ == "__main__":
    main()
