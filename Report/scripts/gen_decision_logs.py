"""Render one decision-log chart per task from data/decision_log_tN.csv.

Output: taskN/figures/decision_log_tN.pdf (vector, 6.3 in wide = \\linewidth on A4 with 25 mm margins).

Design
- One row per experiment, CSV (chronological) order inside each panel.
- Panels group rows by metric family (public LB / CV-OOF / estimate / no score) because the
  units of delta differ; each panel has its own x scale and a zero line. The exact metric of
  each row is printed as a small tag under the experiment name, so nothing is mixed silently.
- Thin bar anchored at zero plus an end dot; colour kept green, dropped grey, rule-excluded
  red with /// hatch. Delta value is a direct label at the bar end (text in ink, not series colour).
Run: python scripts/gen_decision_logs.py [1 2 3 4 5]
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (apply_style, STATUS_COLORS, INK, INK_MUTED, AXIS, GRID)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = [
    ("lb", "Public leaderboard (change in LB score)"),
    ("cv", "Cross-validation / out-of-fold (change in CV score)"),
    ("est", "Estimates, not measured on the leaderboard"),
    ("noscore", "Adopted or dropped without an isolated score"),
    ("none", "Excluded by the rules: no score reported"),
]
RULE_FACE = "#F8DADA"
STATUS_LABEL = {"kept": "Kept", "dropped": "Dropped (no gain or worse)", "rule_excluded": "Excluded by rule"}


def family(metric, delta, status=""):
    m = str(metric).lower()
    if pd.isna(delta):
        return "none" if status == "rule_excluded" else "noscore"
    if "estimate" in m:
        return "est"
    if "lb" in m or "leaderboard" in m:
        return "lb"
    return "cv"


def fmt(v):
    s = f"{abs(v):.5f}".rstrip("0")
    if len(s.split(".")[1]) < 4:
        s = f"{abs(v):.4f}"
    return ("+" if v > 0 else "−" if v < 0 else "") + s


def render(n):
    csv = os.path.join(ROOT, "data", f"decision_log_t{n}.csv")
    if not os.path.exists(csv):
        print(f"skip task {n}: {csv} missing")
        return
    df = pd.read_csv(csv)
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["fam"] = [family(m, d, st) for m, d, st in zip(df["metric"], df["delta"], df["status"])]
    panels = [(k, t, df[df.fam == k]) for k, t in FAMILIES if (df.fam == k).any()]
    row_h = 0.52
    head = 0.34
    uniform = {k: g["metric"].nunique() == 1 for k, _, g in panels}
    rh = {k: (0.29 if uniform[k] else row_h) for k, _, g in panels}
    heights = [len(g) * rh[k] + head for k, _, g in panels]
    fig_h = sum(heights) + 0.75
    fig = plt.figure(figsize=(6.3, fig_h))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.55,
                          left=0.54, right=0.875, top=1 - 0.40 / fig_h, bottom=0.30 / fig_h)
    for (key, title, g), pos in zip(panels, range(len(panels))):
        ax = fig.add_subplot(gs[pos])
        k = len(g)
        ys = np.arange(k)[::-1]
        ax.set_ylim(-0.6, k - 0.4)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        if uniform[key] and key not in ("none", "noscore"):
            title = f"{title}; metric: {g['metric'].iloc[0]}"
        ax.text(-0.54 / 0.335, 1.0, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4,
                fontweight="bold", color=INK)
        if key in ("none", "noscore"):
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.grid(False)
        else:
            vals = g["delta"].to_numpy(float)
            lim = max(abs(vals).max(), 1e-9)
            lo = min(vals.min(), 0) * 1.0
            hi = max(vals.max(), 0) * 1.0
            span = hi - lo
            absv = np.abs(vals[vals != 0]) if (vals != 0).any() else np.array([1.0])
            symlog = absv.max() > 40 * np.median(absv)
            fmtr = matplotlib.ticker.FuncFormatter(
                lambda v, _: "0" if abs(v) < 1e-12 else ("+" if v > 0 else "−") + f"{abs(v):.5f}".rstrip("0").rstrip("."))
            if symlog:
                lt = 10 ** np.floor(np.log10(np.median(absv)))
                ax.set_xscale("symlog", linthresh=lt, linscale=0.7)
                top = int(np.ceil(np.log10(absv.max())))
                ticks = [0.0] + [s_ * 10.0 ** k for k in range(int(np.log10(lt)), top + 1) for s_ in (-1, 1)]
                ticks = [t for t in ticks if lo * 3 <= t <= hi * 3 or t == 0]
                ax.set_xticks(ticks)
                ax.set_xlim(lo * 2.2 if lo < 0 else -0.5 * lt, hi * 2.2 if hi > 0 else 0.5 * lt)
            else:
                ax.set_xlim(lo - 0.06 * span if lo < 0 else -0.04 * span, hi + 0.06 * span)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.axvline(0, color=INK_MUTED, lw=1.0, zorder=2)
            ax.tick_params(axis="x", labelsize=7.5, length=2)
            ax.xaxis.set_major_formatter(fmtr)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("change vs the best result before this experiment"
                          + (" (symmetric-log axis)" if symlog else ""), fontsize=7.4, color=INK_MUTED, labelpad=2)
        for (_, r), y in zip(g.iterrows(), ys):
            col = STATUS_COLORS[r["status"]]
            name = "\n".join(textwrap.wrap(str(r["experiment"]), 54 if not uniform[key] else 58))
            tag = str(r["metric"])
            nlines = name.count("\n") + 1
            dy = (0.12 if nlines == 1 else 0.22) if not uniform[key] else 0.0
            ax.text(-0.025, y + dy, name, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.4, color=INK, linespacing=0.95)
            if not uniform[key]:
                ax.text(-0.025, y - (0.30 if nlines == 1 else 0.30), tag, transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=6.4, color=INK_MUTED)
            if key in ("none", "noscore"):
                ax.barh(y, 0.40, height=0.42, left=0.0, facecolor=RULE_FACE if key == "none" else "white", edgecolor=col,
                        hatch="///" if key == "none" else None, lw=0.8, ls="-" if key == "none" else "--", zorder=3)
                ax.text(0.45, y, "excluded by rule" if key == "none" else "no score measured", va="center", fontsize=7.4, color=INK)
                continue
            d = r["delta"]
            hatch = "///" if r["status"] == "rule_excluded" else None
            ax.barh(y, d, height=0.34 if hatch else 0.24, facecolor=RULE_FACE if hatch else col,
                    edgecolor=col if hatch else "none", hatch=hatch, lw=0.8 if hatch else 0.0, zorder=3)
            ax.plot([d], [y], marker="o", ms=5.0, color=col, mec="white", mew=0.8, zorder=4)
            ax.text(1.02, y, fmt(d), transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=7.6, color=INK, fontweight="bold")
    handles = [Patch(facecolor=RULE_FACE if s == "rule_excluded" else STATUS_COLORS[s], edgecolor=STATUS_COLORS[s], hatch="///" if s == "rule_excluded" else None,
                     label=STATUS_LABEL[s]) for s in ("kept", "dropped", "rule_excluded") if (df.status == s).any()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles), fontsize=7.6,
               handlelength=1.4, columnspacing=1.4, frameon=False)
    out_dir = os.path.join(ROOT, f"task{n}", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"decision_log_t{n}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"({len(df)} rows, {len(panels)} panels, h={fig_h:.2f} in)")


if __name__ == "__main__":
    apply_style()
    plt.rcParams["savefig.bbox"] = "standard"
    tasks = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for t in tasks:
        render(t)
