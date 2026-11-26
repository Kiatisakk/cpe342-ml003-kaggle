"""Shared matplotlib style for the CPE342 report. Colours match Report/macros.tex (see SPEC.md)."""
import matplotlib.pyplot as plt

TASK_COLORS = {1: "#257DE0", 2: "#F47230", 3: "#00B894", 4: "#6A4CE2", 5: "#EE7FAC"}
KEPT = "#008A17"
DROPPED = "#8A8A8A"
RULE_EXCLUDED = "#DC3C3D"   # draw with hatch "///"
INK = "#191919"
INK_MUTED = "#666666"
GRID = "#E7E7E7"
AXIS = "#C2C2C2"
STATUS_COLORS = {"kept": KEPT, "dropped": DROPPED, "rule_excluded": RULE_EXCLUDED}
PALETTE = [TASK_COLORS[i] for i in range(1, 6)]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.labelsize": 9.5, "axes.labelcolor": INK,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "hatch.linewidth": 0.8,
    })
