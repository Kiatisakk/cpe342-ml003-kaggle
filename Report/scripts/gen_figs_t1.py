"""Task 1 operating-point figure (binary F2 and macro F2 against the fraction flagged).

The curve comes from a re-run of the final pipeline's OOF scores (local files under
Kaggle-ML); only the submitted notebook's operating point is annotated.
Set KAGGLE_ML to the pipeline repo if it lives elsewhere.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_style import TASK_COLORS, INK, INK_MUTED, apply_style  # noqa: E402

ROOT = Path(os.environ.get("KAGGLE_ML", r"C:/Users/kiati/Documents/Kaggle-ML"))
OUT = HERE.parent / "task1" / "figures" / "fig_t1_f2_operating_point.png"
C1 = TASK_COLORS[1]

# Submitted notebook (task1.ipynb, cell 13)
NB_THR, NB_F2, NB_OOF_RATE, NB_TEST_RATE = 0.37264, 0.83979, 0.6280, 0.5981


def ecdf(ref):
    s = np.sort(ref)
    return lambda v: np.searchsorted(s, v, side="right") / len(s)


def main():
    apply_style()
    tr = pd.read_csv(ROOT / "cpe342-kaggle-main" / "01_binary_classification" / "train.csv")
    lab_mask = tr["is_cheater"].notna().values
    y = tr.loc[lab_mask, "is_cheater"].astype(int).values

    a = np.load(ROOT / "t1/research/best_oof_allseeds.npy").mean(0)
    o = ecdf(a)(a)
    s_all = np.load(ROOT / "t1/research/blind_sonnet/stack_lr_oof.npy")
    s = s_all[lab_mask] if len(s_all) == len(tr) else s_all
    sc = ecdf(s)(s)
    g = np.zeros(len(y))
    for name, w in {"xgb": 0.1834, "lgb": 0.4848, "cat": 0.3318}.items():
        arr = np.load(ROOT / f"t1/preds/t1_pipeline/oof_{name}.npy")
        g += w * ecdf(arr)(arr)
    p = ecdf(g)(g)
    b = 0.7 * o + 0.2 * sc + 0.1 * p

    rates = np.unique(np.r_[np.linspace(0.25, 0.85, 241), NB_OOF_RATE])
    f2b = np.array([fbeta_score(y, b >= np.quantile(b, 1 - r), beta=2) for r in rates])
    f2m = np.array([fbeta_score(y, b >= np.quantile(b, 1 - r), beta=2, average="macro") for r in rates])
    im = int(np.argmax(f2m))
    r_mac, v_mac = rates[im] * 100, f2m[im]
    v_loc = float(f2b[np.argmin(np.abs(rates - NB_OOF_RATE))])
    print(f"local binary F2 at 62.8%: {v_loc:.5f}; macro optimum {r_mac:.1f}% = {v_mac:.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(rates * 100, f2b, color=C1, lw=2.3, label="OOF binary F2 (positive class, leaderboard metric)")
    ax.plot(rates * 100, f2m, color=INK_MUTED, lw=1.9, ls="--", label="OOF macro F2 (both classes)")

    ax.scatter([r_mac], [v_mac], color=INK_MUTED, s=45, zorder=6, edgecolor="white")
    ax.annotate(f"macro-F2 optimum, about {r_mac:.0f}% flagged", xy=(r_mac, v_mac), xytext=(40, 0.66),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))

    x = NB_OOF_RATE * 100
    ax.axvline(x, color=C1, ls=":", lw=1.2, zorder=2)
    ax.scatter([x], [v_loc], color=C1, s=70, zorder=7, edgecolor="white", lw=1.0)
    ax.annotate(f"submitted notebook: thr {NB_THR}, F2 {NB_F2}\n"
                f"{x:.2f}% of OOF flagged ({NB_TEST_RATE*100:.2f}% of test)",
                xy=(x, v_loc), xytext=(26, 0.858), fontsize=9, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C1, lw=1.1),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=C1, lw=0.9))

    ax.set_xlim(25, 85)
    ax.set_ylim(0.58, 0.885)
    ax.set_xlabel("Fraction of accounts flagged (%)")
    ax.set_ylabel("OOF F2 score")
    ax.legend(loc="lower left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print("saved", OUT)

# ---------------------------------------------------------------- wave 4 additions
LC_N = np.array([7265, 14530, 25428, 36326, 50856, 72652])
LC_F2M = np.array([0.7625, 0.7725, 0.7767, 0.7787, 0.7796, 0.7811])   # t1/research/out/exp18.log
LC_AUC = np.array([0.8705, 0.8851, 0.8894, 0.8911, 0.8921, 0.8932])
LC_ASYMP = (0.7833, 0.8939)                                          # power-law fit reported in exp18.log


def fig_ceiling():
    from scipy.optimize import curve_fit
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, y, asy, lab in [(axs[0], LC_F2M, LC_ASYMP[0], "Nested macro F2"), (axs[1], LC_AUC, LC_ASYMP[1], "ROC-AUC")]:
        f = lambda n, s, a, b: s - a * (n / 1e4) ** (-b)
        try:
            pr, _ = curve_fit(f, LC_N, y, p0=[asy, 0.02, 0.7], maxfev=20000)
        except Exception:
            pr = None
        xs = np.linspace(LC_N[0], 8e5, 300)
        if pr is not None:
            ax.plot(xs, f(xs, *pr), color=C1, lw=1.8, label="power-law fit")
        ax.scatter(LC_N, y, color=C1, s=36, zorder=5, edgecolor="white", label="measured (3 seeds)")
        ax.axhline(asy, color=INK_MUTED, ls="--", lw=1.1)
        ax.text(1.2e5, asy - (0.0025 if lab != "ROC-AUC" else 0.0017), f"asymptote {asy:.4f}", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Training rows per fold")
        ax.set_ylabel(lab)
        ax.legend(loc="lower right")
    fig.tight_layout()
    out = HERE.parent / "task1" / "figures" / "fig_t1_ceiling.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig_missing_auc():
    """OOF AUC of the final LightGBM (3-split-seed averaged OOF) by missingness group."""
    from sklearn.metrics import roc_auc_score
    apply_style()
    tr = pd.read_csv(ROOT / "cpe342-kaggle-main" / "01_binary_classification" / "train.csv")
    d = tr[tr["is_cheater"].notna()].reset_index(drop=True)
    y = d["is_cheater"].astype(int).values
    o = np.load(ROOT / "t1/research/best_oof_avg3.npy")
    key = ["reports_received", "friend_network_size", "account_age_days", "crosshair_placement"]
    nk = d[key].isna().sum(1).values
    rows = [("None of the four\nmissing", nk == 0)]
    names = {"reports_received": "reports", "friend_network_size": "friends",
             "account_age_days": "age", "crosshair_placement": "crosshair"}
    for c in key:
        rows.append((f"{names[c]}\nmissing", d[c].isna().values))
    rows.append(("2+ of the four\nmissing", nk >= 2))
    labs, aucs, ns = [], [], []
    for l, m in rows:
        labs.append(l)
        aucs.append(roc_auc_score(y[m], o[m]))
        ns.append(int(m.sum()))
        print(f"{l!r}: n={m.sum()} AUC={aucs[-1]:.4f}")
    allauc = roc_auc_score(y, o)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    cols = [C1 if i == 0 else INK_MUTED for i in range(len(labs))]
    bars = ax.bar(range(len(labs)), aucs, color=cols, width=0.62)
    for i, (a, n) in enumerate(zip(aucs, ns)):
        ax.text(i, a + 0.002, f"{a:.3f}\n(n={n:,})", ha="center", fontsize=8.3, color=INK)
    ax.axhline(allauc, color=INK, ls=":", lw=1.1)
    ax.text(len(labs) - 0.5, allauc + 0.002, f"all rows {allauc:.3f}", ha="right", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylim(0.80, 0.94)
    ax.set_ylabel("OOF ROC-AUC")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = HERE.parent / "task1" / "figures" / "fig_t1_missing_auc.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
    fig_ceiling()
    fig_missing_auc()
