"""Figures for Task 2 (player segmentation, new 12 Nov data release).

Run: python gen_figs_t2.py   (from anywhere). Data/OOF locations can be overridden with
CPE342_DATA (folder holding 02_multiclass_classification/) and CPE342_OOF (pipeline repo root holding
t2/preds/v2data/nb_oof.npy and work/t2v2/blind/out/{logreg_spline_A,mlp_torch_A}_oof.npy).
Outputs: Report/task2/figures/*.png. Numbers printed to stdout are the ones quoted in task2.tex.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import f_oneway, chi2_contingency
from sklearn.metrics import f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_style as ps

OUT = os.path.join(os.path.dirname(HERE), "task2", "figures")
DATA = os.environ.get("CPE342_DATA", r"C:/Users/kiati/Documents/Kaggle-ML/cpe342-kaggle-main")
OOF = os.environ.get("CPE342_OOF", r"C:/Users/kiati/Documents/Kaggle-ML")
os.makedirs(OUT, exist_ok=True)
ps.apply_style()
ACC = ps.TASK_COLORS[2]
CLS = ["#257DE0", "#00B894", "#F47230", "#6A4CE2"]   # segment 0..3 (direct labels always added)
NAMES = ["Casual", "Grinder", "Social", "Whale"]
NOISE_NUM = ["random_metric_1", "random_metric_2", "random_metric_3", "account_age_days",
             "responds_to_discounts", "item_type_preference_social", "avg_match_length", "preferred_game_mode"]
NOISE_CAT = ["region", "platform", "device_type", "payment_method", "language", "account_status", "player_type_tag", "skill_tier"]
NOISE = NOISE_NUM + NOISE_CAT

tr = pd.read_csv(os.path.join(DATA, "02_multiclass_classification", "train.csv"))
te = pd.read_csv(os.path.join(DATA, "02_multiclass_classification", "test.csv"))
y = tr["segment"].astype(int).values
K = 4
FEATS = [c for c in tr.columns if c not in ["player_id", "segment"] + NOISE]
DISC = [c for c in FEATS if (not pd.api.types.is_numeric_dtype(tr[c])) or tr[c].nunique() <= 12]
CONT = [c for c in FEATS if c not in DISC]
print("train", tr.shape, "test", te.shape, "feats", len(FEATS), "cont", len(CONT), "disc", DISC)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("saved", p)


def fig_balance_missing():
    cnt = np.bincount(y, minlength=K)
    share = cnt / len(y)
    test_pred = np.array([0.4085, 0.2541, 0.1939, 0.1435])  # notebook output: share of the final test prediction
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.95})
    x = np.arange(K)
    a.bar(x, share * 100, color=CLS, width=0.62)
    for i in range(K):
        a.text(i, max(share[i], test_pred[i]) * 100 + 1.6, f"{cnt[i]:,}\n({share[i]*100:.1f}%)", ha="center", va="bottom", fontsize=8.3, color=ps.INK)
    a.scatter(x, test_pred * 100, marker="D", s=34, color="white", edgecolor=ps.INK, zorder=5, label="share predicted on test")
    a.set_xticks(x)
    a.set_xticklabels([f"{i}\n{n}" for i, n in enumerate(NAMES)])
    a.set_ylim(0, 60)
    a.set_ylabel("share of training rows (%)")
    a.set_title("(a) Class balance")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.0))
    miss = tr[FEATS].isna().mean() * 100
    order = miss.sort_values().index.tolist()
    byc = np.array([[tr.loc[y == k, c].isna().mean() * 100 for k in range(K)] for c in order])
    yy = np.arange(len(order))
    b.hlines(yy, byc.min(1), byc.max(1), color=ps.AXIS, lw=3, zorder=2)
    b.scatter(miss[order].values, yy, s=22, color=ACC, zorder=3, label="all rows")
    b.set_yticks(yy)
    b.set_yticklabels(order, fontsize=6.3)
    b.set_xlim(6.8, 8.8)
    b.set_xlabel("missing rate (%)")
    b.set_title("(b) Missing rate per kept feature")
    b.grid(axis="y", visible=False)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ACC, label="all rows"),
                      plt.Line2D([], [], color=ps.AXIS, lw=3, label="range over the 4 classes")], loc="upper left", fontsize=8)
    print("missing: overall range %.2f-%.2f, mean %.2f; across-class gap max %.2f" % (miss.min(), miss.max(), miss.mean(), (byc.max(1) - byc.min(1)).max()))
    save(fig, "fig_t2_balance_missing.png")


def fig_signal():
    nums = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c]) and c not in ("segment", "player_id")]
    cats = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c]) and c != "player_id"]
    lf = {}
    for c in nums:
        g = [tr.loc[y == k, c].dropna().values for k in range(K)]
        lf[c] = np.log1p(max(f_oneway(*g)[0], 0))
    lf = pd.Series(lf).sort_values()
    cv = {}
    for c in cats:
        ct = pd.crosstab(tr[c].fillna("NA"), y)
        chi2 = chi2_contingency(ct)[0]
        cv[c] = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    cv = pd.Series(cv).sort_values()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 5.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.6})
    a.barh(lf.index, lf.values, color=[ps.AXIS if c in NOISE else ACC for c in lf.index], height=0.72)
    a.set_xlabel(r"$\log(1+F)$ of a one-way ANOVA across segments")
    a.set_title("(a) 34 numeric columns")
    a.tick_params(axis="y", labelsize=6.6)
    a.grid(axis="y", visible=False)
    b.barh(cv.index, cv.values, color=[ps.AXIS if c in NOISE else ACC for c in cv.index], height=0.6)
    b.set_xlabel("Cram\u00e9r's V with segment")
    b.set_title("(b) 10 categorical columns")
    b.tick_params(axis="y", labelsize=7.5)
    b.grid(axis="y", visible=False)
    b.legend(handles=[Patch(color=ACC, label="kept (28)"), Patch(color=ps.AXIS, label="dropped as noise (16)")], loc="lower right", fontsize=8)
    print("ANOVA logF noise max %.3f ; kept min %.3f" % (lf[[c for c in lf.index if c in NOISE]].max(), lf[[c for c in lf.index if c not in NOISE]].min()))
    print("CramerV noise max %.4f ; kept:" % cv[[c for c in cv.index if c in NOISE]].max(), cv[[c for c in cv.index if c not in NOISE]].round(3).to_dict())
    save(fig, "fig_t2_feature_signal.png")


def fig_independence():
    X = tr[CONT]
    n = len(CONT)
    off = ~np.eye(n, dtype=bool)
    rp = X.corr(method="spearman").values
    w = np.bincount(y) / len(y)
    rw = sum(wk * X.iloc[y == k].corr(method="spearman").values for k, wk in enumerate(w))
    mp, mxp = np.abs(rp[off]).mean(), np.abs(rp[off]).max()
    mw, mxw = np.abs(rw[off]).mean(), np.abs(rw[off]).max()
    med_w = [np.nanmedian(np.abs(tr.loc[y == k, CONT].corr().values[np.triu_indices(n, 1)])) for k in range(K)]
    print("spearman pooled mean %.4f max %.4f ; within mean %.4f max %.4f ; ratio %.1f ; median |pearson| within per class %s" % (mp, mxp, mw, mxw, mp / mw, np.round(med_w, 4)))
    cm = LinearSegmentedColormap.from_list("div", ["#257DE0", "#FFFFFF", "#F47230"])
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.04], "wspace": 0.12})
    for ax, R, t in [(axs[0], rp, f"Pooled over all rows\nmean |r| = {mp:.3f}, max {mxp:.3f}"),
                     (axs[1], rw, f"Within class (weighted mean of 4)\nmean |r| = {mw:.3f}, max {mxw:.3f}")]:
        Rm = np.where(np.eye(n, dtype=bool), np.nan, R)
        im = ax.imshow(Rm, cmap=cm, vmin=-0.8, vmax=0.8, interpolation="nearest")
        ax.grid(False)
        ax.set_title(t, fontsize=9.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(ps.AXIS)
    axs[0].set_ylabel("23 continuous kept features")
    axs[0].set_xlabel("same 23 features")
    axs[1].set_xlabel("same 23 features")
    cb = fig.colorbar(im, cax=axs[2])
    cb.set_label("Spearman r")
    cb.outline.set_visible(False)
    save(fig, "fig_t2_independence.png")


def fig_latent():
    fig = plt.figure(figsize=(10.2, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 0.8], wspace=0.6)
    a = fig.add_subplot(gs[0])
    med = {k: (tr.loc[y == k, "total_playtime_hours"].median(), tr.loc[y == k, "total_spending_thb"].median()) for k in range(K)}
    print("medians (hours, thb):", {k: (round(v[0], 1), round(v[1], 0)) for k, v in med.items()})
    for p, q in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        a.plot([med[p][0], med[q][0]], [med[p][1] / 1000, med[q][1] / 1000], color=ps.AXIS, lw=1.2, ls=":", zorder=1)
    offs = {0: (8, 8, "left"), 1: (-6, -16, "right"), 2: (8, 6, "left"), 3: (-8, -14, "right")}
    for k in range(K):
        a.scatter(med[k][0], med[k][1] / 1000, s=150, color=CLS[k], edgecolor="white", lw=1.5, zorder=4)
        dx, dy, ha = offs[k]
        a.annotate(f"{k} {NAMES[k]}", (med[k][0], med[k][1] / 1000), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5, color=ps.INK, fontweight="bold")
    a.set_xlabel("median total_playtime_hours")
    a.set_ylabel("median total_spending_thb (thousand)")
    a.set_title("(a) Class medians form a 2$\\times$2 grid", fontsize=9.5)
    a.margins(0.2)
    order_e = ["Low", "Mid", "High", "Veteran"]
    order_l = ["Bronze", "Silver", "Gold", "Platinum", "Diamond"]
    cmap = LinearSegmentedColormap.from_list("o", ["#FFFFFF", ACC])
    for j, (col, order, ttl) in enumerate([("engagement_level", order_e, "(b) engagement_level"), ("loyalty_tier", order_l, "(c) loyalty_tier")]):
        ax = fig.add_subplot(gs[1 + j])
        ct = pd.crosstab(tr[col], y, normalize="index").reindex(order)
        ax.imshow(ct.values, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
        ax.grid(False)
        for i in range(ct.shape[0]):
            for k in range(K):
                v = ct.values[i, k]
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7.3, color="white" if v > 0.5 else ps.INK)
        ax.set_xticks(range(K))
        ax.set_xticklabels(range(K))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("segment")
        ax.set_title(ttl + "\n(row-normalised)", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        print(col, ct.round(2).to_dict("index"))
    save(fig, "fig_t2_latent_2x2.png")


def load_oof():
    nb = np.load(os.path.join(OOF, "t2", "preds", "v2data", "nb_oof.npy"))
    lr = np.load(os.path.join(OOF, "work", "t2v2", "blind", "out", "logreg_spline_A_oof.npy"))
    ml = np.load(os.path.join(OOF, "work", "t2v2", "blind", "out", "mlp_torch_A_oof.npy"))
    return nb, lr, ml


def affine(Bo):
    pi = np.bincount(y, minlength=K) / len(y)
    q = pi.copy()
    Fk = np.full(K, 0.75)
    for _ in range(8):
        aa = 2.0 / (pi + q)
        bb = -Fk / (pi + q)
        pr = (aa * Bo + bb).argmax(1)
        q = np.bincount(pr, minlength=K) / len(y)
        Fk = np.array([2.0 * ((pr == k) & (y == k)).mean() / (pi[k] + q[k]) for k in range(K)])
    return pr, aa, bb


def fig_results_insights():
    nb, lr, ml = load_oof()
    Bo = 0.87 * nb + 0.08 * lr + 0.05 * ml
    pr, aa, bb = affine(Bo)
    f = lambda p: f1_score(y, p, average="macro")
    vals = [("MLP (weight 0.05)", f(ml.argmax(1))), ("Spline LR (0.08)", f(lr.argmax(1))), ("EM-NB (0.87)", f(nb.argmax(1))),
            ("Blend, argmax", f(Bo.argmax(1))), ("Blend + affine rule (final)", f(pr))]
    print("OOF macro-F1 (recomputed):", [(n, round(v, 5)) for n, v in vals])
    print("rows changed by affine rule:", int((pr != Bo.argmax(1)).sum()), " a_k=", aa.round(3), " b_k=", bb.round(3))
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    yy = np.arange(len(vals))[::-1]
    base = 0.750
    for (n, v), yi in zip(vals, yy):
        fin = n.startswith("Blend + affine")
        ax.hlines(yi, base, v, color=ACC if fin else ps.AXIS, lw=3.2, zorder=2)
        ax.scatter(v, yi, s=70 if fin else 48, color=ACC if fin else ps.INK_MUTED, zorder=3)
        ax.text(v + 0.0002, yi, f"{v:.5f}", va="center", fontsize=8.5, fontweight="bold" if fin else "normal", color=ps.INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _ in vals])
    ax.set_xlim(base, 0.7605)
    ax.set_xlabel("5-fold OOF macro F1 (new data, no player_id)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_t2_progression.png")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.05, 1, 1], "wspace": 0.5})
    cmx = confusion_matrix(y, pr, normalize="true")
    cmap = LinearSegmentedColormap.from_list("o", ["#FFFFFF", ACC])
    a.imshow(cmx, cmap=cmap, vmin=0, vmax=0.9)
    a.grid(False)
    for i in range(K):
        for j in range(K):
            a.text(j, i, f"{cmx[i,j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if cmx[i, j] > 0.55 else ps.INK)
    a.set_xticks(range(K))
    a.set_yticks(range(K))
    a.set_xticklabels(NAMES, fontsize=7.8, rotation=25)
    a.set_yticklabels(NAMES, fontsize=7.8)
    a.set_xlabel("predicted")
    a.set_ylabel("true")
    a.set_title("(a) OOF confusion (row-normalised)", fontsize=9)
    for s in a.spines.values():
        s.set_visible(False)
    f1a = f1_score(y, Bo.argmax(1), average=None)
    f1b = f1_score(y, pr, average=None)
    x = np.arange(K)
    w = 0.36
    b.bar(x - w / 2, f1a, w, color=ps.AXIS, label="argmax")
    b.bar(x + w / 2, f1b, w, color=ACC, label="affine rule")
    for i in range(K):
        b.text(i - w / 2, f1a[i] + 0.006, f"{f1a[i]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
        b.text(i + w / 2, f1b[i] + 0.006, f"{f1b[i]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    b.set_xticks(x)
    b.set_xticklabels(NAMES, fontsize=7.8, rotation=25)
    b.set_ylim(0.6, 0.95)
    b.set_ylabel("per-class F1")
    b.set_title("(b) F1 by class", fontsize=9)
    b.legend(fontsize=7.5, loc="upper right")
    b.grid(axis="x", visible=False)
    pi = np.bincount(y) / len(y)
    qa = np.bincount(Bo.argmax(1), minlength=K) / len(y)
    qb = np.bincount(pr, minlength=K) / len(y)
    c.bar(x - w / 2, (qa - pi) * 100, w, color=ps.AXIS, label="argmax")
    c.bar(x + w / 2, (qb - pi) * 100, w, color=ACC, label="affine rule")
    c.axhline(0, color=ps.INK_MUTED, lw=0.8)
    c.set_xticks(x)
    c.set_xticklabels(NAMES, fontsize=7.8, rotation=25)
    c.set_ylabel("predicted minus true share (pp)")
    c.set_title("(c) Class-share bias", fontsize=9)
    c.grid(axis="x", visible=False)
    print("per-class F1 argmax", f1a.round(4), " affine", f1b.round(4))
    print("pred share - prior (pp) argmax", ((qa - pi) * 100).round(2), " affine", ((qb - pi) * 100).round(2))
    print("confusion diag", np.diag(cmx).round(3))
    print("confusion 0->1 %.3f 1->0 %.3f 2->3 %.3f 3->2 %.3f 0->2 %.3f 2->0 %.3f 1->3 %.3f 3->1 %.3f" % (
        cmx[0, 1], cmx[1, 0], cmx[2, 3], cmx[3, 2], cmx[0, 2], cmx[2, 0], cmx[1, 3], cmx[3, 1]))
    save(fig, "fig_t2_errors.png")


def fig_learning_curve():
    """Learning curve of NB / EM-NB (first data release, hand-entered from E11 log; see data/t2_learning_curve.csv)."""
    lc = pd.read_csv(os.path.join(os.path.dirname(HERE), "data", "t2_learning_curve.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for m, col, mk in [("NB", ps.INK_MUTED, "o"), ("EM-NB", ACC, "s")]:
        d = lc[lc["model"] == m]
        ax.errorbar(d["n_train"], d["macro_f1_mean"], yerr=d["sd_3_seeds"], color=col, marker=mk, ms=5, lw=1.6, capsize=2, label=m)
    ax.axhline(0.7557 + 0.002, color=ps.AXIS, ls="--", lw=1)
    ax.text(9000, 0.7557 + 0.0025, "full data + 0.002 (headroom)", fontsize=8, color=ps.INK_MUTED)
    ax.set_ylim(0.745, 0.7595)
    ax.set_xscale("log")
    ax.set_xticks([8320, 20800, 41600, 83200])
    ax.set_xticklabels(["8.3k\n(10%)", "20.8k\n(25%)", "41.6k\n(50%)", "83.2k\n(100%)"])
    ax.minorticks_off()
    ax.set_xlabel("training rows per fold (share of the fold's training part)")
    ax.set_ylabel("5-fold OOF macro F1 (3 seeds)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig_t2_learning_curve.png")


if __name__ == "__main__":
    fig_balance_missing()
    fig_signal()
    fig_independence()
    fig_latent()
    fig_results_insights()
    fig_learning_curve()
