"""Figures for Task 3 (30-day spending, NMAE).

Run: python gen_figs_t3.py. Overrides: CPE342_DATA (folder holding 03_regression/), CPE342_OOF (pipeline repo root holding
t3/preds/seedbag10/repo/t3/research/oof/bag_q10p10_{p,Q,oof}.npy). The saved 10x10 bag OOF reproduces NMAE 0.79537
(the submission notebook prints 0.79535 for the same recipe; difference is numerical, see task3.tex).
Outputs: Report/task3/figures/*.png; stdout carries the numbers quoted in task3.tex.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_style as ps

OUT = os.path.join(os.path.dirname(HERE), "task3", "figures")
DATA = os.environ.get("CPE342_DATA", r"C:/Users/kiati/Documents/Kaggle-ML/cpe342-kaggle-main")
OOF = os.environ.get("CPE342_OOF", r"C:/Users/kiati/Documents/Kaggle-ML")
os.makedirs(OUT, exist_ok=True)
ps.apply_style()
ACC = ps.TASK_COLORS[3]
SEGC = ["#257DE0", "#F47230", "#6A4CE2", "#00B894"]   # segments 0..3 (direct labels always added)
SEGN = ["0 Casual", "1 Grinder", "2 Social", "3 Whale"]
TARGET = "spending_30d"

tr = pd.read_csv(os.path.join(DATA, "03_regression", "train.csv"))
y = tr[TARGET].values.astype(float)
seg = tr["segment"].values
print("train", tr.shape)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("saved", p)


def exact_fill(d):
    d = d.copy()
    for _ in range(3):
        h, tx, av = d["historical_spending"], d["total_transactions"], d["avg_transaction_value"]
        po, dr = d["purchases_on_discount"], d["discount_rate_used"]
        rules = [
            ("historical_spending", h.isna() & tx.notna() & av.notna(), tx * av),
            ("total_transactions", tx.isna() & h.notna() & av.notna() & (av > 0), h / av),
            ("avg_transaction_value", av.isna() & h.notna() & tx.notna() & (tx > 0), h / tx),
            ("purchases_on_discount", po.isna() & tx.notna() & dr.notna(), tx * dr),
            ("total_transactions", tx.isna() & po.notna() & dr.notna() & (dr > 0), po / dr),
            ("discount_rate_used", dr.isna() & po.notna() & tx.notna() & (tx > 0), po / tx),
        ]
        for col, m, val in rules:
            m = m & d[col].isna()
            d.loc[m, col] = val[m]
    return d


def fig_target():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [0.7, 1.4], "wspace": 0.3})
    z, pz = (y == 0).sum(), (y > 0).sum()
    a.bar([0, 1], [z, pz], color=[ps.AXIS, ACC], width=0.6)
    for i, v in enumerate([z, pz]):
        a.text(i, v + 1200, f"{v:,}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    a.set_xticks([0, 1])
    a.set_xticklabels(["y = 0", "y > 0"])
    a.set_ylim(0, 62000)
    a.set_ylabel("accounts")
    a.set_title("(a) Half of the accounts spend nothing")
    a.grid(axis="x", visible=False)
    bins = np.linspace(0, 5.6, 57)
    for k in range(4):
        v = np.log10(y[(seg == k) & (y > 0)])
        a.__class__  # keep linters quiet
        b.hist(v, bins=bins, histtype="step", lw=1.8, color=SEGC[k], density=True, label=SEGN[k])
    b.set_xlabel(r"$\log_{10}$ spending among payers (THB)")
    b.set_ylabel("density within segment")
    b.set_title("(b) Payers span four orders of magnitude, by segment")
    b.legend(loc="upper left", fontsize=8)
    pos = y[y > 0]
    print("zero rate %.4f | positive: median %.1f mean %.1f max %.0f | overall mean %.1f median %.2f p99 %.0f" % ((y == 0).mean(), np.median(pos), pos.mean(), y.max(), y.mean(), np.median(y), np.percentile(y, 99)))
    for k in range(4):
        v = y[(seg == k) & (y > 0)]
        print("seg", k, "payer median %.0f, p10 %.0f, p90 %.0f" % (np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
    save(fig, "fig_t3_target.png")


def fig_whales():
    labs = ["0 Casual", "1 Grinder", "2 Social", "3 Whale", "missing"]
    keys = [0, 1, 2, 3, -1]
    s = np.where(np.isnan(seg), -1, seg)
    rows = np.array([(s == k).mean() * 100 for k in keys])
    spend = np.array([y[s == k].sum() / y.sum() * 100 for k in keys])
    zr = np.array([(y[s == k] == 0).mean() * 100 for k in keys])
    meany = np.array([y[s == k].mean() for k in keys])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.28})
    x = np.arange(5)
    w = 0.38
    a.bar(x - w / 2, rows, w, color=ps.AXIS, label="share of accounts")
    a.bar(x + w / 2, spend, w, color=ACC, label="share of total spending")
    for i in range(5):
        a.text(i - w / 2, rows[i] + 1.2, f"{rows[i]:.1f}", ha="center", fontsize=7.8)
        a.text(i + w / 2, spend[i] + 1.2, f"{spend[i]:.1f}", ha="center", fontsize=7.8, fontweight="bold" if i == 3 else "normal")
    a.set_xticks(x)
    a.set_xticklabels(labs, fontsize=8)
    a.set_ylim(0, 100)
    a.set_ylabel("%")
    a.legend(loc="upper left", fontsize=8)
    a.set_title("(a) Whales: 14% of accounts, 87% of spending")
    a.grid(axis="x", visible=False)
    b.bar(x, zr, color=[SEGC[0], SEGC[1], SEGC[2], SEGC[3], ps.AXIS], width=0.6)
    for i in range(5):
        b.text(i, zr[i] + 1.5, f"{zr[i]:.0f}%", ha="center", fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels(["0", "1", "2", "3", "miss."], fontsize=8)
    b.set_ylim(0, 85)
    b.set_ylabel("share with y = 0 (%)")
    b.set_xlabel("segment")
    b.set_title("(b) Pay rate rises with segment")
    b.grid(axis="x", visible=False)
    print("seg rows %", rows.round(2), "spend %", spend.round(2), "zero rate %", zr.round(1), "mean y", meany.round(0))
    save(fig, "fig_t3_whales.png")


def fig_identities():
    cols = ["historical_spending", "total_transactions", "avg_transaction_value", "purchases_on_discount", "discount_rate_used"]
    d = exact_fill(tr)
    mb, ma = tr[cols].isna().mean() * 100, d[cols].isna().mean() * 100
    m1 = tr[["historical_spending", "total_transactions", "avg_transaction_value"]].notna().all(axis=1)
    e1 = (tr.loc[m1, "historical_spending"] - tr.loc[m1, "total_transactions"] * tr.loc[m1, "avg_transaction_value"]).abs()
    m2 = tr[["purchases_on_discount", "total_transactions", "discount_rate_used"]].notna().all(axis=1)
    e2 = (tr.loc[m2, "purchases_on_discount"] - tr.loc[m2, "total_transactions"] * tr.loc[m2, "discount_rate_used"]).abs()
    print("missing before", mb.round(2).to_dict(), "after", ma.round(2).to_dict())
    print("identity1 exact(<1e-2) %.2f%%, median err %.6f; identity2 exact %.2f%%" % ((e1 < 1e-2).mean() * 100, e1.median(), (e2 < 1e-2).mean() * 100))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 3.7), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    x = np.arange(len(cols))
    w = 0.38
    a.bar(x - w / 2, mb.values, w, color=ps.AXIS, label="before exact fill")
    a.bar(x + w / 2, ma.values, w, color=ACC, label="after exact fill")
    for i in range(len(cols)):
        a.text(i - w / 2, mb.values[i] + 0.1, f"{mb.values[i]:.1f}", ha="center", fontsize=7.5)
        a.text(i + w / 2, ma.values[i] + 0.1, f"{ma.values[i]:.1f}", ha="center", fontsize=7.5)
    a.set_xticks(x)
    a.set_xticklabels([c.replace("_", "_\n", 1) for c in cols], fontsize=6.8)
    a.set_ylabel("missing rate (%)")
    a.set_ylim(0, 8)
    a.legend(fontsize=8, loc="upper right")
    a.set_title("(a) Missing rate before and after")
    a.grid(axis="x", visible=False)
    s = tr.loc[m1].sample(20000, random_state=0)
    xx = np.log1p(s["total_transactions"] * s["avg_transaction_value"])
    yy = np.log1p(s["historical_spending"])
    cm = LinearSegmentedColormap.from_list("o", ["#FFFFFF", ACC])
    b.hexbin(xx, yy, gridsize=45, cmap=cm, mincnt=1, bins="log")
    b.plot([0, 14], [0, 14], color=ps.INK, lw=1, ls="--", label="y = x")
    b.set_xlabel("log1p(total_transactions x avg_transaction_value)", fontsize=8)
    b.set_ylabel("log1p(historical_spending)", fontsize=8)
    b.set_title("(b) Identity holds on complete rows")
    b.legend(fontsize=8, loc="upper left")
    save(fig, "fig_t3_identities.png")


def fig_offset():
    d = exact_fill(tr)
    pos = d[y > 0].copy()
    seg_p = seg[y > 0]
    lp = np.log1p(pos["prev_month_spending"].clip(lower=0).values)
    lh = np.log1p(pos["historical_spending"].clip(lower=0).values)
    off = np.where(np.isnan(lp), lh - 1.05, lp)
    off = np.where(np.isnan(off), 6.5, off)
    r = np.log(pos[TARGET].values) - off
    ok = ~np.isnan(lp)
    print("prev missing among payers %.3f; corr(log1p prev, log y) %.4f; residual: median %.3f robust sd %.3f, share |r|>1: %.4f" % (
        1 - ok.mean(), np.corrcoef(lp[ok], np.log(pos[TARGET].values[ok]))[0, 1], np.median(r), 1.4826 * np.median(np.abs(r - np.median(r))), (np.abs(r) > 1).mean()))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"wspace": 0.28})
    cm = LinearSegmentedColormap.from_list("o", ["#FFFFFF", ACC])
    a.hexbin(lp[ok], np.log1p(pos[TARGET].values[ok]), gridsize=50, cmap=cm, mincnt=1, bins="log")
    a.plot([0, 13], [0, 13], color=ps.INK, lw=1, ls="--", label="y = x")
    a.set_xlabel("log1p(prev_month_spending)")
    a.set_ylabel("log1p(spending_30d)")
    a.set_title("(a) Payers sit on a unit-slope line")
    a.legend(loc="upper left", fontsize=8)
    bins = np.linspace(-1.0, 3.0, 81)
    for k in range(4):
        rk = r[seg_p == k]
        b.hist(rk, bins=bins, histtype="step", lw=1.7, color=SEGC[k], density=True, label=SEGN[k])
        print("seg", k, "residual median %.3f robust sd %.3f" % (np.median(rk), 1.4826 * np.median(np.abs(rk - np.median(rk)))))
    b.axvline(0, color=ps.INK_MUTED, lw=0.8, ls=":")
    b.set_xlabel(r"$\log y - \mathrm{offset}$ (payers)")
    b.set_ylabel("density")
    b.set_title("(b) Residual: tight in each segment, shifted by segment")
    b.legend(fontsize=8, loc="upper right")
    save(fig, "fig_t3_offset.png")


A = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])


def hurdle(p, Q, thr=0.5, qfix=None):
    pp = np.clip(p, 1e-6, 1.0)
    q = np.clip(1 - 1 / (2 * pp), A[0], A[-1]) if qfix is None else np.full(len(p), qfix)
    Qs = np.sort(Q, 1)
    j = np.clip(np.searchsorted(A, q) - 1, 0, len(A) - 2)
    a0, a1 = A[j], A[j + 1]
    r = np.arange(len(q))
    v = Qs[r, j] + (q - a0) / (a1 - a0) * (Qs[r, j + 1] - Qs[r, j])
    return np.where(pp > thr, np.clip(v, 0, 3e5), 0.0)


def load_oof():
    O = os.path.join(OOF, "t3", "preds", "seedbag10", "repo", "t3", "research", "oof")
    n = len(y)
    return np.load(os.path.join(O, "bag_q10p10_p.npy"))[:n], np.load(os.path.join(O, "bag_q10p10_Q.npy"))[:n], np.load(os.path.join(O, "bag_q10p10_oof.npy"))


def nm(yp):
    return 1 / (1 + np.abs(y - yp).sum() / y.sum())


def fig_qrule(p, Q):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})
    pp = np.linspace(0.3, 1.0, 400)
    q = np.where(pp > 0.5, 1 - 1 / (2 * pp), np.nan)
    a.axvspan(0.3, 0.5, color=ps.GRID, zorder=0)
    a.text(0.4, 0.25, "predict 0\n(the median of y\nis 0)", ha="center", va="center", fontsize=8.5, color=ps.INK_MUTED)
    a.plot(pp, q, color=ACC, lw=2.6)
    a.axhline(0.5, color=ps.INK_MUTED, lw=0.8, ls=":")
    a.text(0.995, 0.52, "q = 0.5 only when p = 1", ha="right", fontsize=8, color=ps.INK_MUTED)
    for pv in (0.6, 0.75, 0.9):
        a.scatter([pv], [1 - 1 / (2 * pv)], color=ACC, s=28, zorder=4)
        a.annotate(f"p={pv}: q={1-1/(2*pv):.2f}", (pv, 1 - 1 / (2 * pv)), xytext=(6, -12), textcoords="offset points", fontsize=7.8)
    a.set_xlim(0.3, 1.0)
    a.set_ylim(0, 0.58)
    a.set_xlabel("P(y > 0 | x)")
    a.set_ylabel(r"quantile of the payer distribution: $q = 1 - 1/(2p)$")
    a.set_title("(a) Which payer quantile to predict")
    a.grid(False)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    mp = np.array([p[idx == i].mean() for i in range(10) if (idx == i).sum() > 50])
    ay = np.array([(y[idx == i] > 0).mean() for i in range(10) if (idx == i).sum() > 50])
    cnt = np.array([(idx == i).sum() for i in range(10) if (idx == i).sum() > 50])
    b.plot([0, 1], [0, 1], color=ps.INK_MUTED, lw=1, ls="--")
    b.scatter(mp, ay, s=20 + cnt / 400, color=ACC, zorder=4, edgecolor="white")
    b.set_xlabel("mean OOF P(y > 0) in the bin")
    b.set_ylabel("observed payer rate")
    b.set_title("(b) The payer probability is calibrated")
    b.text(0.04, 0.9, "marker size = rows in bin", fontsize=7.8, color=ps.INK_MUTED)
    print("reliability (mean p, observed, n):", [(round(m, 3), round(o, 3), int(c)) for m, o, c in zip(mp, ay, cnt)])
    save(fig, "fig_t3_qrule.png")


def fig_error(p, Q, oof):
    s = np.where(np.isnan(seg), -1, seg)
    e = np.abs(y - oof)
    keys = [0, 1, 2, 3, -1]
    labs = ["0 Casual", "1 Grinder", "2 Social", "3 Whale", "missing"]
    rows = np.array([(s == k).mean() * 100 for k in keys])
    spend = np.array([y[s == k].sum() / y.sum() * 100 for k in keys])
    err = np.array([e[s == k].sum() / e.sum() * 100 for k in keys])
    ratio = np.array([e[s == k].sum() / y[s == k].sum() for k in keys])
    print("OOF NMAE of saved bag %.5f; error share pct" % nm(oof), err.round(2), "error/spend ratio", ratio.round(3), "MAE %.1f" % e.mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.5})
    x = np.arange(5)
    w = 0.27
    a.bar(x - w, rows, w, color=ps.AXIS, label="accounts")
    a.bar(x, spend, w, color=ACC, alpha=0.55, label="spending")
    a.bar(x + w, err, w, color=ACC, label="absolute error")
    for i in range(5):
        a.text(i + w, err[i] + 1.2, f"{err[i]:.1f}", ha="center", fontsize=7.5, fontweight="bold" if i == 3 else "normal")
    a.set_xticks(x)
    a.set_xticklabels(labs, fontsize=7.8)
    a.set_ylim(0, 100)
    a.set_ylabel("share of total (%)")
    a.legend(fontsize=8, loc="upper left")
    a.set_title("(a) Where the OOF error sits")
    a.grid(axis="x", visible=False)
    cats = ["payer, multiplicative\nnoise on the amount", "y = 0 but we predict > 0\n(coin flip of paying)", "whale spikes\n(266 redrawn rows)", "payer predicted 0"]
    vals = [49.0, 32.9, 15.4, 2.7]
    yy = np.arange(4)[::-1]
    b.barh(yy, vals, color=[ACC, ACC, ACC, ps.AXIS], height=0.55, alpha=0.9)
    for yi, v in zip(yy, vals):
        b.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=8.5)
    b.set_yticks(yy)
    b.set_yticklabels(cats, fontsize=7.8)
    b.set_xlim(0, 62)
    b.set_xlabel("share of total absolute error (%)")
    b.set_title("(b) Error budget of the OOF predictions")
    b.grid(axis="y", visible=False)
    save(fig, "fig_t3_error_budget.png")


def fig_progression():
    rows = [("HGB two-stage (first hurdle)", 0.7843), ("XGB multi-quantile, hurdle (v4 blend)", 0.79191), ("XGB, offset log(y/prev)", 0.79391),
            ("CatBoost depth 4, offset (1 seed)", 0.79505), ("CatBoost + LightGBM, 3x3 seeds", 0.79529), ("Final, 10x10 seeds", 0.79535)]
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    yy = np.arange(len(rows))[::-1]
    base = 0.782
    for (n, v), yi in zip(rows, yy):
        fin = n.startswith("Final")
        ax.hlines(yi, base, v, color=ACC if fin else ps.AXIS, lw=3.2, zorder=2)
        ax.scatter(v, yi, s=70 if fin else 46, color=ACC if fin else ps.INK_MUTED, zorder=3)
        ax.text(v + 0.0003, yi, f"{v:.5f}", va="center", fontsize=8.3, fontweight="bold" if fin else "normal")
    ax.axvspan(0.796, 0.797, color=ps.GRID, zorder=0)
    ax.text(0.7965, len(rows) - 0.45, "estimated ceiling\n0.796-0.797", ha="center", fontsize=7.3, color=ps.INK_MUTED, va="top")
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _ in rows], fontsize=8.2)
    ax.set_xlim(base, 0.7995)
    ax.set_xlabel("5-fold OOF NMAE score")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_t3_progression.png")


def fig_evolution():
    """OOF NMAE per modelling stage, ZILN stages in blue; numbers from data/t3_model_evolution.csv."""
    d = pd.read_csv(os.path.join(os.path.dirname(HERE), "data", "t3_model_evolution.csv"))
    col = {"baseline": ps.INK_MUTED, "ziln": ps.TASK_COLORS[1], "catboost": ACC}
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    yy = np.arange(len(d))[::-1]
    base = 0.780
    for (_, r), yi in zip(d.iterrows(), yy):
        c = col[r.family]
        full = r.submitted == "yes"
        comp = r.submitted == "component"
        ax.hlines(yi, base, r.oof_nmae, color=c, lw=2.6, alpha=0.9, zorder=2)
        ax.scatter(r.oof_nmae, yi, s=85, zorder=3, edgecolor=c, linewidth=1.8,
                   facecolor=c if full else ("white" if not comp else c), marker="o" if not comp else "D")
        ax.text(r.oof_nmae + 0.0003, yi, f"{r.oof_nmae:.4f}" if r.oof_nmae < 0.795 or r.family != "catboost" else f"{r.oof_nmae:.5f}",
                va="center", fontsize=8.2, fontweight="bold" if full else "normal")
    ax.set_yticks(yy)
    ax.set_yticklabels(d.stage.tolist(), fontsize=7.9)
    ax.set_xlim(base, 0.8015)
    ax.set_xlabel("5-fold OOF NMAE score (same folds; first-release data, Task 3 unchanged by the re-release)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=ps.TASK_COLORS[1], mec=ps.TASK_COLORS[1], label="ZILN stage"),
         Line2D([], [], marker="o", ls="", mfc=ACC, mec=ACC, label="CatBoost quantile stage"),
         Line2D([], [], marker="o", ls="", mfc=ps.INK_MUTED, mec=ps.INK_MUTED, label="first hurdle (HGB)"),
         Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="filled = submitted as Task 3"),
         Line2D([], [], marker="D", ls="", mfc="k", mec="k", label="diamond = inside a submitted blend"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="hollow = research only")]
    ax.legend(handles=h, loc="upper right", fontsize=7.4, ncol=1, bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, labelspacing=0.3)
    save(fig, "fig_t3_evolution.png")


def fig_ziln_diag(oof):
    """ZILN (NN K=1, K=3) against the final model on the same OOF rows: error ratio by segment, and the blend curve."""
    nnf = os.path.join(OOF, "t3", "preds", "nn_ziln", "preds", "t3_nn_oof.csv")
    nn = pd.read_csv(nnf)
    assert len(nn) == len(y) and np.abs(nn["y"].values - y).max() == 0, "NN OOF rows do not line up with train"
    s = np.where(np.isnan(seg), -1, seg)
    keys = [0, 1, 2, 3]
    ratio = lambda p, k: np.abs(y[s == k] - p[s == k]).sum() / y[s == k].sum()
    preds = [("NN-ZILN K=1", nn["NN_K1"].values, ps.AXIS), ("NN-ZILN K=3", nn["NN_K3"].values, ps.TASK_COLORS[1]), ("final CatBoost bag", oof, ACC)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 3.7), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
    x = np.arange(4)
    w = 0.26
    for i, (n_, p_, c_) in enumerate(preds):
        v = [ratio(p_, k) for k in keys]
        a.bar(x + (i - 1) * w, v, w, color=c_, label=n_)
        for xi, vi in zip(x, v):
            a.text(xi + (i - 1) * w, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=6.8)
        print(n_, "NMAE %.5f" % nm(p_), "error ratio by segment", np.round(v, 3), "zero share %.3f" % (p_ == 0).mean())
    a.set_xticks(x)
    a.set_xticklabels(SEGN, fontsize=8)
    a.set_ylim(0, 1.18)
    a.set_ylabel(r"$\sum|e|/\sum y$ within the segment (lower is better)")
    a.legend(loc="upper right", fontsize=7.8, ncol=1)
    a.set_title("(a) Error relative to spending, by segment")
    a.grid(axis="x", visible=False)
    ws = np.linspace(0, 1, 21)
    sc = [nm((1 - w_) * oof + w_ * nn["NN_K3"].values) for w_ in ws]
    b.plot(ws, sc, color=ps.TASK_COLORS[1], lw=2.2)
    b.scatter([0], [sc[0]], color=ACC, s=50, zorder=4)
    b.scatter([1], [sc[-1]], color=ps.TASK_COLORS[1], s=50, zorder=4)
    b.annotate("final alone\n%.5f" % sc[0], (0, sc[0]), xytext=(0.06, sc[0] - 0.0017), fontsize=7.8)
    b.annotate("NN-ZILN K=3 alone\n%.5f" % sc[-1], (1, sc[-1]), xytext=(0.52, sc[-1] + 0.0012), fontsize=7.8, ha="center")
    b.set_xlabel("weight of NN-ZILN K=3 in an arithmetic blend with the final model")
    b.set_ylabel("OOF NMAE")
    b.set_title("(b) Adding the NN never helps")
    print("blend with NN_K3 at w=.1/.2/.3/.5:", [round(sc[i], 5) for i in (2, 4, 6, 10)])
    save(fig, "fig_t3_ziln_diag.png")


if __name__ == "__main__":
    fig_target()
    fig_whales()
    fig_identities()
    fig_offset()
    p, Q, oof = load_oof()
    print("recomputed final NMAE", nm(hurdle(p, Q)), "saved", nm(oof))
    fig_qrule(p, Q)
    fig_error(p, Q, oof)
    fig_progression()
    fig_evolution()
    fig_ziln_diag(oof)
