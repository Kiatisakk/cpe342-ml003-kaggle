"""Task 5 figures, recomputed from the 12 Nov test.csv with the exact rules of task5/task5.ipynb.
Data: env T5_TEST_CSV or the local pipeline copy. Output: task5/figures/."""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, TASK_COLORS, INK  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.environ.get("T5_TEST_CSV", "C:/Users/kiati/Documents/Kaggle-ML/cpe342-kaggle-main/05_anomaly_detection/test.csv")
OUT = os.path.join(ROOT, "task5", "figures")
C5 = TASK_COLORS[5]
BASE = "#9AA3AE"
X = pd.read_csv(CSV)
P = [1, 2, 3, 4]
STATIC = ["behavior_stability", "account_age_days"]


def mean4(b):
    return X[[f"{b}_{p}" for p in P]].astype(float).mean(axis=1).values


def cols_of(b):
    return [b] if b in STATIC else [f"{b}_{p}" for p in P]


def inbox(b, lo, hi):
    m = X[cols_of(b)].values.astype(float)
    return (((m >= lo) & (m <= hi)) | np.isnan(m)).all(axis=1)


lat = np.stack([X[f"login_lat_{p}"] for p in P], 1).astype(float)
lon = np.stack([X[f"login_lon_{p}"] for p in P], 1).astype(float)
bad_lat = (np.abs(lat) > 90).any(1)
lt = np.where(np.abs(lat) > 90, np.nan, lat)
hl, hn = np.nanmean(lt[:, :2], 1), np.nanmean(lon[:, :2], 1)
jump = np.nan_to_num(np.nanmax(np.hypot(lt[:, 2:] - hl[:, None], (lon[:, 2:] - hn[:, None] + 180) % 360 - 180), 1))
bases = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")} - {"login_lat", "login_lon"})
Z = []
for b in bases:
    m = X[[f"{b}_{p}" for p in P]].astype(float).values
    ch = np.nanmax(np.abs(m[:, 2:] - np.nanmean(m[:, :2], 1, keepdims=True)), 1)
    med = np.nanmedian(ch)
    mad = 1.4826 * np.nanmedian(np.abs(ch - med)) + 1e-9
    Z.append((ch - med) / mad)
z2 = np.sort(np.nan_to_num(np.stack(Z, 1)), 1)[:, -2]
rule = (jump > 5) | bad_lat | (z2 > 6)
bot = mean4("login_time_variance") < 1.0
lowact = mean4("activity_level") < 0.34
lowcomm = mean4("communication_frequency") < 0.17
lowstab = X["behavior_stability"].values < 0.55
union = rule | bot | lowact | lowcomm | lowstab
acckda = inbox("performance_stability", .85, .95) & inbox("kda_std", .7, 1.) & inbox("accuracy_std", 9., 12.)
bsn = X["behavior_stability"].isna().values
pf, ag = inbox("purchase_frequency", 3., 6.), inbox("account_age_days", 800., 1500.)
cand = bsn & ~union & pf & ag
nor = ~union & ~bsn & pf & ag & (X["behavior_stability"].values >= .6)
lm = mean4("login_count")
ps = X[[f"purchase_frequency_{p}" for p in P]].std(axis=1).values


def logpdf(v, ref, bins):
    h, _ = np.histogram(ref[~np.isnan(ref)], bins=bins)
    h = (h + .5) / (h.sum() + .5 * len(h)) / np.diff(bins)
    o = np.log(h[np.clip(np.digitize(v, bins) - 1, 0, len(h) - 1)])
    o[np.isnan(v)] = 0.
    return o


bl, bp = np.linspace(0, 130, 27), np.linspace(0, 2.5, 26)
LLR = logpdf(lm, lm[lowstab], bl) - logpdf(lm, lm[nor], bl) + logpdf(ps, ps[lowstab], bp) - logpdf(ps, ps[nor], bp)
prior = 14. / cand.sum()
post = np.where(cand, 1 / (1 + np.exp(-(LLR + np.log(prior / (1 - prior))))), 0.)
lsnan = post > .10
final = union | acckda | lsnan
COUNTS = dict(rule=int(rule.sum()), bot=int(bot.sum()), lowact=int(lowact.sum()), lowcomm=int(lowcomm.sum()),
              lowstab=int(lowstab.sum()), union=int(union.sum()), acckda=int(acckda.sum()),
              acckda_new=int((acckda & ~union).sum()), cand=int(cand.sum()), lsnan=int(lsnan.sum()),
              lsnan_new=int((lsnan & ~union & ~acckda).sum()), final=int(final.sum()))


def fig_shock():
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
    spec = [(ax[0], np.log1p(jump), np.log1p(5.), "log(1 + largest period-3/4 jump from home, degrees)",
             f"jump > 5 degrees: {int((jump > 5).sum())}"),
            (ax[1], np.clip(z2, 0, 40), 6., "second-largest robust z of period-3/4 change (clipped at 40)",
             f"z > 6: {int((z2 > 6).sum())}")]
    for a, v, thr, xl, nm in spec:
        a.hist(v, bins=60, color=BASE, edgecolor="white", lw=.3)
        a.set_yscale("log")
        a.axvline(thr, color=C5, lw=1.8)
        a.set_xlabel(xl, fontsize=7.2)
        a.set_ylabel("accounts (log scale)", fontsize=7.5)
        a.text(.97, .93, nm, transform=a.transAxes, ha="right", va="top", fontsize=7.6, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_t5_shock.png"))
    plt.close(fig)


def fig_gaps():
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2))
    panels = [(mean4("login_time_variance"), 1.0, (0, 8), "bot: mean login_time_variance", bot),
              (mean4("activity_level"), .34, (0, 1.8), "lowact: mean activity_level", lowact),
              (mean4("communication_frequency"), .17, (0, 1.5), "lowcomm: mean communication_frequency", lowcomm),
              (X["behavior_stability"].values, .55, (.1, 1.), "lowstab: behavior_stability (static)", lowstab)]
    for a, (v, thr, xl, t, m) in zip(axes.ravel(), panels):
        v = v[~np.isnan(v)]
        v = v[(v >= xl[0]) & (v <= xl[1])]
        _, b, patches = a.hist(v, bins=70, edgecolor="white", lw=.2)
        for pt, left in zip(patches, b[:-1]):
            pt.set_facecolor(C5 if left < thr else BASE)
        a.set_yscale("log")
        a.axvline(thr, color=INK, lw=1.2, ls="--")
        a.set_title(t, fontsize=7.6, loc="left")
        a.text(.97, .93, f"cut {thr}: {int(m.sum())} accounts below", transform=a.transAxes, ha="right", va="top", fontsize=7, color=INK, fontweight="bold",
               bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
        a.tick_params(labelsize=7)
    for a in axes[:, 0]:
        a.set_ylabel("accounts (log scale)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_t5_gaps.png"))
    plt.close(fig)


def fig_box():
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
    a = ax[0]
    mk, ma = mean4("kda_std"), mean4("accuracy_std")
    new = acckda & ~union
    bg = np.where(~union & ~new)[0]
    bg = np.random.RandomState(42).choice(bg, 5000, replace=False)
    a.scatter(ma[bg], mk[bg], s=3, color=BASE, alpha=.35, lw=0, label="other accounts (sample)")
    a.scatter(ma[new], mk[new], s=8, color=C5, lw=0, label=f"3-metric box ({int(new.sum())})")
    a.add_patch(plt.Rectangle((9, .7), 3, .3, fill=False, ec=INK, lw=1.0, ls="--"))
    a.set_xlabel("mean accuracy_std", fontsize=7.5)
    a.set_ylabel("mean kda_std", fontsize=7.5)
    a.legend(fontsize=6.8, loc="lower left")
    a.tick_params(labelsize=7)
    b = ax[1]
    for v, c, lab in [(lm[nor], BASE, f"normal in companion box ({int(nor.sum())})"),
                      (lm[lowstab], C5, f"visible lowstab ({int(lowstab.sum())})")]:
        b.hist(v[~np.isnan(v)], bins=np.linspace(0, 130, 27), density=True, color=c, alpha=.75, label=lab,
               edgecolor="white", lw=.3)
    b.set_xlabel("mean login_count", fontsize=7.5)
    b.set_ylabel("density", fontsize=7.5)
    b.legend(fontsize=6.8, loc="upper right")
    b.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_t5_box_recovery.png"))
    plt.close(fig)


def fig_breakdown():
    items = [("shock / geo rule", COUNTS["rule"]), ("bot", COUNTS["bot"]), ("low activity", COUNTS["lowact"]),
             ("low communication", COUNTS["lowcomm"]), ("low stability (visible)", COUNTS["lowstab"]),
             ("3-metric box (new)", COUNTS["acckda_new"]), ("low stability, NaN (recovered, new)", COUNTS["lsnan_new"])]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v in items], height=.45, color=C5)
    for yi, (_, v) in zip(y, items):
        ax.text(v + 12, yi, f"{v:,}", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in items], fontsize=8)
    ax.set_xlim(0, 1000)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"accounts flagged by each detector (final union {COUNTS['final']:,} = {COUNTS['final'] / len(X) * 100:.2f}% of test)",
                  fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_t5_breakdown.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Per-group EDA figures (added for the expanded Task 5 section)
# ---------------------------------------------------------------------------------------------
DPI = 200
MUTED = "#666666"
BLUE = "#0072B2"     # colour-blind-safe companion hues (Okabe-Ito) for multi-group charts
AMBER = "#E69F00"
PALE5 = "#F6BBD3"
geo = (jump > 5) | bad_lat
shock = z2 > 6
stationary = bot | lowact | lowcomm | lowstab
normal_mask = ~final
BASES = bases + ["login_lat", "login_lon"]
FEATS = {}
for _b in BASES:
    _m = X[[f"{_b}_{p}" for p in P]].astype(float)
    FEATS[f"{_b} mean"] = _m.mean(axis=1).values
    FEATS[f"{_b} sd"] = _m.std(axis=1).values
FEATS["behavior_stability"] = X["behavior_stability"].values
FEATS["account_age_days"] = X["account_age_days"].values


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI)
    plt.close(fig)


def style_ax(a, fs=6.8):
    a.tick_params(labelsize=fs)
    a.xaxis.label.set_size(fs + .3)
    a.yaxis.label.set_size(fs + .3)


def panel_label(a, s):
    a.text(-0.02, 1.04, s, transform=a.transAxes, fontsize=8.5, fontweight="bold", color=INK, ha="right", va="bottom")


def fig_takeover():
    """Group 1: account takeover (geo jump and invalid latitude)."""
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1.1, .9]))
    # (a) home position -> farthest period-3/4 login of geo accounts
    a = ax[0]
    rs = np.random.RandomState(3)
    ok = ~np.isnan(hl) & ~np.isnan(hn)
    bg = rs.choice(np.where(ok & normal_mask)[0], 3000, replace=False)
    a.scatter(hn[bg], hl[bg], s=2, color=BASE, alpha=.35, lw=0)
    gi_ = np.where((jump > 5) & ok)[0]
    sel = rs.choice(gi_, 70, replace=False)
    dl = lt[:, 2:] - hl[:, None]
    dn = (lon[:, 2:] - hn[:, None] + 180) % 360 - 180
    dist = np.nan_to_num(np.hypot(dl, dn), nan=-1)
    for i in sel:
        k = int(np.argmax(dist[i]))
        tx, ty = hn[i] + dn[i, k], lt[i, 2 + k]
        a.annotate("", xy=(tx, ty), xytext=(hn[i], hl[i]), arrowprops=dict(arrowstyle="-|>", color=C5, lw=.55, alpha=.8, mutation_scale=5))
    a.set_xlim(-185, 185)
    a.set_ylim(-95, 95)
    a.set_xlabel("longitude (degrees)")
    a.set_ylabel("latitude (degrees)")
    a.text(.02, .03, "grey: home of normal accounts\narrows: home to period-3/4 login,\n70 of 425 jumpers", transform=a.transAxes, fontsize=6, color=INK, va="bottom",
           bbox=dict(fc="white", ec="none", alpha=.85, pad=1.2))
    style_ax(a)
    panel_label(a, "a")
    # (b) jump histogram with the empty band
    a = ax[1]
    v = np.log1p(jump[~bad_lat])
    bins = np.linspace(0, np.log1p(180), 60)
    _, b, pt = a.hist(v, bins=bins, edgecolor="white", lw=.2)
    for p_, left in zip(pt, b[:-1]):
        p_.set_facecolor(C5 if left >= np.log1p(5) - 1e-9 else BASE)
    nmax = jump[~geo & ~bad_lat].max()
    jmin = jump[(jump > 5)].min()
    a.axvspan(np.log1p(nmax), np.log1p(jmin), color="#F2F2F2", zorder=0)
    a.axvline(np.log1p(5), color=INK, lw=1.1, ls="--")
    a.set_yscale("log")
    a.set_xticks(np.log1p([0, 1, 2, 5, 10, 30, 100]))
    a.set_xticklabels(["0", "1", "2", "5", "10", "30", "100"])
    a.set_xlabel("largest jump from home (degrees)")
    a.set_ylabel("accounts (log scale)")
    a.text(.97, .95, f"empty band\n{nmax:.1f} to {jmin:.1f} degrees", transform=a.transAxes, ha="right", va="top", fontsize=6.3, color=INK)
    a.text(.97, .62, f"cut 5 degrees:\n{int((jump > 5).sum())} accounts\nabove", transform=a.transAxes, ha="right", va="top", fontsize=6.3, color=INK, fontweight="bold")
    style_ax(a)
    panel_label(a, "b")
    # (c) invalid latitudes
    a = ax[2]
    allv = lat[~np.isnan(lat)].ravel()
    bins = np.arange(-180, 181, 6)
    a.hist(np.clip(allv, -180, 180), bins=bins, color=BASE, edgecolor="white", lw=.2)
    inv = allv[np.abs(allv) > 90]
    a.hist(np.clip(inv, -180, 180), bins=bins, color=C5, edgecolor="white", lw=.2)
    a.set_yscale("log")
    a.set_ylim(.5, 3e6)
    for s_ in (-90, 90):
        a.axvline(s_, color=INK, lw=1.0, ls="--")
    a.set_xlabel("login latitude (degrees)")
    a.set_ylabel("login cells (log scale)")
    a.text(.5, .95, f"|lat| > 90: {len(inv)} cells\nin {int(bad_lat.sum())} accounts\n({int((bad_lat & ~(jump > 5)).sum())} not caught by the jump)", transform=a.transAxes, ha="center", va="top", fontsize=6.2, color=INK, fontweight="bold",
           bbox=dict(fc="white", ec="none", alpha=.9, pad=1.2))
    style_ax(a)
    panel_label(a, "c")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_t5_takeover.png")
    return dict(nmax=nmax, jmin=jmin, n_inv_cells=len(inv), inv_min=float(inv.min()), inv_max=float(inv.max()))


def _flag_rate(col, m):
    v = X[col].values.astype(float)
    ok = ~np.isnan(v) & m
    return float((v[ok] > 0).mean()), int(ok.sum())


def fig_selling():
    """Group 2: behaviour shock (account selling) and flag enrichment."""
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.8), gridspec_kw=dict(width_ratios=[1.2, .8, 1.15]))
    both = geo & shock
    shock_only = shock & ~geo
    geo_only = geo & ~shock
    # (a) jump vs z2
    a = ax[0]
    rs = np.random.RandomState(1)
    nb = rs.choice(np.where(normal_mask)[0], 3500, replace=False)
    a.scatter(np.log1p(jump[nb]), np.clip(z2[nb], 0, 40), s=2.5, color=BASE, alpha=.4, lw=0, label="normal (sample)")
    a.scatter(np.log1p(jump[shock_only]), np.clip(z2[shock_only], 0, 40), s=4, color=BLUE, lw=0, alpha=.8, label=f"shock only ({int(shock_only.sum())})")
    a.scatter(np.log1p(jump[both]), np.clip(z2[both], 0, 40), s=4, color=C5, lw=0, alpha=.8, label=f"jump and shock ({int(both.sum())})")
    a.scatter(np.log1p(jump[geo_only]), np.clip(z2[geo_only], 0, 40), s=7, color=AMBER, lw=0, label=f"jump only ({int(geo_only.sum())})")
    a.axhline(6, color=INK, lw=1.0, ls="--")
    a.axvline(np.log1p(5), color=INK, lw=1.0, ls="--")
    a.set_xticks(np.log1p([0, 1, 2, 5, 10, 30, 100]))
    a.set_xticklabels(["0", "1", "2", "5", "10", "30", "100"])
    a.set_xlabel("largest jump from home (degrees)")
    a.set_ylabel("second-largest robust z (clipped 40)")
    a.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, handletextpad=.2, columnspacing=.8, markerscale=1.6)
    style_ax(a)
    panel_label(a, "a")
    # (b) number of shocked metrics
    a = ax[1]
    Zs = ZMAT
    cnt = (Zs > 6).sum(1)
    ks = np.arange(0, 8)
    ctot = np.array([(cnt == k).sum() for k in ks])
    cols = [BASE if k < 2 else C5 for k in ks]
    a.bar(ks, np.maximum(ctot, .6), color=cols, width=.75)
    a.set_yscale("log")
    for k, c in zip(ks, ctot):
        a.text(k, max(c, .6) * 1.25, f"{c:,}", ha="center", va="bottom", fontsize=5.8, color=INK)
    a.set_ylim(.5, 3e5)
    a.set_xticks(ks)
    a.set_xlabel("metrics with z > 6")
    a.set_ylabel("accounts (log scale)")
    a.text(.97, .93, "rule: at least 2\nmetrics (pink)", transform=a.transAxes, ha="right", va="top", fontsize=6.2, color=INK, fontweight="bold")
    style_ax(a)
    panel_label(a, "b")
    # (c) which metrics shock
    a = ax[2]
    rates = []
    for j, bname in enumerate(ZBASES):
        rates.append(((Zs[both, j] > 6).mean(), (Zs[shock_only, j] > 6).mean(), (Zs[:, j] > 6)[shock].sum(), bname))
    rates = [r for r in rates if r[2] > 0]
    rates.sort(key=lambda r: r[0] + r[1])
    y = np.arange(len(rates))
    a.barh(y + .2, [r[0] for r in rates], height=.38, color=C5, label="jump and shock")
    a.barh(y - .2, [r[1] for r in rates], height=.38, color=BLUE, label="shock only")
    a.set_yticks(y)
    a.set_yticklabels([r[3] for r in rates], fontsize=5.6)
    a.set_xlabel("share of accounts with z > 6 in the metric")
    a.legend(fontsize=5.6, loc="lower right", handletextpad=.3, handlelength=.9, borderaxespad=0)
    a.grid(axis="y", visible=False)
    style_ax(a, 6.4)
    a.tick_params(axis="y", labelsize=5.6)
    panel_label(a, "c")
    fig.tight_layout(w_pad=.8)
    save(fig, "fig_t5_selling.png")
    return {"n_metrics_shocked": len(rates), "metrics_never_shocked": len(ZBASES) - len(rates), "z_cnt": {int(k): int(c) for k, c in zip(ks, ctot)}}


def fig_flags():
    """Enrichment of the four security flags inside each group."""
    flags = ["password_changed", "vpn_usage", "mass_item_sale", "suspicious_login_time"]
    groups = [("normal accounts", normal_mask, BASE), ("stationary clusters", stationary & ~rule, PALE5),
              ("shock only", shock & ~geo, BLUE), ("location jump or invalid latitude", geo, C5)]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    w = .2
    out = {}
    for gi_, (nm, m, c) in enumerate(groups):
        r = [_flag_rate(f, m) for f in flags]
        xs = np.arange(len(flags)) + (gi_ - 1.5) * w
        ax.bar(xs, [v for v, _ in r], width=w * .92, color=c, label=f"{nm} (n = {int(m.sum()):,})")
        for x_, (v, _) in zip(xs, r):
            ax.text(x_, v + .01, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
        out[nm] = [round(v, 4) for v, _ in r]
    ax.set_xticks(np.arange(len(flags)))
    ax.set_xticklabels(flags, fontsize=7.5)
    ax.set_ylabel("share of accounts with the flag set")
    ax.set_ylim(0, .9)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.6, loc="upper right")
    style_ax(ax, 7)
    fig.tight_layout()
    save(fig, "fig_t5_flags.png")
    return out


def _ks(a, b):
    from scipy.stats import ks_2samp
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    return ks_2samp(a, b).statistic


def fig_group(name, key, v, thr, xlim, xlabel, second, m, fname, color_below=True):
    """Stationary group figure: (a) log-count histogram with the empty gap, (b) 2-D view with the most telling second
    feature, (c) the features that differ most from normal accounts (KS distance)."""
    fig = plt.figure(figsize=(6.5, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1.15], hspace=.62, wspace=1.05)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    vv = v.copy()
    # (a)
    a = ax[0]
    ok = ~np.isnan(vv)
    inside = m & ok
    outside = ~m & ok
    hi_g = np.nanmax(vv[inside])
    lo_n = np.nanmin(vv[outside & (vv > hi_g)]) if (outside & (vv > hi_g)).any() else xlim[1]
    mask = ok & (vv >= xlim[0]) & (vv <= xlim[1])
    _, b, patches = a.hist(vv[mask], bins=70, edgecolor="white", lw=.2)
    for p_, left in zip(patches, b[:-1]):
        p_.set_facecolor(C5 if left < thr else BASE)
    a.set_yscale("log")
    a.axvspan(hi_g, lo_n, color="#F2F2F2", zorder=0)
    a.axvline(thr, color=INK, lw=1.1, ls="--")
    a.set_xlabel(xlabel)
    a.set_ylabel("accounts (log scale)")
    top_ = max(p_.get_height() for p_ in patches)
    a.set_ylim(.6, top_ * 14)
    a.text(.97, .96, f"{int(m.sum())} accounts below the cut ({thr})\ncluster max {hi_g:.2f}, normal min {lo_n:.2f}", transform=a.transAxes, ha="right", va="top", fontsize=6, color=INK, fontweight="bold")
    style_ax(a)
    panel_label(a, "a")
    # (b)
    a = ax[1]
    sv, sl = second
    s = FEATS[sv]
    rs = np.random.RandomState(7)
    nb = rs.choice(np.where(normal_mask & ~np.isnan(s) & ~np.isnan(vv))[0], 4000, replace=False)
    a.scatter(vv[nb], s[nb], s=2.5, color=BASE, alpha=.4, lw=0, label="normal (sample)")
    mm = m & ~np.isnan(s) & ~np.isnan(vv)
    a.scatter(vv[mm], s[mm], s=5, color=C5, lw=0, label=f"{name} ({int(m.sum())})")
    a.axvline(thr, color=INK, lw=1.0, ls="--")
    a.set_xlabel(xlabel)
    a.set_ylabel(sl)
    a.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, handletextpad=.2, columnspacing=1.0, markerscale=1.6)
    style_ax(a)
    panel_label(a, "b")
    # (c)
    a = ax[2]
    res = []
    for f, fv in FEATS.items():
        if f == key:
            continue
        if (m & ~np.isnan(fv)).sum() < 20:
            continue
        res.append((_ks(fv[m], fv[normal_mask]), f))
    res.sort(reverse=True)
    top = res[:6][::-1]
    y = np.arange(len(top))
    a.barh(y, [t[0] for t in top], color=C5, height=.55)
    a.set_yticks(y)
    a.set_yticklabels([t[1] for t in top], fontsize=6.4)
    a.set_xlim(0, 1.12)
    a.set_xticks([0, .25, .5, .75, 1.0])
    a.set_xlabel("KS distance to normal accounts")
    for yi, t in zip(y, top):
        a.text(t[0] + .015, yi, f"{t[0]:.2f}", va="center", fontsize=6, color=INK)
    a.grid(axis="y", visible=False)
    style_ax(a)
    a.tick_params(axis="y", labelsize=6.4)
    panel_label(a, "c")
    save(fig, fname)
    return dict(hi=float(hi_g), lo=float(lo_n), top=[(t[1], round(float(t[0]), 3)) for t in top[::-1]])


def fig_lsnan():
    """Group 6 (second part): posterior for the accounts whose behavior_stability is missing."""
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    a = ax[0]
    pc = post[cand]
    bins = np.linspace(0, 1, 21)
    _, b, pt = a.hist(pc, bins=bins, edgecolor="white", lw=.3)
    for p_, left in zip(pt, b[:-1]):
        p_.set_facecolor(C5 if left >= .1 - 1e-9 else BASE)
    a.set_yscale("log")
    a.axvline(.10, color=INK, lw=1.1, ls="--")
    a.set_xlabel("posterior of being a hidden lowstab account")
    a.set_ylabel("candidates (log scale)")
    a.text(.97, .95, f"{int(cand.sum())} candidates\n{int(lsnan.sum())} above 0.10", transform=a.transAxes, ha="right", va="top", fontsize=6.4, color=INK, fontweight="bold")
    style_ax(a)
    panel_label(a, "a")
    a = ax[1]
    inb = ~union & ~bsn & pf & ag
    vis = lowstab & pf & ag
    a.scatter(lm[nor], ps[nor], s=3, color=BASE, alpha=.4, lw=0, label=f"normal in companion box ({int(nor.sum())})")
    a.scatter(lm[vis], ps[vis], s=5, color=C5, lw=0, label=f"visible lowstab ({int(vis.sum())})")
    c_out = cand & ~lsnan
    a.scatter(lm[c_out], ps[c_out], s=14, facecolors="none", edgecolors=MUTED, lw=.6, label=f"candidate, not flagged ({int(c_out.sum())})")
    a.scatter(lm[lsnan], ps[lsnan], s=14, facecolors="none", edgecolors=INK, lw=1.0, marker="D", label=f"candidate, flagged ({int(lsnan.sum())})")
    a.set_xlabel("mean login_count")
    a.set_ylabel("sd of purchase_frequency")
    a.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, handletextpad=.2, columnspacing=.8)
    style_ax(a)
    panel_label(a, "b")
    a = ax[2]
    priors = np.arange(4, 31, 2)
    for thr, c, ls_ in [(.05, MUTED, ":"), (.10, C5, "-"), (.15, MUTED, "--")]:
        ns = []
        for n_ in priors:
            pr = n_ / cand.sum()
            pp = np.where(cand, 1 / (1 + np.exp(-(LLR + np.log(pr / (1 - pr))))), 0.)
            ns.append(int((pp > thr).sum()))
        a.plot(priors, ns, color=c, lw=1.6 if thr == .10 else 1.0, ls=ls_)
        off = {.05: 4.5, .10: 0, .15: -5}[thr]
        a.text(priors[-1] + .6, ns[-1] + off, f"p > {thr:.2f}", fontsize=6, color=INK, va="center")
    a.axvline(14, color=INK, lw=.9, ls="--")
    a.text(14.4, a.get_ylim()[0] + 1, "prior 14", fontsize=6, color=INK, va="bottom")
    a.set_xlim(3, 36)
    a.set_xlabel("assumed number of hidden members")
    a.set_ylabel("accounts flagged")
    style_ax(a)
    panel_label(a, "c")
    fig.tight_layout(w_pad=.8)
    save(fig, "fig_t5_lsnan.png")
    sens = {}
    for n_ in (10, 13, 14, 17, 20):
        pr = n_ / cand.sum()
        pp = np.where(cand, 1 / (1 + np.exp(-(LLR + np.log(pr / (1 - pr))))), 0.)
        sens[n_] = [int((pp > t).sum()) for t in (.05, .10, .15)]
    return dict(sens=sens, vis=int(vis.sum()))


def fig_acckda():
    """Group 7: the 3-metric box."""
    mets = [("performance_stability", (.85, .95)), ("kda_std", (.7, 1.0)), ("accuracy_std", (9., 12.))]
    new = acckda & ~union
    fig = plt.figure(figsize=(6.5, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    stats = {}
    for j, (b, (lo, hi)) in enumerate(mets):
        a = fig.add_subplot(gs[0, j])
        v = mean4(b)
        rng = (np.nanpercentile(v, .1), np.nanpercentile(v, 99.9))
        bins = np.linspace(rng[0], rng[1], 36)
        a.hist(v[normal_mask & ~np.isnan(v)], bins=bins, density=True, color=BASE, alpha=.8, edgecolor="white", lw=.2, label="normal accounts")
        a.hist(v[new & ~np.isnan(v)], bins=bins, density=True, color=C5, alpha=.85, edgecolor="white", lw=.2, label=f"box accounts ({int(new.sum())})")
        a.axvspan(lo, hi, color="#F2F2F2", zorder=0)
        a.set_xlabel(f"mean {b}")
        if j == 0:
            a.set_ylabel("density")
            a.legend(fontsize=5.8, loc="center left", handletextpad=.3)
        inb1 = inbox(b, lo, hi)
        stats[b] = float(inb1[normal_mask].mean())
        a.text(.03, .96, f"{stats[b] * 100:.0f}% of normal\naccounts inside", transform=a.transAxes, ha="left", va="top", fontsize=6, color=INK, fontweight="bold")
        style_ax(a)
        panel_label(a, "abc"[j])
    pairs = [(0, 1), (0, 2), (1, 2)]
    rs = np.random.RandomState(5)
    bg = rs.choice(np.where(normal_mask)[0], 4000, replace=False)
    for j, (p, q) in enumerate(pairs):
        a = fig.add_subplot(gs[1, j])
        bp, (lp, hp) = mets[p]
        bq, (lq, hq) = mets[q]
        a.scatter(mean4(bp)[bg], mean4(bq)[bg], s=2.5, color=BASE, alpha=.35, lw=0)
        a.scatter(mean4(bp)[new], mean4(bq)[new], s=5, color=C5, lw=0)
        a.add_patch(plt.Rectangle((lp, lq), hp - lp, hq - lq, fill=False, ec=INK, lw=1.0, ls="--"))
        a.set_xlabel(f"mean {bp}")
        a.set_ylabel(f"mean {bq}")
        style_ax(a)
        panel_label(a, "def"[j])
    fig.tight_layout(w_pad=.8, h_pad=.9)
    save(fig, "fig_t5_acckda.png")
    allin = acckda & normal_mask
    stats["all_three_normal"] = int(allin.sum())
    stats["all_three"] = int(acckda.sum())
    # expected normal accounts in the box if the three metrics were independent
    stats["indep_expected"] = float(np.prod([stats[b] for b, _ in mets]) * normal_mask.sum())
    return stats


def fig_overlap():
    """UpSet-style chart of the overlaps between the detectors (exclusive membership counts on the final set)."""
    names = ["location jump / latitude", "behaviour shock", "bot", "low activity", "low communication", "low stability", "3-metric box", "low stability, NaN"]
    arrs = [geo, shock, bot, lowact, lowcomm, lowstab, acckda, lsnan]
    M = np.stack(arrs, 1)
    keep = M.any(1)
    pats = pd.Series([tuple(r) for r in M[keep]]).value_counts().sort_values(ascending=False)
    fig = plt.figure(figsize=(6.4, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.25], hspace=.04)
    ax_bar = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)
    xs = np.arange(len(pats))
    ax_bar.bar(xs, pats.values, color=C5, width=.62)
    for x_, v_ in zip(xs, pats.values):
        ax_bar.text(x_, v_ + 8, f"{v_:,}", ha="center", va="bottom", fontsize=6.8, color=INK, fontweight="bold")
    ax_bar.set_ylim(0, pats.values.max() * 1.15)
    ax_bar.set_ylabel("accounts in exactly this combination", fontsize=7)
    ax_bar.tick_params(labelbottom=False, labelsize=6.5)
    ax_bar.grid(axis="x", visible=False)
    ny = len(names)
    for i in range(ny):
        ax_dot.axhspan(i - .5, i + .5, color="#F6F6F6" if i % 2 == 0 else "white", zorder=0)
    for x_, pat in zip(xs, pats.index):
        idx = [i for i, f in enumerate(pat) if f]
        ax_dot.scatter([x_] * ny, np.arange(ny), s=14, color="#DADADA", zorder=2, lw=0)
        ax_dot.scatter([x_] * len(idx), idx, s=18, color=C5 if len(idx) == 1 else INK, zorder=3, lw=0)
        if len(idx) > 1:
            ax_dot.plot([x_, x_], [min(idx), max(idx)], color=INK, lw=1.4, zorder=2)
    ax_dot.set_ylim(ny - .5, -.5)
    ax_dot.set_yticks(np.arange(ny))
    ax_dot.set_yticklabels([f"{n_} ({int(a_.sum()):,})" for n_, a_ in zip(names, arrs)], fontsize=6.6)
    ax_dot.tick_params(axis="y", length=0)
    ax_dot.set_xticks([])
    ax_dot.grid(False)
    for s_ in ("left", "bottom"):
        ax_dot.spines[s_].set_visible(False)
    ax_dot.set_xlim(-.6, len(pats) - .4)
    fig.subplots_adjust(left=.30, right=.99, top=.98, bottom=.03)
    save(fig, "fig_t5_overlap.png")
    return {"patterns": [(tuple(int(x) for x in p), int(c)) for p, c in pats.items()], "total": int(pats.sum())}



def fig_composition():
    """Per-group size on the final 2,104-row set: build-up from the rule to the final file."""
    steps = [("location jump /\nlatitude / shock", int(rule.sum())),
             ("bot", int((bot & ~rule).sum())), ("low activity", int((lowact & ~rule & ~bot).sum())),
             ("low\ncommunication", int((lowcomm & ~rule & ~bot & ~lowact).sum())),
             ("low stability\n(visible)", int((lowstab & ~(rule | bot | lowact | lowcomm)).sum())),
             ("3-metric box", int((acckda & ~union).sum())), ("low stability\nNaN (posterior)", int((lsnan & ~union & ~acckda).sum()))]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    cum = 0
    xs = np.arange(len(steps) + 1)
    for i, (nm, n) in enumerate(steps):
        ax.bar(i, n, bottom=cum, color=C5, width=.62)
        ax.text(i, cum + n + 45, f"+{n:,}" if i else f"{n:,}", ha="center", va="bottom", fontsize=7.2, color=INK, fontweight="bold")
        if i < len(steps):
            ax.plot([i + .31, i + .69], [cum + n, cum + n], color=MUTED, lw=.7)
        cum += n
    ax.bar(len(steps), cum, color=INK, width=.62)
    ax.text(len(steps), cum + 45, f"{cum:,}", ha="center", va="bottom", fontsize=7.6, color=INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps] + ["final file\n(8.13%)"], fontsize=6.4)
    ax.set_ylabel("accounts flagged (cumulative)")
    ax.set_ylim(0, 2500)
    ax.grid(axis="x", visible=False)
    style_ax(ax, 7)
    ax.tick_params(axis="x", labelsize=6.2)
    fig.tight_layout()
    save(fig, "fig_t5_composition.png")
    return dict(steps=steps, total=cum)


def fig_gapscan():
    """Full gap scan of every mean, standard deviation and static feature."""
    known = dict(rule=rule, bot=bot, lowact=lowact, lowcomm=lowcomm, lowstab=lowstab, box=acckda)
    int_base = set()
    for b in BASES:
        v = X[[f"{b}_{p}" for p in P]].astype(float).values
        v = v[~np.isnan(v)]
        if len(v) and np.mean(np.abs(v - np.round(v)) < 1e-9) > .995:
            int_base.add(b)
    MINC, MAXC = 20, 777
    rows = []
    for f, v in FEATS.items():
        idx = np.where(~np.isnan(v))[0]
        o = idx[np.argsort(v[idx])]
        s = v[o]
        n = len(s)
        scale = 1.4826 * np.median(np.abs(s - np.median(s))) + 1e-12
        d = np.diff(s)
        ks = np.r_[np.arange(MINC, MAXC + 1), np.arange(n - MAXC, n - MINC + 1)]
        gi_ = d[ks - 1]
        j = int(np.argmax(gi_))
        kk = ks[j]
        side = o[:kk] if kk < n / 2 else o[kk:]
        ov = {kn: float(km[side].mean()) for kn, km in known.items()}
        kn = max(ov, key=ov.get)
        base = f.rsplit(" ", 1)[0]
        rows.append(dict(feat=f, gap=gi_[j] / scale, size=len(side), grp=kn, frac=ov[kn], discrete=base in int_base))
    df = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)
    top = df.head(24).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = np.arange(len(top))
    for yi, r in zip(y, top.itertuples()):
        if r.frac >= .9:
            col, fc = C5, C5
        else:
            col, fc = MUTED, "white"
        ax.hlines(yi, 0, r.gap, color=col, lw=1.0)
        ax.scatter([r.gap], [yi], s=26, facecolor=fc, edgecolor=col, lw=1.0, zorder=3)
        if r.frac >= .9:
            lab = f"{r.size} accounts, {'shock rule' if r.grp == 'rule' else r.grp if r.grp != 'box' else '3-metric box'}"
        else:
            lab = f"{r.size} accounts, in no group" + (" (integer-valued feature)" if r.discrete else "")
        ax.text(r.gap + .5, yi, lab, va="center", fontsize=5.8, color=INK if r.frac >= .9 else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(top.feat, fontsize=6.2)
    ax.set_xlim(0, 40)
    ax.set_xlabel("widest gap that isolates 20 to 777 accounts (units of robust standard deviation)")
    ax.grid(axis="y", visible=False)
    style_ax(ax, 7)
    ax.tick_params(axis="y", labelsize=6.2)
    fig.tight_layout()
    save(fig, "fig_t5_gapscan.png")
    unexpl = df[df.frac < .9]
    return dict(n_feats=len(df), n_explained=int((df.frac >= .9).sum()), unexpl_top=[(r.feat, round(r.gap, 2), int(r.size), bool(r.discrete)) for r in unexpl.head(6).itertuples()],
                max_unexpl_gap=float(unexpl.gap.max()), min_expl_gap_top=float(df[df.frac >= .9].gap.min()), df=df)


# z matrix for the selling figure (same computation as the rule)
ZBASES = bases[:len([b for b in bases])]
ZMAT = np.nan_to_num(np.stack(Z, 1))
ZBASES = [b for b in bases]

if __name__ == "__main__":
    apply_style()
    plt.rcParams["savefig.bbox"] = "tight"
    os.makedirs(OUT, exist_ok=True)
    print(COUNTS)
    fig_shock()
    fig_gaps()
    fig_box()
    fig_breakdown()
    print("takeover", fig_takeover())
    print("selling", fig_selling())
    print("flags", fig_flags())
    print("bot", fig_group("bot", "login_time_variance mean", mean4("login_time_variance"), 1.0, (0, 8), "mean login_time_variance",
                           ("ip_hash_entropy mean", "mean ip_hash_entropy"), bot, "fig_t5_bot.png"))
    print("lowact", fig_group("low activity", "activity_level mean", mean4("activity_level"), .34, (0, 1.8), "mean activity_level",
                              ("skill_rating mean", "mean skill_rating"), lowact, "fig_t5_lowact.png"))
    print("lowcomm", fig_group("low communication", "communication_frequency mean", mean4("communication_frequency"), .17, (0, 1.5), "mean communication_frequency",
                               ("friend_adds mean", "mean friend_adds"), lowcomm, "fig_t5_lowcomm.png"))
    print("lowstab", fig_group("low stability", "behavior_stability", X["behavior_stability"].values, .55, (.1, 1.), "behavior_stability (static)",
                               ("purchase_frequency mean", "mean purchase_frequency"), lowstab, "fig_t5_lowstab.png"))
    print("lsnan", fig_lsnan())
    print("acckda", fig_acckda())
    print("overlap", fig_overlap())
    print("composition", fig_composition())
    gs_ = fig_gapscan()
    gs_.pop("df")
    print("gapscan", gs_)
