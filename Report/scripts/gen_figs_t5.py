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

# ---------------------------------------------------------------------------------------------
# Discovery path of the 3-metric box, box precision and negative scans (12 Nov release)
# ---------------------------------------------------------------------------------------------
import itertools  # noqa: E402

DISC = ["device_count", "refund_count", "guild_changes"]
FLAGS = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
PMET = [b for b in bases]                     # 27 behavioural metrics (no lat/lon)
CONTM = [b for b in PMET if b not in DISC]    # 24 continuous metrics
MEANS = pd.DataFrame({b: mean4(b) for b in CONTM})
MEANS["account_age_days"] = X["account_age_days"].values
MEANS["behavior_stability"] = X["behavior_stability"].values   # 26 features -> C(26,3)=2600, C(26,4)=14950
SDS = pd.DataFrame({b: X[[f"{b}_{p}" for p in P]].astype(float).std(axis=1).values for b in CONTM})


def pairwise_tails(o):
    """5%-tail co-occurrence of every cross-metric pair outside the union (as in the discovery round)."""
    Mo = MEANS[o]
    med = Mo.median()
    sc = (Mo.quantile(.75) - Mo.quantile(.25)) / 1.349
    Zr_ = (Mo - med) / sc
    FF = pd.concat([Zr_.add_suffix("|mean"), SDS[o].add_suffix("|sd")], axis=1)
    T = {}
    for c in FF.columns:
        T[c + "<"] = (FF[c] < FF[c].quantile(.05)).values
        T[c + ">"] = (FF[c] > FF[c].quantile(.95)).values
    for f in FLAGS:
        T[f] = (X.loc[o, f] == 1).values
    T = pd.DataFrame(T).astype(float)
    cnt = (T.T @ T).values
    pr = T.mean().values
    E = np.outer(pr, pr) * len(T)
    Zp = (cnt - E) / np.sqrt(E + 1e-9)
    nm = list(T.columns)
    base_of = lambda s: s.rstrip("<>").split("|")[0]  # noqa: E731
    return nm, cnt, E, Zp, base_of


def triple_scan(F, B=8):
    R = F.rank(pct=True, method="first").values
    ok = ~F.isna().values
    Q = np.minimum((np.nan_to_num(R) * B).astype(int), B - 1)
    out = []
    for i, j, k in itertools.combinations(range(F.shape[1]), 3):
        v = ok[:, i] & ok[:, j] & ok[:, k]
        c = np.bincount(Q[v, i] * B * B + Q[v, j] * B + Q[v, k], minlength=B ** 3)
        e = v.sum() / B ** 3
        out.append(((c.max() - e) / np.sqrt(e), F.columns[i], F.columns[j], F.columns[k], int(c.max()), e))
    return sorted(out, key=lambda t: -t[0])


def scan4(D, B=5):
    R = D.rank(pct=True, method="first").values
    ok = ~D.isna().values
    Q = np.minimum((np.nan_to_num(R) * B).astype(int), B - 1)
    best = (-1, None, 0, 0)
    for c in itertools.combinations(range(D.shape[1]), 4):
        v = ok[:, c].all(1)
        idx = ((Q[v, c[0]] * B + Q[v, c[1]]) * B + Q[v, c[2]]) * B + Q[v, c[3]]
        cnt = np.bincount(idx, minlength=B ** 4)
        e = v.sum() / B ** 4
        z = (cnt.max() - e) / np.sqrt(e)
        if z > best[0]:
            best = (z, [D.columns[i] for i in c], int(cnt.max()), e)
    return best


def box_pair_scan(rows, K=13):
    """Indicators 'all 4 periods in [a,b]' on a K-point grid (intervals up to half the range), cross-metric pairs."""
    ind, lab = [], []
    for b in CONTM:
        m = X.loc[rows, [f"{b}_{p}" for p in P]].values.astype(float)
        lo, hi = np.nanpercentile(m, .5), np.nanpercentile(m, 99.5)
        g = np.linspace(lo, hi, K)
        for i in range(K):
            for j in range(i + 1, K):
                if g[j] - g[i] > (hi - lo) / 2:
                    continue
                ind.append((((m >= g[i]) & (m <= g[j])) | np.isnan(m)).all(1))
                lab.append((b, g[i], g[j]))
    I = np.array(ind, dtype=np.float32)
    cnt = I @ I.T
    pr = I.mean(1)
    n = I.shape[1]
    E = np.outer(pr, pr) * n
    Zb = (cnt - E) / np.sqrt(E + 1)
    names = np.array([l[0] for l in lab])
    valid = (names[:, None] != names[None, :]) & (E >= 2) & np.triu(np.ones(E.shape, dtype=bool), 1)
    zz = np.where(valid, Zb, -np.inf)
    a, c = np.unravel_index(np.argmax(zz), zz.shape)
    return float(zz[a, c]), lab[a], lab[c], int(cnt[a, c]), float(E[a, c]), int(valid.sum()), len(lab)


def fig_discovery():
    res = {}
    o = ~union
    nm, cnt, E, Zp, base_of = pairwise_tails(o)
    bs = [base_of(s) for s in nm]
    ub = sorted({b for b in bs if b in CONTM + ["account_age_days", "behavior_stability"]})
    H_ = np.full((len(ub), len(ub)), np.nan)
    for i in range(len(nm)):
        for j in range(i + 1, len(nm)):
            if bs[i] == bs[j] or bs[i] not in ub or bs[j] not in ub:
                continue
            ii, jj = ub.index(bs[i]), ub.index(bs[j])
            if np.isnan(H_[ii, jj]) or Zp[i, j] > H_[ii, jj]:
                H_[ii, jj] = H_[jj, ii] = Zp[i, j]
    flat = sorted([(Zp[i, j], nm[i], nm[j], int(cnt[i, j]), float(E[i, j])) for i in range(len(nm)) for j in range(i + 1, len(nm))
                   if bs[i] != bs[j]], reverse=True)
    res["pair_top"] = flat[:3]
    res["pair_second_z"] = [f for f in flat if {"accuracy_std", "kda_std"} != {base_of(f[1]), base_of(f[2])}][0][:3]
    res["n_pairs_base"] = len(ub) * (len(ub) - 1) // 2
    ia, ik = ub.index("accuracy_std"), ub.index("kda_std")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(H_, cmap="Greys", vmin=0, vmax=8, origin="lower")
    ax.set_xticks(range(len(ub)))
    ax.set_yticks(range(len(ub)))
    ax.set_xticklabels(ub, rotation=90, fontsize=5.6)
    ax.set_yticklabels(ub, fontsize=5.6)
    ax.grid(False)
    for (r_, c_) in [(ia, ik), (ik, ia)]:
        ax.add_patch(plt.Rectangle((c_ - .5, r_ - .5), 1, 1, fill=False, ec=C5, lw=2.2))
    t0 = res["pair_top"][0]
    ax.annotate(f"accuracy_std x kda_std\nobserved {t0[3]} vs {t0[4]:.0f} expected\nz = {t0[0]:.1f}",
                xy=(min(ik, ia) + .5, max(ik, ia) + .5), xytext=(1, len(ub) - 3.5), fontsize=7, color=INK, fontweight="bold",
                bbox=dict(fc="white", ec=C5, lw=1.2, pad=3))
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("largest tail co-occurrence z of the pair (colour clipped at 8)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"{len(ub)} features, {res['n_pairs_base']} pairs, outside the {int(union.sum()):,}-row union", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig_t5_pairheat.png")

    t6 = triple_scan(MEANS[o], 8)
    zs = np.array([t[0] for t in t6])
    res["triple_n"] = len(t6)
    res["triple_top"] = t6[0]
    res["triple_median"] = float(np.median(zs))
    res["triple_next"] = [t for t in t6 if {"accuracy_std", "kda_std", "performance_stability"} != {t[1], t[2], t[3]}][0]
    res["triple_withpair_n"] = len([t for t in t6 if {"accuracy_std", "kda_std"} <= {t[1], t[2], t[3]}])
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.hist(zs, bins=60, color=BASE, edgecolor="white", lw=.3)
    ax.set_yscale("log")
    ax.axvline(res["triple_median"], color=INK, lw=1, ls="--")
    ax.text(res["triple_median"] + .2, 300, f"median z = {res['triple_median']:.1f}", fontsize=7.2, color=INK)
    t0 = res["triple_top"]
    ax.axvline(t0[0], color=C5, lw=2)
    ax.annotate(f"({t0[1]}, {t0[2]}, {t0[3]})\nz = {t0[0]:.1f}: {t0[4]} rows in one cell,\n{t0[5]:.0f} expected",
                xy=(t0[0], 1.5), xytext=(t0[0] - 9, 25), fontsize=7, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C5, lw=1.2))
    ax.set_xlabel(f"grid-scan z of each of the {len(t6):,} feature triples (8 x 8 x 8 rank cells, fullest cell)", fontsize=7.5)
    ax.set_ylabel("triples (log scale)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_t5_triplescan.png")
    return res


def box_stats():
    """Edges, null precision and sensitivity of the 3-metric box."""
    mets = [("performance_stability", (.85, .95)), ("kda_std", (.7, 1.0)), ("accuracy_std", (9., 12.))]
    members = acckda & ~union
    res = {"members": int(members.sum())}
    res["edges"] = {}
    for b, (lo, hi) in mets:
        v = X.loc[members, [f"{b}_{p}" for p in P]].values.astype(float)
        res["edges"][b] = (float(np.nanmin(v)), float(np.nanmax(v)))
    base_rows = ~union & ~acckda
    dists = []
    for b, (lo, hi) in mets:
        m = X.loc[base_rows, [f"{b}_{p}" for p in P]].values.astype(float)
        out_ = ((m < lo) | (m > hi)).sum(1)
        dists.append(np.bincount(out_, minlength=5)[:5] / len(m))
    conv = np.convolve(np.convolve(dists[0], dists[1]), dists[2])
    nrows = int((~union).sum())
    tot = 0
    for b, (lo, hi) in mets:
        m = X.loc[~union, [f"{b}_{p}" for p in P]].values.astype(float)
        tot = tot + ((m < lo) | (m > hi)).sum(1)
    res["null_table"] = [(k, int((tot == k).sum()), float(conv[k] * nrows)) for k in range(4)]
    res["precision"] = 1 - res["null_table"][0][2] / res["null_table"][0][1]
    scales = np.linspace(.7, 1.3, 61)
    def count_at(scl):
        ok = np.ones(len(X), bool)
        for b, (lo, hi) in mets:
            c_, h_ = (lo + hi) / 2, (hi - lo) / 2 * scl[b]
            ok &= inbox(b, c_ - h_, c_ + h_)
        return int((ok & ~union).sum())
    curves = {"all three": [count_at({b: s_ for b, _ in mets}) for s_ in scales]}
    for b0, _ in mets:
        curves[b0] = [count_at({b: (s_ if b == b0 else 1.) for b, _ in mets}) for s_ in scales]
    res["n_at"] = {k: {sc: v[int(np.argmin(np.abs(scales - sc)))] for sc in (.8, .9, 1.0, 1.1, 1.3)} for k, v in curves.items()}
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    sty = {"all three": (C5, 2.2, "-"), "performance_stability": (INK, 1.2, "--"), "kda_std": (BLUE, 1.2, "-."), "accuracy_std": (AMBER, 1.2, ":")}
    for k, v in curves.items():
        ax.plot(scales * 100, v, color=sty[k][0], lw=sty[k][1], ls=sty[k][2], label=("only " + k + " scaled") if k != "all three" else "all three intervals scaled")
    ax.axvline(100, color=INK, lw=.8, ls="--")
    ax.set_xlabel("scale of the interval(s) around the centre (100% = the box of the notebook)", fontsize=7.5)
    ax.set_ylabel("rows inside, outside the union", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_t5_boxsens.png")
    return res


def negative_scans():
    res = {}
    o = ~final
    res["boxpair_all"] = box_pair_scan(np.ones(len(X), bool))
    res["boxpair_out"] = box_pair_scan(o)
    Fn = MEANS[o].reset_index(drop=True)
    res["scan4_real"] = scan4(Fn)
    nul = []
    for s in (1, 2):
        g = np.random.default_rng(s)
        nul.append(scan4(pd.DataFrame({c: g.permutation(Fn[c].values) for c in Fn.columns}))[0])
    res["scan4_null"] = nul
    res["scan4_cal"] = scan4(MEANS[o | (acckda & ~union)].reset_index(drop=True))[0]
    ok = o & ~np.isnan(hl) & ~np.isnan(hn)
    h, _, _ = np.histogram2d(hl[ok], hn[ok], bins=[np.arange(-90, 90.5, .5), np.arange(-180, 180.5, .5)])
    res["geo_max"] = int(h.max())
    try:
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler
        D = Fn.fillna(Fn.median())
        sub = StandardScaler().fit_transform(D.sample(6000, random_state=0).values)
        lab_r = HDBSCAN(min_cluster_size=60).fit_predict(sub)
        gs = np.random.default_rng(3)
        shf = np.stack([gs.permutation(sub[:, i]) for i in range(sub.shape[1])], 1)
        lab_s = HDBSCAN(min_cluster_size=60).fit_predict(shf)
        res["hdb_real"] = (int(lab_r.max() + 1), int((lab_r == -1).sum()))
        res["hdb_shuf"] = (int(lab_s.max() + 1), int((lab_s == -1).sum()))
    except Exception as e:  # pragma: no cover
        res["hdb_err"] = str(e)
    return res


def write_scan_csv(disc, bx, neg):
    R = "release 12 Nov"
    rows = []
    t = disc["pair_top"][0]
    rows.append(("pairwise_tail_top", R, f"z={t[0]:.2f}", f"{t[1]} x {t[2]} observed {t[3]} expected {t[4]:.1f}"))
    rows.append(("pairwise_tail_next_best", R, f"z={disc['pair_second_z'][0]:.2f}", f"{disc['pair_second_z'][1]} x {disc['pair_second_z'][2]}"))
    t = disc["triple_top"]
    rows.append(("triple_scan_top", R, f"z={t[0]:.2f}", f"{t[1]},{t[2]},{t[3]} cell {t[4]} expected {t[5]:.1f}; {disc['triple_n']} triples, median z {disc['triple_median']:.2f}"))
    t = disc["triple_next"]
    rows.append(("triple_scan_best_other", R, f"z={t[0]:.2f}", f"{t[1]},{t[2]},{t[3]}"))
    rows.append(("box_members", R, str(bx["members"]), "rows inside the box outside the union"))
    for b, (lo, hi) in bx["edges"].items():
        rows.append((f"box_edges_{b}", R, f"[{lo:.4f},{hi:.4f}]", "min/max of the per-period values of the box members"))
    for k, obs, exp in bx["null_table"]:
        rows.append((f"null_out_of_box_{k}", R, f"obs {obs} exp {exp:.2f}", "rows (outside the union) with k observed out-of-box period values"))
    for k, d in bx["n_at"].items():
        rows.append((f"sensitivity_{k}", R, f"80%: {d[.8]}, 90%: {d[.9]}, 100%: {d[1.0]}, 110%: {d[1.1]}, 130%: {d[1.3]}", "rows inside after scaling the interval(s) around the centre"))
    for key in ("boxpair_all", "boxpair_out"):
        r = neg[key]
        rows.append((key, R, f"z={r[0]:.2f}", f"{r[1][0]} x {r[2][0]}; {r[5]} valid pairs of {r[6]} indicators"))
    r = neg["scan4_real"]
    rows.append(("scan4_real", R, f"z={r[0]:.2f}", ", ".join(r[1])))
    rows.append(("scan4_null", R, str([round(float(x), 2) for x in neg["scan4_null"]]), "column-shuffled nulls"))
    rows.append(("scan4_calibration", R, f"z={neg['scan4_cal']:.2f}", "box rows put back"))
    rows.append(("geo_densest_cell", R, str(neg["geo_max"]), "rows in the densest 0.5 degree home cell outside the final set"))
    if "hdb_real" in neg:
        rows.append(("hdbscan_real_vs_shuffled", R, f"{neg['hdb_real']} / {neg['hdb_shuf']}", "(clusters, noise rows), 6,000 normal rows"))
    pd.DataFrame(rows, columns=["quantity", "release", "value", "note"]).to_csv(os.path.join(ROOT, "data", "t5_scan_summary.csv"), index=False)


def run_discovery():
    disc = fig_discovery()
    bx = box_stats()
    neg = negative_scans()
    write_scan_csv(disc, bx, neg)
    for k, v in {**disc, **{"bx_" + k: v for k, v in bx.items()}, **{"neg_" + k: v for k, v in neg.items()}}.items():
        print(k, v)


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
    run_discovery()
