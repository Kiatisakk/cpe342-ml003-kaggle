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


if __name__ == "__main__":
    apply_style()
    plt.rcParams["savefig.bbox"] = "tight"
    os.makedirs(OUT, exist_ok=True)
    print(COUNTS)
    fig_shock()
    fig_gaps()
    fig_box()
    fig_breakdown()
