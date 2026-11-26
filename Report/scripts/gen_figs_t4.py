"""Figures for Task 4 (train images and logged numbers only). Run from Report/: python scripts/gen_figs_t4.py
Needs the competition data folder; set T4_DATA to override."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import *

apply_style()
DATA = os.environ.get("T4_DATA", r"C:/Users/kiati/Documents/Kaggle-ML/cpe342-kaggle-main/04_image_classification")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task4", "figures")
os.makedirs(OUT, exist_ok=True)
T4C = TASK_COLORS[4]
NAMES = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]
CCOL = PALETTE  # one colour per class, always direct-labelled
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
K = 5
cnt = tr.label.value_counts().sort_index().values
prior = cnt / cnt.sum()


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# 1. class balance + the two decisions it drives
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
b = ax[0].bar(NAMES, cnt, color=T4C)
for r, c, p in zip(b, cnt, prior):
    ax[0].text(r.get_x() + r.get_width() / 2, c + 200, f"{c:,}\n{p*100:.1f}%", ha="center", fontsize=8.5, color=INK)
ax[0].set_ylim(0, cnt.max() * 1.2)
ax[0].set_ylabel("training images")
ax[0].set_title("(a) class counts, 31,546 train images")
ax[0].tick_params(axis="x", labelsize=8)
w = 1 / np.sqrt(cnt)
w = w / w.mean()
x = np.arange(K)
wd = 0.38
ax[1].bar(x - wd / 2, w, wd, color=T4C, label=r"loss weight $\propto 1/\sqrt{n_k}$ (mean 1)")
ax[1].bar(x + wd / 2, -np.log(prior), wd, color=DROPPED, label=r"inference offset $-\log\pi_k$")
ax[1].set_xticks(x)
ax[1].set_xticklabels(NAMES, fontsize=8, rotation=15)
ax[1].legend(loc="upper left", fontsize=7.8)
ax[1].set_ylim(0, 2.9)
ax[1].set_title("(b) what we derive from the counts")
fig.tight_layout()
save(fig, "fig_t4_class_balance.png")

# 2. sample grid, 2 train images per class
fig, axes = plt.subplots(2, K, figsize=(10, 2.75))
for k in range(K):
    fns = tr.loc[tr.label == k, "file_name"].sample(2, random_state=7 + k).tolist()
    for j, fn in enumerate(fns):
        a = axes[j, k]
        a.imshow(Image.open(os.path.join(DATA, "train", fn)).convert("RGB"))
        a.axis("off")
        if j == 0:
            a.set_title(f"{k}: {NAMES[k]}", fontsize=9.5, color=INK)
fig.subplots_adjust(wspace=0.04, hspace=0.02, top=0.92, bottom=0.01, left=0.01, right=0.99)
save(fig, "fig_t4_samples.png")

# 3. native size and colour statistics (train only)
S = pd.concat([g.sample(250, random_state=42) for _, g in tr.groupby("label")], ignore_index=True)
br, sa, ws, hs = [], [], [], []
for fn in S.file_name:
    im = Image.open(os.path.join(DATA, "train", fn))
    ws.append(im.size[0])
    hs.append(im.size[1])
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    br.append(a.mean())
    mx, mn = a.max(2), a.min(2)
    sa.append(np.mean((mx - mn) / np.maximum(mx, 1e-6)))
S["br"], S["sa"] = br, sa
print("sizes", set(zip(ws, hs)), "n", len(S))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
for k in range(K):
    for a, col, lab in ((ax[0], "br", "mean brightness"), (ax[1], "sa", "mean saturation")):
        v = S.loc[S.label == k, col].values
        xs = np.linspace(0, 1, 300)
        a.plot(xs, gaussian_kde(v)(xs), color=CCOL[k], lw=1.8, label=NAMES[k])
        a.set_xlabel(lab)
ax[0].set_ylabel("density")
ax[0].legend(fontsize=7.5)
ax[0].set_title("(a) brightness by title")
ax[1].set_title("(b) saturation by title")
fig.tight_layout()
save(fig, "fig_t4_appearance.png")

# 4. near-duplicate scenes (train -> train thumbnails, 500 per class)
D = pd.concat([g.sample(500, random_state=0) for _, g in tr.groupby("label")], ignore_index=True)
V = []
for fn in D.file_name:
    im = Image.open(os.path.join(DATA, "train", fn)).convert("RGB").resize((28, 16), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float32).ravel()
    v = (v - v.mean()) / (v.std() + 1e-6)
    V.append(v / (np.linalg.norm(v) + 1e-6))
V = np.stack(V)
sim = V @ V.T
np.fill_diagonal(sim, -1)
nn = sim.argmax(1)
ns = sim[np.arange(len(V)), nn]
same = D.label.values == D.label.values[nn]
print("median nn", np.median(ns), ">0.7", (ns > .7).mean())
q = pd.qcut(ns, 5)
pur = pd.Series(same).groupby(q, observed=False).mean() * 100
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3))
ax[0].hist(ns, bins=40, color=T4C, edgecolor="white")
ax[0].axvline(np.median(ns), color=INK, ls="--", lw=1.2)
ax[0].text(np.median(ns) + .01, ax[0].get_ylim()[1] * .9, f"median {np.median(ns):.2f}", fontsize=8.5)
ax[0].set_xlabel("cosine to nearest other train thumbnail")
ax[0].set_ylabel("images")
ax[0].set_title("(a) scene redundancy inside train")
ax[1].bar(range(5), pur.values, color=T4C)
ax[1].set_xticks(range(5))
ax[1].set_xticklabels([f"Q{i+1}\n{iv.left:.2f}-{iv.right:.2f}" for i, iv in enumerate(pur.index)], fontsize=8)
for i, v in enumerate(pur.values):
    ax[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5)
ax[1].set_ylim(0, 100)
ax[1].set_ylabel("neighbour has same title (%)")
ax[1].set_title("(b) label purity by similarity quintile")
fig.tight_layout()
save(fig, "fig_t4_neardup.png")

# 5. random vs scene-group CV, per family (numbers from the pipeline log; see caption)
rows = [  # name, random 5-fold OOF, scene-group (8 clusters per class)
    ("SigLIP2 head", 0.9289, 0.83814), ("PE-Core-L head", 0.9504, 0.85329), ("C-RADIOv4-H head", 0.9643, 0.87896),
    ("DINOv3 ViT-H+ head", 0.9639, 0.89274), ("C-RADIOv3-H head", 0.9712, 0.88558), ("DINOv2-g head", 0.9588, 0.88731),
    ("DINOv3 ViT-7B head (last block)", 0.97824, 0.9206), ("DINOv3 ViT-7B head (75% depth)", 0.9862, 0.9343),
]
rows.sort(key=lambda r: r[2])
fig, ax = plt.subplots(figsize=(8.6, 3.9))
for i, (n, r, s) in enumerate(rows):
    ax.plot([s, r], [i, i], color=AXIS, lw=2, zorder=1)
    ax.scatter(r, i, color=DROPPED, s=46, zorder=2, label="random 5-fold OOF" if i == 0 else None)
    ax.scatter(s, i, color=T4C, s=50, zorder=3, label="scene-group CV (8 clusters per class)" if i == 0 else None)
    ax.text(r + .002, i, f"gap {r-s:.3f}", va="center", fontsize=7.8, color=INK_MUTED)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.3)
ax.set_xlim(0.82, 1.0); ax.set_ylim(-0.6, len(rows) + 0.4)
ax.set_xlabel("macro F1 (linear head on frozen features)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y", visible=False)
save(fig, "fig_t4_cv_gap.png")

# 6. soup fold-0 errors and LoRA weight curve
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
names = ["l75 LoRA\n(avg head)", "l75 +\nrefit head", "c30f10 +\nrefit head", "soup\n(avg head)", "soup +\nrefit head", "soup + 2-view\nrefit head"]
raw = [42, 36, 34, 41, 30, 28]
pri = [30, 22, 32, 26, 21, 20]
x = np.arange(len(names))
w = .38
ax[0].bar(x - w / 2, raw, w, color=DROPPED, label="argmax")
ax[0].bar(x + w / 2, pri, w, color=T4C, label=r"with train prior ($a=1$)")
for i in range(len(x)):
    ax[0].text(x[i] - w / 2, raw[i] + .6, raw[i], ha="center", fontsize=8)
    ax[0].text(x[i] + w / 2, pri[i] + .6, pri[i], ha="center", fontsize=8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(names, fontsize=7.4)
ax[0].set_ylabel("errors on scene fold 0 (5,113 images)")
ax[0].set_title("(a) LoRA head and soup, fold-0 errors")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_ylim(0, 50)
lw = [.5, .7, .85, 1.0]
lb = [.87028, .87104, .87109, .87074]
ax[1].plot(lw, lb, "-o", color=T4C, lw=2)
for a, b in zip(lw, lb):
    ax[1].text(a, b + .00006, f"{b:.5f}", ha="center", fontsize=7.6)
ax[1].set_xlabel("LoRA weight in the blend")
ax[1].set_ylabel("public LB")
ax[1].set_ylim(.8698, .8716)
ax[1].set_xlim(.4, 1.1)
ax[1].set_title("(b) LoRA weight, public LB")
fig.tight_layout()
save(fig, "fig_t4_lora_soup.png")


# =====================================================================================================
# Part 2 (expanded Task 4): timeline, backbones, OOF-vs-LB, LB curves, scene-CV, error analysis.
# Inputs: data/t4_lb_history.csv, t4_backbones.csv, t4_oof_vs_lb.csv (hand-entered from the CLAUDE.md log,
# see the source column) and two small caches made once from the local pipeline repo:
#   data/t4_fold0_cache.npz : scene-fold-0 out-of-fold log-probs of the LoRA soup (refit head), labels, train prior
#   data/t4_scene_cache.npz : scene group / scene fold ids, train->train nearest-neighbour cosine (RADIOv3), labels
# Only train data and logged numbers are used.
# =====================================================================================================
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

DD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SC = {"kept": KEPT, "dropped": DROPPED, "rule_excluded": RULE_EXCLUDED}
F1_OF_LB = lambda lb: 5 * lb - 3.389           # implied public Task 4 macro F1 (CLAUDE.md: other tasks at their validation values)
LB_OF_F1 = lambda f: (f + 3.389) / 5


# ---- 7. LB timeline of every Task-4 submission ------------------------------------------------------
H = pd.read_csv(os.path.join(DD, "t4_lb_history.csv"))
H["x"] = H.day + (H.order - 3) * 0.15
fig, ax = plt.subplots(figsize=(10.4, 5.9))
kp = H[H.status == "kept"]
ax.plot(kp.x, kp.lb, color=T4C, lw=1.6, alpha=.55, zorder=1)
for st, mk in (("dropped", "o"), ("rule_excluded", "s"), ("kept", "o")):
    d = H[H.status == st]
    if st == "rule_excluded":
        ax.scatter(d.x, d.lb, s=46, marker=mk, facecolor="white", edgecolor=RULE_EXCLUDED, hatch="////", linewidth=1.1, zorder=3)
    else:
        ax.scatter(d.x, d.lb, s=40 if st == "dropped" else 62, marker=mk, color=SC[st], zorder=4 if st == "kept" else 2,
                   edgecolor="white", linewidth=.6)
LAB = {  # file: (label, dx, dy, ha)
    "v25a": ("v25a realigned\n0.85787", 10, 26, "left"),
    "v28": ("v28 + ConvNeXt-S DINOv3 0.85906", 14, -30, "left"),
    "v30": ("v30 + ConvNeXt-B 0.86169", -14, -6, "right"),
    "v32": ("v31/v32 frozen heads\n30% / 50%  0.86430", -16, 6, "right"),
    "v34_em_full": ("v33/v34 EM prior 0.86750", -12, 12, "right"),
    "v37_em": ("v37 EM 0.86837", -8, 22, "right"),
    "v37n_noEM": ("v37n no EM 0.86494", 4, -34, "left"),
    "v74b_clean": ("v74b_clean 0.86812\nfirst clean 7B-head file", 8, -34, "left"),
    "v79a_clean": ("v79a LoRA .5 0.87028", 10, -32, "left"),
    "v79c_clean": ("v79b/v79c LoRA .7/.85\n0.87109", -90, 30, "right"),
    "v81b_prior": ("v82b/v81b\ntrain prior 0.87307", -10, 22, "right"),
    "v85a_soup": ("v85a soup 0.87346", -24, 36, "right"),
    "v90f": ("v90f final 0.87351", -8, 22, "right"),
    "v79b_em": ("EM on the LoRA blend 0.85993", -8, 0, "right"),
    "v26_eq3effv2": ("v26 + EffNetV2-S 0.85630", 10, -14, "left"),
}
for fn, (t, dx, dy, ha) in LAB.items():
    r = H[H.file == fn].iloc[0]
    ax.annotate(t, (r.x, r.lb), xytext=(dx, dy), textcoords="offset points", fontsize=7.8, color=INK, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=.7, shrinkA=0, shrinkB=3))
ax.set_ylim(0.8535, 0.8785)
ax.set_xlim(11.4, 24.6)
ax.set_xticks(range(12, 25))
ax.set_xlabel("day of November 2026 (submissions of one day are spread horizontally in order)")
ax.set_ylabel("public leaderboard score")
sec = ax.secondary_yaxis("right", functions=(F1_OF_LB, LB_OF_F1))
sec.set_ylabel("implied public Task 4 macro F1  $=5\\cdot$LB$-3.389$", color=INK_MUTED, fontsize=8.5)
sec.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98])
sec.tick_params(labelsize=8)
for x0, x1, t in ((11.4, 14.5, "fine-tuned\nensembles"), (14.5, 16.5, "EM prior\nstudy"), (16.5, 19.5, "ViT-7B heads\nand LoRA"),
                  (19.5, 24.6, "train prior, soup,\ncalibration")):
    ax.text((x0 + x1) / 2, 0.8782, t, ha="center", va="top", fontsize=8, color=INK_MUTED, style="italic")
for xv in (14.5, 16.5, 19.5):
    ax.axvline(xv, color=AXIS, lw=.8, ls=":")
hd = [Line2D([], [], marker="o", color="w", markerfacecolor=KEPT, markersize=8, label="kept (path to the final recipe)"),
      Line2D([], [], marker="o", color="w", markerfacecolor=DROPPED, markersize=7, label="tested, dropped"),
      Line2D([], [], marker="s", color="w", markerfacecolor="white", markeredgecolor=RULE_EXCLUDED, markersize=8,
             label="uses a test-derived EM prior (rule-excluded)")]
ax.legend(handles=hd, loc="lower right", fontsize=8, bbox_to_anchor=(1.0, 0.1))
save(fig, "fig_t4_timeline.png")


# ---- 8. backbone comparison -------------------------------------------------------------------------
B = pd.read_csv(os.path.join(DD, "t4_backbones.csv"))
fig = plt.figure(figsize=(10.2, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], width_ratios=[1.3, 1], hspace=.3, wspace=1.05)
a0 = fig.add_subplot(gs[0, 0])
ft = B[(B.kind == "finetune") & B.rand_oof.notna()].sort_values("rand_oof")
a0.barh(ft.name, ft.rand_oof, color=[SC[r] for r in ft.role], height=.62)
for i, v in enumerate(ft.rand_oof):
    a0.text(v + .0004, i, f"{v:.4f}", va="center", fontsize=7.8)
a0.set_xlim(.965, .9955)
a0.set_xlabel("random 5-fold OOF macro F1")
a0.set_title("(a) fine-tuned models")
a0.tick_params(axis="y", labelsize=7.8)
a0.grid(axis="y", visible=False)
a2 = fig.add_subplot(gs[0, 1])
lo = B[B.kind == "lora"].copy()
lo["err"] = lo.fold0_err
lo = lo.sort_values("err", ascending=True)
lab = ["c40f28\n(all blocks)", "c24f12\n(60% depth)", "l75\n(20-29)", "c30f10\n(10-29)", "l75la\n(logit-adj.)", "soup +\nrefit"]
nm = {"LoRA 7B c40f28 (all 40 blocks)": "c40f28\nall blocks", "LoRA 7B c24f12 (60% depth)": "c24f12\n60% depth", "LoRA 7B l75 (blocks 20-29)": "l75\nblocks 20-29",
      "LoRA 7B c30f10 (blocks 10-29)": "c30f10\nblocks 10-29", "LoRA 7B l75la (logit-adjusted)": "l75la\nlogit-adj.", "LoRA soup + refit head": "soup +\nrefit head"}
a2.barh([nm[n].replace("\n", " ") for n in lo.name], lo.err, color=[SC[r] for r in lo.role], height=.66)
for i, v in enumerate(lo.err):
    a2.text(v + 1.2, i, int(v), va="center", fontsize=8)
a2.set_xlim(0, 80)
a2.set_xlabel("errors on scene fold 0 (5,113 images)")
a2.set_title("(b) LoRA ViT-7B variants")
a2.tick_params(axis="y", labelsize=7.6)
a2.grid(axis="y", visible=False)
a1 = fig.add_subplot(gs[1, :])
fr = B[(B.kind == "frozen") & B.scene_sig8.notna()].sort_values("scene_sig8").reset_index(drop=True)
for i, r in fr.iterrows():
    a1.plot([r.scene_sig8, r.rand_oof if pd.notna(r.rand_oof) else r.scene_sig8], [i, i], color=AXIS, lw=2, zorder=1)
    if pd.notna(r.rand_oof):
        a1.scatter(r.rand_oof, i, s=40, facecolor="white", edgecolor=INK_MUTED, linewidth=1.2, zorder=2)
    a1.scatter(r.scene_sig8, i, s=58, color=SC[r.role], zorder=3)
    if pd.notna(r.scene_sgkf):
        a1.scatter(r.scene_sgkf, i, s=30, marker="D", color=T4C, zorder=3)
a1.set_yticks(range(len(fr)))
a1.set_yticklabels(fr.name, fontsize=8)
a1.set_xlim(.74, 1.0)
a1.set_xlabel("macro F1 of a linear head on frozen features")
a1.set_title("(c) frozen-feature linear heads: random CV versus held-out scenes")
a1.title.set_fontsize(9.2)
a1.grid(axis="y", visible=False)
hd = [Line2D([], [], marker="s", color="w", markerfacecolor=KEPT, markersize=8, label="in the final blend"),
      Line2D([], [], marker="s", color="w", markerfacecolor=DROPPED, markersize=8, label="tested, not in the final blend"),
      Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK_MUTED, markersize=7, label="random 5-fold OOF"),
      Line2D([], [], marker="D", color="w", markerfacecolor=T4C, markersize=6, label="scene folds (SGKF, 3 folds)"),
      Line2D([], [], marker="o", color="w", markerfacecolor=DROPPED, markersize=8, label="filled: scene-group CV, 8 clusters per title")]
a1.legend(handles=hd, loc="upper left", fontsize=7.8, ncol=1)
save(fig, "fig_t4_backbones.png")


# ---- 9. OOF versus LB ---------------------------------------------------------------------------------
O = pd.read_csv(os.path.join(DD, "t4_oof_vs_lb.csv"))
from scipy.stats import spearmanr
fig = plt.figure(figsize=(10.0, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.42, wspace=.28)
for j, (col, ttl) in enumerate((("oof_random", "(a) random 5-fold OOF"), ("oof_hard", "(b) hard rows only (25% least duplicated)"))):
    a = fig.add_subplot(gs[0, j])
    a.plot(O[col], O.lb, "-", color=AXIS, lw=1, zorder=1)
    a.scatter(O[col], O.lb, color=T4C, s=46, zorder=3)
    for _, r in O.iterrows():
        a.annotate(r.short, (r[col], r.lb), xytext=(5, -9), textcoords="offset points", fontsize=7.8)
    rho = spearmanr(O[col], O.lb).correlation
    a.text(.04, .93, f"Spearman {rho:.2f}", transform=a.transAxes, fontsize=9, color=INK, fontweight="bold")
    a.set_xlabel("OOF macro F1 of the recipe")
    a.set_ylabel("public LB")
    a.set_title(ttl)
    a.xaxis.set_major_locator(plt.MaxNLocator(4))
    a.tick_params(axis="x", labelsize=7.8)
a = fig.add_subplot(gs[1, :])
steps = [(O.short[i - 1] + "→" + O.short[i], O.oof_random[i] - O.oof_random[i - 1], O.oof_hard[i] - O.oof_hard[i - 1],
          5 * (O.lb[i] - O.lb[i - 1])) for i in range(1, len(O))]
x = np.arange(len(steps)); w = .26
a.bar(x - w, [s[1] for s in steps], w, color=DROPPED, label="random OOF")
a.bar(x, [s[2] for s in steps], w, color="#B8A9F2", label="hard-row OOF")
a.bar(x + w, [s[3] for s in steps], w, color=T4C, label=r"public LB step $\times5$ (= Task 4 F1 change)")
a.axhline(0, color=INK_MUTED, lw=.9)
a.set_xticks(x)
a.set_xticklabels([s[0] for s in steps], fontsize=8)
a.set_ylabel("change in macro F1")
a.set_title("(c) step by step: where the two OOF variants point the other way from the leaderboard")
for i, s in enumerate(steps):
    if s[3] > 0 and s[1] < 0:
        a.text(i - w, s[1] - .0013, "OOF\ndown", ha="center", va="top", fontsize=7, color=RULE_EXCLUDED)
    if s[3] > 0 and s[2] < 0:
        a.text(i, s[2] - .0013, "hard\ndown", ha="center", va="top", fontsize=7, color=RULE_EXCLUDED)
a.set_ylim(-.0095, .0148)
a.legend(loc="upper left", fontsize=8)
save(fig, "fig_t4_oof_vs_lb.png")


# ---- 10. LB curves of the blend weights --------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.0))
ax = ax.ravel()
def curve(a, x, y, title, xl, xt=None, labs=True, dx=0, extra=None):
    a.plot(x, y, "-o", color=T4C, lw=1.8, ms=5.5)
    if labs:
        for u, v in zip(x, y):
            a.text(u, v + (max(y) - min(y)) * .09 + 1e-6, f"{v:.5f}", ha="center", fontsize=7.4)
    a.set_title(title, fontsize=9.3)
    a.set_xlabel(xl)
    a.set_ylabel("public LB")
    pad = (max(y) - min(y)) * .35
    a.set_ylim(min(y) - pad * .6, max(y) + pad)
    a.tick_params(axis="y", labelsize=7.5)
    if xt is not None:
        a.set_xticks(xt)
    r = max(x) - min(x)
    a.set_xlim(min(x) - .18 * r, max(x) + .18 * r)
curve(ax[0], [0, .3, .5], [.86169, .86340, .86430], "(a) frozen-head share (v30, v31, v32)", "frozen share of the blend", [0, .3, .5])
curve(ax[1], [.2, .4, .7], [.86997, .87016, .87046], "(b) ViT-7B share of the frozen half", "7B share of frozen weight", [.2, .4, .7])
curve(ax[2], [.5, .7, .85, 1.0], [.87028, .87104, .87109, .87074], "(c) LoRA weight (clean files)", "LoRA weight in the blend", [.5, .7, .85, 1.0])
ax[3].plot([0, .5, 1], [.86430, .86669, .86750], "-o", color=RULE_EXCLUDED, lw=1.8, ms=5.5, label="on v32")
ax[3].plot([1, 1.5], [.86837, .86643], "-s", color=RULE_EXCLUDED, lw=1.8, ms=5.5, mfc="white", label="on v37")
for u, v in zip([0, .5, 1], [.86430, .86669, .86750]):
    ax[3].text(u, v + .00028, f"{v:.5f}", ha="center", fontsize=7.4)
for u, v in zip([1, 1.5], [.86837, .86643]):
    ax[3].text(u, v + .00028, f"{v:.5f}", ha="center", fontsize=7.4)
ax[3].set_title("(d) EM prior exponent (rule-excluded)", fontsize=9.3)
ax[3].set_xlabel(r"exponent $a$ in $p\,(\pi_{EM}/\pi_{train})^a$"); ax[3].set_ylabel("public LB")
ax[3].set_xticks([0, .5, 1, 1.5]); ax[3].set_ylim(.8635, .8702); ax[3].legend(loc="lower right", fontsize=7.5)
ax[3].tick_params(axis="y", labelsize=7.5)
curve(ax[4], [0, .5, 1], [.87087, .87224, .87300], "(e) train-prior exponent (allowed)", r"exponent $a$ in $p\,\pi_{train}^{-a}$", [0, .5, 1])
curve(ax[5], [.4, .5], [.87012, .87046], "(f) fine-tuned share (7B share .7)", "fine-tuned share of the blend", [.4, .5])
ax[3].set_xlim(-.25, 1.75)
fig.tight_layout(h_pad=1.8, w_pad=1.2)
save(fig, "fig_t4_lb_curves.png")


# ---- 11. scene clusters: thumbnails of one title ------------------------------------------------------
Z = np.load(os.path.join(DD, "t4_scene_cache.npz"))
grp, fold, nn1, ys = Z["group"], Z["fold"], Z["nn1"], Z["y"]
CLS = 1
gids = sorted(np.unique(grp[ys == CLS]))
fig, axes = plt.subplots(4, 6, figsize=(10.2, 4.55))
rng = np.random.RandomState(3)
for gi, g in enumerate(gids):
    idx = np.where(grp == g)[0]
    pick = rng.choice(idx, 3, replace=False)
    r, c0 = gi // 2, (gi % 2) * 3
    for j, p in enumerate(pick):
        a = axes[r, c0 + j]
        a.imshow(Image.open(os.path.join(DATA, "train", tr.file_name[p])).convert("RGB"))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(T4C if fold[idx[0]] == 0 else AXIS); s.set_linewidth(2.2 if fold[idx[0]] == 0 else .8)
        if j == 0:
            a.set_title(f"cluster {gi + 1}  (n={len(idx):,}, fold {int(np.bincount(fold[idx]).argmax())})", fontsize=7.8, loc="left", color=INK)
fig.subplots_adjust(wspace=.04, hspace=.34, top=.94, bottom=.01, left=.01, right=.99)
save(fig, "fig_t4_scene_clusters.png")


# ---- 12. scene folds: redundancy and fold composition --------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})
ax[0].hist(nn1, bins=50, color=T4C, edgecolor="white")
q25 = np.quantile(nn1, .25)
ax[0].axvline(np.median(nn1), color=INK, ls="--", lw=1.2)
ax[0].axvline(q25, color=RULE_EXCLUDED, ls=":", lw=1.4)
ax[0].text(np.median(nn1) + .01, ax[0].get_ylim()[1] * .92, f"median {np.median(nn1):.2f}", fontsize=8.5)
ax[0].text(q25 - .02, ax[0].get_ylim()[1] * .75, f"25th percentile {q25:.2f}\n= the 'hard row' cut", fontsize=8, ha="right", color=RULE_EXCLUDED)
ax[0].set_xlabel("cosine to the nearest other train image (C-RADIOv3-H features)")
ax[0].set_ylabel("train images")
ax[0].set_title("(a) redundancy of the training set")
bot = np.zeros(5)
for k in range(K):
    cnts = np.array([((fold == f) & (ys == k)).sum() for f in range(5)])
    ax[1].bar(range(5), cnts, bottom=bot, color=CCOL[k], width=.68, label=NAMES[k])
    bot += cnts
for f in range(5):
    ax[1].text(f, bot[f] + 120, f"{int(bot[f]):,}", ha="center", fontsize=8)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f"fold {f}" for f in range(5)])
ax[1].set_ylim(0, 8700)
ax[1].set_ylabel("images")
ax[1].set_title("(b) scene folds: 40 clusters, whole clusters per fold")
ax[1].legend(fontsize=7.4, ncol=3, loc="upper left")
fig.tight_layout()
save(fig, "fig_t4_scene_folds.png")


# ---- 13. what scene CV predicted versus what the leaderboard did --------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
a.plot([.3, .5, .7, 1.0], [.9679, .9697, .9677, .9635], "-o", color=DROPPED, lw=1.8, label="4 frozen heads")
a.plot([.3, .4, .5], [.9754, .9742, .9724], "-o", color=T4C, lw=1.8, label="frozen heads with ViT-7B (share .7)")
for u, v in zip([.3, .5, .7, 1.0], [.9679, .9697, .9677, .9635]):
    a.text(u, v - .0009, f"{v:.4f}", ha="center", va="top", fontsize=7.5, color=INK_MUTED)
for u, v in zip([.3, .4, .5], [.9754, .9742, .9724]):
    a.text(u, v + .0005, f"{v:.4f}", ha="center", fontsize=7.5, color=T4C)
a.set_xlabel("fine-tuned share of the blend (scene folds 0-2)")
a.set_ylabel("macro F1 on held-out scenes")
a.set_title("(a) fine-tuned / frozen share on scene folds")
a.set_ylim(.961, .978); a.set_xlim(.22, 1.08)
a.legend(loc="lower left", fontsize=7.8)
a = ax[1]
dec = [("kNN smoothing", .0045, 5 * -.00077), ("fine-tuned share .5→.4", .0018, 5 * -.00034), ("multi-scale TTA", .00086, 5 * -.00013),
       ("frozen-4 heads (v36)", .0006, 5 * .00055), ("soup + prior (v85a)", .0022, 5 * .00039), ("LoRA at .5 (v79a)", .0082, 5 * .00216)]
dec = dec[::-1]
for i, (n, p, l) in enumerate(dec):
    a.plot([p, l], [i, i], color=AXIS, lw=2, zorder=1)
    a.scatter(p, i, s=52, color=T4C, zorder=3)
    a.scatter(l, i, s=58, marker="s", color=RULE_EXCLUDED if (p > 0) != (l > 0) else KEPT, zorder=3)
a.axvline(0, color=INK_MUTED, lw=1)
a.set_yticks(range(len(dec))); a.set_yticklabels([d[0] for d in dec], fontsize=8.3)
a.set_xlabel("change in Task 4 macro F1")
a.set_title("(b) scene CV or fold 0 (violet) vs leaderboard ×5 (square)")
a.grid(axis="y", visible=False)
a.legend(handles=[Line2D([], [], marker="s", color="w", markerfacecolor=KEPT, markersize=8, label="same sign"),
                  Line2D([], [], marker="s", color="w", markerfacecolor=RULE_EXCLUDED, markersize=8, label="opposite sign")],
         loc="center right", fontsize=8)
fig.tight_layout()
save(fig, "fig_t4_scene_vs_lb.png")


# ---- 14. error analysis on scene fold 0 ---------------------------------------------------------------
F0 = np.load(os.path.join(DD, "t4_fold0_cache.npz"))
L0, y0, pi0 = F0["logp"].astype(np.float64), F0["y"].astype(int), F0["prior"]
cmap = LinearSegmentedColormap.from_list("t4", ["#FFFFFF", T4C])
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.55), gridspec_kw={"width_ratios": [1, 1, 1.05]})
short = ["7-11", "FiveN", "Free\nFried", "PUBE", "Roblock"]
Lp = L0 - np.log(pi0)
for a, L, ttl in ((ax[0], L0, "(a) soup + refit head, argmax"), (ax[1], Lp, "(b) with the train prior")):
    cm = confusion_matrix(y0, L.argmax(1), labels=range(5))
    nrm = cm / cm.sum(1, keepdims=True)
    a.imshow(np.where(np.eye(5) > 0, 0, nrm) ** .5, cmap=cmap, vmin=0, vmax=.12 ** .5)
    for i in range(5):
        for j in range(5):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8.3, color=INK if i != j else INK_MUTED, fontweight="normal" if i == j else "bold")
    a.set_xticks(range(5)); a.set_xticklabels(short, fontsize=7.4)
    a.set_yticks(range(5)); a.set_yticklabels(short, fontsize=7.4)
    a.set_xlabel("predicted"); a.set_ylabel("true")
    a.set_title(f"{ttl}: {int((L.argmax(1) != y0).sum())} errors", fontsize=9)
    a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
pp = sm(Lp); conf = pp.max(1); err = pp.argmax(1) != y0
a = ax[2]
bins = np.linspace(.2, 1, 33)
a.hist(conf[~err], bins=bins, color=KEPT, alpha=.85, label=f"correct ({(~err).sum():,})")
a.hist(conf[err], bins=bins, color=RULE_EXCLUDED, alpha=.95, label=f"errors ({err.sum()})")
a.set_yscale("log")
a.set_xlabel("confidence (max probability, after prior)")
a.set_ylabel("images (log)")
a.set_title("(c) confidence on fold 0")
a.legend(loc="upper left", fontsize=7.8)
print("fold0: conf<.6 share", (conf < .6).mean(), " errors with conf>.9:", int((err & (conf > .9)).sum()), " errors", int(err.sum()))
fig.tight_layout()
save(fig, "fig_t4_confusion.png")


# ---- 15. example train images of every scene fold (research folds: SigLIP2 k-means, sig8, SGKF) -------------
# Selection is deterministic (seed 11) and saved to data/t4_scene_fold_samples.csv so the figure can be redrawn
# from the file list alone. Row order of the cache equals train.csv (checked below).
assert (ys == tr.label.values).all()
rngs = np.random.RandomState(11)
rows_out = []
for f in range(5):
    for k in range(K):
        gs_ = sorted(set(grp[(fold == f) & (ys == k)]))
        order = rngs.permutation(len(gs_))
        chosen = [gs_[i] for i in order][:2]
        for j in range(2):
            g = chosen[j % len(chosen)]
            idx = np.where((grp == g) & (fold == f) & (ys == k))[0]
            if len(chosen) == 1 and j == 1:      # only one scene group in the cell: take a second, different image
                idx = np.setdiff1d(idx, [rows_out[-1]["row"]])
            p = int(rngs.choice(idx))
            rows_out.append(dict(fold=f, label=k, class_name=NAMES[k], group=int(g), file_name=tr.file_name[p], row=p))
S15 = pd.DataFrame(rows_out)
S15.drop(columns="row").to_csv(os.path.join(DD, "t4_scene_fold_samples.csv"), index=False)
nfold = np.bincount(fold, minlength=5)
fig = plt.figure(figsize=(10.2, 3.45))
gsp = fig.add_gridspec(5, K * 2, left=.105, right=.995, top=.915, bottom=.01, wspace=.035, hspace=.04)
for f in range(5):
    for k in range(K):
        for j in range(2):
            a = fig.add_subplot(gsp[f, 2 * k + j])
            fn = S15[(S15.fold == f) & (S15.label == k)].file_name.iloc[j]
            a.imshow(Image.open(os.path.join(DATA, "train", fn)).convert("RGB"))
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(T4C if f == 0 else AXIS); s.set_linewidth(1.4 if f == 0 else .6)
            if f == 0 and j == 0:
                a.annotate(NAMES[k], (1.0, 1.08), xycoords="axes fraction", ha="center", va="bottom", fontsize=8.8, color=INK)
            if k == 0 and j == 0:
                a.annotate(f"fold {f}\n({nfold[f]:,} images)", (-0.1, .5), xycoords="axes fraction", ha="right", va="center",
                           fontsize=8, color=T4C if f == 0 else INK)
fp = os.path.join(OUT, "fig_t4_scene_fold_samples.png")
fig.savefig(fp, dpi=200)          # photographic content: 200 dpi and a 256-colour palette keep the PNG under 2 MB
plt.close(fig)
Image.open(fp).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(fp, optimize=True)
print("wrote fig_t4_scene_fold_samples.png", os.path.getsize(fp) // 1024, "KB")


# ---- 16. constant-base probes of the first data release (numbers from the pipeline repo's LB log) -------------
# base = T1 all 1, T2 = T3 = T5 = 0, T4 = datacard names (measured 0.27401 = T4 scores zero); bar = probe LB - base.
BASE = 0.27401
probes = [  # label, LB, chance-model prediction of the LB (or None)
    ("T4 = datacard names (reference)", 0.27401, None),
    ("7 label formats round-robin", 0.27935, None),
    ("only fiveN rows = 1, rest 'LOL'", 0.28336, 0.2833),
    ("integer text only in some groups", 0.28420, None),
    ("ints, mapping M1 (one name row)", 0.30552, None),
    ("ints, mapping M4", 0.31254, None),
    ("ints, mapping M3", 0.31336, None),
    ("ints, mapping M2b", 0.31337, None),
    ("ints, rows sorted by id", 0.31327, 0.3129),
    ("ints, contiguous row blocks", 0.31307, 0.3127),
]
fig, a = plt.subplots(figsize=(8.4, 3.7))
yy = np.arange(len(probes))[::-1]
a.axvspan(.2 * .19, .2 * .20, color=T4C, alpha=.16, lw=0)
a.text(.2 * .195, len(probes) - .35, "chance:\nmacro F1\n0.19-0.20", ha="center", va="top", fontsize=7.4, color=T4C)
for yi, (lab, lb, pred) in zip(yy, probes):
    a.barh(yi, lb - BASE, color=T4C if lb - BASE > .02 else DROPPED, height=.62)
    a.text(max(lb - BASE, 0) + .0008, yi, f"{lb - BASE:+.4f}", va="center", fontsize=7.8, color=INK)
    if pred is not None:
        a.plot(pred - BASE, yi, "D", color=INK, ms=4.5, zorder=5)
a.set_yticks(yy); a.set_yticklabels([p[0] for p in probes], fontsize=8)
a.set_xlim(-.002, .05)
a.set_xlabel("public LB minus constant base (0.27401)  =  0.2 x (Task 4 macro F1)")
a.set_title("Constant-base probes, first release: only integer text scores, and only at chance", fontsize=9.5)
a.plot([], [], "D", color=INK, ms=4.5, label="chance-model prediction")
a.legend(loc="center right", bbox_to_anchor=(1.0, 0.7), fontsize=7.8)
a.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "fig_t4_probes.png")
