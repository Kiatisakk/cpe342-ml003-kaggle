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
