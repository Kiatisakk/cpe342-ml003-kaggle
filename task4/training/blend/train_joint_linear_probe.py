"""Task 4 joint 10-backbone linear probe (the "joint" component of the final blend).

Concatenates frozen [CLS, mean patch] features of ten backbones / layers (plain and hflip averaged), centres each
backbone block on the training mean, L2-normalises it, and trains one L2-regularised linear classifier
(AdamW, weight decay 0.15, 1/sqrt(count) class weights, label smoothing 0.05, 30 epochs).

Writes two files used by build_final_blend.py:
  <OUT_DIR>/allfam_joint_wd015_val0.npy  trained on scene folds 1-4, predicted on scene fold 0
  <OUT_DIR>/allfam_joint_wd015_test.npy  trained on all training images (two seeds), predicted on the test set
Training images and labels only. Runs on CPU (needs ~16 GB RAM).
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PRED_ROOT = os.environ.get("T4_PRED_ROOT", ".")
DATA_DIR = os.environ.get("T4_DATA_DIR", "cpe342-kaggle-main/04_image_classification")
TEST_ORDER = os.environ.get("T4_TEST_ORDER", "t4/preds/v3_5f/preds/task4_v3_5f.csv")
FOLDS = os.environ.get("T4_SCENE_FOLDS", "t4/research/groupcv/sgkf_sig8_fold.npy")
OUT_DIR = os.environ.get("T4_JOINT_OUT", "t4/preds/joint_mb_linear")
P = lambda *a: os.path.join(PRED_ROOT, *a)

# (name, feature folder, file tag) -- outputs of frozen_features/* and colab/frozen_vit7b_*.ipynb
SPECS = [
    ("7b_l75", "t4/preds/frozen_7b_ms/feats", "vit7b_l75"),
    ("7b_s15", "t4/preds/frozen_7b_ms/feats", "vit7b_s15"),
    ("7b_s10", "t4/preds/frozen_7b/feats", "vit_7b_patch16_dinov3"),
    ("7b_l50", "t4/preds/frozen_7b_ms/feats", "vit7b_l50"),
    ("radio3g", "t4/preds/frozen_e/feats", "radio3g"),
    ("radio3", "t4/preds/frozen_c/feats", "radio3"),
    ("radio", "t4/preds/frozen_b/feats", "radio"),
    ("dinov2g", "t4/preds/frozen_c/feats", "dinov2g"),
    ("vithp", "t4/preds/dinov3_feats/feats", "vithp"),
    ("webssl1b", "t4/preds/frozen_e/feats", "webssl1b"),
]

torch.set_num_threads(os.cpu_count() or 4)
tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
y = tr["label"].astype(int).values
K = 5
tr_names = tr["file_name"].str.replace(".jpg", "", regex=False).values
test_order = pd.read_csv(P(TEST_ORDER))["id_img"].astype(str).values
fold = np.load(P(FOLDS))
va0, tr0 = np.where(fold == 0)[0], np.where(fold != 0)[0]


def load_features(mean_idx, need_test):
    tr_parts, te_parts = [], []
    for name, d, tag in SPECS:
        if name == "vithp" and not os.path.exists(P(d, f"{tag}_train_plain.npy")):
            tag = "vit_huge_plus_patch16_dinov3"
        assert (np.load(P(d, "train_names.npy"), allow_pickle=True) == tr_names).all()
        a_p = np.load(P(d, f"{tag}_train_plain.npy"), mmap_mode="r")
        a_h = np.load(P(d, f"{tag}_train_hflip.npy"), mmap_mode="r")
        x = np.empty((len(tr), a_p.shape[1]), dtype=np.float32)
        for i in range(0, len(tr), 4096):
            x[i:i + 4096] = 0.5 * (a_p[i:i + 4096].astype(np.float32) + a_h[i:i + 4096].astype(np.float32))
        mu = x[mean_idx].mean(0, keepdims=True)          # training mean only
        x -= mu
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
        tr_parts.append(x.astype(np.float16))
        if need_test:
            names = np.load(P(d, "test_names.npy"), allow_pickle=True)
            pos = pd.Series(np.arange(len(names)), index=names).loc[test_order].values
            t_p = np.load(P(d, f"{tag}_test_plain.npy"), mmap_mode="r")
            t_h = np.load(P(d, f"{tag}_test_hflip.npy"), mmap_mode="r")
            xt = np.empty((len(test_order), a_p.shape[1]), dtype=np.float32)
            for i in range(0, len(test_order), 4096):
                idx = pos[i:i + 4096]
                xt[i:i + 4096] = 0.5 * (t_p[idx].astype(np.float32) + t_h[idx].astype(np.float32))
            xt -= mu
            xt /= np.linalg.norm(xt, axis=1, keepdims=True) + 1e-6
            te_parts.append(xt.astype(np.float16))
    return np.concatenate(tr_parts, 1), (np.concatenate(te_parts, 1) if need_test else None)


def train_linear(x, idx_fit, x_eval, seed, wd=0.15, epochs=30, lr=1e-2, ls=0.05):
    torch.manual_seed(seed)
    xa, ya = torch.from_numpy(x[idx_fit]), torch.from_numpy(y[idx_fit]).long()
    cw = torch.tensor(1.0 / np.sqrt(np.bincount(y[idx_fit], minlength=K)), dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], K)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw / cw.mean(), label_smoothing=ls)
    for _ in range(epochs):
        perm = torch.randperm(len(xa))
        for i in range(0, len(xa), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(xa[b].float()), ya[b]).backward()
            opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.from_numpy(x_eval[i:i + 4096]).float()), 1).numpy()
                               for i in range(0, len(x_eval), 4096)])


os.makedirs(P(OUT_DIR), exist_ok=True)
x0, _ = load_features(tr0, need_test=False)
p_val = train_linear(x0, tr0, x0[va0], seed=42)
np.save(P(OUT_DIR, "allfam_joint_wd015_val0.npy"), p_val.astype(np.float32))
print(f"scene fold 0 macro F1 {f1_score(y[va0], p_val.argmax(1), average='macro'):.5f}")
del x0

all_idx = np.arange(len(tr))
x_all, x_te = load_features(all_idx, need_test=True)
p_te = 0.5 * (train_linear(x_all, all_idx, x_te, seed=42) + train_linear(x_all, all_idx, x_te, seed=7))
np.save(P(OUT_DIR, "allfam_joint_wd015_test.npy"), p_te.astype(np.float32))
print("wrote", P(OUT_DIR, "allfam_joint_wd015_test.npy"))
