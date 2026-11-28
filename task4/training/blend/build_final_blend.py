"""Task 4 final blend (the Task 4 column of sub_v90f_entropy_dualhead_md.csv).

Inputs are probability files written by the training code in this folder:
  * LoRA ViT-7B models (colab/lora_vit7b_soup.ipynb): model soup with a single-view head and a two-view head,
    and the two multi-depth LoRA models (cut 24 / blocks 12-23, cut 40 / blocks 28-39).
  * Fine-tuned companions (finetune_kaggle/*, colab/*_fulltrain.ipynb).
  * Linear heads on frozen features (frozen_features/train_linear_heads.py).
  * The joint 10-backbone linear probe (blend/train_joint_linear_probe.py).
For every component we need its predictions on scene fold 0 (for the temperature) and on the test set.

Steps (all parameters come from training data only):
  1. Fit one temperature per component on scene fold 0 (NLL-optimal).
  2. Entropy-weighted pooling: each component's weight on an image is its base weight times
     exp(-entropy of its calibrated prediction).
  3. Uniform-target prior adjustment with the training class frequencies.

Set PRED_ROOT to the folder that holds the run outputs, laid out as in RUNS below.
TEST_ORDER lists the image names in the order the test probability files were written.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

PRED_ROOT = os.environ.get("T4_PRED_ROOT", ".")
DATA_DIR = os.environ.get("T4_DATA_DIR", "cpe342-kaggle-main/04_image_classification")
TEST_ORDER = os.environ.get("T4_TEST_ORDER", "t4/preds/v3_5f/preds/task4_v3_5f.csv")  # column id_img
VAL0_IDX = os.environ.get("T4_VAL0_IDX", "t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy")
OUT_CSV = os.environ.get("T4_OUT", "task4_predictions.csv")

P = lambda *a: os.path.join(PRED_ROOT, *a)

# ---------------------------------------------------------------- data (train only)
tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y = tr["label"].astype(int).values
K = 5
prior = np.bincount(y, minlength=K) / len(y)
va0 = np.load(P(VAL0_IDX))
yv = y[va0]
order = pd.read_csv(P(TEST_ORDER))["id_img"].astype(str).values
to_test_csv = pd.Series(np.arange(len(order)), index=order).loc[te["file_name"].str.replace(".jpg", "", regex=False).values].values

lg = lambda p: np.log(np.clip(p, 1e-9, 1.0))
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)

# ---------------------------------------------------------------- loaders
def lora_val(d, t):
    return sm(np.mean([lg(np.load(P(d, f"lora7b_{t}_oof_{v}.npy"))) for v in ("plain", "hflip")], 0))

def lora_test(d, t):
    return sm(np.mean([lg(np.load(P(d, f"{t}_test_full_{v}.npy"))) for v in ("plain", "hflip")], 0))

V3 = ("plain", "hflip", "up")
def ft_test(d, t):
    if os.path.exists(P(d, f"{t}_test_full_plain.npy")):
        return np.mean([np.load(P(d, f"{t}_test_full_{v}.npy")) for v in V3], 0)
    return np.mean([np.mean([np.load(P(d, f"{t}_test_fold{k}_{v}.npy")) for v in V3], 0)
                    for k in range(5) if os.path.exists(P(d, f"{t}_test_fold{k}_plain.npy"))], 0)

head_test = lambda name: np.mean([np.load(P("t4/preds/dinov3_head", name, f"d3lin_{name}_test_fold{k}.npy")) for k in range(5)], 0)
head_val = lambda name: np.load(P("t4/research/groupcv/oof_sgkf", f"{name}.npy"))[va0]

# ---------------------------------------------------------------- scene fold 0 predictions
V = {
    "soup_1v": lora_val("t4/preds/v4f_lora7b_q4", "soup2_val0_refit"),
    "soup_2v": lora_val("t4/preds/v4f_lora7b_q5", "soup2b_val0_refit2v"),
    "md_c24": lora_val("t4/preds/v4f_lora7b_q1/preds", "c24f12r16e3_val0"),
    "md_c40": lora_val("t4/preds/v4f_lora7b_q1/preds", "c40f28r16e3_val0"),
    "joint": np.load(P("t4/preds/joint_mb_linear/allfam_joint_wd015_val0.npy")),
}
# multi-architecture base on fold 0: scene-fold ConvNeXt-S stands in for the fine-tuned half (the full-train
# companions have no held-out predictions), plus the scene-fold OOF of the frozen heads
V_ft = np.mean([np.load(P("t4/preds/scenecv_s/preds", f"v4g_cnxt_s_d3_base_oof_{v}.npy")) for v in V3], 0)[va0]
V["multi_arch"] = sm(0.55 * lg(V_ft)
                     + 0.25 * (0.55 * lg(head_val("vit7b_l75")) + 0.20 * lg(head_val("vit7b_s15"))
                               + 0.15 * lg(head_val("vit7b")) + 0.10 * lg(head_val("vit7b_s05")))
                     + 0.20 * (0.25 * lg(head_val("vithp")) + 0.25 * lg(head_val("radio3"))
                               + 0.25 * lg(head_val("radio")) + 0.25 * lg(head_val("dinov2g"))))

# ---------------------------------------------------------------- test predictions
L_ft = (0.25 * lg(ft_test("t4/preds/v4f_vit_l_d3", "v4f_vit_l_d3_v4aug_s42"))
        + 0.25 * (0.75 * lg(ft_test("t4/preds/v4f_eva02_l", "v4f_eva02_l_v4aug_s42"))
                  + 0.25 * lg(ft_test("t4/preds/v4_eva02_b/preds", "v4_eva02_b")))
        + 0.25 * (0.25 * lg(ft_test("t4/preds/v4f_cnxt_l_d3c", "v4f_cnxt_l_d3c_s42"))
                  + 0.25 * lg(ft_test("t4/preds/v4f_cnxt_l_d3c_kg/preds", "v4f_cnxt_l_d3c_s42"))
                  + 0.20 * lg(ft_test("t4/preds/v4f_cnxt_l_d3c_v3aug", "v4f_cnxt_l_d3c_v3aug_s42"))
                  + 0.15 * lg(ft_test("t4/preds/v4f_cnxt_l_d3_v3aug_kg/preds", "v4f_cnxt_l_d3c_v3aug_s42"))
                  + 0.15 * lg(ft_test("t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c")))
        + 0.15 * (0.5 * lg(ft_test("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3"))
                  + 0.5 * lg(ft_test("t4/preds/v4f_cnxt_b_d3_nolq/preds", "v4f_cnxt_b_d3_s42_nolq")))
        + 0.10 * lg(ft_test("t4/preds/v4_dinov2_l/preds", "v4_dinov2_l")))
L_7b = (0.55 * lg(head_test("vit7b_l75_ph")) + 0.20 * lg(head_test("vit7b_s15_ph"))
        + 0.15 * lg(head_test("vit_7b_patch16_dinov3_ph")) + 0.10 * lg(head_test("vit7b_s05_ph")))
L_frozen = (0.30 * lg(head_test("radio3g_ph")) + 0.20 * lg(head_test("vithp_ph")) + 0.18 * lg(head_test("radio3_ph"))
            + 0.14 * lg(head_test("radio_ph")) + 0.13 * lg(head_test("dinov2g_ph")) + 0.05 * lg(head_test("webssl1b_ph")))
T = {
    "soup_1v": lora_test("t4/preds/v4f_lora7b_q4", "v4f_lora7b_soup2_s42"),
    "soup_2v": lora_test("t4/preds/v4f_lora7b_q5", "v4f_lora7b_soup2hv2_s42"),
    "md_c24": lora_test("t4/preds/v4f_lora7b_q2/preds", "v4f_lora7b_c24f12r16e3_s42"),
    "md_c40": lora_test("t4/preds/v4f_lora7b_q1/preds", "v4f_lora7b_c40f28r16e3_s42"),
    "joint": np.load(P("t4/preds/joint_mb_linear/allfam_joint_wd015_test.npy")),
    "multi_arch": sm(0.55 * L_ft + 0.25 * L_7b + 0.20 * L_frozen),
}

# ---------------------------------------------------------------- calibrate and pool
WEIGHTS = {"soup_1v": 0.52, "soup_2v": 0.25, "md_c24": 0.04, "md_c40": 0.04, "multi_arch": 0.09, "joint": 0.06}

def fit_temperature(p):
    return float(minimize_scalar(lambda t: log_loss(yv, sm(lg(p) / t)), bounds=(0.1, 3.0), method="bounded").x)

TEMP = {k: fit_temperature(V[k]) for k in WEIGHTS}

def pool(preds):
    num = np.zeros_like(next(iter(preds.values())), dtype=np.float64)
    scale = np.zeros((len(num), 1))
    for k, w in WEIGHTS.items():
        z = lg(preds[k]) / TEMP[k]
        w_img = w * np.exp(-ent(sm(z)))[:, None]
        num += w_img * z
        scale += w_img / TEMP[k]
    return sm(num - scale * np.log(prior))

f0 = pool(V).argmax(1)
print("temperatures:", {k: round(v, 3) for k, v in TEMP.items()})
print(f"scene fold 0: macro F1 {f1_score(yv, f0, average='macro'):.5f}, errors {int((f0 != yv).sum())} / {len(yv)}")

pred = pool(T).argmax(1)[to_test_csv]
pd.DataFrame({"id": te["id"], "task4": pred}).to_csv(OUT_CSV, index=False, lineterminator="\n")
print("wrote", OUT_CSV, "class share", np.round(np.bincount(pred, minlength=K) / len(pred), 3).tolist())
