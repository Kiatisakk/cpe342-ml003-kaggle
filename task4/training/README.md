# Task 4 — training code for the ensemble members

`task4/task4.ipynb` is the final pipeline: it trains the LoRA ViT-7B models and fuses everything into `task4_predictions.csv`.
The other members of the ensemble (fine-tuned companion backbones and frozen-feature heads) were trained in separate GPU sessions
on Kaggle and Colab, because each run needs several hours of GPU time.
This folder holds the code for those runs, so that every component of the final submission can be retrained.

All scripts train on **training images and labels only**. The test set is only predicted, never used to fit or choose anything.

## Folder map

| Folder / file | What it trains | Where it ran | Output used in the final blend |
|---|---|---|---|
| `colab/lora_vit7b_soup.ipynb` | LoRA on DINOv3 ViT-7B cut at block 30 (two adapters), multi-depth LoRA (cut 24, cut 40), exact weight-space model soup and head refit | Colab A100 (~17–20 min/epoch) | `lora7b_soup2_*`, `lora7b_c24f12`, `lora7b_c40f28` |
| `colab/convnext_l_fulltrain.ipynb` | ConvNeXt-L DINOv3, full-train refit (v4 recipe) | Colab A100 | companion |
| `colab/convnext_l_fulltrain_v3aug.ipynb` | ConvNeXt-L DINOv3, full-train refit (v3 augmentation recipe) | Colab A100 | companion |
| `colab/eva02_l_fulltrain.ipynb` | EVA02-L, full-train refit | Colab A100 | companion |
| `colab/vit_l_dinov3_fulltrain.ipynb` | DINOv3 ViT-L/16, full-train refit | Colab A100 | companion |
| `colab/frozen_vit7b_features.ipynb` | Frozen DINOv3 ViT-7B `[CLS, mean patch]` features (plain + hflip) | Colab A100 | frozen head, joint linear probe |
| `colab/frozen_vit7b_multiscale_layers.ipynb` | Frozen ViT-7B features at other scales and depths (block 30 = 75%, block 20 = 50%, scale 1.5×) | Colab A100 | frozen heads, joint linear probe |
| `finetune_kaggle/convnext_s_dinov3_3fold.py` | Shared 3-fold fine-tune script (`T4_BACKBONE` preset); the ConvNeXt-B 3-fold run used this script with a `cnxt_b_d3` preset | Kaggle T4 | companion |
| `finetune_kaggle/convnext_l_dinov3_3fold.py` | ConvNeXt-L DINOv3, 3 folds | Kaggle T4 | companion |
| `finetune_kaggle/eva02_b_3fold.py` | EVA02-B, 3 folds | Kaggle T4 | companion |
| `finetune_kaggle/dinov2_l_3fold.py` | DINOv2-L, 3 folds | Kaggle T4 | companion |
| `finetune_kaggle/convnext_fulltrain.py` | Full-train refit; `T4_BACKBONE=cnxt_b_d3` (Base) or `cnxt_l_d3c` (Large) | Kaggle T4 | companion |
| `finetune_kaggle/convnext_l_fulltrain_v3aug.py` | ConvNeXt-L full-train with the v3 augmentation recipe | Kaggle T4 | companion |
| `frozen_features/frozen_feats_dinov3_vith_vitl.py` | Frozen DINOv3 ViT-H+ and ViT-L features | Kaggle T4 | frozen head, joint linear probe |
| `frozen_features/frozen_feats_b_radio4_siglip2_pe.py` | Frozen C-RADIOv4-H, SigLIP2, PE-Core-L features | Kaggle T4 | C-RADIOv4-H head, joint probe |
| `frozen_features/frozen_feats_c_radio3_dinov2g.py` | Frozen C-RADIOv3-H and DINOv2-g features | Kaggle T4 | frozen heads, joint probe |
| `frozen_features/frozen_feats_e_radio3g_webssl_aimv2.py` | Frozen C-RADIOv3-g, Web-SSL 1B, AIMv2-H features | Kaggle T4 | frozen head, joint probe |
| `frozen_features/train_linear_heads.py` | Linear heads on the frozen features (5 folds, train labels only, CPU) | local CPU | frozen-head probabilities |
| `scene_cv/scene_group_cv.py` | Scene clusters (k-means per class on frozen embeddings) and scene-grouped CV of the heads | local CPU | scene folds |
| `scene_cv/scene_group_folds_sgkf.py` | Class-balanced scene-grouped folds (`StratifiedGroupKFold`) and head OOF on them | local CPU | fold 0 for calibration |
| `blend/build_final_blend_v90.py` | Joint 10-backbone linear probe, temperature calibration on scene fold 0, entropy-weighted pooling, train-prior adjustment | local CPU | `sub_v90f_entropy_dualhead_md.csv` (Task 4 column) |

## How to run

1. **Data.** Every script expects the competition data (`04_image_classification/train.csv`, `test.csv`, `train/`, `test/`).
   Kaggle scripts read it from `/kaggle/input/`. Colab notebooks read a zip of the same folder from Google Drive (`MyDrive/cpe342_t4/task4_data.zip`; the first cell explains the layout).
2. **GPU and secrets.** DINOv3 weights are gated on Hugging Face, so the DINOv3 runs need an `HF_TOKEN` secret.
   On Kaggle: Add-ons → Secrets → `HF_TOKEN`, GPU T4, Internet on, then run from the notebook UI (a CLI push cannot read secrets).
   On Colab: add `HF_TOKEN` in the Secrets panel. EVA02, DINOv2, RADIO, SigLIP2, PE-Core and Web-SSL are not gated.
   Never write a token into the code.
3. **Settings.** The Kaggle scripts are configured by environment variables at the top of each file (`T4_BACKBONE`, `T4_FOLDS`, `T4_SEEDS`, `T4_FULL_EPOCHS`, `T4_RECIPE`).
   A 3-fold run of a large backbone takes about two folds per 12-hour Kaggle session; set `T4_FOLDS` to the remaining fold for the next session.
4. **Outputs.** Each run writes test (and, for fold runs, out-of-fold) probability files `*_test_*_{plain,hflip,up}.npy` in the original `test.csv` row order.
5. **Blend.** `blend/build_final_blend_v90.py` reads these files from the research folder layout (`t4/preds/<run>/...`) that we used during the competition; adjust the paths at the top of the script to where you saved the outputs.
   `task4/task4.ipynb` can also load pre-computed companion probabilities from its `task4_artifacts/` folder (`multi_arch_base_*`, `joint_10mb_lin_*`) and fuses them with the LoRA models.
