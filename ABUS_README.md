# SegMamba for ABUS 3D Ultrasound Tumour Segmentation

This document describes how to run the full SegMamba pipeline on the **ABUS (Automated Breast Ultrasound)** dataset for binary tumour segmentation.

The four ABUS-specific scripts (`abus_*.py`) sit alongside the original SegMamba codebase. **No original files are modified.**

---

## 1  Overview

### 1.1  Pipeline at a Glance

```
Step 0   Environment setup          (one-time)
Step 1   abus_preprocessing.py      NRRD  -->  NPZ + PKL
Step 2   abus_train.py              Train SegMamba (single GPU)
Step 3   abus_predict.py            Sliding-window inference + TTA
Step 4   abus_compute_metrics.py    Dice & HD95 evaluation
```

### 1.2  ABUS Dataset

| Property | Value |
|---|---|
| Source path | `/Volumes/Autzoko/ABUS` |
| Format | NRRD, single-channel, uint8 \[0, 255\] |
| Task | Binary segmentation (0 = background, 1 = tumour) |
| Train / Val / Test | 100 / 30 / 70 cases |
| Volume size (typical) | ~330 x 608 x 865 voxels (numpy z,y,x after SimpleITK) |
| Spacing | (1.0, 1.0, 1.0) — isotropic |
| Class imbalance | Tumour occupies ~0.1 – 0.3 % of the volume |

### 1.3  Model Adaptations (BraTS --> ABUS)

| | BraTS (original `3_train.py`) | ABUS (`abus_train.py`) |
|---|---|---|
| Input channels | 4 (T1, T1c, T2, T2-FLAIR) | **1** (ultrasound) |
| Output classes | 4 (bg + TC / WT / ET) | **2** (bg + tumour) |
| Label mapping | TC = 1\|3, WT = 1\|2\|3, ET = 3 | **None** (already binary) |
| Loss function | CrossEntropyLoss | **DiceLoss + CrossEntropyLoss** |
| Data splitting | Random 70/10/20 from one directory | **Pre-split** train / val / test directories |
| Resampling | To 1 mm isotropic | **Skipped** (already 1 mm) |

Everything else (patch size 128^3, SGD + poly schedule, augmentation, sliding-window inference, 8-way TTA) is kept identical.

---

## 2  Environment Setup (Step 0)

### 2.1  CUDA Dependencies

SegMamba requires an NVIDIA GPU with CUDA. The `mamba_ssm` and `causal-conv1d` packages must be compiled from source.

```bash
cd SegMamba

# 1. Install causal-conv1d
cd causal-conv1d
python setup.py install
cd ..

# 2. Install mamba
cd mamba
python setup.py install
cd ..
```

### 2.2  Python Dependencies

```bash
pip install acvl-utils medpy SimpleITK tqdm scikit-image
```

`monai` is already bundled inside the repository (the `monai/` directory). No separate install needed.

### 2.3  Quick Smoke Test

```bash
python 0_inference.py
```

If this prints a tensor shape `(1, 4, 128, 128, 128)` without errors, the environment is ready.

---

## 3  Data Preprocessing (Step 1)

### 3.1  What It Does

For every NRRD volume in Train / Validation / Test:

1. **Read** `DATA_XXX.nrrd` and `MASK_XXX.nrrd` with SimpleITK.
2. **Convert** image to float32, shape `(1, D, H, W)`.
3. **Collect** foreground intensity statistics (inside mask > 0).
4. **Crop to nonzero** bounding box (removes pure-zero borders).
5. **Z-score normalise** the entire image: `(x - mean) / std`.
6. **Compute** foreground voxel locations (for oversampling during training).
7. **Save** as compressed `ABUS_XXX.npz` + properties `ABUS_XXX.pkl`.

Resampling is intentionally skipped because the spacing is already (1, 1, 1).

### 3.2  Run

```bash
python abus_preprocessing.py
```

### 3.3  Configuration

All settings are at the bottom of the file. The most important ones:

| Variable | Default | Notes |
|---|---|---|
| `abus_root` | `/Volumes/Autzoko/ABUS` | Root of raw ABUS data |
| `output_base` | `./data/abus` | Where preprocessed files are written |
| `num_processes` | `4` | Parallel workers — reduce if running out of RAM |

### 3.4  Output

```
./data/abus/
    train/      100 x (.npz + .pkl)
    val/         30 x (.npz + .pkl)
    test/        70 x (.npz + .pkl)
```

Each `.npz` contains two arrays:
- `data` — float32, shape `(1, D', H', W')` (normalised image)
- `seg`  — int8, shape `(1, D', H', W')` (mask with values -1, 0, 1)

Each `.pkl` stores a properties dict with spacing, bounding box, class locations, etc.

### 3.5  Disk Space

Each compressed `.npz` is roughly 100 – 300 MB depending on volume size and compressibility. The dataset also unpacks `.npy` files at training time. Budget approximately **100–200 GB** total.

---

## 4  Training (Step 2)

### 4.1  What It Does

- Loads preprocessed train and validation data.
- Instantiates `SegMamba(in_chans=1, out_chans=2)`.
- Trains for 1000 epochs using **DiceLoss + CrossEntropyLoss**.
- Validates every 2 epochs (patch-based Dice on random crops).
- Saves `best_model_*.pt` and `final_model_*.pt` checkpoints.
- Logs scalars (loss, dice, lr) to TensorBoard.

### 4.2  Run

```bash
python abus_train.py
```

### 4.3  Configuration

Edit the variables at the top of `abus_train.py`:

| Variable | Default | Notes |
|---|---|---|
| `data_dir_train` | `./data/abus/train` | Preprocessed training data |
| `data_dir_val` | `./data/abus/val` | Preprocessed validation data |
| `logdir` | `./logs/segmamba_abus` | TensorBoard logs + checkpoints |
| `max_epoch` | `1000` | Total training epochs |
| `batch_size` | `2` | Per-GPU batch size |
| `val_every` | `2` | Validate every N epochs |
| `device` | `cuda:0` | GPU device |
| `roi_size` | `[128, 128, 128]` | Patch size |
| `augmentation` | `True` | Full augmentation pipeline |

### 4.4  Loss Function

```
L = CrossEntropyLoss(pred, label) + DiceLoss(pred, label)
```

`DiceLoss` is configured with `include_background=False` so it only optimises the tumour class. This is critical because the tumour can be as small as 0.1 % of the volume.

### 4.5  Optimiser

| Parameter | Value |
|---|---|
| Optimiser | SGD (Nesterov) |
| Learning rate | 0.01 |
| Momentum | 0.99 |
| Weight decay | 3e-5 |
| LR schedule | Polynomial decay (exponent 0.9) |

### 4.6  Output

```
./logs/segmamba_abus/
    model/
        best_model_0.XXXX.pt       Best validation Dice
        final_model_0.XXXX.pt      Latest checkpoint
        tmp_model_ep99_0.XXXX.pt   Every 100 epochs
    events.out.tfevents.*          TensorBoard logs
```

### 4.7  Monitor Training

```bash
tensorboard --logdir ./logs/segmamba_abus
```

Tracked scalars: `training_loss`, `ce_loss`, `dice_loss`, `mean_dice`, `lr`.

---

## 5  Prediction / Inference (Step 3)

### 5.1  What It Does

For each preprocessed test case:

1. Load the volume from `./data/abus/test/`.
2. Run **sliding-window inference** (128^3 patches, 50 % overlap, Gaussian weighting).
3. Apply **8-way test-time augmentation** (mirroring along all 3 axes).
4. Resample the output back to the pre-crop shape.
5. Restore the original (non-cropped) volume size.
6. Save the binary prediction as `ABUS_XXX.nii.gz`.

### 5.2  Before You Run

Open `abus_predict.py` and update the checkpoint path:

```python
model_path = "./logs/segmamba_abus/model/best_model_0.XXXX.pt"
```

Replace `0.XXXX` with the actual best Dice value from training.

### 5.3  Run

```bash
python abus_predict.py
```

### 5.4  Output

```
./prediction_results/segmamba_abus/
    ABUS_130.nii.gz
    ABUS_131.nii.gz
    ...
    ABUS_199.nii.gz        (70 test cases)
```

Each file is a binary NIfTI volume (0 = background, 1 = tumour) matching the original NRRD dimensions.

---

## 6  Metrics Computation (Step 4)

### 6.1  What It Does

Compares each predicted `.nii.gz` against its ground-truth `MASK_XXX.nrrd` and computes:

- **Dice coefficient** — overlap between prediction and ground truth.
- **HD95** (95th-percentile Hausdorff distance) — surface distance in voxels.

### 6.2  Run

```bash
python abus_compute_metrics.py
```

Optional arguments:

```bash
python abus_compute_metrics.py --pred_name segmamba_abus --split Test
```

### 6.3  Output

Per-case results are printed to the console. Aggregated results are saved to:

```
./prediction_results/result_metrics/segmamba_abus.npy
```

This is a numpy array of shape `(N, 2)` where column 0 = Dice and column 1 = HD95.

---

## 7  File Reference

### Original SegMamba (unchanged)

| File | Purpose |
|---|---|
| `0_inference.py` | Quick forward-pass smoke test |
| `1_rename_mri_data.py` | Rename BraTS filenames |
| `2_preprocessing_mri.py` | BraTS preprocessing |
| `3_train.py` | BraTS training |
| `4_predict.py` | BraTS inference |
| `5_compute_metrics.py` | BraTS metrics |
| `model_segmamba/` | SegMamba model definition |
| `light_training/` | Training framework (Trainer, DataLoader, augmentation, etc.) |
| `monai/` | Bundled MONAI components |
| `mamba/` | Mamba SSM source (requires CUDA compilation) |
| `causal-conv1d/` | Causal convolution source (requires CUDA compilation) |

### ABUS Adaptation (new files)

| File | Purpose |
|---|---|
| `abus_preprocessing.py` | NRRD --> NPZ, crop, normalise, save properties |
| `abus_train.py` | Train SegMamba (1ch input, 2-class output, DiceCE loss) |
| `abus_predict.py` | Sliding-window inference + TTA, save NIfTI predictions |
| `abus_compute_metrics.py` | Dice + HD95 evaluation against NRRD ground truth |
| `ABUS_README.md` | This document |

---

## 8  Directory Layout After All Steps

```
SegMamba/
├── abus_preprocessing.py
├── abus_train.py
├── abus_predict.py
├── abus_compute_metrics.py
├── ABUS_README.md
│
├── data/
│   └── abus/
│       ├── train/           ABUS_000.npz/.pkl ... ABUS_099.npz/.pkl
│       ├── val/             ABUS_100.npz/.pkl ... ABUS_129.npz/.pkl
│       └── test/            ABUS_130.npz/.pkl ... ABUS_199.npz/.pkl
│
├── logs/
│   └── segmamba_abus/
│       ├── model/           best_model_*.pt, final_model_*.pt
│       └── events.*         TensorBoard logs
│
├── prediction_results/
│   ├── segmamba_abus/       ABUS_130.nii.gz ... ABUS_199.nii.gz
│   └── result_metrics/      segmamba_abus.npy
│
├── (original BraTS files: 0_inference.py ... 5_compute_metrics.py)
├── model_segmamba/
├── light_training/
├── monai/
├── mamba/
└── causal-conv1d/
```

---

## 9  Quick-Start (All Commands)

```bash
# ---- one-time setup ----
cd SegMamba
cd causal-conv1d && python setup.py install && cd ..
cd mamba && python setup.py install && cd ..
pip install acvl-utils medpy SimpleITK tqdm scikit-image

# ---- run pipeline ----
python abus_preprocessing.py          # Step 1  (~1-2 hours)
python abus_train.py                  # Step 2  (~days, GPU)

# edit model_path in abus_predict.py, then:
python abus_predict.py                # Step 3  (~hours, GPU)
python abus_compute_metrics.py        # Step 4  (~minutes)
```

---

## 10  Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mamba_ssm` | mamba not compiled | `cd mamba && python setup.py install` (requires CUDA) |
| `ModuleNotFoundError: acvl_utils` | Missing dependency | `pip install acvl-utils` |
| `ModuleNotFoundError: medpy` | Missing dependency | `pip install medpy` |
| Preprocessing worker dies | Out of RAM | Reduce `num_processes` in `abus_preprocessing.py` |
| CUDA OOM during training | GPU memory | Reduce `batch_size` to 1, or reduce `train_process` to 4 |
| CUDA OOM during inference | GPU memory | Set `sw_batch_size=1` (already default) |
| Shape mismatch in metrics | Prediction not full-size | Check that `predict_noncrop_probability` ran correctly |
| All Dice = 0 | Model not trained / wrong checkpoint path | Verify `model_path` in `abus_predict.py` |
