# SegMamba for ABUS 3D Ultrasound Tumour Segmentation

This document is a detailed, step-by-step tutorial for running the complete
SegMamba pipeline on the **ABUS (Automated Breast Ultrasound)** dataset for
binary tumour segmentation. It covers the theoretical background of every
component, the exact data flow through each stage, and practical instructions
for reproducing the experiments.

The four ABUS-specific scripts (`abus_*.py`) sit alongside the original
SegMamba codebase. **No original files are modified.**

---

## Table of Contents

1. [Background — SegMamba Architecture](#1--background--segmamba-architecture)
2. [ABUS Dataset](#2--abus-dataset)
3. [Adaptation Decisions — BraTS to ABUS](#3--adaptation-decisions--brats-to-abus)
4. [Environment Setup (Step 0)](#4--environment-setup-step-0)
5. [Data Preprocessing (Step 1)](#5--data-preprocessing-step-1)
6. [Training (Step 2)](#6--training-step-2)
7. [Prediction / Inference (Step 3)](#7--prediction--inference-step-3)
8. [Metrics Computation (Step 4)](#8--metrics-computation-step-4)
9. [File Reference](#9--file-reference)
10. [Directory Layout After All Steps](#10--directory-layout-after-all-steps)
11. [Quick-Start — All Commands](#11--quick-start--all-commands)
12. [Hyperparameter Tuning Guide](#12--hyperparameter-tuning-guide)
13. [Troubleshooting](#13--troubleshooting)

---

## 1  Background — SegMamba Architecture

SegMamba ([arXiv 2401.13560](https://arxiv.org/abs/2401.13560)) brings the
**Mamba** state-space model into 3D medical image segmentation. Mamba provides
linear-complexity long-range sequence modelling — a crucial advantage for the
enormous token counts in volumetric images (a single 128^3 patch has over
2 million voxels).

### 1.1  High-Level Architecture

SegMamba follows an encoder–decoder design inspired by UNETR. The encoder is a
hierarchical **MambaEncoder**; the decoder uses standard UNETR up-blocks with
skip connections from every encoder level.

```
                        SegMamba Architecture  (for ABUS: in=1, out=2)

  Input (1, 128, 128, 128)
         │
         ├──────────────────────────────────────────────────── enc1 (UnetrBasicBlock)
         │                                                          1 ch → 48 ch
         ▼                                                          128^3
  ┌─────────────────────────── MambaEncoder ──────────────────────────────┐
  │  Stem: Conv3d(1→48, k=7, s=2, p=3)           64^3, 48 ch            │
  │  GSC(48) → 2×MambaLayer(48, nslices=64) → IN → MLP(48→96→48) ──┐   │
  │         │                                                        │   │
  │  DownSample: IN(48) → Conv3d(48→96, k=2, s=2)  32^3, 96 ch     │   │
  │  GSC(96) → 2×MambaLayer(96, nslices=32) → IN → MLP(96→192→96) ─┤   │
  │         │                                                        │   │
  │  DownSample: IN(96) → Conv3d(96→192, k=2, s=2)  16^3, 192 ch   │   │
  │  GSC(192) → 2×MambaLayer(192, nslices=16) → IN → MLP(192→384→192)  │
  │         │                                                        │   │
  │  DownSample: IN(192) → Conv3d(192→384, k=2, s=2)  8^3, 384 ch  │   │
  │  GSC(384) → 2×MambaLayer(384, nslices=8) → IN → MLP(384→768→384)   │
  └──────────┼──────────────────────────────────┼────────────────────┘   │
             │                                  │                        │
             ▼                                  ▼                        ▼
  outs[0]  64^3,48ch              outs[1]  32^3,96ch          outs[2]  16^3,192ch
      │                               │                           │
      ▼                               ▼                           ▼
  enc2 (UnetrBasicBlock)       enc3 (UnetrBasicBlock)      enc4 (UnetrBasicBlock)
      48→96 ch                     96→192 ch                   192→384 ch
                                                                   │
                                                          outs[3]  8^3, 384ch
                                                                   │
                                                                   ▼
                                                          enc5 (UnetrBasicBlock)
                                                              384→768 ch
                                                         enc_hidden  8^3, 768ch
                                                                   │
         ┌──────────── Decoder Path (bottom → top) ───────────────┘
         ▼
  dec3 = decoder5(enc_hidden=768, skip=enc4=384) → 16^3, 384 ch
  dec2 = decoder4(dec3=384,      skip=enc3=192) → 32^3, 192 ch
  dec1 = decoder3(dec2=192,      skip=enc2=96)  → 64^3, 96 ch
  dec0 = decoder2(dec1=96,       skip=enc1=48)  → 128^3, 48 ch
  out  = decoder1(dec0=48)                       → 128^3, 48 ch
         │
         ▼
  UnetOutBlock(48 → 2)  →  Output (2, 128, 128, 128)
```

### 1.2  MambaLayer — Core Building Block

Each MambaLayer processes a 3D feature map as a 1D sequence via the Mamba SSM:

```
Input x: (B, C, D, H, W)
  │
  ├── x_skip = x                          # save for residual
  │
  ├── Flatten spatial:  (B, C, D*H*W)
  ├── Transpose:        (B, D*H*W, C)     # tokens × channels
  ├── LayerNorm(C)
  ├── Mamba(d_model=C, d_state=16, d_conv=4, expand=2, bimamba_type="v3")
  ├── Transpose back:   (B, C, D*H*W)
  ├── Reshape:          (B, C, D, H, W)
  │
  └── Output = reshaped + x_skip          # residual connection
```

**Mamba SSM parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `d_model` | Channel dim (48/96/192/384) | Input/output feature dimension |
| `d_state` | 16 | SSM hidden state dimension |
| `d_conv` | 4 | Local convolution width in SSM |
| `expand` | 2 | Inner dimension = 2 × d_model |
| `bimamba_type` | `"v3"` | Bidirectional Mamba variant |
| `nslices` | 64 / 32 / 16 / 8 | Number of depth slices for the 3D scan order |

The `nslices` parameter is tied to the **first spatial dimension** of each
level's feature map. For a 128^3 input patch:

| Encoder Level | Feature Map | nslices |
|---|---|---|
| 0 (after stem) | 64^3 | 64 |
| 1 | 32^3 | 32 |
| 2 | 16^3 | 16 |
| 3 | 8^3 | 8 |

### 1.3  GSC Block — Gated Spatial Convolution

Every encoder stage begins with a GSC block before the Mamba layers. GSC acts
as a local feature enhancer through two parallel paths fused with a residual:

```
Input x: (B, C, D, H, W)
  │
  ├── x_residual = x
  │
  ├── Path A (3×3 conv × 2):
  │     Conv3d(C→C, k=3, p=1) → InstanceNorm → ReLU
  │     Conv3d(C→C, k=3, p=1) → InstanceNorm → ReLU
  │
  ├── Path B (1×1 conv):
  │     Conv3d(C→C, k=1) → InstanceNorm → ReLU
  │
  ├── Merge = Path_A + Path_B
  ├── Conv3d(C→C, k=1) → InstanceNorm → ReLU
  │
  └── Output = Merge + x_residual
```

### 1.4  Decoder Blocks

The decoder uses MONAI's `UnetrUpBlock` modules. Each up-block:

1. Transposed convolution (upsample kernel = 2) to double spatial dimensions.
2. Concatenates with the encoder skip connection along the channel axis.
3. Two residual convolution sub-blocks (3×3 conv, InstanceNorm, LeakyReLU).

### 1.5  Model Size

With default `feat_size=[48, 96, 192, 384]` and `depths=[2,2,2,2]`:

| Configuration | Parameters |
|---|---|
| BraTS (in=4, out=4) | ~62.8 M |
| **ABUS (in=1, out=2)** | **~62.2 M** |

The channel count difference in the first conv and last out-block is small
relative to the full model. Most parameters reside in the Mamba layers and
UNETR blocks.

---

## 2  ABUS Dataset

### 2.1  Overview

ABUS (Automated Breast Ultrasound) is a 3D ultrasound imaging modality for
breast cancer screening. Each volume captures a full breast in a single sweep,
producing a large 3D image.

| Property | Value |
|---|---|
| Source path | `/Volumes/Autzoko/ABUS` |
| Format | NRRD (`.nrrd`), single-channel, uint8 \[0, 255\] |
| Task | Binary segmentation: 0 = background, 1 = tumour |
| Train / Val / Test | 100 / 30 / 70 cases |
| Spacing | (1.0, 1.0, 1.0) — isotropic |
| Class imbalance | Tumour ≈ 0.01–0.3 % of total voxels |

### 2.2  Volume Sizes

Volumes are large and vary slightly in shape:

| Case | NRRD Sizes Field | NumPy Shape (z, y, x via SimpleITK) |
|---|---|---|
| DATA\_000 (Train) | 865 × 608 × 330 | (330, 608, 865) |
| DATA\_002 (Train) | 865 × 682 × 354 | (354, 682, 865) |
| DATA\_131 (Test) | 843 × 546 × 353 | (353, 546, 843) |

> **Note on axis order:** SimpleITK's `GetArrayFromImage()` returns arrays in
> **(z, y, x)** order, which is the **reverse** of the NRRD sizes field
> (x, y, z). All SegMamba processing uses the NumPy (z, y, x) convention.

### 2.3  Data Layout on Disk

```
/Volumes/Autzoko/ABUS/data/
├── Train/
│   ├── DATA/          DATA_000.nrrd ... DATA_099.nrrd   (100 volumes)
│   ├── MASK/          MASK_000.nrrd ... MASK_099.nrrd   (100 masks)
│   ├── labels.csv     case_id, label (M=malignant / B=benign), paths
│   └── bbx_labels.csv bounding-box annotations (centre + extent)
├── Validation/
│   ├── DATA/          DATA_100.nrrd ... DATA_129.nrrd   (30 volumes)
│   ├── MASK/          MASK_100.nrrd ... MASK_129.nrrd
│   ├── labels.csv
│   └── bbx_labels.csv
└── Test/
    ├── DATA/          DATA_130.nrrd ... DATA_199.nrrd   (70 volumes)
    ├── MASK/          MASK_130.nrrd ... MASK_199.nrrd
    ├── labels.csv
    └── bbx_labels.csv
```

### 2.4  Class Imbalance

Tumour regions are very small relative to the full volume. Examples:

| Case | Total Voxels | Tumour Voxels | Tumour % |
|---|---|---|---|
| MASK\_000 | 173,578,400 | 241,350 | 0.14 % |
| MASK\_001 | 173,578,400 | 21,181 | 0.01 % |
| MASK\_002 | 208,889,580 | 515,503 | 0.25 % |

This severe imbalance motivates the use of DiceLoss in addition to
CrossEntropyLoss (see [Section 6.4](#64--loss-function-design)).

---

## 3  Adaptation Decisions — BraTS to ABUS

| Aspect | BraTS (original `3_train.py`) | ABUS (`abus_train.py`) | Reason |
|---|---|---|---|
| Input channels | 4 (T1, T1c, T2, T2-FLAIR) | **1** (ultrasound) | ABUS is single-modality |
| Output classes | 4 (bg + TC / WT / ET) | **2** (bg + tumour) | Binary segmentation task |
| Label conversion | TC = 1\|3, WT = 1\|2\|3, ET = 3 | **None** | Masks already binary {0, 1} |
| Loss function | CrossEntropyLoss only | **DiceLoss + CE** | Handles severe class imbalance |
| DiceLoss background | N/A | `include_background=False` | Optimise tumour Dice directly |
| Data splitting | Random 70/10/20 from one folder | **Pre-split** directories | ABUS provides official splits |
| Data loader | `get_train_val_test_loader_from_train` | `get_train_val_test_loader_seperate` | Loads from 3 separate dirs |
| Resampling | To 1 mm isotropic | **Skipped** | Spacing already (1, 1, 1) |
| Validation metric | 3-class Dice (TC, WT, ET) | **Single binary Dice** | One foreground class |
| Data workers | `train_process = 18` | `train_process = 8` | ABUS volumes are larger, use fewer workers to save RAM |

Everything else is kept identical: patch size 128^3, SGD + polynomial LR
schedule, full data augmentation, sliding-window inference, 8-way TTA.

---

## 4  Environment Setup (Step 0)

### 4.1  System Requirements

- **GPU:** NVIDIA with CUDA support (tested on A100 / RTX 3090 / V100).
- **VRAM:** ≥ 16 GB recommended for batch\_size=2, patch 128^3.
- **RAM:** ≥ 32 GB (each ABUS volume is ~700 MB as float32).
- **Disk:** ~100–200 GB for preprocessed data (compressed NPZ + unpacked NPY).
- **Python:** 3.8+, PyTorch 1.12+.

### 4.2  Install CUDA-Only Dependencies

SegMamba depends on two packages that must be compiled from source with CUDA:

```bash
cd SegMamba

# 1. causal-conv1d (local convolution kernel used inside Mamba)
cd causal-conv1d
python setup.py install
cd ..

# 2. mamba-ssm (the Mamba state-space model)
cd mamba
python setup.py install
cd ..
```

If compilation fails, check that `nvcc --version` matches your PyTorch CUDA
version. For example, PyTorch built with CUDA 11.8 needs nvcc from the
CUDA 11.8 toolkit.

### 4.3  Install Python Dependencies

```bash
pip install acvl-utils medpy SimpleITK tqdm scikit-image batchgenerators
```

| Package | Used By |
|---|---|
| `acvl-utils` | `crop_to_nonzero` in preprocessing |
| `medpy` | HD95 metric computation |
| `SimpleITK` | NRRD / NIfTI reading and writing |
| `batchgenerators` | Data augmentation pipeline |
| `scikit-image` | Connected-component post-processing |

`monai` is **already bundled** inside the repository (`monai/` directory) — do
**not** install it separately via pip; the bundled version contains
SegMamba-specific patches.

### 4.4  Verify Installation

```bash
python 0_inference.py
```

Expected output: a tensor of shape `(1, 4, 128, 128, 128)` printed without
errors. If you see `ModuleNotFoundError: mamba_ssm`, the Mamba compilation
in Section 4.2 did not succeed.

---

## 5  Data Preprocessing (Step 1)

### 5.1  Purpose

Convert raw NRRD volumes into the format that SegMamba's training framework
expects: compressed NumPy archives (`.npz`) paired with metadata pickle files
(`.pkl`).

### 5.2  Pipeline — Per Case

```
 DATA_XXX.nrrd                   MASK_XXX.nrrd
       │                               │
       ▼                               ▼
  sitk.ReadImage()               sitk.ReadImage()
  GetArrayFromImage()            GetArrayFromImage()
  .astype(float32)               .astype(float32)
  Add channel dim: (1,D,H,W)    Add channel dim: (1,D,H,W)
       │                               │
       ├───────────┬───────────────────┘
       ▼           ▼
  Collect foreground intensity statistics
  (sample 10,000 voxels where mask > 0)
       │
       ▼
  crop_to_nonzero(data, seg)
  ├── Build nonzero mask (data != 0, per-channel OR, fill holes)
  ├── Compute bounding box of nonzero region
  ├── Crop data and seg to bounding box
  └── Mark seg voxels that are 0 AND outside nonzero mask as -1
       │
       ▼
  Z-score normalise: x = (x - mean) / std    [global, not per-channel]
       │
       ▼
  Sample foreground voxel locations           [for oversampling during training]
  (up to 10,000 voxel coords where seg == 1)
       │
       ▼
  Save:  ABUS_XXX.npz  (data=float32, seg=int8)
         ABUS_XXX.pkl  (properties dict)
```

### 5.3  What Is Inside the Properties (`.pkl`) File?

| Key | Type | Description |
|---|---|---|
| `spacing` | tuple (3,) | Original voxel spacing from NRRD header (x,y,z) |
| `raw_size` | tuple (3,) | Volume shape before any processing (D,H,W) |
| `name` | str | Case name, e.g. `"ABUS_000"` |
| `intensities_per_channel` | list[ndarray] | 10,000 sampled foreground pixel values |
| `intensity_statistics_per_channel` | list[dict] | mean, median, min, max, p0.5, p99.5 |
| `original_spacing_trans` | list | Spacing in (z,y,x) order |
| `target_spacing_trans` | list | Target spacing \[1,1,1\] |
| `shape_before_cropping` | tuple (3,) | Shape before crop\_to\_nonzero |
| `bbox_used_for_cropping` | list of \[lo,hi\] | Bounding box coords per axis |
| `shape_after_cropping_before_resample` | tuple (3,) | Shape after crop |
| `shape_after_resample` | tuple (3,) | Same as above (no resampling for ABUS) |
| `class_locations` | dict {1: ndarray} | Foreground voxel coords for oversampling |

### 5.4  The `-1` Label — Background Inside the Crop

`crop_to_nonzero` marks certain segmentation voxels as **-1**:

```
Original volume:       ┌─────────────────────────────┐
                       │  0 0 0 0 0 0 0 0 0 0 0 0 0  │
                       │  0 0 ┌───────────────┐ 0 0  │
                       │  0 0 │ tissue region  │ 0 0  │  ← data != 0
                       │  0 0 │   ●tumour●     │ 0 0  │  ← seg == 1
                       │  0 0 │               │ 0 0  │
                       │  0 0 └───────────────┘ 0 0  │
                       │  0 0 0 0 0 0 0 0 0 0 0 0 0  │
                       └─────────────────────────────┘

After crop (bbox around nonzero mask):
                       ┌───────────────┐
                       │-1  tissue  -1 │  ← corners may be -1
                       │   ●tumour●    │     (inside bbox but outside tissue)
                       │-1  tissue  -1 │
                       └───────────────┘

Labels:  1 = tumour,  0 = background inside tissue,  -1 = background outside tissue
```

During training, the `RemoveLabelTransform(-1, 0)` in the augmentation pipeline
converts -1 back to 0 so the network only sees labels {0, 1}.

> **ABUS-specific note:** Many ABUS volumes have a minimum pixel value > 0
> (e.g. 12–14), which means the entire volume is "nonzero" and
> `crop_to_nonzero` does **not** reduce the size. For cases with min=0, the
> zero-valued border regions are removed.

### 5.5  Run Preprocessing

```bash
python abus_preprocessing.py
```

### 5.6  Configuration

Edit the `if __name__` block at the bottom of `abus_preprocessing.py`:

| Variable | Default | Notes |
|---|---|---|
| `abus_root` | `/Volumes/Autzoko/ABUS` | Path to raw ABUS data |
| `output_base` | `./data/abus` | Output directory for preprocessed files |
| `num_processes` | `4` | Parallel workers — **reduce to 1–2 if running out of RAM** (each worker loads one full volume ≈ 700 MB) |

### 5.7  Output

```
./data/abus/
├── train/       ABUS_000.npz + .pkl  ...  ABUS_099.npz + .pkl   (100 cases)
├── val/         ABUS_100.npz + .pkl  ...  ABUS_129.npz + .pkl   (30 cases)
└── test/        ABUS_130.npz + .pkl  ...  ABUS_199.npz + .pkl   (70 cases)
```

### 5.8  Disk Space Estimate

| Component | Per Case | Total (200 cases) |
|---|---|---|
| Compressed `.npz` | 100–300 MB | ~20–60 GB |
| Unpacked `.npy` (data, float32) | ~700 MB | ~140 GB |
| Unpacked `_seg.npy` (int8) | ~170 MB | ~34 GB |
| `.pkl` (properties) | < 1 MB | negligible |
| **Total** | | **~100–200 GB** |

The `.npy` files are created automatically by `MedicalDataset` when the
training script first loads the data. They enable memory-mapped reading so that
only the requested patch is loaded into RAM.

---

## 6  Training (Step 2)

### 6.1  Data Loading Pipeline

The training data loader is a multi-stage pipeline:

```
MedicalDataset
  │  Reads .npy files via memory-mapping (np.load "r+" mode)
  │  Returns: {"data": (1,D,H,W), "seg": (1,D,H,W), "properties": dict}
  ▼
DataLoaderMultiProcess
  │  Random sampling with foreground oversampling (33%)
  │  Extracts 128^3 patches from full volumes
  │  Output: {"data": (B,1,128,128,128), "seg": (B,1,128,128,128)}
  ▼
LimitedLenWrapper (batchgenerators multi-threaded augmentation)
  │  Applies spatial + intensity transforms (see 6.3)
  │  250 steps per epoch, 8 worker threads
  ▼
Trainer.to_device()
  │  Moves batch to GPU, converts to torch tensors
  ▼
Training step
```

### 6.2  Foreground Oversampling

Because tumours are so small, purely random patches would rarely contain
tumour tissue. The `DataLoaderMultiProcess` uses **foreground oversampling** to
ensure the model regularly sees tumour voxels:

- **Oversample rate:** 33 % of each batch is guaranteed to contain foreground.
- **Mechanism:** For a `batch_size=2`, the last `round(2 × 0.33) = 1` sample
  is forced to be centred on a foreground voxel.
- **How the centre is picked:**
  1. Randomly select a foreground class from `class_locations` (for ABUS: class 1).
  2. Randomly pick one of the pre-computed foreground voxel coordinates.
  3. Centre the 128^3 patch on that voxel (clamped to stay within the volume).
- **Remaining samples** (the first sample in a batch of 2) are cropped randomly
  from anywhere in the volume.

### 6.3  Data Augmentation

The augmentation pipeline is applied on-the-fly to every training batch. It
consists of **spatial transforms**, **intensity transforms**, and **mirroring**:

#### Spatial Transforms (20% per sample)

| Transform | Parameters |
|---|---|
| **Rotation** | ±30° (±π/6 rad) around each of the 3 axes independently; 100% per-axis probability |
| **Scaling** | 0.7× to 1.4×; 20% probability |
| **Elastic deformation** | Disabled (α=0, σ=0) |
| **Interpolation** | order=3 (cubic) for images, order=1 (linear) for segmentation |
| **Border padding** | Constant 0 for data, constant -1 for segmentation |

#### Intensity Transforms

| Transform | Parameters | Probability |
|---|---|---|
| **Gaussian noise** | Additive noise | 10% |
| **Gaussian blur** | σ ∈ \[0.5, 1.0\], 50% per-channel | 20% |
| **Brightness** | Multiplicative factor ∈ \[0.75, 1.25\] | 15% |
| **Contrast** | Contrast adjustment | 15% |
| **Low-resolution simulation** | Downsample to 0.5–1.0× then upsample back; 50% per-channel | 25% |
| **Gamma correction (×2)** | γ ∈ \[0.7, 1.5\], retain\_stats=True | 10% + 30% |

#### Mirroring

| Transform | Parameters |
|---|---|
| **Random mirror** | Axes \[0, 1, 2\] (depth, height, width) — 50% per axis |

#### Clean-Up

| Transform | Description |
|---|---|
| **RemoveLabelTransform** | Converts label -1 → 0 |
| **NumpyToTensor** | Converts data and seg to PyTorch float tensors |

### 6.4  Loss Function Design

```
L_total = L_CE + L_Dice
```

| Component | Formula | Why |
|---|---|---|
| `CrossEntropyLoss` | Standard per-voxel CE on 2-class logits | Stable gradient signal across all voxels; prevents mode collapse |
| `DiceLoss` | 1 − (2·\|P∩G\| / (\|P\|+\|G\|)), **tumour class only** | Directly optimises the Dice metric; counteracts class imbalance |

**Critical setting:** `DiceLoss(include_background=False)`.
If background were included, the Dice would be dominated by the huge
background region (>99.7% of voxels), and the loss gradient for the tumour
class would be negligible.

### 6.5  Optimiser and Learning Rate Schedule

| Parameter | Value |
|---|---|
| Optimiser | SGD with Nesterov momentum |
| Initial learning rate | 0.01 |
| Momentum | 0.99 |
| Weight decay | 3 × 10⁻⁵ |
| LR schedule | Polynomial decay: `lr = lr_init × (1 − step/max_steps)^0.9` |
| Gradient clipping | Max norm = 12 |
| Mixed precision | AMP (automatic mixed precision) with GradScaler |

### 6.6  Validation

- **Frequency:** Every 2 epochs.
- **Method:** 100 random patches (128^3) from validation volumes.
- **Metric:** Binary Dice coefficient (argmax of 2-class output vs. label).
- **Checkpointing:** `best_model_*.pt` saved when validation Dice improves;
  `final_model_*.pt` overwritten every validation; `tmp_model_ep*.pt` saved
  every 100 epochs as milestone.

> **Note:** Patch-based validation Dice during training is an approximation.
> The true test-set evaluation (Step 3–4) uses full-volume sliding-window
> inference with TTA for accurate results.

### 6.7  Run Training

```bash
python abus_train.py
```

### 6.8  Configuration

Edit the top of `abus_train.py`:

| Variable | Default | Notes |
|---|---|---|
| `data_dir_train` | `./data/abus/train` | Preprocessed training data |
| `data_dir_val` | `./data/abus/val` | Preprocessed validation data |
| `logdir` | `./logs/segmamba_abus` | TensorBoard logs + checkpoints |
| `max_epoch` | `1000` | Total training epochs |
| `batch_size` | `2` | Per-GPU batch size — reduce to 1 if OOM |
| `val_every` | `2` | Validate every N epochs |
| `device` | `cuda:0` | GPU device |
| `roi_size` | `[128, 128, 128]` | Training patch size |
| `augmentation` | `True` | Full augmentation. Alternatives: `"nomirror"`, `"onlymirror"`, `"onlyspatial"` |

### 6.9  Monitor Training

```bash
tensorboard --logdir ./logs/segmamba_abus
```

Tracked scalars:

| Scalar | Meaning |
|---|---|
| `training_loss` | DiceLoss + CE combined |
| `ce_loss` | CrossEntropyLoss alone |
| `dice_loss` | DiceLoss alone |
| `mean_dice` | Validation binary Dice |
| `lr` | Current learning rate |

### 6.10  Output

```
./logs/segmamba_abus/
├── model/
│   ├── best_model_0.XXXX.pt       ← Best validation Dice
│   ├── final_model_0.XXXX.pt      ← Latest checkpoint
│   └── tmp_model_ep99_0.XXXX.pt   ← Every 100 epochs
└── events.out.tfevents.*           ← TensorBoard log files
```

---

## 7  Prediction / Inference (Step 3)

### 7.1  Inference Strategy

Full-volume inference on the large ABUS volumes uses three techniques:

#### 7.1.1  Sliding-Window Inference

The volume is too large to process in one forward pass. Instead, MONAI's
`SlidingWindowInferer` tiles the volume into overlapping 128^3 patches:

```
Full preprocessed volume (1, D', H', W')
  │
  ├── Split into 128^3 patches with 50% overlap
  ├── Forward-pass each patch through the model
  ├── Aggregate with Gaussian weighting (centre pixels weighted more)
  │
  └── Fused output (2, D', H', W')
```

| Parameter | Value |
|---|---|
| `roi_size` | \[128, 128, 128\] |
| `sw_batch_size` | 1 (patches processed one at a time to save VRAM) |
| `overlap` | 0.5 (50 %) |
| `mode` | `"gaussian"` (smooth blending of overlapping patches) |

#### 7.1.2  Test-Time Augmentation (8-Way Mirroring)

Each volume is predicted 8 times under all combinations of flipping along the
3 spatial axes. The 8 predictions are averaged for a more robust result.

```
        mirror_axes = [0, 1, 2]  →  2^3 = 8 combinations

        Prediction 1:  original input
        Prediction 2:  flip depth     (torch.flip dim 2)
        Prediction 3:  flip height    (torch.flip dim 3)
        Prediction 4:  flip width     (torch.flip dim 4)
        Prediction 5:  flip depth  + height
        Prediction 6:  flip depth  + width
        Prediction 7:  flip height + width
        Prediction 8:  flip depth  + height + width

        Final = (P1 + P2 + ... + P8) / 8
```

Each prediction is flipped back to the original orientation before averaging.

#### 7.1.3  Post-Processing — Restore Original Resolution

```
Model output (2, D', H', W')       ← preprocessed (cropped) shape
  │
  ├── argmax(dim=0) → binary mask (D', H', W')
  │
  ├── Resample to pre-crop shape via trilinear interpolation
  │   (in ABUS this is typically a no-op since no resampling was done)
  │
  ├── Restore original volume size by padding with zeros
  │   (undo the crop_to_nonzero bounding box)
  │
  └── Save as NIfTI: ABUS_XXX.nii.gz  (uint8, values 0/1)
```

### 7.2  Before You Run

Open `abus_predict.py` and update the checkpoint path:

```python
model_path = "./logs/segmamba_abus/model/best_model_0.XXXX.pt"
```

Replace `0.XXXX` with the actual best Dice value shown in the filename.

### 7.3  Run

```bash
python abus_predict.py
```

### 7.4  Output

```
./prediction_results/segmamba_abus/
├── ABUS_130.nii.gz
├── ABUS_131.nii.gz
├── ...
└── ABUS_199.nii.gz        (70 test cases)
```

Each file is a binary NIfTI volume (0 = background, 1 = tumour) with the same
dimensions and spacing as the original NRRD.

---

## 8  Metrics Computation (Step 4)

### 8.1  Metrics

| Metric | Definition | Ideal Value |
|---|---|---|
| **Dice coefficient** | 2 \|P ∩ G\| / (\|P\| + \|G\|) | 1.0 |
| **HD95** | 95th percentile of Hausdorff surface distances (in voxels) | 0.0 |

Edge cases:

| Prediction | Ground Truth | Dice | HD95 |
|---|---|---|---|
| Non-empty | Non-empty | Computed normally | Computed normally |
| Empty | Empty | 1.0 | 0.0 |
| One empty, one non-empty | — | 0.0 | 50.0 (penalty) |

### 8.2  How It Works

For each test case:

1. Load the predicted `.nii.gz` with SimpleITK.
2. Load the ground-truth `MASK_XXX.nrrd` from the original ABUS directory.
3. Binarise both (> 0).
4. Verify shapes match.
5. Compute Dice and HD95 using `medpy.metric.binary`.

### 8.3  Run

```bash
python abus_compute_metrics.py
```

Optional arguments:

```bash
python abus_compute_metrics.py --pred_name segmamba_abus --split Test
```

| Argument | Default | Options |
|---|---|---|
| `--pred_name` | `segmamba_abus` | Name of the prediction folder under `prediction_results/` |
| `--split` | `Test` | `Train`, `Validation`, or `Test` |

### 8.4  Output

Per-case results are printed to the console:

```
  ABUS_130  Dice=0.7823  HD95=12.45
  ABUS_131  Dice=0.6512  HD95=18.20
  ...
============================================================
  Results for segmamba_abus  (Test split)
  Cases evaluated: 70
============================================================
  Dice  — mean: 0.XXXX  std: 0.XXXX
  HD95  — mean: XX.XX   std: XX.XX
============================================================
```

Aggregated metric array saved to:

```
./prediction_results/result_metrics/segmamba_abus.npy     # shape (N, 2)
```

Column 0 = Dice, column 1 = HD95. Load with `np.load(...)` for further analysis.

---

## 9  File Reference

### Original SegMamba (unchanged)

| File | Purpose |
|---|---|
| `0_inference.py` | Quick forward-pass smoke test |
| `1_rename_mri_data.py` | Rename BraTS filenames |
| `2_preprocessing_mri.py` | BraTS preprocessing (4-modality MRI) |
| `3_train.py` | BraTS training (4-class segmentation) |
| `4_predict.py` | BraTS sliding-window inference |
| `5_compute_metrics.py` | BraTS TC/WT/ET metrics |
| `model_segmamba/` | **SegMamba model definition** (`segmamba.py`) |
| `light_training/` | Training framework: `trainer.py`, `dataloading/`, `augment/`, `preprocessing/`, `evaluation/`, `prediction.py` |
| `monai/` | Bundled MONAI components (SlidingWindowInferer, UnetrBlocks, DiceLoss, etc.) |
| `mamba/` | Mamba SSM source (requires CUDA compilation) |
| `causal-conv1d/` | Causal convolution source (requires CUDA compilation) |

### ABUS Adaptation (new files)

| File | Lines | Purpose |
|---|---|---|
| `abus_preprocessing.py` | 321 | NRRD → NPZ: read, crop, normalise, save properties |
| `abus_train.py` | 203 | Train SegMamba (1-ch input, 2-class output, DiceCE loss) |
| `abus_predict.py` | 209 | Sliding-window inference + 8-way TTA, save NIfTI |
| `abus_compute_metrics.py` | 125 | Dice + HD95 evaluation against NRRD ground truth |
| `ABUS_README.md` | — | This document |

---

## 10  Directory Layout After All Steps

```
SegMamba/
│
│  # ──── ABUS adaptation scripts ────
├── abus_preprocessing.py
├── abus_train.py
├── abus_predict.py
├── abus_compute_metrics.py
├── ABUS_README.md
│
│  # ──── Preprocessed data ────
├── data/
│   └── abus/
│       ├── train/           ABUS_000.npz/.npy/.pkl  ...  ABUS_099.*
│       ├── val/             ABUS_100.*  ...  ABUS_129.*
│       └── test/            ABUS_130.*  ...  ABUS_199.*
│
│  # ──── Training outputs ────
├── logs/
│   └── segmamba_abus/
│       ├── model/
│       │   ├── best_model_0.XXXX.pt
│       │   ├── final_model_0.XXXX.pt
│       │   └── tmp_model_ep99_0.XXXX.pt
│       └── events.out.tfevents.*
│
│  # ──── Prediction outputs ────
├── prediction_results/
│   ├── segmamba_abus/       ABUS_130.nii.gz  ...  ABUS_199.nii.gz
│   └── result_metrics/      segmamba_abus.npy
│
│  # ──── Original SegMamba (unchanged) ────
├── 0_inference.py  ...  5_compute_metrics.py
├── model_segmamba/
├── light_training/
├── monai/
├── mamba/
├── causal-conv1d/
└── README.md
```

---

## 11  Quick-Start — All Commands

```bash
# ================================================================
# Step 0 — Environment (one-time)
# ================================================================
cd SegMamba
cd causal-conv1d && python setup.py install && cd ..
cd mamba         && python setup.py install && cd ..
pip install acvl-utils medpy SimpleITK tqdm scikit-image batchgenerators
python 0_inference.py                          # verify installation

# ================================================================
# Step 1 — Preprocessing                        (~1–2 hours, CPU)
# ================================================================
python abus_preprocessing.py

# ================================================================
# Step 2 — Training                              (~days, 1× GPU)
# ================================================================
python abus_train.py
# Monitor:  tensorboard --logdir ./logs/segmamba_abus

# ================================================================
# Step 3 — Prediction                            (~hours, 1× GPU)
# ================================================================
# First, edit model_path in abus_predict.py to point to your best checkpoint
python abus_predict.py

# ================================================================
# Step 4 — Evaluation                            (~minutes, CPU)
# ================================================================
python abus_compute_metrics.py
```

---

## 12  Hyperparameter Tuning Guide

### Things to try if Dice is low

| Change | How | Expected Effect |
|---|---|---|
| **Increase foreground oversampling** | In `abus_train.py`, set `self.oversample_foreground_percent = 0.67` in the DataLoaderMultiProcess (via subclassing or patching) | Model sees more tumour patches; may improve recall |
| **Lower learning rate** | Change `lr=1e-2` → `lr=1e-3` | More stable convergence; useful if loss is oscillating |
| **Longer warm-up** | Change `scheduler_type = "poly"` → `"poly_with_warmup"` and set `self.warmup = 0.05` | Gentler start; helps with large initial gradients |
| **More augmentation** | Already using full augmentation; try `"onlyspatial"` to reduce noise | Sometimes less augmentation helps on small datasets |
| **Weight the CE loss** | Replace `nn.CrossEntropyLoss()` with `nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))` | Penalise missing tumours more heavily |
| **Larger patch size** | Change `roi_size = [160, 160, 160]` and update `num_slices_list` in model | Larger context; requires more VRAM |

### Things to try if training is too slow / OOM

| Change | How | Expected Effect |
|---|---|---|
| **Reduce batch size** | `batch_size = 1` | Halves VRAM; may need LR adjustment |
| **Reduce data workers** | `self.train_process = 4` | Less RAM usage from augmentation threads |
| **Reduce patch size** | `roi_size = [96, 96, 96]` | Faster training; less context per patch |
| **Disable TTA at inference** | In `abus_predict.py`, set `mirror_axes=None` | 8× faster inference; slightly lower Dice |

---

## 13  Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mamba_ssm` | Mamba not compiled | `cd mamba && python setup.py install` — requires CUDA toolkit matching your PyTorch CUDA version |
| `ModuleNotFoundError: causal_conv1d` | causal-conv1d not compiled | `cd causal-conv1d && python setup.py install` |
| `ModuleNotFoundError: acvl_utils` | Missing pip package | `pip install acvl-utils` |
| `ModuleNotFoundError: medpy` | Missing pip package | `pip install medpy` |
| `No module named 'batchgenerators'` | Missing pip package | `pip install batchgenerators` |
| Preprocessing worker dies silently | Out of RAM | Reduce `num_processes` to 1 or 2 |
| `CUDA out of memory` during training | GPU VRAM exceeded | Reduce `batch_size` to 1; reduce `train_process` to 4; try smaller `roi_size` |
| `CUDA out of memory` during inference | GPU VRAM exceeded | `sw_batch_size` is already 1; the Predictor falls back to CPU interpolation automatically |
| Shape mismatch in metrics | Prediction not restored to original size | Verify that `predict_noncrop_probability` ran without error during inference |
| All Dice = 0.0 during evaluation | Wrong checkpoint or empty predictions | Check `model_path` in `abus_predict.py`; inspect a prediction with e.g. `python -c "import SimpleITK as sitk; print(sitk.GetArrayFromImage(sitk.ReadImage('prediction_results/segmamba_abus/ABUS_130.nii.gz')).sum())"` |
| Validation Dice stuck at 0.0 during training | Model predicting all-background | Expected in early epochs; if persists beyond epoch 50, check that foreground oversampling is working (class\_locations in .pkl should be non-empty) |
| `nvcc` version mismatch | CUDA toolkit ≠ PyTorch CUDA | Run `python -c "import torch; print(torch.version.cuda)"` and install matching CUDA toolkit |
| TensorBoard shows no data | Wrong logdir | Ensure you run `tensorboard --logdir ./logs/segmamba_abus` (not `./logs/segmamba`) |
