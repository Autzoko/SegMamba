# SegMamba for ABUS 3D Ultrasound

Six pipelines for the ABUS (Automated Breast Ultrasound) dataset:

- **Pipeline A — Segmentation** (`abus_*.py`) — per-voxel tumour masks via SegMamba encoder-decoder
- **Pipeline B — Detection FCOS** (`abus_det_*.py`) — 3D bounding boxes via MambaEncoder + FPN + FCOS head
- **Pipeline C — Detection DETR** (`abus_detr_*.py`) — 3D bounding boxes via MambaEncoder + DETR transformer decoder
- **Pipeline D — Multi-task BoxHead** (`abus_boxhead_*.py`) — segmentation + attention-based box regression from decoder features
- **Pipeline E — Patch-Set Global Fusion** (`abus_patch_fusion_*.py`) — multi-task with differentiable patch-to-global box fusion
- **Pipeline F — Two-Stage Training** (`abus_stage2_boxhead_*.py`) — **recommended**: train segmentation first, then BoxHead on frozen features

---

## Environment Setup

### 1. Install CUDA dependencies (requires NVIDIA GPU + CUDA toolkit)

```bash
cd SegMamba
cd causal-conv1d && python setup.py install && cd ..
cd mamba         && python setup.py install && cd ..
```

### 2. Install Python packages

```bash
pip install acvl-utils medpy SimpleITK tqdm scikit-image batchgenerators einops
```

> Do **not** `pip install monai` — it is bundled in the repo.

### 3. Verify

```bash
python 0_inference.py
```

---

# Pipeline A — Segmentation

## Step 1 — Preprocessing

Convert raw NRRD volumes to SegMamba NPZ format.

```bash
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus
```

| Argument | Default | Description |
|---|---|---|
| `--abus_root` | `/Volumes/Autzoko/ABUS` | Path to raw ABUS dataset |
| `--output_base` | `./data/abus` | Output directory |
| `--num_processes` | `4` | Parallel workers (reduce if OOM) |

Output:
```
./data/abus/{train,val,test}/ABUS_XXX.npz + .pkl
```

---

## Step 2 — Training

```bash
python abus_train.py --data_dir_train ./data/abus/train --data_dir_val ./data/abus/val
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir_train` | `./data/abus/train` | Training data directory |
| `--data_dir_val` | `./data/abus/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus` | Log and checkpoint directory |
| `--max_epoch` | `1000` | Total training epochs |
| `--batch_size` | `2` | Per-GPU batch size |
| `--val_every` | `2` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.01` | Initial learning rate |
| `--num_workers` | `8` | Data loader workers |

Monitor with TensorBoard:
```bash
tensorboard --logdir ./logs/segmamba_abus
```

Output:
```
./logs/segmamba_abus/model/best_model_*.pt
```

---

## Step 3 — Prediction

```bash
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model_XXXX.pt
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | **(required)** | Path to trained checkpoint |
| `--data_dir_test` | `./data/abus/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus` | Output directory |
| `--device` | `cuda:0` | GPU device |

Output:
```
./prediction_results/segmamba_abus/ABUS_XXX.nii.gz
```

---

## Step 4 — Evaluation

```bash
python abus_compute_metrics.py --pred_name segmamba_abus --split Test
```

| Argument | Default | Description |
|---|---|---|
| `--pred_name` | `segmamba_abus` | Prediction folder name |
| `--split` | `Test` | `Train`, `Validation`, or `Test` |

Reports per-case and mean **Dice** and **HD95**.

---

## Segmentation Quick-Start

```bash
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_train.py
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model_XXXX.pt
python abus_compute_metrics.py
```

---

# Pipeline B — Detection (SegMamba-Det)

SegMamba-Det reuses the MambaEncoder backbone from SegMamba and replaces
the UNETR decoder with an FPN neck + FCOS-style 3D anchor-free detection
head. It predicts bounding boxes directly (not derived from masks).

Architecture: `MambaEncoder → FPN3D → FCOS3DHead`

The entire volume is resized to 128^3 for detection (required by
MambaEncoder's hardcoded `num_slices_list`).

## Det Step 1 — Preprocessing

Derive bounding-box GT from segmentation masks, resize volumes to 128^3.

```bash
python abus_det_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus_det
```

| Argument | Default | Description |
|---|---|---|
| `--abus_root` | `/Volumes/Autzoko/ABUS` | Path to raw ABUS dataset |
| `--output_base` | `./data/abus_det` | Output directory |

Output:
```
./data/abus_det/{train,val,test}/ABUS_XXX.npz
```

Each NPZ contains: `data` (1,128,128,128), `boxes` (N,6), `original_shape`, `spacing`.

## Det Step 2 — Training

```bash
python abus_det_train.py --data_dir_train ./data/abus_det/train --data_dir_val ./data/abus_det/val
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir_train` | `./data/abus_det/train` | Training data directory |
| `--data_dir_val` | `./data/abus_det/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus_det` | Log and checkpoint directory |
| `--max_epoch` | `300` | Total training epochs |
| `--batch_size` | `2` | Per-GPU batch size |
| `--val_every` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.0001` | Learning rate (AdamW) |
| `--num_workers` | `4` | Data loader workers |
| `--pretrained_backbone` | `""` | Optional: path to segmentation checkpoint to init MambaEncoder |

Monitor:
```bash
tensorboard --logdir ./logs/segmamba_abus_det
```

## Det Step 3 — Prediction

```bash
python abus_det_predict.py --model_path ./logs/segmamba_abus_det/model/best_model_XXXX.pt
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | **(required)** | Path to trained checkpoint |
| `--data_dir_test` | `./data/abus_det/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus_det` | Output directory |
| `--device` | `cuda:0` | GPU device |
| `--score_thresh` | `0.05` | Score filter threshold |
| `--nms_thresh` | `0.3` | NMS IoU threshold |

Output: `./prediction_results/segmamba_abus_det/detections.json`

## Det Step 4 — Evaluation

```bash
python abus_det_compute_metrics.py --abus_root /Volumes/Autzoko/ABUS --split Test
```

| Argument | Default | Description |
|---|---|---|
| `--pred_file` | `./prediction_results/segmamba_abus_det/detections.json` | Predictions file |
| `--abus_root` | `/Volumes/Autzoko/ABUS` | Raw dataset root (for GT masks) |
| `--split` | `Test` | `Train`, `Validation`, or `Test` |

Reports **AP@0.1**, **AP@0.25**, **AP@0.5**, recall, and mean best IoU.

## Detection Quick-Start

```bash
python abus_det_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_det_train.py
python abus_det_predict.py --model_path ./logs/segmamba_abus_det/model/best_model_XXXX.pt
python abus_det_compute_metrics.py --abus_root /Volumes/Autzoko/ABUS
```

---

# Pipeline C — Detection with DETR (SegMamba-DETR)

SegMamba-DETR replaces the FPN + FCOS head from Pipeline B with a DETR-style
transformer encoder-decoder. Learned object queries directly output a fixed-size
set of (class, box) predictions via Hungarian matching — **no NMS required**.

Architecture: `MambaEncoder (level 3, 8^3) → 1×1 proj → Transformer Encoder (4L) → Transformer Decoder (4L, 20 queries) → Class + Box heads`

## DETR Step 1 — Preprocessing

**Same as Pipeline B.** Uses `abus_det_preprocessing.py`.

```bash
python abus_det_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus_det
```

## DETR Step 2 — Training

```bash
python abus_detr_train.py --data_dir_train ./data/abus_det/train --data_dir_val ./data/abus_det/val
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir_train` | `./data/abus_det/train` | Training data directory |
| `--data_dir_val` | `./data/abus_det/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus_detr` | Log and checkpoint directory |
| `--max_epoch` | `300` | Total training epochs |
| `--batch_size` | `2` | Per-GPU batch size |
| `--val_every` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.0001` | Learning rate (AdamW) |
| `--num_queries` | `20` | Learned object queries |
| `--d_model` | `256` | Transformer hidden dimension |
| `--nhead` | `8` | Attention heads |
| `--enc_layers` | `4` | Transformer encoder layers |
| `--dec_layers` | `4` | Transformer decoder layers |
| `--dim_feedforward` | `1024` | FFN intermediate dimension |
| `--pretrained_backbone` | `""` | Path to segmentation checkpoint to init MambaEncoder |

Monitor:
```bash
tensorboard --logdir ./logs/segmamba_abus_detr
```

## DETR Step 3 — Prediction

```bash
python abus_detr_predict.py --model_path ./logs/segmamba_abus_detr/model/best_model_XXXX.pt
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | **(required)** | Path to trained checkpoint |
| `--data_dir_test` | `./data/abus_det/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus_detr` | Output directory |
| `--device` | `cuda:0` | GPU device |
| `--score_thresh` | `0.05` | Score filter threshold |

Output: `./prediction_results/segmamba_abus_detr/detections.json`

## DETR Step 4 — Evaluation

**Same as Pipeline B.** Uses `abus_det_compute_metrics.py`.

```bash
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_detr/detections.json \
    --abus_root /Volumes/Autzoko/ABUS --split Test
```

Reports **AP@0.1**, **AP@0.25**, **AP@0.5**, recall, and mean best IoU.

## DETR Quick-Start

```bash
python abus_det_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_detr_train.py
python abus_detr_predict.py --model_path ./logs/segmamba_abus_detr/model/best_model_XXXX.pt
python abus_det_compute_metrics.py --pred_file ./prediction_results/segmamba_abus_detr/detections.json --abus_root /Volumes/Autzoko/ABUS
```

---

# Pipeline D — Multi-task Detection (SegMamba-BoxHead)

SegMamba-BoxHead extends the original SegMamba segmentation model with a
lightweight attention-based Box Head that branches off the decoder's
full-resolution feature map. The Box Head learns a spatial attention map over
all voxels, aggregates features into a global vector, and regresses a single 3D
bounding box via MLP — all while retaining the original segmentation
supervision (Dice + CE).

Architecture: `SegMamba encoder-decoder → decoder1 (B,48,128,128,128) → UnetOutBlock (seg) + BoxHead (det)`

The BoxHead applies 3D convolutions for feature compression, predicts a
single-channel attention weight map with softmax over the full voxel space,
performs weighted aggregation, and regresses normalised box parameters through
an MLP.

**Note:** Standard BoxHead training uses 128³ resized volumes (~30 sec/epoch).
For full-resolution detection, see [Full-Resolution Detection](#full-resolution-detection) below.

## BoxHead Step 1 — Preprocessing

Combines segmentation masks and detection boxes at 128^3 resolution.

```bash
python abus_boxhead_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus_boxhead
```

| Argument | Default | Description |
|---|---|---|
| `--abus_root` | `/Volumes/Autzoko/ABUS` | Path to raw ABUS dataset |
| `--output_base` | `./data/abus_boxhead` | Output directory |

Output:
```
./data/abus_boxhead/{train,val,test}/ABUS_XXX.npz
```

Each NPZ contains: `data` (1,128,128,128), `seg` (1,128,128,128), `boxes` (N,6), `original_shape`, `spacing`.

## BoxHead Step 2 — Training

```bash
python abus_boxhead_train.py --data_dir_train ./data/abus_boxhead/train --data_dir_val ./data/abus_boxhead/val
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir_train` | `./data/abus_boxhead/train` | Training data directory |
| `--data_dir_val` | `./data/abus_boxhead/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus_boxhead` | Log and checkpoint directory |
| `--max_epoch` | `500` | Total training epochs |
| `--batch_size` | `2` | Per-GPU batch size |
| `--val_every` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.01` | Learning rate (SGD) |
| `--det_weight` | `5.0` | Detection loss weight relative to seg loss |
| `--num_workers` | `4` | Data loader workers |
| `--pretrained_seg` | `""` | Path to SegMamba segmentation checkpoint |
| `--freeze_backbone` | (flag) | Freeze encoder/decoder, train BoxHead only |

Loss: `total = (Dice + CE) + det_weight × (SmoothL1 + GIoU)`

Monitor:
```bash
tensorboard --logdir ./logs/segmamba_abus_boxhead
```

## BoxHead Step 3 — Prediction

```bash
python abus_boxhead_predict.py --model_path ./logs/segmamba_abus_boxhead/model/best_model_XXXX.pt
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | **(required)** | Path to trained checkpoint |
| `--data_dir_test` | `./data/abus_boxhead/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus_boxhead` | Output directory |
| `--save_seg` | (flag) | Also save segmentation masks as NIfTI |
| `--device` | `cuda:0` | GPU device |

Output: `./prediction_results/segmamba_abus_boxhead/detections.json`

## BoxHead Step 4 — Evaluation

**Same as Pipeline B/C.** Uses `abus_det_compute_metrics.py`.

```bash
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_boxhead/detections.json \
    --abus_root /Volumes/Autzoko/ABUS --split Test
```

Reports **AP@0.1**, **AP@0.25**, **AP@0.5**, recall, and mean best IoU.

## BoxHead Quick-Start

```bash
python abus_boxhead_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_boxhead_train.py
python abus_boxhead_predict.py --model_path ./logs/segmamba_abus_boxhead/model/best_model_XXXX.pt
python abus_det_compute_metrics.py --pred_file ./prediction_results/segmamba_abus_boxhead/detections.json --abus_root /Volumes/Autzoko/ABUS
```

---

# Pipeline E — Patch-Set Global Fusion (Recommended)

SegMamba with Patch-Set Global Fusion combines native patch-based segmentation
training with end-to-end global detection optimization. **This is the recommended
approach for multi-task segmentation + detection.**

## Key Idea

Each 128³ patch produces:
- Segmentation logits (same as original SegMamba)
- Local box prediction `b_i` (mapped to global coordinates)
- Objectness score `o_i` (is target present in this patch?)
- Quality score `q_i` (how reliable is this patch's prediction?)

Multiple patches from the same volume are **fused via differentiable soft-weighted
aggregation** into a single global box:

```
b_global = Σ(o_i × q_i × b_i) / Σ(o_i × q_i)
```

**Benefits:**
- A patch doesn't need to fully contain the target to contribute
- Network learns which patches are reliable via auxiliary supervision
- End-to-end global detection without post-processing
- Maintains native SegMamba segmentation accuracy
- Robust to patch truncation at volume boundaries

## Architecture

```
Input Patch (B, 1, 128, 128, 128)
    │
    ▼
SegMamba Encoder-Decoder (unchanged)
    │
    ├─→ decoder1 (B, 48, 128, 128, 128)
    │     │
    │     ├─→ UnetOutBlock → Segmentation logits
    │     │
    │     └─→ PatchBoxHead:
    │           Conv3d(48→64, s=1) + IN + ReLU
    │           Conv3d(64→64, s=2) + IN + ReLU  (128→64)
    │           Conv3d(64→64, s=2) + IN + ReLU  (64→32)
    │           Global Avg Pool → (B, 64)
    │           ├─→ MLP → box (B, 6)
    │           ├─→ MLP → objectness (B, 1)
    │           └─→ MLP → quality (B, 1)
    │
    ▼
Global Fusion (across all patches from same volume):
    boxes_global = transform_to_global(boxes_local, patch_positions)
    fused_box = soft_weighted_average(boxes_global, objectness × quality)
    │
    ▼
Losses:
    seg_loss = DiceCE (per-patch, same as original)
    det_loss = SmoothL1 + GIoU (fused_box vs GT_box)
    obj_loss = BCE (objectness vs has_overlap)
    qual_loss = MSE (quality vs centerness, for overlapping patches)
```

## Patch Fusion Step 1 — Preprocessing

**Uses the same preprocessing as Pipeline A (original SegMamba).**

```bash
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus
```

## Patch Fusion Step 2 — Training

```bash
python abus_patch_fusion_train.py --data_dir_train ./data/abus/train \
                                   --data_dir_val ./data/abus/val
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir_train` | `./data/abus/train` | Training data directory |
| `--data_dir_val` | `./data/abus/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus_patch_fusion` | Log and checkpoint directory |
| `--max_epoch` | `1000` | Total training epochs |
| `--patches_per_volume` | `4` | Number of patches sampled per volume |
| `--val_every` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.01` | Learning rate (SGD) |
| `--pretrained_seg` | `""` | Path to pretrained SegMamba checkpoint |

Monitor:
```bash
tensorboard --logdir ./logs/segmamba_abus_patch_fusion
```

## Patch Fusion Step 3 — Prediction

```bash
python abus_patch_fusion_predict.py \
    --model_path ./logs/segmamba_abus_patch_fusion/model/best_model_XXXX.pt \
    --save_seg
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | **(required)** | Path to trained checkpoint |
| `--data_dir_test` | `./data/abus/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus_patch_fusion` | Output directory |
| `--save_seg` | (flag) | Also save segmentation masks as NIfTI |
| `--overlap` | `0.5` | Sliding window overlap |

Outputs:
- `detections.json`: Detection boxes (one fused box per volume)
- `*.nii.gz`: Segmentation masks (if `--save_seg`)

## Patch Fusion Step 4 — Evaluation

```bash
# Segmentation metrics (Dice, HD95)
python abus_compute_metrics.py --pred_name segmamba_abus_patch_fusion

# Detection metrics (AP, IoU)
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_patch_fusion/detections.json \
    --abus_root /Volumes/Autzoko/ABUS --split Test
```

## Patch Fusion Quick-Start

```bash
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_patch_fusion_train.py
python abus_patch_fusion_predict.py --model_path ./logs/segmamba_abus_patch_fusion/model/best_model_XXXX.pt --save_seg
python abus_compute_metrics.py --pred_name segmamba_abus_patch_fusion
python abus_det_compute_metrics.py --pred_file ./prediction_results/segmamba_abus_patch_fusion/detections.json --abus_root /Volumes/Autzoko/ABUS
```

---

# Full-Resolution Detection

The standard detection pipelines (B, C, D) resize volumes to 128³. For full-resolution
detection, derive bounding boxes from segmentation predictions.

## Seg-to-Boxes: Extract Boxes from Segmentation (Recommended)

Uses the exact same pipeline as original SegMamba, with one additional step to
extract bounding boxes from the predicted masks via connected component analysis.

```bash
# Step 1: Preprocessing (same as original SegMamba)
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS

# Step 2: Training (same as original SegMamba)
python abus_train.py --max_epoch 1000

# Step 3: Prediction (same as original SegMamba)
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model_XXXX.pt

# Step 4: Extract bounding boxes from segmentation masks (NEW)
python abus_seg_to_boxes.py --pred_dir ./prediction_results/segmamba_abus

# Step 5: Evaluate detection metrics
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus/detections.json \
    --abus_root /Volumes/Autzoko/ABUS --split Test
```

| Argument | Default | Description |
|---|---|---|
| `--pred_dir` | `./prediction_results/segmamba_abus` | Directory with .nii.gz predictions |
| `--output_file` | `{pred_dir}/detections.json` | Output JSON path |
| `--min_volume` | `100` | Minimum component volume (filters noise) |

This approach leverages full-resolution segmentation to derive accurate bounding
boxes without any changes to the original SegMamba training pipeline.

---

---

# Pipeline F — Two-Stage Training (Recommended)

Two-stage training separates segmentation and detection training for **stable and
high-quality results**. This avoids the training instability seen with joint multi-task
optimization.

## Motivation

Joint training of segmentation and detection (Pipeline E) can be unstable due to:
- Conflicting gradient signals from different loss functions
- Detection loss interfering with segmentation convergence early in training
- Numerical instability from auxiliary losses under AMP

The two-stage approach:
1. **Stage 1**: Train SegMamba purely for segmentation until stable, high-quality masks
2. **Stage 2**: Freeze SegMamba, train only a lightweight BoxHead for detection

## Architecture (Stage 2)

```
Input Patch (B, 1, 128, 128, 128)
    │
    ▼
Frozen SegMamba Encoder-Decoder
    │
    ├─→ decoder1 (B, 48, 128, 128, 128) ─→ [frozen] UnetOutBlock → Seg logits
    │
    └─→ [trainable] PatchBoxHead:
          Conv3d(48→64, s=1) + IN + ReLU
          Conv3d(64→64, s=2) + IN + ReLU
          Conv3d(64→64, s=2) + IN + ReLU
          Global Avg Pool → (B, 64)
          ├─→ MLP → box (B, 6)
          ├─→ MLP → objectness (B, 1)
          └─→ MLP → quality (B, 1)
    │
    ▼
Global Fusion (same as Pipeline E):
    fused_box = soft_weighted_average(boxes_global, objectness × quality)
    │
    ▼
Detection Losses only:
    det_loss = SmoothL1 + GIoU
    obj_loss = BCE
    qual_loss = MSE
```

**Benefits:**
- Segmentation performance is preserved (frozen weights)
- BoxHead learns to read geometric structure from stable features
- More stable training, no multi-task loss balancing needed
- Faster Stage 2 training (~0.5M trainable params vs ~75M total)
- Provides clean foundation for using boxes as prompts in downstream tasks

## Stage 1 — Segmentation Training

Use the standard Pipeline A training:

```bash
# Preprocessing
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS --output_base ./data/abus

# Training (train until Dice > 0.7)
python abus_train.py --data_dir_train ./data/abus/train \
                     --data_dir_val ./data/abus/val \
                     --max_epoch 1000

# Verify segmentation quality
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model_XXXX.pt
python abus_compute_metrics.py --pred_name segmamba_abus
```

## Stage 2 — BoxHead Training

Train only the BoxHead with frozen SegMamba:

```bash
python abus_stage2_boxhead_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model_XXXX.pt \
    --data_dir_train ./data/abus/train \
    --data_dir_val ./data/abus/val
```

| Argument | Default | Description |
|---|---|---|
| `--pretrained_seg` | **(required)** | Path to Stage 1 SegMamba checkpoint |
| `--data_dir_train` | `./data/abus/train` | Training data directory |
| `--data_dir_val` | `./data/abus/val` | Validation data directory |
| `--logdir` | `./logs/segmamba_abus_stage2_boxhead` | Log directory |
| `--max_epoch` | `200` | Training epochs (fewer needed) |
| `--patches_per_volume` | `4` | Patches sampled per volume |
| `--val_every` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |
| `--lr` | `0.001` | Learning rate (lower than Stage 1) |

Monitor:
```bash
tensorboard --logdir ./logs/segmamba_abus_stage2_boxhead
```

## Stage 2 — Prediction

```bash
python abus_stage2_boxhead_predict.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model_XXXX.pt \
    --boxhead_path ./logs/segmamba_abus_stage2_boxhead/model/best_model_giouX.XXXX.pt \
    --save_seg
```

| Argument | Default | Description |
|---|---|---|
| `--pretrained_seg` | **(required)** | Path to Stage 1 SegMamba checkpoint |
| `--boxhead_path` | **(required)** | Path to Stage 2 BoxHead checkpoint |
| `--data_dir_test` | `./data/abus/test` | Test data directory |
| `--save_path` | `./prediction_results/segmamba_abus_stage2_boxhead` | Output directory |
| `--save_seg` | (flag) | Save segmentation masks as NIfTI |
| `--overlap` | `0.5` | Sliding window overlap |

Outputs:
- `detections.json`: Detection boxes
- `*.nii.gz`: Segmentation masks (if `--save_seg`)

## Stage 2 — Evaluation

```bash
# Segmentation (if --save_seg was used)
python abus_compute_metrics.py --pred_name segmamba_abus_stage2_boxhead

# Detection
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_stage2_boxhead/detections.json \
    --abus_root /Volumes/Autzoko/ABUS --split Test
```

## Two-Stage Quick-Start

```bash
# Stage 1: Segmentation
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
python abus_train.py --max_epoch 1000
python abus_compute_metrics.py  # Verify Dice > 0.7

# Stage 2: Detection
python abus_stage2_boxhead_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model_XXXX.pt

# Predict & Evaluate
python abus_stage2_boxhead_predict.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model_XXXX.pt \
    --boxhead_path ./logs/segmamba_abus_stage2_boxhead/model/best_model_giouX.XXXX.pt \
    --save_seg
python abus_compute_metrics.py --pred_name segmamba_abus_stage2_boxhead
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_stage2_boxhead/detections.json \
    --abus_root /Volumes/Autzoko/ABUS
```
