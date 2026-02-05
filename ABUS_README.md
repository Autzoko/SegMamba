# SegMamba for ABUS 3D Ultrasound

Four pipelines for the ABUS (Automated Breast Ultrasound) dataset:

- **Pipeline A — Segmentation** (`abus_*.py`) — per-voxel tumour masks via SegMamba encoder-decoder
- **Pipeline B — Detection FCOS** (`abus_det_*.py`) — 3D bounding boxes via MambaEncoder + FPN + FCOS head
- **Pipeline C — Detection DETR** (`abus_detr_*.py`) — 3D bounding boxes via MambaEncoder + DETR transformer decoder
- **Pipeline D — Multi-task BoxHead** (`abus_boxhead_*.py`) — segmentation + attention-based box regression from decoder features

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
