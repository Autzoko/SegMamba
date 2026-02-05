# SegMamba for ABUS 3D Ultrasound

Two pipelines for the ABUS (Automated Breast Ultrasound) dataset:

- **Segmentation** (`abus_*.py`) — per-voxel tumour masks via SegMamba encoder-decoder
- **Detection** (`abus_det_*.py`) — direct 3D bounding box prediction via SegMamba-Det (MambaEncoder + FPN + FCOS head)

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
