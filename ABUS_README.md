# SegMamba for ABUS 3D Ultrasound Tumour Segmentation

Train and evaluate SegMamba on the ABUS (Automated Breast Ultrasound) dataset
for binary tumour segmentation. Four scripts handle the full pipeline:
preprocessing, training, prediction, and evaluation.

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
pip install acvl-utils medpy SimpleITK tqdm scikit-image batchgenerators
```

> Do **not** `pip install monai` — it is bundled in the repo.

### 3. Verify

```bash
python 0_inference.py
```

---

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

## Quick-Start (all commands)

```bash
# Environment
cd SegMamba
cd causal-conv1d && python setup.py install && cd ..
cd mamba         && python setup.py install && cd ..
pip install acvl-utils medpy SimpleITK tqdm scikit-image batchgenerators

# Preprocessing
python abus_preprocessing.py --abus_root /Volumes/Autzoko/ABUS

# Training
python abus_train.py

# Prediction
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model_XXXX.pt

# Evaluation
python abus_compute_metrics.py
```
