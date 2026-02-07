# SegMamba

**Recent news: If you are interested in the research about vision language models, please refers to the latest work: https://github.com/MrGiovanni/RadGPT (ICCV2025)**

**Now we have open-sourced the pre-processing, training, inference, and metrics computation codes.**

SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation

[https://arxiv.org/abs/2401.13560](https://arxiv.org/abs/2401.13560)

![](images/method_figure.jpg)

![](images/modules.jpg)

Our advantage in speed and memory.
![](images/segmamba_ablation.jpg)

## Contact 
If you have any questions about our project, please feel free to contact us by email at zxing565@connect.hkust-gz.edu.cn or via WeChat at 18340097191. Furthermore, the data underlying this article will be shared on reasonable request to gaof57@mail.sysu.edu.cn.

## Environment install
Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/ge-xing/SegMamba.git

cd SegMamba
```
### Install causal-conv1d

```bash
cd causal-conv1d

python setup.py install
```

### Install mamba

```bash
cd mamba

python setup.py install
```

### Install monai 

```bash
pip install monai
```

## Simple test

```bash
python 0_inference.py
```

## Preprocessing, training, testing, inference, and metrics computation

### Data downloading 

Data is from [https://arxiv.org/abs/2305.17033](https://arxiv.org/abs/2305.17033)

Download from Baidu Disk  [https://pan.baidu.com/s/1C0FUHdDtWNaYWLtDDP9TnA?pwd=ty22提取码ty22](https://pan.baidu.com/s/1C0FUHdDtWNaYWLtDDP9TnA?pwd=ty22) 

Download from OneDrive [https://hkustgz-my.sharepoint.com/:f:/g/personal/zxing565_connect_hkust-gz_edu_cn/EqqaINbHRxREuIj0XGicY2EBv8hjwEFKgFOhF_Ub0mvENw?e=yTpE9B](https://hkustgz-my.sharepoint.com/:f:/g/personal/zxing565_connect_hkust-gz_edu_cn/EqqaINbHRxREuIj0XGicY2EBv8hjwEFKgFOhF_Ub0mvENw?e=yTpE9B)

### Preprocessing
In my setting, the data directory of BraTS2023 is : "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"

First, we need to run the rename process.

```bash 
python 1_rename_mri_data.py
```

Then, we need to run the pre-processing code to do resample, normalization, and crop processes.

```bash
python 2_preprocessing_mri.py
```

After pre-processing, the data structure will be in this format:

![](images/data_structure.jpg)
### Training 

When the pre-processing process is done, we can train our model.

We mainly use the pre-processde data from last step: **data_dir = "./data/fullres/train"**


```bash 
python 3_train.py
```

The training logs and checkpoints are saved in:
**logdir = f"./logs/segmamba"**




### Inference 

When we have trained our models, we can inference all the data in testing set.

```bash 
python 4_predict.py
```

When this process is done, the prediction cases will be put in this path:
**save_path = "./prediction_results/segmamba"**

### Metrics computation
We can obtain the Dice score and HD95 on each segmentation target (WT, TC, ET for BraTS2023 dataset) using this code:

```bash
python 5_compute_metrics.py --pred_name="segmamba"
```



---

## ABUS Tumor Detection & Segmentation (SegMamba-Retina)

This section describes the complete workflow for **Automated Breast Ultrasound (ABUS)** tumor detection and segmentation using the **SegMamba-Retina** pipeline.

### Quick Start (Complete Pipeline)

```bash
# Step 1: Preprocess ABUS data (80/10/10 split, only cases with lesions)
python abus_preprocessing.py --abus_root /path/to/ABUS --seed 42

# Step 2: Train segmentation model
python abus_train.py --max_epoch 1000

# Step 3: Train detection (with frozen backbone to preserve segmentation)
python abus_retina_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
    --freeze_backbone \
    --full_volume_val \
    --max_epoch 500

# Step 4: Run inference
python abus_retina_predict.py \
    --model_path ./logs/segmamba_abus_retina/model/best_model.pt \
    --save_seg

# Step 5: Evaluate
python abus_compute_metrics.py --pred_name segmamba_abus_retina  # Segmentation
python abus_det_compute_metrics.py --pred_file ./prediction_results/segmamba_abus_retina/detections.json  # Detection
```

---

### ABUS Dataset Setup

The ABUS dataset should be organized as:
```
/path/to/ABUS/data/
├── Train/
│   ├── DATA/DATA_XXX.nrrd
│   └── MASK/MASK_XXX.nrrd
├── Validation/
│   ├── DATA/...
│   └── MASK/...
└── Test/
    ├── DATA/...
    └── MASK/...
```

---

### Step 1: Preprocessing

The preprocessing script:
- Collects ALL cases from Train/Validation/Test folders
- **Filters out cases without lesions** (empty masks)
- Splits data **80% train / 10% val / 10% test** with random seed
- Applies Z-score normalization and crops to non-zero region

```bash
python abus_preprocessing.py \
    --abus_root /path/to/ABUS \
    --output_base ./data/abus \
    --seed 42 \
    --num_processes 4
```

**Output:**
```
./data/abus/
├── train/          # 80% of cases with lesions
├── val/            # 10% of cases with lesions
├── test/           # 10% of cases with lesions
├── split_info.pkl  # Split details for reproducibility
└── split_info.txt  # Human-readable split info
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--abus_root` | `/Volumes/Autzoko/ABUS` | Path to raw ABUS dataset |
| `--output_base` | `./data/abus` | Output directory |
| `--seed` | `42` | Random seed for split |
| `--train_ratio` | `0.8` | Training set ratio |
| `--val_ratio` | `0.1` | Validation set ratio |
| `--num_processes` | `4` | Parallel workers |

---

### Step 2: Train Segmentation Model

Train the base SegMamba model for tumor segmentation:

```bash
python abus_train.py \
    --data_dir_train ./data/abus/train \
    --data_dir_val ./data/abus/val \
    --max_epoch 1000 \
    --batch_size 2
```

**Output:** `./logs/segmamba_abus/model/best_model.pt`

---

### Step 3: Train SegMamba-Retina (Detection)

Train the detection head with pretrained segmentation weights:

```bash
# Option A: Freeze backbone (recommended - preserves segmentation quality)
python abus_retina_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
    --freeze_backbone \
    --full_volume_val \
    --max_epoch 500 \
    --fg_ratio 0.5

# Option B: Joint training (fine-tunes both seg and det)
python abus_retina_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
    --full_volume_val \
    --max_epoch 500 \
    --fg_ratio 0.5
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_seg` | `""` | Path to pretrained SegMamba checkpoint |
| `--freeze_backbone` | `False` | Freeze backbone, only train detection head |
| `--full_volume_val` | `False` | Use full-volume validation (slower but accurate) |
| `--fg_ratio` | `0.5` | Fraction of patches containing tumors |
| `--det_warmup_epochs` | `50` | Epochs to ramp up detection loss |
| `--max_epoch` | `500` | Total training epochs |

**Output:** `./logs/segmamba_abus_retina/model/best_model.pt`

---

### Step 4: Inference

Run sliding window inference with NMS:

```bash
python abus_retina_predict.py \
    --model_path ./logs/segmamba_abus_retina/model/best_model.pt \
    --data_dir_test ./data/abus/test \
    --save_seg \
    --overlap 0.5
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained model |
| `--data_dir_test` | `./data/abus/test` | Test data directory |
| `--save_seg` | `False` | Save segmentation masks as NIfTI |
| `--overlap` | `0.5` | Sliding window overlap |
| `--score_threshold` | `0.05` | Detection score threshold |
| `--nms_threshold` | `0.5` | NMS IoU threshold |

**Output:**
```
./prediction_results/segmamba_abus_retina/
├── detections.json     # Detection boxes + scores
└── ABUS_XXX.nii.gz     # Segmentation masks (if --save_seg)
```

---

### Step 5: Evaluation

**Segmentation Metrics (Dice, HD95):**
```bash
python abus_compute_metrics.py --pred_name segmamba_abus_retina
```

**Detection Metrics (AP, Recall, Precision):**
```bash
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_retina/detections.json \
    --abus_root /path/to/ABUS
```

---

### SegMamba-Retina Architecture

```
Input (B, 1, 128, 128, 128)
    │
    ▼
┌───────────────────────────────────┐
│ MambaEncoder (shared backbone)    │
│   outs[0]: (B, 48, 64, 64, 64)    │
│   outs[1]: (B, 96, 32, 32, 32)    │  ← P4 (stride 4)
│   outs[2]: (B, 192, 16, 16, 16)   │  ← P8 (stride 8)
│   outs[3]: (B, 384, 8, 8, 8)      │  ← P16 (stride 16)
└───────────────────────────────────┘
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Segmentation Decoder│            │ Detection Pathway   │
│ (UNETR-style)       │            │                     │
│                     │            │ FPN3D (128 channels)│
│ → Dice + CE loss    │            │ Retina3DHead        │
└─────────────────────┘            │                     │
                                   │ → Focal + L1 + GIoU │
                                   └─────────────────────┘
```

**Key Features:**
- Multi-scale anchor generation (strides 4/8/16/32)
- ATSS (Adaptive Training Sample Selection) matching
- Hard negative mining for class imbalance
- Biased patch sampling (50% foreground)
- Detection loss warmup (50 epochs)
- `--freeze_backbone` to train detection only

---

### Detection Module

The `detection/` directory contains reusable 3D detection components:

| File | Description |
|------|-------------|
| `anchors.py` | Multi-scale 3D anchor generation |
| `fpn.py` | 3D Feature Pyramid Network |
| `retina_head.py` | RetinaNet-style cls/reg heads |
| `atss_matcher.py` | Adaptive Training Sample Selection |
| `sampler.py` | Hard negative mining |
| `losses.py` | Focal loss, GIoU loss, 3D NMS |
| `box_coder.py` | Box encoding/decoding |

---

### Other Detection Pipelines (Legacy)

Additional detection approaches are available but SegMamba-Retina is recommended:

| Pipeline | Script | Description |
|----------|--------|-------------|
| A: Seg→Box | `abus_seg_box_eval.py` | Extract boxes from segmentation masks |
| B: FCOS | `abus_det_train.py` | Anchor-free per-voxel detection |
| C: DETR | `abus_detr_train.py` | Transformer decoder with set prediction |
| D: BoxHead | `abus_boxhead_train.py` | Attention-based single box prediction |
| E: PatchFusion | `abus_patch_fusion_train.py` | Patch-based with global fusion |

---

## Acknowledgement
Many thanks for these repos for their great contribution!

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)

