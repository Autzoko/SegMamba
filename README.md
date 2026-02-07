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

## ABUS Tumor Detection & Segmentation

This repository has been extended to support **Automated Breast Ultrasound (ABUS)** tumor detection and segmentation with multiple detection pipelines.

### ABUS Dataset Setup

The ABUS dataset should be organized as:
```
/path/to/ABUS/
├── Train/          # 100 cases
├── Validation/     # 30 cases
├── Test/           # 70 cases
└── abus_labels.csv # Contains case_id, diagnosis (M/B)
```

### Preprocessing

```bash
# Preprocess ABUS data (NRRD → NPZ with normalization and cropping)
python abus_preprocessing.py --abus_root /path/to/ABUS --output_dir ./data/abus
```

### Segmentation Training & Inference

```bash
# Train segmentation model
python abus_train.py --max_epoch 1000

# Run inference
python abus_predict.py --model_path ./logs/segmamba_abus/model/best_model.pt

# Compute metrics (Dice, HD95)
python abus_compute_metrics.py --pred_name segmamba_abus
```

### Detection Pipelines

Multiple detection approaches are available:

#### Pipeline A: Segmentation → Box Extraction
Extract bounding boxes from predicted segmentation masks:
```bash
python abus_seg_to_boxes.py --pred_dir ./prediction_results/segmamba_abus
python abus_seg_box_eval.py  # Evaluate seg-derived boxes vs GT
```

#### Pipeline B: FCOS-style Detection (SegMamba-Det)
Per-voxel anchor-free detection with FPN:
```bash
python abus_det_train.py --max_epoch 500
python abus_det_predict.py --model_path ./logs/segmamba_det/model/best_model.pt
python abus_det_compute_metrics.py --pred_file ./prediction_results/.../detections.json
```

#### Pipeline C: DETR-style Detection (SegMamba-DETR)
Transformer decoder with set prediction:
```bash
python abus_detr_train.py --max_epoch 500
python abus_detr_predict.py --model_path ./logs/segmamba_detr/model/best_model.pt
```

#### Pipeline D: BoxHead Detection
Attention-based single box prediction per volume:
```bash
python abus_boxhead_train.py --pretrained_seg ./logs/segmamba_abus/model/best_model.pt
python abus_boxhead_predict.py --model_path ./logs/segmamba_boxhead/model/best_model.pt
```

#### Pipeline E: Patch Fusion Detection
Full-resolution patch-based training with global box fusion:
```bash
python abus_patch_fusion_train.py --pretrained_seg ./logs/segmamba_abus/model/best_model.pt
python abus_patch_fusion_predict.py --model_path ./logs/segmamba_patch_fusion/model/best_model.pt
```

#### Pipeline F: SegMamba-Retina (nnDetection-style) ⭐ **NEW**
Anchor-based 3D RetinaNet with FPN, ATSS matching, and focal loss:
```bash
# Train with pretrained segmentation weights
python abus_retina_train.py \
    --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
    --max_epoch 500 \
    --fg_ratio 0.5

# Inference with sliding window + NMS
python abus_retina_predict.py \
    --model_path ./logs/segmamba_abus_retina/model/best_model.pt \
    --save_seg

# Evaluate detection
python abus_det_compute_metrics.py \
    --pred_file ./prediction_results/segmamba_abus_retina/detections.json
```

**SegMamba-Retina Architecture:**
```
Input (B, 1, 128, 128, 128)
    │
    ▼
MambaEncoder (shared backbone)
    │
    ├── Segmentation Decoder → Dice+CE loss
    │
    └── Detection Pathway
        ├── FPN3D (strides 4/8/16/32)
        └── Retina3DHead → Focal + L1 + GIoU loss
```

Key features:
- Multi-scale anchor generation with ATSS matching
- Hard negative mining for class imbalance
- Biased patch sampling (50% foreground)
- Detection loss warmup
- Sliding window inference with global NMS

### Detection Module

The `detection/` directory contains reusable 3D detection components:
- `anchors.py` - Multi-scale 3D anchor generation
- `fpn.py` - 3D Feature Pyramid Network
- `retina_head.py` - RetinaNet-style detection head
- `atss_matcher.py` - Adaptive Training Sample Selection
- `sampler.py` - Hard negative mining
- `losses.py` - Focal loss, GIoU loss, 3D NMS
- `box_coder.py` - Box encoding/decoding

---

## Acknowledgement
Many thanks for these repos for their great contribution!

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)

