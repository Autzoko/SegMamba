"""
SegMamba Patch-Set Global Fusion Prediction for ABUS.

Performs sliding window inference and fuses patch-level box predictions
into a single global bounding box per volume.

Outputs:
  - Segmentation masks as NIfTI files (same as original SegMamba)
  - Detection boxes as JSON (same format as other detection pipelines)

Usage:
    python abus_patch_fusion_predict.py \
        --model_path ./logs/segmamba_abus_patch_fusion/model/best_model_XXXX.pt
"""

import os
import glob
import argparse
import json
import numpy as np
import pickle
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torch.cuda.amp import autocast

from model_segmamba.segmamba_patch_fusion import (
    SegMambaWithPatchFusion,
    transform_box_to_global,
    fuse_patch_boxes,
)

PATCH_SIZE = (128, 128, 128)


def sliding_window_positions(volume_shape, patch_size, overlap=0.5):
    """Generate sliding window positions with given overlap."""
    positions = []
    stride = [int(p * (1 - overlap)) for p in patch_size]

    for z in range(0, max(1, volume_shape[0] - patch_size[0] + 1), stride[0]):
        for y in range(0, max(1, volume_shape[1] - patch_size[1] + 1), stride[1]):
            for x in range(0, max(1, volume_shape[2] - patch_size[2] + 1), stride[2]):
                positions.append((z, y, x))

    # Ensure we cover the corners
    if len(positions) == 0:
        positions.append((0, 0, 0))

    # Add corner positions if not covered
    z_max = max(0, volume_shape[0] - patch_size[0])
    y_max = max(0, volume_shape[1] - patch_size[1])
    x_max = max(0, volume_shape[2] - patch_size[2])

    corner_positions = [
        (0, 0, 0), (0, 0, x_max), (0, y_max, 0), (0, y_max, x_max),
        (z_max, 0, 0), (z_max, 0, x_max), (z_max, y_max, 0), (z_max, y_max, x_max),
    ]
    for pos in corner_positions:
        if pos not in positions and all(p >= 0 for p in pos):
            positions.append(pos)

    return positions


def extract_patch(volume, position, patch_size):
    """Extract a patch, padding if necessary."""
    z, y, x = position
    patch = volume[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]

    # Pad if at volume boundary
    if patch.shape[1:] != tuple(patch_size):
        pad_d = patch_size[0] - patch.shape[1]
        pad_h = patch_size[1] - patch.shape[2]
        pad_w = patch_size[2] - patch.shape[3]
        patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    return patch


def aggregate_segmentation(seg_patches, positions, volume_shape, patch_size):
    """Aggregate patch segmentation predictions using Gaussian weighting."""
    # Create Gaussian weight map for blending
    sigma = [s / 4 for s in patch_size]
    zz, yy, xx = np.mgrid[:patch_size[0], :patch_size[1], :patch_size[2]]
    center = [s / 2 for s in patch_size]
    gaussian = np.exp(-((zz - center[0])**2 / (2*sigma[0]**2) +
                        (yy - center[1])**2 / (2*sigma[1]**2) +
                        (xx - center[2])**2 / (2*sigma[2]**2)))

    # Accumulate predictions with weights
    output = np.zeros((2, *volume_shape), dtype=np.float32)
    weight_sum = np.zeros(volume_shape, dtype=np.float32)

    for seg_patch, (z, y, x) in zip(seg_patches, positions):
        # Crop to valid region
        d_end = min(z + patch_size[0], volume_shape[0]) - z
        h_end = min(y + patch_size[1], volume_shape[1]) - y
        w_end = min(x + patch_size[2], volume_shape[2]) - x

        output[:, z:z+d_end, y:y+h_end, x:x+w_end] += (
            seg_patch[:, :d_end, :h_end, :w_end] * gaussian[:d_end, :h_end, :w_end])
        weight_sum[z:z+d_end, y:y+h_end, x:x+w_end] += gaussian[:d_end, :h_end, :w_end]

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-6)
    output = output / weight_sum

    return output


@torch.no_grad()
def predict_volume(model, volume, device, overlap=0.5, batch_size=4):
    """Run inference on a full volume.

    Returns:
        seg_output: (2, D, H, W) softmax probabilities
        fused_box: (6,) global box in CORNER format [z1, y1, x1, z2, y2, x2]
        confidence: float, detection confidence
    """
    model.eval()
    volume_shape = volume.shape[1:]  # (D, H, W)

    positions = sliding_window_positions(volume_shape, PATCH_SIZE, overlap)

    all_seg_patches = []
    all_boxes_local = []
    all_objectness = []
    all_quality = []
    all_positions = []

    # Process in batches
    for i in range(0, len(positions), batch_size):
        batch_positions = positions[i:i+batch_size]
        patches = []

        for pos in batch_positions:
            patch = extract_patch(volume, pos, PATCH_SIZE)
            patches.append(patch)

        patches = torch.from_numpy(np.stack(patches)).to(device)
        batch_pos = torch.tensor(batch_positions, dtype=torch.float32, device=device)
        vol_shape = torch.tensor(volume_shape, dtype=torch.float32, device=device)

        with autocast():
            seg_logits, boxes_local, objectness, quality = model(
                patches, patch_pos=batch_pos, volume_shape=vol_shape)

        # Softmax for segmentation
        seg_probs = torch.softmax(seg_logits, dim=1).cpu().numpy()

        all_seg_patches.extend(list(seg_probs))
        all_boxes_local.append(boxes_local.cpu())
        all_objectness.append(objectness.cpu())
        all_quality.append(quality.cpu())
        all_positions.extend(batch_positions)

    # Aggregate segmentation
    seg_output = aggregate_segmentation(all_seg_patches, all_positions, volume_shape, PATCH_SIZE)

    # Fuse detection predictions (all in CORNER format)
    boxes_local = torch.cat(all_boxes_local, dim=0)
    objectness = torch.cat(all_objectness, dim=0)
    quality = torch.cat(all_quality, dim=0)
    positions_tensor = torch.tensor(all_positions, dtype=torch.float32)

    boxes_global = transform_box_to_global(
        boxes_local, positions_tensor, PATCH_SIZE, volume_shape)

    fused_box, weights = fuse_patch_boxes(boxes_global, objectness, quality)

    # Confidence is the sum of weights from reliable patches
    confidence = float((objectness * quality).max())

    # fused_box is already in CORNER format [z1, y1, x1, z2, y2, x2]
    return seg_output, fused_box.numpy(), confidence


def main():
    parser = argparse.ArgumentParser(
        description="SegMamba Patch-Fusion prediction")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--data_dir_test", type=str, default="./data/abus/test",
                        help="Test data directory (preprocessed)")
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_patch_fusion",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap")
    parser.add_argument("--save_seg", action="store_true",
                        help="Save segmentation masks as NIfTI")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    # Load model
    model = SegMambaWithPatchFusion(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    sd = torch.load(args.model_path, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Find test files
    test_files = sorted(glob.glob(os.path.join(args.data_dir_test, "*.npz")))
    if len(test_files) == 0:
        print(f"No .npz files found in {args.data_dir_test}")
        return

    print(f"Processing {len(test_files)} test cases...")

    all_detections = {}

    for npz_path in tqdm(test_files, desc="Predicting"):
        # Load data
        data = np.load(npz_path)
        volume = data['data'].astype(np.float32)  # (1, D, H, W)

        # Load properties
        pkl_path = npz_path.replace('.npz', '.pkl')
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)

        case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
        case_id = case_name.replace('ABUS_', '')

        # Predict
        seg_output, fused_box, confidence = predict_volume(
            model, volume, device, overlap=args.overlap)

        # fused_box is already in CORNER format [z1, y1, x1, z2, y2, x2]
        box_corners = list(fused_box)

        # Clip to volume bounds
        volume_shape = volume.shape[1:]
        box_corners = [
            max(0, box_corners[0]), max(0, box_corners[1]), max(0, box_corners[2]),
            min(volume_shape[0], box_corners[3]),
            min(volume_shape[1], box_corners[4]),
            min(volume_shape[2], box_corners[5]),
        ]

        # Scale box back to original (non-cropped) coordinates if needed
        if 'crop_bbox' in props:
            crop_bbox = props['crop_bbox']
            box_corners = [
                box_corners[0] + crop_bbox[0][0],
                box_corners[1] + crop_bbox[1][0],
                box_corners[2] + crop_bbox[2][0],
                box_corners[3] + crop_bbox[0][0],
                box_corners[4] + crop_bbox[1][0],
                box_corners[5] + crop_bbox[2][0],
            ]

        # Store detection
        all_detections[case_id] = {
            "boxes": [box_corners],
            "scores": [float(confidence)],
        }

        # Save segmentation if requested
        if args.save_seg:
            seg_pred = (seg_output[1] > 0.5).astype(np.uint8)

            # Restore to original shape if cropped
            if 'shape_before_cropping' in props and 'crop_bbox' in props:
                full_shape = props['shape_before_cropping']
                crop_bbox = props['crop_bbox']
                full_seg = np.zeros(full_shape, dtype=np.uint8)
                full_seg[crop_bbox[0][0]:crop_bbox[0][1],
                         crop_bbox[1][0]:crop_bbox[1][1],
                         crop_bbox[2][0]:crop_bbox[2][1]] = seg_pred
                seg_pred = full_seg

            spacing = props.get('spacing', [1.0, 1.0, 1.0])
            if isinstance(spacing[0], torch.Tensor):
                spacing = [s.item() for s in spacing]

            seg_itk = sitk.GetImageFromArray(seg_pred)
            seg_itk.SetSpacing(list(spacing)[::-1])  # xyz order for SimpleITK
            sitk.WriteImage(seg_itk, os.path.join(args.save_path, f"{case_name}.nii.gz"))

    # Save detections
    det_path = os.path.join(args.save_path, "detections.json")
    with open(det_path, 'w') as f:
        json.dump(all_detections, f, indent=2)

    print(f"\nSaved {len(all_detections)} detections to {det_path}")
    if args.save_seg:
        print(f"Saved segmentation masks to {args.save_path}/")

    print(f"\nEvaluate segmentation with:")
    print(f"  python abus_compute_metrics.py --pred_name segmamba_abus_patch_fusion")
    print(f"\nEvaluate detection with:")
    print(f"  python abus_det_compute_metrics.py --pred_file {det_path}")


if __name__ == "__main__":
    main()
