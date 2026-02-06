"""
Stage 2 BoxHead Prediction for ABUS Detection.

Performs sliding window inference with the frozen SegMamba + trained BoxHead,
fusing patch-level predictions into global bounding boxes.

Outputs:
  - Detection boxes as JSON (same format as other detection pipelines)
  - Optionally: segmentation masks as NIfTI

Usage:
    python abus_stage2_boxhead_predict.py \
        --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
        --boxhead_path ./logs/segmamba_abus_stage2_boxhead/model/best_model_giou0.5.pt
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

    if patch.shape[1:] != tuple(patch_size):
        pad_d = patch_size[0] - patch.shape[1]
        pad_h = patch_size[1] - patch.shape[2]
        pad_w = patch_size[2] - patch.shape[3]
        patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    return patch


def aggregate_segmentation(seg_patches, positions, volume_shape, patch_size):
    """Aggregate patch segmentation predictions using Gaussian weighting."""
    sigma = [s / 4 for s in patch_size]
    zz, yy, xx = np.mgrid[:patch_size[0], :patch_size[1], :patch_size[2]]
    center = [s / 2 for s in patch_size]
    gaussian = np.exp(-((zz - center[0])**2 / (2*sigma[0]**2) +
                        (yy - center[1])**2 / (2*sigma[1]**2) +
                        (xx - center[2])**2 / (2*sigma[2]**2)))

    output = np.zeros((2, *volume_shape), dtype=np.float32)
    weight_sum = np.zeros(volume_shape, dtype=np.float32)

    for seg_patch, (z, y, x) in zip(seg_patches, positions):
        d_end = min(z + patch_size[0], volume_shape[0]) - z
        h_end = min(y + patch_size[1], volume_shape[1]) - y
        w_end = min(x + patch_size[2], volume_shape[2]) - x

        output[:, z:z+d_end, y:y+h_end, x:x+w_end] += (
            seg_patch[:, :d_end, :h_end, :w_end] * gaussian[:d_end, :h_end, :w_end])
        weight_sum[z:z+d_end, y:y+h_end, x:x+w_end] += gaussian[:d_end, :h_end, :w_end]

    weight_sum = np.maximum(weight_sum, 1e-6)
    output = output / weight_sum

    return output


@torch.no_grad()
def predict_volume(model, volume, device, overlap=0.5, batch_size=4):
    """Run inference on a full volume.

    Returns:
        seg_output: (2, D, H, W) softmax probabilities
        fused_box: (6,) global box in voxel coordinates [cz, cy, cx, dz, dy, dx]
        confidence: float, detection confidence
    """
    model.eval()
    volume_shape = volume.shape[1:]

    positions = sliding_window_positions(volume_shape, PATCH_SIZE, overlap)

    all_seg_patches = []
    all_boxes_local = []
    all_objectness = []
    all_quality = []
    all_positions = []

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

        seg_probs = torch.softmax(seg_logits, dim=1).cpu().numpy()

        all_seg_patches.extend(list(seg_probs))
        all_boxes_local.append(boxes_local.cpu())
        all_objectness.append(objectness.cpu())
        all_quality.append(quality.cpu())
        all_positions.extend(batch_positions)

    # Aggregate segmentation
    seg_output = aggregate_segmentation(all_seg_patches, all_positions, volume_shape, PATCH_SIZE)

    # Fuse detection predictions
    boxes_local = torch.cat(all_boxes_local, dim=0)
    objectness = torch.cat(all_objectness, dim=0)
    quality = torch.cat(all_quality, dim=0)
    positions_tensor = torch.tensor(all_positions, dtype=torch.float32)

    boxes_global = transform_box_to_global(
        boxes_local, positions_tensor, PATCH_SIZE, volume_shape)

    fused_box, weights = fuse_patch_boxes(boxes_global, objectness, quality)

    confidence = float((objectness * quality).max())

    return seg_output, fused_box.numpy(), confidence


def box_center_to_corners(box):
    """Convert [cz, cy, cx, dz, dy, dx] to [z1, y1, x1, z2, y2, x2]."""
    cz, cy, cx, dz, dy, dx = box
    return [
        cz - dz/2, cy - dy/2, cx - dx/2,
        cz + dz/2, cy + dy/2, cx + dx/2
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 BoxHead prediction for ABUS")
    parser.add_argument("--pretrained_seg", type=str, required=True,
                        help="Path to pretrained SegMamba checkpoint")
    parser.add_argument("--boxhead_path", type=str, required=True,
                        help="Path to trained BoxHead checkpoint")
    parser.add_argument("--data_dir_test", type=str, default="./data/abus/test")
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_stage2_boxhead")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlap", type=float, default=0.5)
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

    # Load pretrained SegMamba
    print(f"Loading SegMamba from: {args.pretrained_seg}")
    sd = torch.load(args.pretrained_seg, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)

    # Load BoxHead
    print(f"Loading BoxHead from: {args.boxhead_path}")
    boxhead_sd = torch.load(args.boxhead_path, map_location='cpu')
    if 'patch_box_head' in boxhead_sd:
        model.patch_box_head.load_state_dict(boxhead_sd['patch_box_head'])
    elif 'full_model' in boxhead_sd:
        # Load full model state dict
        model.load_state_dict(boxhead_sd['full_model'])
    else:
        # Assume it's the full model state dict
        model.load_state_dict(boxhead_sd, strict=False)

    model.eval()
    print("Model loaded successfully")

    # Find test files
    test_files = sorted(glob.glob(os.path.join(args.data_dir_test, "*.npz")))
    if len(test_files) == 0:
        print(f"No .npz files found in {args.data_dir_test}")
        return

    print(f"Processing {len(test_files)} test cases...")

    all_detections = {}

    for npz_path in tqdm(test_files, desc="Predicting"):
        data = np.load(npz_path)
        volume = data['data'].astype(np.float32)

        pkl_path = npz_path.replace('.npz', '.pkl')
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)

        case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
        case_id = case_name.replace('ABUS_', '')

        # Predict
        seg_output, fused_box, confidence = predict_volume(
            model, volume, device, overlap=args.overlap)

        # Convert box to corner format
        box_corners = box_center_to_corners(fused_box)

        # Clip to volume bounds
        volume_shape = volume.shape[1:]
        box_corners = [
            max(0, box_corners[0]), max(0, box_corners[1]), max(0, box_corners[2]),
            min(volume_shape[0], box_corners[3]),
            min(volume_shape[1], box_corners[4]),
            min(volume_shape[2], box_corners[5]),
        ]

        # Scale box back to original coordinates if needed
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

        all_detections[case_id] = {
            "boxes": [box_corners],
            "scores": [float(confidence)],
        }

        # Save segmentation if requested
        if args.save_seg:
            seg_pred = (seg_output[1] > 0.5).astype(np.uint8)

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
            seg_itk.SetSpacing(list(spacing)[::-1])
            sitk.WriteImage(seg_itk, os.path.join(args.save_path, f"{case_name}.nii.gz"))

    # Save detections
    det_path = os.path.join(args.save_path, "detections.json")
    with open(det_path, 'w') as f:
        json.dump(all_detections, f, indent=2)

    print(f"\nSaved {len(all_detections)} detections to {det_path}")
    if args.save_seg:
        print(f"Saved segmentation masks to {args.save_path}/")

    print(f"\nEvaluate with:")
    print(f"  python abus_det_compute_metrics.py --pred_file {det_path}")


if __name__ == "__main__":
    main()
