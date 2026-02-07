"""
Inference script for SegMamba-Retina detection.

Performs sliding window inference with global NMS to produce detection boxes
and segmentation masks.

Outputs:
    - Detection boxes as JSON (compatible with abus_det_compute_metrics.py)
    - Optionally: segmentation masks as NIfTI

Usage:
    python abus_retina_predict.py \
        --model_path ./logs/segmamba_abus_retina/model/best_model.pt \
        --save_seg
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

from model_segmamba.segmamba_retina import (
    SegMambaWithRetina,
    DetectionPostProcessor,
)
from detection.retina_head import reshape_head_outputs
from detection.box_coder import BoxCoder3D
from detection.losses import nms_3d


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


def transform_boxes_to_global(boxes, position, patch_size):
    """
    Transform local patch boxes to global volume coordinates.

    Args:
        boxes: (N, 6) boxes in [x1, y1, x2, y2, z1, z2] format
        position: (z, y, x) patch start position
        patch_size: (D, H, W) patch size

    Returns:
        global_boxes: (N, 6) boxes in global coordinates
    """
    z, y, x = position

    # Add offset to convert to global coordinates
    # Box format: [x1, y1, x2, y2, z1, z2]
    offset = torch.tensor([x, y, x, y, z, z], device=boxes.device, dtype=boxes.dtype)

    global_boxes = boxes + offset

    return global_boxes


@torch.no_grad()
def predict_volume(
    model: SegMambaWithRetina,
    volume: np.ndarray,
    device: torch.device,
    box_coder: BoxCoder3D,
    overlap: float = 0.5,
    batch_size: int = 2,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
):
    """
    Run inference on a full volume.

    Args:
        model: SegMambaWithRetina model
        volume: (1, D, H, W) input volume
        device: Torch device
        box_coder: Box coder for decoding
        overlap: Sliding window overlap
        batch_size: Batch size for inference
        score_threshold: Minimum score for detections
        nms_threshold: IoU threshold for NMS

    Returns:
        seg_output: (2, D, H, W) softmax probabilities
        final_boxes: (K, 6) detected boxes in [x1, y1, x2, y2, z1, z2] format
        final_scores: (K,) detection scores
    """
    model.eval()
    volume_shape = volume.shape[1:]  # (D, H, W)

    positions = sliding_window_positions(volume_shape, PATCH_SIZE, overlap)

    all_seg_patches = []
    all_boxes = []
    all_scores = []
    all_positions = []

    # Process in batches
    for i in range(0, len(positions), batch_size):
        batch_positions = positions[i:i+batch_size]
        patches = []

        for pos in batch_positions:
            patch = extract_patch(volume, pos, PATCH_SIZE)
            patches.append(patch)

        patches = torch.from_numpy(np.stack(patches)).to(device)

        with autocast():
            outputs = model(patches, return_seg=True, return_det=True)

        # Get segmentation
        seg_probs = torch.softmax(outputs['seg_logits'], dim=1).cpu().numpy()
        all_seg_patches.extend(list(seg_probs))
        all_positions.extend(batch_positions)

        # Get detection outputs
        cls_flat, box_flat, ctr_flat, num_per_level = reshape_head_outputs(
            outputs['cls_logits'],
            outputs['box_deltas'],
            outputs['centerness'],
            model.num_anchors,
            model.num_classes,
        )
        anchors = outputs['anchors']

        # Process each patch
        for b, pos in enumerate(batch_positions):
            cls_scores = torch.sigmoid(cls_flat[b].view(-1))
            ctr_scores = torch.sigmoid(ctr_flat[b].view(-1))

            # Combine classification and centerness
            scores = cls_scores * ctr_scores

            # Filter by score
            keep = scores > score_threshold
            if keep.sum() == 0:
                continue

            scores_keep = scores[keep]
            deltas_keep = box_flat[b][keep]
            anchors_keep = anchors[keep]

            # Decode boxes
            boxes_local = box_coder.decode(deltas_keep, anchors_keep)

            # Transform to global coordinates
            boxes_global = transform_boxes_to_global(boxes_local, pos, PATCH_SIZE)

            all_boxes.append(boxes_global.cpu())
            all_scores.append(scores_keep.cpu())

    # Aggregate segmentation
    seg_output = aggregate_segmentation(all_seg_patches, all_positions, volume_shape, PATCH_SIZE)

    # Global NMS on all detections
    if len(all_boxes) > 0:
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)

        # Clip boxes to volume bounds
        all_boxes[:, 0] = all_boxes[:, 0].clamp(0, volume_shape[2])  # x1
        all_boxes[:, 1] = all_boxes[:, 1].clamp(0, volume_shape[1])  # y1
        all_boxes[:, 2] = all_boxes[:, 2].clamp(0, volume_shape[2])  # x2
        all_boxes[:, 3] = all_boxes[:, 3].clamp(0, volume_shape[1])  # y2
        all_boxes[:, 4] = all_boxes[:, 4].clamp(0, volume_shape[0])  # z1
        all_boxes[:, 5] = all_boxes[:, 5].clamp(0, volume_shape[0])  # z2

        # NMS
        keep = nms_3d(all_boxes, all_scores, nms_threshold)

        final_boxes = all_boxes[keep].numpy()
        final_scores = all_scores[keep].numpy()
    else:
        final_boxes = np.zeros((0, 6))
        final_scores = np.zeros(0)

    return seg_output, final_boxes, final_scores


def convert_to_corner_format(boxes):
    """
    Convert boxes from [x1, y1, x2, y2, z1, z2] to [z1, y1, x1, z2, y2, x2].

    This matches the format expected by abus_det_compute_metrics.py.
    """
    if len(boxes) == 0:
        return []

    result = []
    for box in boxes:
        x1, y1, x2, y2, z1, z2 = box
        result.append([z1, y1, x1, z2, y2, x2])
    return result


def main():
    parser = argparse.ArgumentParser(description="SegMamba-Retina inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir_test", type=str, default="./data/abus/test")
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_retina")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--save_seg", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    # Load model
    model = SegMambaWithRetina(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'module' in state_dict:
        state_dict = state_dict['module']
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    box_coder = BoxCoder3D()

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
        volume = data['data'].astype(np.float32)

        # Load properties
        pkl_path = npz_path.replace('.npz', '.pkl')
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)

        case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
        case_id = case_name.replace('ABUS_', '')

        # Predict
        seg_output, boxes, scores = predict_volume(
            model, volume, device, box_coder,
            overlap=args.overlap,
            batch_size=args.batch_size,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
        )

        # Scale boxes back to original coordinates if needed
        if 'crop_bbox' in props:
            crop_bbox = props['crop_bbox']
            # Add crop offset to boxes
            # Current format: [x1, y1, x2, y2, z1, z2]
            if len(boxes) > 0:
                boxes[:, 0] += crop_bbox[2][0]  # x1
                boxes[:, 1] += crop_bbox[1][0]  # y1
                boxes[:, 2] += crop_bbox[2][0]  # x2
                boxes[:, 3] += crop_bbox[1][0]  # y2
                boxes[:, 4] += crop_bbox[0][0]  # z1
                boxes[:, 5] += crop_bbox[0][0]  # z2

        # Convert to corner format [z1, y1, x1, z2, y2, x2]
        boxes_corner = convert_to_corner_format(boxes)

        all_detections[case_id] = {
            "boxes": boxes_corner,
            "scores": [float(s) for s in scores],
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
            seg_itk.SetSpacing(list(spacing)[::-1])
            sitk.WriteImage(seg_itk, os.path.join(args.save_path, f"{case_name}.nii.gz"))

    # Save detections
    det_path = os.path.join(args.save_path, "detections.json")
    with open(det_path, 'w') as f:
        json.dump(all_detections, f, indent=2)

    print(f"\nSaved {len(all_detections)} detections to {det_path}")
    if args.save_seg:
        print(f"Saved segmentation masks to {args.save_path}/")

    print(f"\nEvaluate segmentation with:")
    print(f"  python abus_compute_metrics.py --pred_name segmamba_abus_retina")
    print(f"\nEvaluate detection with:")
    print(f"  python abus_det_compute_metrics.py --pred_file {det_path}")


if __name__ == "__main__":
    main()
