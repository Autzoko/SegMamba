"""
Evaluation metrics for SegMamba-Det ABUS bounding box predictions.

Compares predicted JSON detections against ground-truth boxes derived
from NRRD masks. Reports AP at multiple IoU thresholds and per-case
recall/best-IoU.

Usage:
    python abus_det_compute_metrics.py
"""

import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from abus_det_utils import compute_iou_3d, compute_ap_for_dataset


def extract_bboxes_from_mask(mask_arr):
    """Extract 3D bounding boxes from a binary mask."""
    labeled, num = ndimage.label(mask_arr > 0)
    boxes = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) == 0:
            continue
        z1, y1, x1 = coords.min(axis=0).astype(float)
        z2, y2, x2 = (coords.max(axis=0) + 1).astype(float)
        boxes.append([z1, y1, x1, z2, y2, x2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros(
        (0, 6), dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SegMamba-Det predictions")
    parser.add_argument("--pred_file", type=str,
                        default="./prediction_results/segmamba_abus_det/"
                                "detections.json")
    parser.add_argument("--abus_root", type=str,
                        default="/Volumes/Autzoko/ABUS")
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Validation", "Test"])
    args = parser.parse_args()

    # Load predictions
    with open(args.pred_file) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.pred_file}")

    gt_mask_dir = os.path.join(args.abus_root, "data", args.split, "MASK")

    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    case_results = []

    for entry in tqdm(predictions, desc="Evaluating"):
        case_name = entry['case_name']
        case_id = case_name.replace("ABUS_", "")

        # Predicted boxes (already in original resolution)
        dets = entry['detections']
        if len(dets) > 0:
            pred_boxes = np.array(
                [[d['z1'], d['y1'], d['x1'],
                  d['z2'], d['y2'], d['x2']] for d in dets],
                dtype=np.float32)
            pred_scores = np.array(
                [d['score'] for d in dets], dtype=np.float32)
        else:
            pred_boxes = np.zeros((0, 6), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)

        # Ground-truth boxes from mask
        gt_path = os.path.join(gt_mask_dir, f"MASK_{case_id}.nrrd")
        if not os.path.exists(gt_path):
            print(f"  Warning: GT not found for {case_name}, skipping")
            continue

        gt_itk = sitk.ReadImage(gt_path)
        gt_arr = sitk.GetArrayFromImage(gt_itk).astype(np.uint8)
        gt_boxes = extract_bboxes_from_mask(gt_arr)

        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_gt_boxes.append(gt_boxes)

        # Per-case best IoU
        best_iou = 0.0
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = compute_iou_3d(pred_boxes, gt_boxes)
            best_iou = ious.max()
        detected = best_iou >= 0.25

        case_results.append({
            'case': case_name,
            'n_pred': len(pred_boxes),
            'n_gt': len(gt_boxes),
            'best_iou': best_iou,
            'detected': detected,
        })

        print(f"  {case_name}  pred={len(pred_boxes)}  gt={len(gt_boxes)}  "
              f"best_IoU={best_iou:.4f}  {'HIT' if detected else 'MISS'}")

    # Aggregate metrics
    iou_thresholds = [0.1, 0.25, 0.5]
    aps = {}
    for t in iou_thresholds:
        aps[t] = compute_ap_for_dataset(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            iou_threshold=t)

    # Recall
    total_gt = sum(len(g) for g in all_gt_boxes)
    recall_25 = sum(1 for r in case_results if r['detected']) / max(
        len(case_results), 1)
    mean_best_iou = np.mean(
        [r['best_iou'] for r in case_results]) if case_results else 0.0

    print(f"\n{'='*60}")
    print(f"  Results  ({args.split} split, {len(case_results)} cases)")
    print(f"{'='*60}")
    for t in iou_thresholds:
        print(f"  AP@{t:.2f}   = {aps[t]:.4f}")
    print(f"  Recall@0.25 = {recall_25:.4f}")
    print(f"  Mean best IoU = {mean_best_iou:.4f}")
    print(f"{'='*60}")

    # Save
    out_dir = os.path.join(os.path.dirname(args.pred_file), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "detection_metrics.json")
    with open(out_path, 'w') as f:
        json.dump({
            'AP': {str(t): aps[t] for t in iou_thresholds},
            'recall_at_0.25': recall_25,
            'mean_best_iou': mean_best_iou,
            'per_case': case_results,
        }, f, indent=2)
    print(f"  Saved -> {out_path}")
