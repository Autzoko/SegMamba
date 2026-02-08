"""
Segmentation-to-Box Detection Evaluation for ABUS (CPU only).

Evaluates detection performance by extracting bounding boxes from
segmentation predictions and comparing against GT boxes.

This script does NOT require GPU - run on CPU nodes after inference.

Workflow:
1. Run inference on GPU node:
   python abus_seg_inference.py --model_path ... --output_dir ./predictions

2. Run evaluation on CPU node (this script):
   python abus_seg_box_eval.py --pred_dir ./predictions

Output:
    - detections.json: Predicted boxes in standard format
    - seg_box_metrics.json: IoU metrics and per-case results
"""

import os
import argparse
import json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm


def extract_boxes_from_mask(mask):
    """Extract bounding boxes from a binary mask using connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (D, H, W)

    Returns
    -------
    boxes : np.ndarray (N, 6)
        Bounding boxes in CORNER format [z1, y1, x1, z2, y2, x2]
    volumes : list
        Volume of each connected component
    """
    if mask.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    # Label connected components
    labeled, num_features = ndimage.label(mask > 0)

    boxes = []
    volumes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        indices = np.where(component)

        if len(indices[0]) == 0:
            continue

        z1, y1, x1 = indices[0].min(), indices[1].min(), indices[2].min()
        z2, y2, x2 = indices[0].max() + 1, indices[1].max() + 1, indices[2].max() + 1

        boxes.append([z1, y1, x1, z2, y2, x2])
        volumes.append(int(component.sum()))

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    return np.array(boxes, dtype=np.float32), volumes


def compute_iou_3d(box1, box2):
    """Compute IoU between two boxes in corner format [z1,y1,x1,z2,y2,x2]."""
    # Intersection
    z1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    z2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    x2 = min(box1[5], box2[5])

    inter_z = max(0, z2 - z1)
    inter_y = max(0, y2 - y1)
    inter_x = max(0, x2 - x1)
    intersection = inter_z * inter_y * inter_x

    # Volumes
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU matrix between all pred and gt boxes.

    Returns
    -------
    iou_matrix : np.ndarray (N_pred, N_gt)
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou_3d(pb, gb)

    return iou_matrix


def build_gt_box_cache(abus_root, split="Test", cache_path=None):
    """Build or load cached GT boxes to avoid repeated NRRD reads.

    Parameters
    ----------
    abus_root : str
        Root directory of ABUS dataset
    split : str
        Dataset split (Train, Validation, Test)
    cache_path : str, optional
        Path to cache file. If None, uses default location.

    Returns
    -------
    gt_boxes_cache : dict
        Mapping from case_id to {'boxes': np.ndarray, 'volumes': list}
    """
    if cache_path is None:
        cache_path = os.path.join(abus_root, f"gt_boxes_cache_{split.lower()}.json")

    # Try to load existing cache
    if os.path.exists(cache_path):
        print(f"Loading GT boxes from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        # Convert lists back to numpy arrays
        gt_boxes_cache = {}
        for case_id, data in cache_data.items():
            gt_boxes_cache[case_id] = {
                'boxes': np.array(data['boxes'], dtype=np.float32),
                'volumes': data['volumes'],
            }
        print(f"  Loaded {len(gt_boxes_cache)} cached GT boxes")
        return gt_boxes_cache

    # Build cache from scratch
    gt_mask_dir = os.path.join(abus_root, "data", split, "MASK")
    if not os.path.exists(gt_mask_dir):
        print(f"GT mask directory not found: {gt_mask_dir}")
        return {}

    mask_files = sorted([f for f in os.listdir(gt_mask_dir) if f.endswith('.nrrd')])
    print(f"Building GT box cache from {len(mask_files)} masks...")
    print(f"  (This is slow but only needs to be done once)")

    gt_boxes_cache = {}
    cache_data = {}

    for mf in tqdm(mask_files, desc="Extracting GT boxes"):
        # Extract case_id from filename like "MASK_001.nrrd"
        case_id = mf.replace("MASK_", "").replace(".nrrd", "")

        gt_path = os.path.join(gt_mask_dir, mf)
        gt_itk = sitk.ReadImage(gt_path)
        gt_mask = sitk.GetArrayFromImage(gt_itk).astype(np.uint8)

        boxes, volumes = extract_boxes_from_mask(gt_mask)

        gt_boxes_cache[case_id] = {
            'boxes': boxes,
            'volumes': volumes,
        }
        cache_data[case_id] = {
            'boxes': boxes.tolist(),
            'volumes': volumes,
        }

    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    print(f"  Saved GT box cache to: {cache_path}")

    return gt_boxes_cache


def evaluate_detections(pred_dir, gt_cache, min_volume=100):
    """Evaluate detection metrics from segmentation predictions.

    Parameters
    ----------
    pred_dir : str
        Directory with segmentation predictions (.nii.gz)
    gt_cache : dict
        Pre-loaded GT boxes cache from build_gt_box_cache()
    min_volume : int
        Minimum component volume to consider as detection

    Returns
    -------
    results : dict
        Evaluation metrics and per-case results
    all_detections : dict
        Detections in standard format for saving
    """
    # Find prediction files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    if len(pred_files) == 0:
        print(f"No .nii.gz files found in {pred_dir}")
        return None, None

    if len(gt_cache) == 0:
        print("No GT boxes available")
        return None, None

    print(f"Evaluating {len(pred_files)} predictions against {len(gt_cache)} cached GT boxes...")

    all_detections = {}
    case_results = []

    # IoU thresholds for metrics
    iou_thresholds = [0.1, 0.25, 0.5, 0.75]

    for pf in tqdm(pred_files, desc="Evaluating"):
        case_name = pf.replace(".nii.gz", "")
        case_id = case_name.replace("ABUS_", "")

        # Load prediction
        pred_path = os.path.join(pred_dir, pf)
        pred_itk = sitk.ReadImage(pred_path)
        pred_mask = sitk.GetArrayFromImage(pred_itk).astype(np.uint8)

        # Extract predicted boxes
        pred_boxes, pred_volumes = extract_boxes_from_mask(pred_mask)

        # Filter small components
        if min_volume > 0 and len(pred_boxes) > 0:
            keep = [i for i, v in enumerate(pred_volumes) if v >= min_volume]
            if len(keep) > 0:
                pred_boxes = pred_boxes[keep]
                pred_volumes = [pred_volumes[i] for i in keep]
            else:
                pred_boxes = np.zeros((0, 6), dtype=np.float32)
                pred_volumes = []

        # Compute confidence scores based on volume
        pred_scores = []
        for vol in pred_volumes:
            score = min(1.0, vol / 10000.0)
            pred_scores.append(float(max(score, 0.01)))

        # Get GT boxes from cache
        if case_id not in gt_cache:
            print(f"  Warning: GT not found in cache for {case_name}, skipping")
            continue

        gt_data = gt_cache[case_id]
        gt_boxes = gt_data['boxes']

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Best IoU for each GT box
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            best_iou_per_gt = iou_matrix.max(axis=0)  # (N_gt,)
            best_iou_per_pred = iou_matrix.max(axis=1)  # (N_pred,)
            overall_best_iou = iou_matrix.max()
        else:
            best_iou_per_gt = np.zeros(len(gt_boxes))
            best_iou_per_pred = np.zeros(len(pred_boxes))
            overall_best_iou = 0.0

        # Mean IoU across GT boxes
        mean_gt_iou = float(best_iou_per_gt.mean()) if len(gt_boxes) > 0 else 0.0

        # Detection rates at different thresholds
        detected_at = {}
        for t in iou_thresholds:
            if len(gt_boxes) > 0:
                detected_at[t] = int((best_iou_per_gt >= t).sum())
            else:
                detected_at[t] = 0

        # Store results
        case_results.append({
            'case_id': case_id,
            'case_name': case_name,
            'n_pred': len(pred_boxes),
            'n_gt': len(gt_boxes),
            'best_iou': float(overall_best_iou),
            'mean_gt_iou': mean_gt_iou,
            'detected_at': detected_at,
            'pred_boxes': pred_boxes.tolist(),
            'gt_boxes': gt_boxes.tolist(),
        })

        # Store for JSON output
        all_detections[case_id] = {
            "boxes": pred_boxes.tolist(),
            "scores": pred_scores,
        }

    # Aggregate metrics
    total_gt = sum(r['n_gt'] for r in case_results)
    total_pred = sum(r['n_pred'] for r in case_results)

    # Detection rate (at least one GT detected per case)
    detection_rates = {}
    for t in iou_thresholds:
        detected_cases = sum(1 for r in case_results if r['detected_at'][t] > 0 and r['n_gt'] > 0)
        cases_with_gt = sum(1 for r in case_results if r['n_gt'] > 0)
        detection_rates[t] = detected_cases / max(cases_with_gt, 1)

    # Recall (fraction of GT boxes detected)
    recall_at = {}
    for t in iou_thresholds:
        detected_gt = sum(r['detected_at'][t] for r in case_results)
        recall_at[t] = detected_gt / max(total_gt, 1)

    # Mean best IoU
    mean_best_iou = np.mean([r['best_iou'] for r in case_results if r['n_gt'] > 0]) if case_results else 0.0
    mean_gt_iou_all = np.mean([r['mean_gt_iou'] for r in case_results if r['n_gt'] > 0]) if case_results else 0.0

    results = {
        'summary': {
            'n_cases': len(case_results),
            'total_pred_boxes': total_pred,
            'total_gt_boxes': total_gt,
            'mean_best_iou': float(mean_best_iou),
            'mean_gt_iou': float(mean_gt_iou_all),
        },
        'detection_rate': {str(t): detection_rates[t] for t in iou_thresholds},
        'recall': {str(t): recall_at[t] for t in iou_thresholds},
        'per_case': case_results,
    }

    return results, all_detections


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation-to-box detection (CPU only)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory with segmentation predictions (.nii.gz)")
    parser.add_argument("--abus_root", type=str, default="/Volumes/Autzoko/ABUS",
                        help="Root directory of ABUS dataset (for GT masks)")
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Validation", "Test"],
                        help="Dataset split for GT masks")
    parser.add_argument("--min_volume", type=int, default=100,
                        help="Minimum component volume to consider as detection")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild GT box cache")
    args = parser.parse_args()

    print("=" * 60)
    print("  Segmentation-to-Box Evaluation (CPU only)")
    print("=" * 60)

    # Build or load GT box cache
    cache_path = os.path.join(args.abus_root, f"gt_boxes_cache_{args.split.lower()}.json")
    if args.rebuild_cache and os.path.exists(cache_path):
        print(f"Removing existing cache: {cache_path}")
        os.remove(cache_path)

    gt_cache = build_gt_box_cache(args.abus_root, args.split, cache_path)

    if len(gt_cache) == 0:
        print("ERROR: No GT boxes available. Check --abus_root path.")
        return

    # Evaluate
    results, detections = evaluate_detections(
        args.pred_dir, gt_cache, args.min_volume)

    if results is None or detections is None:
        print("Evaluation failed. Check paths and try again.")
        return

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Segmentation-to-Box Detection Evaluation ({args.split} split)")
    print(f"{'='*70}")
    print(f"  Cases:            {results['summary']['n_cases']}")
    print(f"  Total Pred Boxes: {results['summary']['total_pred_boxes']}")
    print(f"  Total GT Boxes:   {results['summary']['total_gt_boxes']}")
    print(f"\n  Mean Best IoU:    {results['summary']['mean_best_iou']:.4f}")
    print(f"  Mean GT IoU:      {results['summary']['mean_gt_iou']:.4f}")
    print(f"\n  Detection Rate (case-level, at least one GT detected):")
    for t in ['0.1', '0.25', '0.5', '0.75']:
        print(f"    @IoU={t}:  {results['detection_rate'][t]:.4f}")
    print(f"\n  Recall (box-level, fraction of GT boxes detected):")
    for t in ['0.1', '0.25', '0.5', '0.75']:
        print(f"    @IoU={t}:  {results['recall'][t]:.4f}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(args.pred_dir, exist_ok=True)

    # Save detections JSON
    det_path = os.path.join(args.pred_dir, "detections.json")
    with open(det_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"\nSaved detections to: {det_path}")

    # Save metrics
    metrics_path = os.path.join(args.pred_dir, "seg_box_metrics.json")
    with open(metrics_path, 'w') as f:
        # Remove per-case boxes to reduce file size
        results_clean = {
            'summary': results['summary'],
            'detection_rate': results['detection_rate'],
            'recall': results['recall'],
            'per_case': [
                {k: v for k, v in r.items() if k not in ['pred_boxes', 'gt_boxes']}
                for r in results['per_case']
            ]
        }
        json.dump(results_clean, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Print per-case details for cases with low IoU
    print(f"\n  Per-case results (cases with best_IoU < 0.5):")
    low_iou_cases = [r for r in results['per_case'] if r['best_iou'] < 0.5 and r['n_gt'] > 0]
    for r in sorted(low_iou_cases, key=lambda x: x['best_iou'])[:10]:
        print(f"    {r['case_name']}: pred={r['n_pred']} gt={r['n_gt']} "
              f"best_IoU={r['best_iou']:.4f}")


if __name__ == "__main__":
    main()
