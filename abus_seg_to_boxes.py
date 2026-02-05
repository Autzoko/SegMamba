"""
Extract 3D bounding boxes from SegMamba segmentation predictions.

This script converts binary segmentation masks to bounding boxes using
connected component analysis. This allows full-resolution detection by:
1. Training segmentation at full resolution with abus_train.py
2. Generating predictions with abus_predict.py
3. Extracting boxes with this script
4. Evaluating with abus_det_compute_metrics.py

Usage:
    python abus_seg_to_boxes.py --pred_dir ./prediction_results/segmamba_abus \
                                --abus_root /Volumes/Autzoko/ABUS \
                                --split Test

Output: detections.json in the same format as abus_det_predict.py
"""

import os
import argparse
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm


def extract_boxes_from_mask(mask):
    """Extract bounding boxes from a binary mask using connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 1)

    Returns
    -------
    boxes : np.ndarray (N, 6)
        Bounding boxes [z1, y1, x1, z2, y2, x2]
    """
    if mask.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Label connected components
    labeled, num_features = ndimage.label(mask)

    boxes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        indices = np.where(component)

        if len(indices[0]) == 0:
            continue

        z1, y1, x1 = indices[0].min(), indices[1].min(), indices[2].min()
        z2, y2, x2 = indices[0].max() + 1, indices[1].max() + 1, indices[2].max() + 1

        boxes.append([z1, y1, x1, z2, y2, x2])

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    return np.array(boxes, dtype=np.float32)


def get_case_ids(abus_root, split):
    """Get case IDs for a given split."""
    split_dir = os.path.join(abus_root, split)
    case_ids = []

    for name in sorted(os.listdir(split_dir)):
        case_path = os.path.join(split_dir, name)
        if os.path.isdir(case_path):
            case_ids.append(name)

    return case_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from segmentation predictions")
    parser.add_argument("--pred_dir", type=str,
                        default="./prediction_results/segmamba_abus",
                        help="Directory with segmentation predictions (.nii.gz)")
    parser.add_argument("--abus_root", type=str,
                        default="/Volumes/Autzoko/ABUS",
                        help="ABUS dataset root (for case IDs)")
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Validation", "Test"],
                        help="Dataset split")
    parser.add_argument("--output_file", type=str, default="",
                        help="Output JSON path (default: pred_dir/detections.json)")
    parser.add_argument("--min_volume", type=int, default=100,
                        help="Minimum component volume in voxels (filters noise)")
    args = parser.parse_args()

    output_file = args.output_file
    if not output_file:
        output_file = os.path.join(args.pred_dir, "detections.json")

    case_ids = get_case_ids(args.abus_root, args.split)
    print(f"Found {len(case_ids)} cases in {args.split} split")

    all_detections = {}

    for case_id in tqdm(case_ids, desc="Extracting boxes"):
        # Try different naming conventions
        pred_path = None
        for pattern in [f"ABUS_{case_id}.nii.gz", f"{case_id}.nii.gz"]:
            candidate = os.path.join(args.pred_dir, pattern)
            if os.path.exists(candidate):
                pred_path = candidate
                break

        if pred_path is None:
            print(f"Warning: No prediction found for {case_id}")
            all_detections[case_id] = {"boxes": [], "scores": []}
            continue

        # Load prediction
        nii = nib.load(pred_path)
        mask = nii.get_fdata().astype(np.uint8)

        # Filter small components
        if args.min_volume > 0:
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                if (labeled == i).sum() < args.min_volume:
                    mask[labeled == i] = 0

        # Extract boxes
        boxes = extract_boxes_from_mask(mask)

        # Compute confidence scores based on component volume
        scores = []
        if len(boxes) > 0:
            labeled, _ = ndimage.label(mask)
            for i, box in enumerate(boxes, 1):
                component_volume = (labeled == i).sum()
                # Normalize score by typical tumor volume (heuristic)
                score = min(1.0, component_volume / 10000.0)
                scores.append(float(score))

        all_detections[case_id] = {
            "boxes": boxes.tolist(),
            "scores": scores
        }

    # Save detections
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_detections, f, indent=2)

    # Summary statistics
    total_boxes = sum(len(d["boxes"]) for d in all_detections.values())
    cases_with_boxes = sum(1 for d in all_detections.values() if len(d["boxes"]) > 0)

    print(f"\nSaved {total_boxes} boxes from {cases_with_boxes}/{len(case_ids)} cases")
    print(f"Output: {output_file}")
    print(f"\nEvaluate with:")
    print(f"  python abus_det_compute_metrics.py --pred_file {output_file} "
          f"--abus_root {args.abus_root} --split {args.split}")


if __name__ == "__main__":
    main()
