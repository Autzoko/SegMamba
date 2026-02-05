"""
Extract 3D bounding boxes from SegMamba segmentation predictions.

This script converts binary segmentation masks to bounding boxes using
connected component analysis. This allows full-resolution detection by:
1. Training segmentation at full resolution with abus_train.py
2. Generating predictions with abus_predict.py
3. Extracting boxes with this script
4. Evaluating with abus_det_compute_metrics.py

Usage:
    python abus_seg_to_boxes.py --pred_dir ./prediction_results/segmamba_abus

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
    volumes : list
        Volume of each connected component (for confidence scoring)
    """
    if mask.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    # Label connected components
    labeled, num_features = ndimage.label(mask)

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


def main():
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from segmentation predictions")
    parser.add_argument("--pred_dir", type=str,
                        default="./prediction_results/segmamba_abus",
                        help="Directory with segmentation predictions (.nii.gz)")
    parser.add_argument("--output_file", type=str, default="",
                        help="Output JSON path (default: pred_dir/detections.json)")
    parser.add_argument("--min_volume", type=int, default=100,
                        help="Minimum component volume in voxels (filters noise)")
    args = parser.parse_args()

    output_file = args.output_file
    if not output_file:
        output_file = os.path.join(args.pred_dir, "detections.json")

    # Find all prediction files (same pattern as abus_compute_metrics.py)
    pred_files = sorted([
        f for f in os.listdir(args.pred_dir) if f.endswith('.nii.gz')
    ])

    if len(pred_files) == 0:
        print(f"No .nii.gz files found in {args.pred_dir}")
        print("Run abus_predict.py first to generate segmentation predictions.")
        return

    print(f"Found {len(pred_files)} predictions in {args.pred_dir}")

    all_detections = {}

    for pf in tqdm(pred_files, desc="Extracting boxes"):
        # Extract case name: ABUS_130.nii.gz -> ABUS_130
        case_name = pf.replace(".nii.gz", "")
        # Extract case ID for JSON key: ABUS_130 -> 130
        case_id = case_name.replace("ABUS_", "")

        # Load prediction
        pred_path = os.path.join(args.pred_dir, pf)
        nii = nib.load(pred_path)
        mask = nii.get_fdata().astype(np.uint8)

        # Extract boxes with volume filtering
        boxes, volumes = extract_boxes_from_mask(mask)

        # Filter small components
        if args.min_volume > 0 and len(boxes) > 0:
            keep = [i for i, v in enumerate(volumes) if v >= args.min_volume]
            boxes = boxes[keep] if len(keep) > 0 else np.zeros((0, 6), dtype=np.float32)
            volumes = [volumes[i] for i in keep]

        # Compute confidence scores based on component volume
        scores = []
        for vol in volumes:
            # Normalize score by typical tumor volume (heuristic)
            # Larger components get higher scores, capped at 1.0
            score = min(1.0, vol / 10000.0)
            scores.append(float(max(score, 0.01)))

        # Use case_id as key (matches abus_det_compute_metrics.py format)
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

    print(f"\nExtracted {total_boxes} boxes from {cases_with_boxes}/{len(pred_files)} cases")
    print(f"Output: {output_file}")
    print(f"\nEvaluate with:")
    print(f"  python abus_det_compute_metrics.py --pred_file {output_file}")


if __name__ == "__main__":
    main()
