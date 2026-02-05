"""
Compute evaluation metrics for ABUS SegMamba predictions.

Compares predicted NIfTI segmentations against ground-truth NRRD masks.
Reports per-case and aggregate Dice and HD95.

Usage:
    python abus_compute_metrics.py

Prerequisites:
    Run  abus_predict.py  first to generate predictions in
        ./prediction_results/segmamba_abus/
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from medpy import metric

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
abus_root    = "/Volumes/Autzoko/ABUS"
results_root = "prediction_results"
pred_name    = "segmamba_abus"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def cal_metric(gt, pred, voxel_spacing):
    """Compute binary Dice and HD95 for a single case."""
    if pred.sum() > 0 and gt.sum() > 0:
        dc   = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dc, hd95])
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 50.0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, default=pred_name)
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Validation", "Test"])
    args = parser.parse_args()

    pred_name  = args.pred_name
    split_name = args.split
    pred_dir   = os.path.join(results_root, pred_name)

    # --- locate predictions ---
    pred_files = sorted([
        f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')
    ])
    print(f"Found {len(pred_files)} predictions in {pred_dir}")

    # --- ground-truth mask directory ---
    gt_mask_dir = os.path.join(abus_root, "data", split_name, "MASK")

    all_results = []
    case_names  = []

    for pf in tqdm(pred_files, desc="Computing metrics"):
        case_name = pf.replace(".nii.gz", "")      # e.g. ABUS_130
        case_id   = case_name.replace("ABUS_", "")  # e.g. 130

        # --- load prediction ---
        pred_itk   = sitk.ReadImage(os.path.join(pred_dir, pf))
        pred_array = sitk.GetArrayFromImage(pred_itk).astype(np.uint8)

        # --- load ground-truth ---
        gt_path = os.path.join(gt_mask_dir, f"MASK_{case_id}.nrrd")
        if not os.path.exists(gt_path):
            print(f"  Warning: GT not found for {case_name}, skipping")
            continue

        gt_itk   = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.uint8)

        # --- ensure shapes match ---
        if pred_array.shape != gt_array.shape:
            print(f"  Warning: shape mismatch for {case_name}: "
                  f"pred {pred_array.shape} vs gt {gt_array.shape}")
            continue

        # voxel spacing (from GT, in zyx order for metric computation)
        spacing_xyz = gt_itk.GetSpacing()   # (sx, sy, sz)
        voxel_spacing = list(spacing_xyz)[::-1]  # -> (sz, sy, sx) = zyx

        # --- compute binary metrics ---
        m = cal_metric(gt_array > 0, pred_array > 0, voxel_spacing)
        all_results.append(m)
        case_names.append(case_name)

        print(f"  {case_name}  Dice={m[0]:.4f}  HD95={m[1]:.2f}")

    # --- aggregate ---
    all_results = np.array(all_results)   # (N, 2)

    print(f"\n{'='*60}")
    print(f"  Results for {pred_name}  ({split_name} split)")
    print(f"  Cases evaluated: {len(all_results)}")
    print(f"{'='*60}")
    print(f"  Dice  — mean: {all_results[:, 0].mean():.4f}  "
          f"std: {all_results[:, 0].std():.4f}")
    print(f"  HD95  — mean: {all_results[:, 1].mean():.2f}  "
          f"std: {all_results[:, 1].std():.2f}")
    print(f"{'='*60}")

    # --- save ---
    out_dir = os.path.join(results_root, "result_metrics")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{pred_name}.npy"), all_results)
    print(f"  Saved metric array -> {out_dir}/{pred_name}.npy")
