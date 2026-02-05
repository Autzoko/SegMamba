"""
ABUS Detection Preprocessing: derive bounding-box GT from masks.

For each case:
    1. Load NRRD volume and mask
    2. Extract bounding boxes from mask via connected components
    3. Resize volume to 128^3 (required by MambaEncoder)
    4. Z-score normalize
    5. Scale box coordinates to 128^3 space
    6. Save as compressed NPZ

Usage:
    python abus_det_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_bboxes_from_mask(mask_arr):
    """Extract 3D bounding boxes from a binary mask.

    Uses connected-component labelling so that multiple disjoint tumour
    regions produce separate boxes.

    Parameters
    ----------
    mask_arr : np.ndarray (D, H, W)

    Returns
    -------
    boxes : list of [z1, y1, x1, z2, y2, x2]  (float, exclusive upper)
    """
    labeled, num = ndimage.label(mask_arr > 0)
    boxes = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) == 0:
            continue
        z1, y1, x1 = coords.min(axis=0).astype(float)
        z2, y2, x2 = (coords.max(axis=0) + 1).astype(float)
        boxes.append([z1, y1, x1, z2, y2, x2])
    return boxes


def resize_volume(arr, target_shape):
    """Trilinear resize of a 3D volume."""
    t = torch.from_numpy(arr[None, None].astype(np.float32))
    out = F.interpolate(t, size=target_shape, mode='trilinear',
                        align_corners=False)
    return out[0, 0].numpy()


def scale_boxes(boxes, orig_shape, target_shape):
    """Scale box coordinates from orig_shape to target_shape."""
    sz = [t / o for t, o in zip(target_shape, orig_shape)]
    scaled = []
    for z1, y1, x1, z2, y2, x2 in boxes:
        scaled.append([
            z1 * sz[0], y1 * sz[1], x1 * sz[2],
            z2 * sz[0], y2 * sz[1], x2 * sz[2],
        ])
    return scaled


# ---------------------------------------------------------------------------
# Per-case preprocessing
# ---------------------------------------------------------------------------

def preprocess_case_det(data_path, mask_path, case_name, output_dir,
                        target_shape=(128, 128, 128)):
    data_itk = sitk.ReadImage(data_path)
    spacing = data_itk.GetSpacing()
    data_arr = sitk.GetArrayFromImage(data_itk).astype(np.float32)
    orig_shape = data_arr.shape

    mask_itk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)

    boxes_orig = extract_bboxes_from_mask(mask_arr)

    data_resized = resize_volume(data_arr, target_shape)

    mean = data_resized.mean()
    std = data_resized.std()
    data_resized = (data_resized - mean) / max(std, 1e-8)

    boxes_scaled = scale_boxes(boxes_orig, orig_shape, target_shape)
    boxes_arr = (np.array(boxes_scaled, dtype=np.float32)
                 if len(boxes_scaled) > 0
                 else np.zeros((0, 6), dtype=np.float32))

    np.savez_compressed(
        os.path.join(output_dir, f"{case_name}.npz"),
        data=data_resized[None].astype(np.float32),   # (1,D,H,W)
        boxes=boxes_arr,
        original_shape=np.array(orig_shape, dtype=np.int64),
        spacing=np.array(list(spacing), dtype=np.float64),
    )

    print(f"  [{case_name}]  {orig_shape} -> {target_shape}  "
          f"boxes={len(boxes_orig)}")


# ---------------------------------------------------------------------------
# Process a split
# ---------------------------------------------------------------------------

def process_split_det(abus_root, split_name, output_dir,
                      target_shape=(128, 128, 128)):
    data_dir = os.path.join(abus_root, "data", split_name, "DATA")
    mask_dir = os.path.join(abus_root, "data", split_name, "MASK")
    os.makedirs(output_dir, exist_ok=True)

    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.nrrd')])

    print(f"\n{'='*60}")
    print(f"  Detection preprocessing — {split_name}: {len(data_files)} cases")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}")

    for df in tqdm(data_files, desc=split_name):
        case_id = df.replace("DATA_", "").replace(".nrrd", "")
        mask_file = f"MASK_{case_id}.nrrd"
        data_path = os.path.join(data_dir, df)
        mask_path = os.path.join(mask_dir, mask_file)
        case_name = f"ABUS_{case_id}"

        if not os.path.exists(mask_path):
            print(f"  Warning: mask not found for {df}, skipping")
            continue

        preprocess_case_det(data_path, mask_path, case_name, output_dir,
                            target_shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABUS detection preprocessing: NRRD -> NPZ with boxes")
    parser.add_argument("--abus_root", type=str,
                        default="/Volumes/Autzoko/ABUS")
    parser.add_argument("--output_base", type=str,
                        default="./data/abus_det")
    args = parser.parse_args()

    target = (128, 128, 128)
    process_split_det(args.abus_root, "Train",
                      os.path.join(args.output_base, "train"), target)
    process_split_det(args.abus_root, "Validation",
                      os.path.join(args.output_base, "val"), target)
    process_split_det(args.abus_root, "Test",
                      os.path.join(args.output_base, "test"), target)

    print("=" * 60)
    print("  Detection preprocessing complete!")
    print("=" * 60)
