"""
ABUS BoxHead Preprocessing: volume + mask + boxes at 128^3.

For each case:
    1. Load NRRD volume and mask
    2. Extract bounding boxes from mask (connected components)
    3. Resize volume to 128^3 (trilinear)
    4. Resize mask to 128^3 (nearest-neighbour)
    5. Z-score normalise volume
    6. Scale box coordinates to 128^3 space
    7. Save as compressed NPZ

Usage:
    python abus_boxhead_preprocessing.py --abus_root /Volumes/Autzoko/ABUS
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

from abus_det_preprocessing import (extract_bboxes_from_mask,
                                     resize_volume, scale_boxes)


def resize_mask(arr, target_shape):
    """Nearest-neighbour resize of a 3D binary mask."""
    t = torch.from_numpy(arr[None, None].astype(np.float32))
    out = F.interpolate(t, size=target_shape, mode='nearest')
    return out[0, 0].numpy().astype(np.uint8)


def preprocess_case(data_path, mask_path, case_name, output_dir,
                    target_shape=(128, 128, 128)):
    data_itk = sitk.ReadImage(data_path)
    spacing = data_itk.GetSpacing()
    data_arr = sitk.GetArrayFromImage(data_itk).astype(np.float32)
    orig_shape = data_arr.shape

    mask_itk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)

    # Bounding boxes from original mask
    boxes_orig = extract_bboxes_from_mask(mask_arr)

    # Resize volume (trilinear) and mask (nearest)
    data_resized = resize_volume(data_arr, target_shape)
    mask_resized = resize_mask(mask_arr, target_shape)

    # Z-score normalise
    mean = data_resized.mean()
    std = data_resized.std()
    data_resized = (data_resized - mean) / max(std, 1e-8)

    # Scale boxes to 128^3 space
    boxes_scaled = scale_boxes(boxes_orig, orig_shape, target_shape)
    boxes_arr = (np.array(boxes_scaled, dtype=np.float32)
                 if len(boxes_scaled) > 0
                 else np.zeros((0, 6), dtype=np.float32))

    np.savez_compressed(
        os.path.join(output_dir, f"{case_name}.npz"),
        data=data_resized[None].astype(np.float32),    # (1,128,128,128)
        seg=mask_resized[None].astype(np.uint8),        # (1,128,128,128)
        boxes=boxes_arr,                                 # (N,6)
        original_shape=np.array(orig_shape, dtype=np.int64),
        spacing=np.array(list(spacing), dtype=np.float64),
    )

    fg = mask_resized.sum()
    print(f"  [{case_name}]  {orig_shape} -> {target_shape}  "
          f"boxes={len(boxes_orig)}  fg_voxels={fg}")


def process_split(abus_root, split_name, output_dir,
                  target_shape=(128, 128, 128)):
    data_dir = os.path.join(abus_root, "data", split_name, "DATA")
    mask_dir = os.path.join(abus_root, "data", split_name, "MASK")
    os.makedirs(output_dir, exist_ok=True)

    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.nrrd')])

    print(f"\n{'='*60}")
    print(f"  BoxHead preprocessing — {split_name}: {len(data_files)} cases")
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

        preprocess_case(data_path, mask_path, case_name, output_dir,
                        target_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABUS boxhead preprocessing: NRRD -> NPZ "
                    "with volume + mask + boxes at 128^3")
    parser.add_argument("--abus_root", type=str,
                        default="/Volumes/Autzoko/ABUS")
    parser.add_argument("--output_base", type=str,
                        default="./data/abus_boxhead")
    args = parser.parse_args()

    target = (128, 128, 128)
    process_split(args.abus_root, "Train",
                  os.path.join(args.output_base, "train"), target)
    process_split(args.abus_root, "Validation",
                  os.path.join(args.output_base, "val"), target)
    process_split(args.abus_root, "Test",
                  os.path.join(args.output_base, "test"), target)

    print("=" * 60)
    print("  BoxHead preprocessing complete!")
    print("=" * 60)
