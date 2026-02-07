"""
ABUS 3D Ultrasound Dataset Preprocessing for SegMamba.

Converts NRRD format ABUS data into SegMamba's NPZ + PKL format.
Handles: single-channel 3D ultrasound, binary segmentation masks.

Key features:
- Collects ALL cases from Train/Validation/Test folders
- Filters out cases with NO lesion (empty mask)
- Splits data 80/10/10 (train/val/test) with random seed
- Z-score normalization, crop to nonzero

Pipeline per case:
    Load NRRD -> Verify lesion exists -> float32 -> Crop to nonzero
    -> Z-score normalize -> Compute class locations -> Save NPZ + PKL

Usage:
    python abus_preprocessing.py --abus_root /path/to/ABUS --seed 42

Data layout expected:
    /path/to/ABUS/data/
        Train/DATA/DATA_XXX.nrrd   Train/MASK/MASK_XXX.nrrd
        Validation/DATA/...        Validation/MASK/...
        Test/DATA/...              Test/MASK/...

Output layout:
    ./data/abus/train/ABUS_XXX.npz  + .pkl  (80% of data)
    ./data/abus/val/ABUS_XXX.npz    + .pkl  (10% of data)
    ./data/abus/test/ABUS_XXX.npz   + .pkl  (10% of data)
"""

import os
import sys
import multiprocessing
from time import sleep
import numpy as np
import pickle
import SimpleITK as sitk
from tqdm import tqdm
from light_training.preprocessing.cropping.cropping import crop_to_nonzero
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collect_foreground_intensities(segmentation, images, seed=1234,
                                   num_samples=10000):
    """Collect intensity statistics from the foreground region (seg > 0).

    Parameters
    ----------
    segmentation : np.ndarray, shape (1, D, H, W)
    images       : np.ndarray, shape (C, D, H, W)

    Returns
    -------
    intensities_per_channel           : list[np.ndarray]
    intensity_statistics_per_channel   : list[dict]
    """
    assert len(images.shape) == 4 and len(segmentation.shape) == 4

    rs = np.random.RandomState(seed)
    intensities_per_channel = []
    intensity_statistics_per_channel = []

    foreground_mask = segmentation[0] > 0

    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        num_fg = len(foreground_pixels)

        intensities_per_channel.append(
            rs.choice(foreground_pixels, num_samples, replace=True)
            if num_fg > 0 else []
        )
        intensity_statistics_per_channel.append({
            'mean':
            float(np.mean(foreground_pixels)) if num_fg > 0 else np.nan,
            'median':
            float(np.median(foreground_pixels)) if num_fg > 0 else np.nan,
            'min':
            float(np.min(foreground_pixels)) if num_fg > 0 else np.nan,
            'max':
            float(np.max(foreground_pixels)) if num_fg > 0 else np.nan,
            'percentile_99_5':
            float(np.percentile(foreground_pixels, 99.5))
            if num_fg > 0 else np.nan,
            'percentile_00_5':
            float(np.percentile(foreground_pixels, 0.5))
            if num_fg > 0 else np.nan,
        })

    return intensities_per_channel, intensity_statistics_per_channel


def sample_foreground_locations(seg, classes_or_regions, seed=1234):
    """Sample foreground voxel locations used for oversampling during training.

    Parameters
    ----------
    seg                : np.ndarray, shape (1, D, H, W)
    classes_or_regions : list of int or list of tuple

    Returns
    -------
    class_locs : dict  {label_value : np.ndarray of shape (N, 4)}
    """
    num_samples = 10000
    min_percent_coverage = 0.01
    rndst = np.random.RandomState(seed)
    class_locs = {}

    for c in classes_or_regions:
        k = c if not isinstance(c, list) else tuple(c)
        if isinstance(c, (tuple, list)):
            mask = seg == c[0]
            for cc in c[1:]:
                mask = mask | (seg == cc)
            all_locs = np.argwhere(mask)
        else:
            all_locs = np.argwhere(seg == c)

        if len(all_locs) == 0:
            class_locs[k] = []
            continue

        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(
            target_num_samples,
            int(np.ceil(len(all_locs) * min_percent_coverage)))

        selected = all_locs[rndst.choice(
            len(all_locs), target_num_samples, replace=False)]
        class_locs[k] = selected

    return class_locs


def check_mask_has_lesion(mask_path):
    """Check if a mask file contains any lesion (non-zero voxels).

    Returns
    -------
    has_lesion : bool
    num_voxels : int
    """
    mask_itk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_itk)
    num_voxels = int((mask_arr > 0).sum())
    return num_voxels > 0, num_voxels


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_case(data_path, mask_path, case_name, output_dir,
                    all_labels=None):
    """Preprocess a single ABUS case.

    Steps
    -----
    1. Read NRRD with SimpleITK  ->  float32 (1, D, H, W)
    2. Collect foreground intensity statistics
    3. Crop to nonzero bounding box
    4. Z-score normalise (global mean / std)
    5. Compute class locations for foreground oversampling
    6. Save compressed NPZ  +  properties PKL
    """
    if all_labels is None:
        all_labels = [1]

    # --- read image ---
    data_itk = sitk.ReadImage(data_path)
    spacing = data_itk.GetSpacing()              # (sx, sy, sz)
    data_arr = sitk.GetArrayFromImage(data_itk)  # (z, y, x)
    data_arr = data_arr.astype(np.float32)[None]  # (1, z, y, x)

    # --- read mask ---
    mask_itk = sitk.ReadImage(mask_path)
    seg_arr = sitk.GetArrayFromImage(mask_itk).astype(np.float32)[None]

    # --- intensity statistics ---
    intensities_per_channel, intensity_statistics_per_channel = \
        collect_foreground_intensities(seg_arr, data_arr)

    original_spacing_trans = list(spacing)[::-1]   # xyz -> zyx
    target_spacing = [1.0, 1.0, 1.0]

    properties = {
        "spacing": spacing,
        "raw_size": data_arr.shape[1:],
        "name": case_name,
        "intensities_per_channel": intensities_per_channel,
        "intensity_statistics_per_channel": intensity_statistics_per_channel,
        "original_spacing_trans": original_spacing_trans,
        "target_spacing_trans": target_spacing,
    }

    # --- crop to nonzero ---
    shape_before_cropping = data_arr.shape[1:]
    properties['shape_before_cropping'] = shape_before_cropping

    data_arr, seg_arr, bbox = crop_to_nonzero(data_arr, seg_arr)
    properties['bbox_used_for_cropping'] = bbox
    properties['crop_bbox'] = bbox  # Also store as crop_bbox for compatibility

    shape_after_cropping = data_arr.shape[1:]
    properties['shape_after_cropping_before_resample'] = shape_after_cropping

    # --- Z-score normalisation (global) ---
    mean = data_arr.mean()
    std = data_arr.std()
    data_arr = (data_arr - mean) / max(std, 1e-8)

    # ABUS spacing is already [1,1,1] — skip resampling.
    properties['shape_after_resample'] = shape_after_cropping

    # --- class locations for foreground oversampling ---
    properties['class_locations'] = sample_foreground_locations(
        seg_arr, all_labels)

    # --- convert seg dtype for storage ---
    if np.max(seg_arr) > 127:
        seg_arr = seg_arr.astype(np.int16)
    else:
        seg_arr = seg_arr.astype(np.int8)

    # --- save ---
    out_npz = os.path.join(output_dir, case_name + '.npz')
    out_pkl = os.path.join(output_dir, case_name + '.pkl')

    np.savez_compressed(out_npz, data=data_arr, seg=seg_arr)
    with open(out_pkl, 'wb') as f:
        pickle.dump(properties, f)

    return case_name, shape_before_cropping, shape_after_cropping, spacing


# ---------------------------------------------------------------------------
# Collect all cases from all splits
# ---------------------------------------------------------------------------

def collect_all_cases(abus_root):
    """Collect all valid cases from Train/Validation/Test folders.

    Returns list of (data_path, mask_path, case_id, original_split).
    Only includes cases WITH lesions (non-empty masks).
    """
    splits = ["Train", "Validation", "Test"]
    all_cases = []
    skipped_no_lesion = []
    skipped_no_mask = []

    for split_name in splits:
        split_dir = os.path.join(abus_root, "data", split_name)
        data_dir = os.path.join(split_dir, "DATA")
        mask_dir = os.path.join(split_dir, "MASK")

        if not os.path.exists(data_dir):
            print(f"  Warning: {data_dir} not found, skipping")
            continue

        data_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.nrrd')])

        for df in data_files:
            case_id = df.replace("DATA_", "").replace(".nrrd", "")
            mask_file = f"MASK_{case_id}.nrrd"
            data_path = os.path.join(data_dir, df)
            mask_path = os.path.join(mask_dir, mask_file)

            if not os.path.exists(mask_path):
                skipped_no_mask.append(case_id)
                continue

            # Check if mask has lesion
            has_lesion, num_voxels = check_mask_has_lesion(mask_path)
            if not has_lesion:
                skipped_no_lesion.append(case_id)
                continue

            all_cases.append({
                'data_path': data_path,
                'mask_path': mask_path,
                'case_id': case_id,
                'original_split': split_name,
                'lesion_voxels': num_voxels,
            })

    print(f"\n{'='*60}")
    print(f"  Case Collection Summary")
    print(f"{'='*60}")
    print(f"  Valid cases (with lesion): {len(all_cases)}")
    print(f"  Skipped (no mask file):    {len(skipped_no_mask)}")
    print(f"  Skipped (no lesion):       {len(skipped_no_lesion)}")

    if skipped_no_lesion:
        print(f"  Cases without lesion: {skipped_no_lesion[:10]}{'...' if len(skipped_no_lesion) > 10 else ''}")

    return all_cases


def split_cases(all_cases, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split cases into train/val/test with given ratios.

    Parameters
    ----------
    all_cases : list of dict
    train_ratio : float (default 0.8)
    val_ratio : float (default 0.1)
    seed : int (default 42)

    Returns
    -------
    train_cases, val_cases, test_cases : lists
    """
    np.random.seed(seed)

    n_total = len(all_cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Shuffle
    indices = np.random.permutation(n_total)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_cases = [all_cases[i] for i in train_indices]
    val_cases = [all_cases[i] for i in val_indices]
    test_cases = [all_cases[i] for i in test_indices]

    print(f"\n  Split (seed={seed}):")
    print(f"    Train:      {len(train_cases)} cases ({len(train_cases)/n_total*100:.1f}%)")
    print(f"    Validation: {len(val_cases)} cases ({len(val_cases)/n_total*100:.1f}%)")
    print(f"    Test:       {len(test_cases)} cases ({len(test_cases)/n_total*100:.1f}%)")

    return train_cases, val_cases, test_cases


# ---------------------------------------------------------------------------
# Process cases
# ---------------------------------------------------------------------------

def process_cases(cases, output_dir, split_name, num_processes=4):
    """Process a list of cases and save to output_dir."""
    maybe_mkdir_p(output_dir)

    if len(cases) == 0:
        print(f"  No cases to process for {split_name}")
        return

    print(f"\n{'='*60}")
    print(f"  Processing {split_name}: {len(cases)} cases")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}")

    # Prepare arguments
    case_args = []
    for case in cases:
        case_name = f"ABUS_{case['case_id']}"
        case_args.append((
            case['data_path'],
            case['mask_path'],
            case_name,
            output_dir
        ))

    # Process first case sequentially (sanity check)
    result = preprocess_case(*case_args[0])
    print(f"  [{result[0]}] orig {result[1]} -> crop {result[2]} spacing {result[3]}")

    # Process remaining cases
    remaining_args = case_args[1:]
    if len(remaining_args) == 0:
        return

    if num_processes > 1:
        results = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for args in remaining_args:
                results.append(p.apply_async(preprocess_case, args))

            workers = [j for j in p._pool]
            with tqdm(desc=split_name, total=len(remaining_args)) as pbar:
                remaining_idx = list(range(len(results)))
                while len(remaining_idx) > 0:
                    all_alive = all(j.is_alive() for j in workers)
                    if not all_alive:
                        raise RuntimeError(
                            "A background worker died. Possibly out of RAM. "
                            "Try reducing num_processes.")
                    done = [i for i in remaining_idx if results[i].ready()]
                    for i in done:
                        results[i].get()  # raises if worker raised
                    for _ in done:
                        pbar.update()
                    remaining_idx = [i for i in remaining_idx if i not in done]
                    sleep(0.1)
    else:
        for args in tqdm(remaining_args, desc=split_name):
            result = preprocess_case(*args)

    print(f"  Done: {split_name} ({len(cases)} cases)\n")


def save_split_info(output_base, train_cases, val_cases, test_cases, seed):
    """Save split information for reproducibility."""
    split_info = {
        'seed': seed,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'train_cases': [c['case_id'] for c in train_cases],
        'val_cases': [c['case_id'] for c in val_cases],
        'test_cases': [c['case_id'] for c in test_cases],
        'total_cases': len(train_cases) + len(val_cases) + len(test_cases),
    }

    info_path = os.path.join(output_base, 'split_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(split_info, f)

    # Also save as readable text
    txt_path = os.path.join(output_base, 'split_info.txt')
    with open(txt_path, 'w') as f:
        f.write(f"ABUS Dataset Split (seed={seed})\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total cases (with lesion): {split_info['total_cases']}\n")
        f.write(f"Train: {len(train_cases)} (80%)\n")
        f.write(f"Val:   {len(val_cases)} (10%)\n")
        f.write(f"Test:  {len(test_cases)} (10%)\n\n")

        f.write(f"Train cases:\n")
        for c in sorted(train_cases, key=lambda x: x['case_id']):
            f.write(f"  {c['case_id']} (from {c['original_split']}, {c['lesion_voxels']} voxels)\n")

        f.write(f"\nValidation cases:\n")
        for c in sorted(val_cases, key=lambda x: x['case_id']):
            f.write(f"  {c['case_id']} (from {c['original_split']}, {c['lesion_voxels']} voxels)\n")

        f.write(f"\nTest cases:\n")
        for c in sorted(test_cases, key=lambda x: x['case_id']):
            f.write(f"  {c['case_id']} (from {c['original_split']}, {c['lesion_voxels']} voxels)\n")

    print(f"  Saved split info to {info_path}")
    print(f"  Saved readable split info to {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ABUS preprocessing: NRRD -> NPZ for SegMamba (80/10/10 split)")
    parser.add_argument("--abus_root", type=str,
                        default="/Volumes/Autzoko/ABUS",
                        help="Path to raw ABUS dataset root")
    parser.add_argument("--output_base", type=str,
                        default="./data/abus",
                        help="Output directory for preprocessed files")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Parallel workers (reduce if OOM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of data for training (default 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default 0.1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ABUS Preprocessing Pipeline")
    print("  - Collects ALL cases from Train/Validation/Test folders")
    print("  - Filters cases WITHOUT lesion (empty masks)")
    print(f"  - Splits: {args.train_ratio*100:.0f}% train / {args.val_ratio*100:.0f}% val / {(1-args.train_ratio-args.val_ratio)*100:.0f}% test")
    print(f"  - Random seed: {args.seed}")
    print("=" * 60)

    # Step 1: Collect all valid cases
    print("\nStep 1: Collecting cases with lesions...")
    all_cases = collect_all_cases(args.abus_root)

    if len(all_cases) == 0:
        print("ERROR: No valid cases found!")
        sys.exit(1)

    # Step 2: Split into train/val/test
    print("\nStep 2: Splitting data...")
    train_cases, val_cases, test_cases = split_cases(
        all_cases,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Step 3: Create output directories
    maybe_mkdir_p(args.output_base)
    train_dir = os.path.join(args.output_base, "train")
    val_dir = os.path.join(args.output_base, "val")
    test_dir = os.path.join(args.output_base, "test")

    # Step 4: Process each split
    print("\nStep 3: Processing cases...")
    process_cases(train_cases, train_dir, "Train", args.num_processes)
    process_cases(val_cases, val_dir, "Validation", args.num_processes)
    process_cases(test_cases, test_dir, "Test", args.num_processes)

    # Step 5: Save split information
    print("\nStep 4: Saving split information...")
    save_split_info(args.output_base, train_cases, val_cases, test_cases, args.seed)

    print("\n" + "=" * 60)
    print("  Preprocessing Complete!")
    print("=" * 60)
    print(f"  Output directory: {args.output_base}")
    print(f"  Train: {len(train_cases)} cases -> {train_dir}")
    print(f"  Val:   {len(val_cases)} cases -> {val_dir}")
    print(f"  Test:  {len(test_cases)} cases -> {test_dir}")
    print("=" * 60)
