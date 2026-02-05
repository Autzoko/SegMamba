"""
ABUS 3D Ultrasound Dataset Preprocessing for SegMamba.

Converts NRRD format ABUS data into SegMamba's NPZ + PKL format.
Handles: single-channel 3D ultrasound, binary segmentation masks.

Pipeline per case:
    Load NRRD -> float32 -> Crop to nonzero -> Z-score normalize
    -> Compute class locations -> Save NPZ + PKL

Usage:
    python abus_preprocessing.py

Data layout expected:
    /Volumes/Autzoko/ABUS/data/
        Train/DATA/DATA_XXX.nrrd   Train/MASK/MASK_XXX.nrrd
        Validation/DATA/...        Validation/MASK/...
        Test/DATA/...              Test/MASK/...

Output layout:
    ./data/abus/train/ABUS_XXX.npz  + .pkl
    ./data/abus/val/ABUS_XXX.npz    + .pkl
    ./data/abus/test/ABUS_XXX.npz   + .pkl
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
# Helper functions (mirrored from DefaultPreprocessor to avoid modifying it)
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

    print(f"  [{case_name}]  orig {shape_before_cropping} -> "
          f"crop {shape_after_cropping}  spacing {spacing}")


# ---------------------------------------------------------------------------
# Process an entire split
# ---------------------------------------------------------------------------

def process_split(abus_root, split_name, output_dir, num_processes=4):
    """Process all cases in one split (Train / Validation / Test)."""
    split_dir = os.path.join(abus_root, "data", split_name)
    data_dir = os.path.join(split_dir, "DATA")
    mask_dir = os.path.join(split_dir, "MASK")

    maybe_mkdir_p(output_dir)

    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.nrrd')])

    cases = []
    for df in data_files:
        case_id = df.replace("DATA_", "").replace(".nrrd", "")
        mask_file = f"MASK_{case_id}.nrrd"
        data_path = os.path.join(data_dir, df)
        mask_path = os.path.join(mask_dir, mask_file)
        case_name = f"ABUS_{case_id}"

        if not os.path.exists(mask_path):
            print(f"  Warning: mask not found for {df}, skipping")
            continue

        cases.append((data_path, mask_path, case_name, output_dir))

    print(f"\n{'='*60}")
    print(f"  Processing {split_name}: {len(cases)} cases")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}")

    if len(cases) == 0:
        return

    # --- first case sequentially (sanity check) ---
    preprocess_case(*cases[0])

    # --- remaining cases with multiprocessing ---
    remaining_cases = cases[1:]
    if len(remaining_cases) == 0:
        return

    if num_processes > 1:
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for case_args in remaining_cases:
                r.append(p.apply_async(preprocess_case, case_args))

            workers = [j for j in p._pool]
            with tqdm(desc=split_name, total=len(remaining_cases)) as pbar:
                remaining_idx = list(range(len(r)))
                while len(remaining_idx) > 0:
                    all_alive = all(j.is_alive() for j in workers)
                    if not all_alive:
                        raise RuntimeError(
                            "A background worker died. Possibly out of RAM. "
                            "Try reducing num_processes.")
                    done = [i for i in remaining_idx if r[i].ready()]
                    # Check for exceptions
                    for i in done:
                        r[i].get()  # raises if the worker raised
                    for _ in done:
                        pbar.update()
                    remaining_idx = [
                        i for i in remaining_idx if i not in done
                    ]
                    sleep(0.1)
    else:
        for case_args in tqdm(remaining_cases, desc=split_name):
            preprocess_case(*case_args)

    print(f"  Done: {split_name} ({len(cases)} cases)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    abus_root = "/Volumes/Autzoko/ABUS"
    output_base = "./data/abus"

    # Number of parallel workers.  Reduce if running out of RAM (each
    # worker loads a full 3D volume into memory).
    num_processes = 4

    process_split(abus_root, "Train",
                  os.path.join(output_base, "train"), num_processes)
    process_split(abus_root, "Validation",
                  os.path.join(output_base, "val"), num_processes)
    process_split(abus_root, "Test",
                  os.path.join(output_base, "test"), num_processes)

    print("=" * 60)
    print("  All preprocessing complete!")
    print("=" * 60)
