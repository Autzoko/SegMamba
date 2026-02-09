"""
SegMamba Inference + 2D Slice Box Prompt Generation for UltraSAM.

This script:
1. Runs SegMamba 3D segmentation inference on ABUS test volumes
2. Extracts 2D slices that have predicted masks
3. Computes minimum enclosing bounding rectangles as box prompts
4. Saves data in COCO format for UltraSAM inference

The output format matches UltraSAM's expected input:
- images/{split}/DATA_XXX_slice_YYYY.png
- annotations/{split}.coco.json (with bbox from predicted masks)

Usage:
    python abus_segmamba_to_ultrasam.py \
        --model_path ./logs/segmamba_abus/model/best_model.pt \
        --abus_root /Volumes/Autzoko/ABUS \
        --data_dir ./data/abus/test \
        --output_dir ./ultrasam_input \
        --split test

After running this, use UltraSAM for inference:
    cd /Volumes/Autzoko/MS\ Thesis/UltraSam
    python evaluate_abus_bbox_prompt.py \
        --config configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine_ABUS.py \
        --checkpoint UltraSam.pth \
        --split test \
        --data_root <output_dir>
"""

import os
import glob
import argparse
import json
import pickle
import numpy as np
import cv2
import torch
import SimpleITK as sitk
from torch.cuda.amp import autocast
from tqdm import tqdm
from pycocotools import mask as maskUtils


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

    # Add corner positions
    z_max = max(0, volume_shape[0] - patch_size[0])
    y_max = max(0, volume_shape[1] - patch_size[1])
    x_max = max(0, volume_shape[2] - patch_size[2])

    corners = [
        (0, 0, 0), (0, 0, x_max), (0, y_max, 0), (0, y_max, x_max),
        (z_max, 0, 0), (z_max, 0, x_max), (z_max, y_max, 0), (z_max, y_max, x_max),
    ]
    for pos in corners:
        if pos not in positions and all(p >= 0 for p in pos):
            positions.append(pos)

    return positions


def compute_bbox_from_mask(mask_2d):
    """Compute tight axis-aligned bounding box from a binary 2D mask.

    Returns [x, y, w, h] in COCO format (x=col_min, y=row_min).
    Returns None if mask is empty.
    """
    if not mask_2d.any():
        return None

    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def encode_mask_rle(mask_2d):
    """Encode a binary 2D mask as COCO RLE."""
    mask_fortran = np.asfortranarray(mask_2d.astype(np.uint8))
    rle = maskUtils.encode(mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def run_inference_3d(model, volume, device, patch_size, overlap=0.5):
    """Run sliding window inference on a 3D volume.

    Parameters
    ----------
    model : torch.nn.Module
        SegMamba model
    volume : np.ndarray
        Input volume of shape (1, D, H, W)
    device : str
        Device to use
    patch_size : tuple
        Patch size (D, H, W)
    overlap : float
        Overlap between patches

    Returns
    -------
    seg_pred : np.ndarray
        Binary segmentation prediction of shape (D, H, W)
    """
    volume_shape = volume.shape[1:]

    # Pre-compute Gaussian weight
    sigma = [s / 4 for s in patch_size]
    zz, yy, xx = np.mgrid[:patch_size[0], :patch_size[1], :patch_size[2]]
    center = [s / 2 for s in patch_size]
    gaussian = np.exp(-((zz - center[0])**2 / (2*sigma[0]**2) +
                        (yy - center[1])**2 / (2*sigma[1]**2) +
                        (xx - center[2])**2 / (2*sigma[2]**2))).astype(np.float32)

    # Sliding window inference
    positions = sliding_window_positions(volume_shape, patch_size, overlap)

    output = np.zeros((2, *volume_shape), dtype=np.float32)
    weight_sum = np.zeros(volume_shape, dtype=np.float32)

    with torch.no_grad():
        for pos in positions:
            z, y, x = pos
            patch = volume[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]

            # Pad if necessary
            if patch.shape[1:] != tuple(patch_size):
                pad_d = patch_size[0] - patch.shape[1]
                pad_h = patch_size[1] - patch.shape[2]
                pad_w = patch_size[2] - patch.shape[3]
                patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))

            patch_t = torch.from_numpy(patch[np.newaxis]).to(device)

            with autocast():
                logits = model(patch_t)

            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Crop to valid region
            d_end = min(z + patch_size[0], volume_shape[0]) - z
            h_end = min(y + patch_size[1], volume_shape[1]) - y
            w_end = min(x + patch_size[2], volume_shape[2]) - x

            output[:, z:z+d_end, y:y+h_end, x:x+w_end] += (
                probs[:, :d_end, :h_end, :w_end] * gaussian[:d_end, :h_end, :w_end])
            weight_sum[z:z+d_end, y:y+h_end, x:x+w_end] += gaussian[:d_end, :h_end, :w_end]

    # Normalize and threshold
    weight_sum = np.maximum(weight_sum, 1e-6)
    output = output / weight_sum
    seg_pred = (output[1] > 0.5).astype(np.uint8)

    return seg_pred


def find_original_nrrd(case_id, abus_root):
    """Find the original NRRD files for a case ID.

    Searches in Train/Validation/Test folders.

    Returns (data_path, mask_path) or (None, None) if not found.
    """
    for split in ["Train", "Validation", "Test"]:
        data_path = os.path.join(abus_root, "data", split, "DATA", f"DATA_{case_id}.nrrd")
        mask_path = os.path.join(abus_root, "data", split, "MASK", f"MASK_{case_id}.nrrd")
        if os.path.exists(data_path) and os.path.exists(mask_path):
            return data_path, mask_path
    return None, None


def process_volume(
    model, npz_path, pkl_path, abus_root, output_dir, split_name,
    device, patch_size, overlap, slice_axis=2
):
    """Process a single volume: inference + 2D slice extraction.

    Parameters
    ----------
    model : torch.nn.Module
        SegMamba model
    npz_path : str
        Path to preprocessed NPZ file
    pkl_path : str
        Path to properties PKL file
    abus_root : str
        Root directory of original ABUS data
    output_dir : str
        Output directory for UltraSAM data
    split_name : str
        Split name (train/val/test)
    device : str
        Device for inference
    patch_size : tuple
        Patch size for sliding window
    overlap : float
        Overlap for sliding window
    slice_axis : int
        Axis along which to extract 2D slices (default: 2 = elevation)

    Returns
    -------
    slices_info : list
        List of dicts with slice info for COCO annotations
    """
    # Load preprocessed data
    data = np.load(npz_path)
    volume = data['data'].astype(np.float32)  # (1, D, H, W)

    with open(pkl_path, 'rb') as f:
        props = pickle.load(f)

    case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
    case_id = case_name.replace('ABUS_', '')

    # Find original NRRD for image data
    data_nrrd_path, mask_nrrd_path = find_original_nrrd(case_id, abus_root)
    if data_nrrd_path is None:
        print(f"  Warning: Original NRRD not found for {case_name}, skipping")
        return []

    # Load original image (for PNG export)
    data_itk = sitk.ReadImage(data_nrrd_path)
    original_image = sitk.GetArrayFromImage(data_itk)  # (D, H, W) in original space

    # Load GT mask for reference
    mask_itk = sitk.ReadImage(mask_nrrd_path)
    original_gt_mask = sitk.GetArrayFromImage(mask_itk)

    # Run 3D inference
    seg_pred_cropped = run_inference_3d(model, volume, device, patch_size, overlap)

    # Restore to original shape if cropped during preprocessing
    if 'shape_before_cropping' in props and 'crop_bbox' in props:
        full_shape = props['shape_before_cropping']
        crop_bbox = props['crop_bbox']
        seg_pred = np.zeros(full_shape, dtype=np.uint8)
        seg_pred[crop_bbox[0][0]:crop_bbox[0][1],
                 crop_bbox[1][0]:crop_bbox[1][1],
                 crop_bbox[2][0]:crop_bbox[2][1]] = seg_pred_cropped
    else:
        seg_pred = seg_pred_cropped

    # Verify shapes match
    if seg_pred.shape != original_image.shape:
        print(f"  Warning: Shape mismatch for {case_name}: "
              f"pred {seg_pred.shape} vs original {original_image.shape}")
        # Try to crop/pad to match
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(seg_pred.shape, original_image.shape))
        seg_pred = seg_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        original_image = original_image[:min_shape[0], :min_shape[1], :min_shape[2]]
        original_gt_mask = original_gt_mask[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Extract 2D slices with predicted masks
    slices_info = []
    img_out_dir = os.path.join(output_dir, 'images', split_name)
    os.makedirs(img_out_dir, exist_ok=True)

    n_slices = seg_pred.shape[slice_axis]
    slices_with_pred = 0

    for z in range(n_slices):
        # Extract slice based on axis
        if slice_axis == 0:
            pred_slice = seg_pred[z, :, :]
            img_slice = original_image[z, :, :]
            gt_slice = original_gt_mask[z, :, :]
        elif slice_axis == 1:
            pred_slice = seg_pred[:, z, :]
            img_slice = original_image[:, z, :]
            gt_slice = original_gt_mask[:, z, :]
        else:  # slice_axis == 2
            pred_slice = seg_pred[:, :, z]
            img_slice = original_image[:, :, z]
            gt_slice = original_gt_mask[:, :, z]

        # Skip slices without predictions
        if not pred_slice.any():
            continue

        # Compute bounding box from predicted mask
        bbox = compute_bbox_from_mask(pred_slice)
        if bbox is None:
            continue

        height, width = img_slice.shape[:2]

        # Save image as PNG
        filename = f"DATA_{case_id}_slice_{z:04d}.png"
        cv2.imwrite(os.path.join(img_out_dir, filename), img_slice)

        # Encode predicted mask as RLE (for reference)
        pred_rle = encode_mask_rle(pred_slice)

        # Encode GT mask as RLE (for evaluation)
        gt_rle = encode_mask_rle(gt_slice) if gt_slice.any() else None

        slices_info.append({
            'case_id': case_id,
            'slice_idx': z,
            'filename': filename,
            'height': height,
            'width': width,
            'bbox': bbox,  # From predicted mask
            'pred_area': float(pred_slice.sum()),
            'pred_segmentation': pred_rle,
            'gt_area': float(gt_slice.sum()),
            'gt_segmentation': gt_rle,
            'has_gt': gt_slice.any(),
        })
        slices_with_pred += 1

    return slices_info


def build_coco_annotations(all_slices_info, output_dir, split_name):
    """Build COCO format annotations from slice info.

    Creates two annotation files:
    1. {split}.coco.json - Uses bbox from SegMamba predictions (for UltraSAM prompts)
    2. {split}_gt.coco.json - Uses GT masks (for evaluation reference)
    """
    ann_out_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(ann_out_dir, exist_ok=True)

    # Build annotations using predicted bbox (for UltraSAM prompts)
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for info in all_slices_info:
        # Image entry
        images.append({
            'id': image_id,
            'file_name': info['filename'],
            'height': info['height'],
            'width': info['width'],
        })

        # Annotation entry with predicted bbox but GT mask for evaluation
        ann_entry = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,
            'bbox': info['bbox'],  # From SegMamba prediction
            'area': info['gt_area'] if info['has_gt'] else info['pred_area'],
            'iscrowd': 0,
        }

        # Use GT segmentation if available, otherwise use predicted
        if info['gt_segmentation'] is not None:
            ann_entry['segmentation'] = info['gt_segmentation']
        else:
            ann_entry['segmentation'] = info['pred_segmentation']

        annotations.append(ann_entry)

        image_id += 1
        annotation_id += 1

    # Build COCO dict
    coco_dict = {
        'info': {
            'description': f'ABUS 2D slices with SegMamba bbox prompts - {split_name} split',
            'source': 'SegMamba predictions',
        },
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'tumor', 'supercategory': 'object'}],
    }

    # Save main COCO JSON
    json_path = os.path.join(ann_out_dir, f'{split_name}.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_dict, f)

    print(f"Saved {len(images)} annotations to {json_path}")

    # Also save detailed info for analysis
    info_path = os.path.join(ann_out_dir, f'{split_name}_slice_info.json')
    with open(info_path, 'w') as f:
        # Remove RLE segmentations to reduce file size
        info_clean = []
        for s in all_slices_info:
            s_clean = {k: v for k, v in s.items()
                       if k not in ['pred_segmentation', 'gt_segmentation']}
            info_clean.append(s_clean)
        json.dump(info_clean, f, indent=2)

    return len(images)


def main():
    parser = argparse.ArgumentParser(
        description="SegMamba inference + 2D slice box prompt generation for UltraSAM")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained SegMamba checkpoint")
    parser.add_argument("--abus_root", type=str, default="/Volumes/Autzoko/ABUS",
                        help="Root directory of original ABUS dataset (for image data)")
    parser.add_argument("--data_dir", type=str, default="./data/abus/test",
                        help="Directory with preprocessed test data (.npz)")
    parser.add_argument("--output_dir", type=str, default="./ultrasam_input",
                        help="Output directory for UltraSAM input data")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Split name for output organization")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for inference")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap (0.0-0.9)")
    parser.add_argument("--slice_axis", type=int, default=2,
                        help="Axis for 2D slicing (0=axial, 1=coronal, 2=elevation)")

    args = parser.parse_args()

    print("=" * 70)
    print("  SegMamba -> UltraSAM Box Prompt Generation")
    print("=" * 70)
    print(f"  Model:      {args.model_path}")
    print(f"  ABUS root:  {args.abus_root}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Split:      {args.split}")
    print(f"  Slice axis: {args.slice_axis}")
    print("=" * 70)

    # Load model
    from model_segmamba.segmamba import SegMamba

    print("\nLoading SegMamba model...")
    model = SegMamba(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(args.device)

    sd = torch.load(args.model_path, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print("Model loaded successfully")

    # Find test files
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if len(npz_files) == 0:
        print(f"No .npz files found in {args.data_dir}")
        return

    print(f"\nProcessing {len(npz_files)} volumes...")

    # Process each volume
    all_slices_info = []

    for npz_path in tqdm(npz_files, desc="Volumes"):
        pkl_path = npz_path.replace('.npz', '.pkl')

        if not os.path.exists(pkl_path):
            print(f"  Warning: PKL not found for {npz_path}, skipping")
            continue

        slices_info = process_volume(
            model, npz_path, pkl_path, args.abus_root, args.output_dir,
            args.split, args.device, PATCH_SIZE, args.overlap, args.slice_axis
        )
        all_slices_info.extend(slices_info)

    # Build COCO annotations
    print(f"\nBuilding COCO annotations...")
    n_slices = build_coco_annotations(all_slices_info, args.output_dir, args.split)

    # Summary
    n_volumes = len(npz_files)
    n_with_gt = sum(1 for s in all_slices_info if s['has_gt'])

    print(f"\n{'='*70}")
    print(f"  Processing Complete!")
    print(f"{'='*70}")
    print(f"  Volumes processed:   {n_volumes}")
    print(f"  Slices with pred:    {n_slices}")
    print(f"  Slices with GT:      {n_with_gt}")
    print(f"\n  Output directory:    {args.output_dir}")
    print(f"  Images:              {args.output_dir}/images/{args.split}/")
    print(f"  Annotations:         {args.output_dir}/annotations/{args.split}.coco.json")
    print(f"\n  Next step - Run UltraSAM inference:")
    print(f"    cd /Volumes/Autzoko/MS\\ Thesis/UltraSam")
    print(f"    python evaluate_abus_bbox_prompt.py \\")
    print(f"        --config configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \\")
    print(f"        --checkpoint weights/UltraSam.pth \\")
    print(f"        --split {args.split} \\")
    print(f"        --data_root {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
