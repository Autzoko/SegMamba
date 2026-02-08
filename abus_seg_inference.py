"""
SegMamba Segmentation Inference for ABUS (GPU).

Runs sliding window inference on ABUS volumes and saves predictions as NIfTI.
This is the GPU-intensive phase that should be run on GPU nodes.

Usage:
    python abus_seg_inference.py \
        --model_path ./logs/segmamba_abus/model/best_model.pt \
        --data_dir ./data/abus/test \
        --output_dir ./prediction_results/segmamba_abus

After inference, run evaluation (CPU only):
    python abus_seg_box_eval.py \
        --pred_dir ./prediction_results/segmamba_abus
"""

import os
import glob
import argparse
import numpy as np
import pickle
import torch
import SimpleITK as sitk
from torch.cuda.amp import autocast
from tqdm import tqdm


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

    # Add corner positions if not covered
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


def run_inference(model_path, data_dir, output_dir, device="cuda:0", overlap=0.5):
    """Run SegMamba segmentation inference with sliding window."""
    from model_segmamba.segmamba import SegMamba

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = SegMamba(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    sd = torch.load(model_path, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print(f"Model loaded successfully")

    # Find test files
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Processing {len(npz_files)} cases...")

    # Pre-compute Gaussian weight
    sigma = [s / 4 for s in PATCH_SIZE]
    zz, yy, xx = np.mgrid[:PATCH_SIZE[0], :PATCH_SIZE[1], :PATCH_SIZE[2]]
    center = [s / 2 for s in PATCH_SIZE]
    gaussian = np.exp(-((zz - center[0])**2 / (2*sigma[0]**2) +
                        (yy - center[1])**2 / (2*sigma[1]**2) +
                        (xx - center[2])**2 / (2*sigma[2]**2))).astype(np.float32)

    for npz_path in tqdm(npz_files, desc="Inference"):
        data = np.load(npz_path)
        volume = data['data'].astype(np.float32)  # (1, D, H, W)

        pkl_path = npz_path.replace('.npz', '.pkl')
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)

        case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
        volume_shape = volume.shape[1:]

        # Sliding window inference with Gaussian weighting
        positions = sliding_window_positions(volume_shape, PATCH_SIZE, overlap)

        output = np.zeros((2, *volume_shape), dtype=np.float32)
        weight_sum = np.zeros(volume_shape, dtype=np.float32)

        with torch.no_grad():
            for pos in positions:
                z, y, x = pos
                patch = volume[:, z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]

                # Pad if necessary
                if patch.shape[1:] != tuple(PATCH_SIZE):
                    pad_d = PATCH_SIZE[0] - patch.shape[1]
                    pad_h = PATCH_SIZE[1] - patch.shape[2]
                    pad_w = PATCH_SIZE[2] - patch.shape[3]
                    patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))

                patch_t = torch.from_numpy(patch[np.newaxis]).to(device)

                with autocast():
                    logits = model(patch_t)

                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                # Crop to valid region
                d_end = min(z + PATCH_SIZE[0], volume_shape[0]) - z
                h_end = min(y + PATCH_SIZE[1], volume_shape[1]) - y
                w_end = min(x + PATCH_SIZE[2], volume_shape[2]) - x

                output[:, z:z+d_end, y:y+h_end, x:x+w_end] += (
                    probs[:, :d_end, :h_end, :w_end] * gaussian[:d_end, :h_end, :w_end])
                weight_sum[z:z+d_end, y:y+h_end, x:x+w_end] += gaussian[:d_end, :h_end, :w_end]

        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-6)
        output = output / weight_sum

        # Get prediction
        seg_pred = (output[1] > 0.5).astype(np.uint8)

        # Restore to original shape if cropped
        if 'shape_before_cropping' in props and 'crop_bbox' in props:
            full_shape = props['shape_before_cropping']
            crop_bbox = props['crop_bbox']
            full_seg = np.zeros(full_shape, dtype=np.uint8)
            full_seg[crop_bbox[0][0]:crop_bbox[0][1],
                     crop_bbox[1][0]:crop_bbox[1][1],
                     crop_bbox[2][0]:crop_bbox[2][1]] = seg_pred
            seg_pred = full_seg

        # Save as NIfTI
        spacing = props.get('spacing', [1.0, 1.0, 1.0])
        if isinstance(spacing[0], torch.Tensor):
            spacing = [s.item() for s in spacing]

        seg_itk = sitk.GetImageFromArray(seg_pred)
        seg_itk.SetSpacing(list(spacing)[::-1])
        sitk.WriteImage(seg_itk, os.path.join(output_dir, f"{case_name}.nii.gz"))

    print(f"\n{'='*60}")
    print(f"  Inference Complete!")
    print(f"{'='*60}")
    print(f"  Saved {len(npz_files)} predictions to: {output_dir}")
    print(f"\n  Next step - Run evaluation (CPU only):")
    print(f"    python abus_seg_box_eval.py --pred_dir {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SegMamba segmentation inference (GPU)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained SegMamba checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/abus/test",
                        help="Directory with preprocessed test data (.npz)")
    parser.add_argument("--output_dir", type=str,
                        default="./prediction_results/segmamba_abus",
                        help="Output directory for predictions (.nii.gz)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:0, cuda:1, etc.)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap (0.0-0.9)")
    args = parser.parse_args()

    run_inference(
        args.model_path,
        args.data_dir,
        args.output_dir,
        args.device,
        args.overlap
    )


if __name__ == "__main__":
    main()
