"""
SegMamba-BoxHead Prediction Script for ABUS 3D Detection.

Multi-task model: outputs segmentation mask + bounding box.
Saves detection JSON (compatible with abus_det_compute_metrics.py)
and optionally segmentation masks as NIfTI.

Usage:
    python abus_boxhead_predict.py \
        --model_path ./logs/segmamba_abus_boxhead/model/best_model_XXXX.pt
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import SimpleITK as sitk
from torch.cuda.amp import autocast

from model_segmamba.segmamba_boxhead import SegMambaWithBoxHead
from abus_detr_utils import box_cxcyczdhwd_to_zyxzyx

INPUT_SIZE = 128


def scale_boxes_to_original(boxes, original_shape, input_size=128):
    """Scale boxes from 128^3 space back to original volume coordinates."""
    if len(boxes) == 0:
        return boxes
    sz = np.array(original_shape, dtype=np.float64) / input_size
    scaled = boxes.copy().astype(np.float64)
    scaled[:, 0] *= sz[0]
    scaled[:, 1] *= sz[1]
    scaled[:, 2] *= sz[2]
    scaled[:, 3] *= sz[0]
    scaled[:, 4] *= sz[1]
    scaled[:, 5] *= sz[2]
    return scaled


@torch.no_grad()
def predict_all(model, data_dir, device, save_seg_path=None):
    """Run BoxHead inference on all test cases."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(files) == 0:
        print(f"No test data in {data_dir}.")
        return []

    model.eval()
    results = []

    for fp in files:
        d = np.load(fp)
        volume = torch.from_numpy(d['data'][None]).to(device)  # (1,1,D,H,W)
        orig_shape = d['original_shape'].tolist()
        spacing = d['spacing'].tolist()
        case_name = os.path.basename(fp).replace('.npz', '')

        with autocast():
            seg_logits, box_pred = model(volume)

        # --- Segmentation ---
        seg_probs = torch.softmax(seg_logits, dim=1)
        seg_mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        fg_voxels = int(seg_mask.sum())

        # Derive confidence from max tumour probability
        max_tumor_prob = float(seg_probs[0, 1].max().cpu())
        score = max_tumor_prob

        # Save seg mask as NIfTI
        if save_seg_path is not None:
            seg_itk = sitk.GetImageFromArray(seg_mask)
            # Spacing in SimpleITK is (x, y, z); orig_shape is (z, y, x)
            ds_spacing = [
                orig_shape[2] / INPUT_SIZE,  # x
                orig_shape[1] / INPUT_SIZE,  # y
                orig_shape[0] / INPUT_SIZE,  # z
            ]
            seg_itk.SetSpacing(ds_spacing)
            sitk.WriteImage(
                seg_itk,
                os.path.join(save_seg_path, f"{case_name}.nii.gz"))

        # --- Detection ---
        bp = box_pred[0].cpu()                          # (6,) normalised cxcycz
        bp_corner = box_cxcyczdhwd_to_zyxzyx(bp.unsqueeze(0))
        bp_abs = (bp_corner * INPUT_SIZE).numpy()
        bp_abs = np.clip(bp_abs, 0, INPUT_SIZE)

        # Scale to original resolution
        bp_orig = scale_boxes_to_original(bp_abs, orig_shape)

        det_list = [{
            'z1': float(bp_orig[0, 0]),
            'y1': float(bp_orig[0, 1]),
            'x1': float(bp_orig[0, 2]),
            'z2': float(bp_orig[0, 3]),
            'y2': float(bp_orig[0, 4]),
            'x2': float(bp_orig[0, 5]),
            'score': score,
        }]

        results.append({
            'case_name': case_name,
            'original_shape': orig_shape,
            'detections': det_list,
        })

        print(f"  {case_name}  fg={fg_voxels}  score={score:.3f}  "
              f"box=[{bp_abs[0,0]:.1f},{bp_abs[0,1]:.1f},{bp_abs[0,2]:.1f},"
              f"{bp_abs[0,3]:.1f},{bp_abs[0,4]:.1f},{bp_abs[0,5]:.1f}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SegMamba-BoxHead inference on ABUS test set")
    parser.add_argument("--data_dir_test", type=str,
                        default="./data/abus_boxhead/test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_boxhead")
    parser.add_argument("--save_seg", action="store_true",
                        help="Also save segmentation masks as NIfTI")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    model = SegMambaWithBoxHead(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f"Loaded checkpoint: {args.model_path}")

    seg_path = None
    if args.save_seg:
        seg_path = os.path.join(args.save_path, "seg_masks")
        os.makedirs(seg_path, exist_ok=True)

    results = predict_all(
        model, args.data_dir_test, device, save_seg_path=seg_path)

    out_file = os.path.join(args.save_path, "detections.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {len(results)} cases processed")
    print(f"  Detections -> {out_file}")
    if seg_path:
        print(f"  Seg masks  -> {seg_path}/")
    print(f"{'='*60}")
