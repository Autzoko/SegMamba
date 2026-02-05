"""
SegMamba-DETR Prediction Script for ABUS 3D Object Detection.

Loads preprocessed 128^3 test volumes, runs DETR inference, converts boxes
to original resolution, and saves as JSON (same format as abus_det_predict.py).

Usage:
    python abus_detr_predict.py --model_path ./logs/segmamba_abus_detr/model/best_model_XXXX.pt
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast

from model_segmamba.segmamba_detr import SegMambaDETR
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
def predict_all(model, data_dir, device, score_thresh=0.05, max_det=20):
    """Run DETR detection on all cases in data_dir."""
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
        case_name = os.path.basename(fp).replace('.npz', '')

        with autocast():
            outputs = model(volume)

        pred_logits = outputs['pred_logits'][0]   # (N, 2)
        pred_boxes = outputs['pred_boxes'][0]     # (N, 6)

        # Score = softmax probability of tumour class (index 0)
        probs = pred_logits.softmax(-1)
        scores = probs[:, 0].cpu().numpy()

        # Convert from normalised cxcycz to corner format in 128^3 space
        boxes_corner = box_cxcyczdhwd_to_zyxzyx(pred_boxes.cpu())
        boxes_abs = (boxes_corner * INPUT_SIZE).numpy()
        boxes_abs = np.clip(boxes_abs, 0, INPUT_SIZE)

        # Filter by score
        mask = scores > score_thresh
        boxes_abs = boxes_abs[mask]
        scores = scores[mask]

        # Sort by score descending, keep top detections
        order = scores.argsort()[::-1][:max_det]
        boxes_abs = boxes_abs[order]
        scores = scores[order]

        # Scale to original resolution
        boxes_orig = scale_boxes_to_original(boxes_abs, orig_shape)

        det_list = []
        for i in range(len(scores)):
            det_list.append({
                'z1': float(boxes_orig[i, 0]),
                'y1': float(boxes_orig[i, 1]),
                'x1': float(boxes_orig[i, 2]),
                'z2': float(boxes_orig[i, 3]),
                'y2': float(boxes_orig[i, 4]),
                'x2': float(boxes_orig[i, 5]),
                'score': float(scores[i]),
            })

        results.append({
            'case_name': case_name,
            'original_shape': orig_shape,
            'detections': det_list,
        })

        n = len(det_list)
        top_s = f"{det_list[0]['score']:.3f}" if n > 0 else "-"
        print(f"  {case_name}  detections={n}  top_score={top_s}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SegMamba-DETR inference on ABUS test set")
    parser.add_argument("--data_dir_test", type=str,
                        default="./data/abus_det/test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_detr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_thresh", type=float, default=0.05)
    # DETR model args (must match training)
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    model = SegMambaDETR(
        in_chans=1, num_classes=1,
        num_queries=args.num_queries,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        aux_loss=False,
    ).to(device)

    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f"Loaded checkpoint: {args.model_path}")

    results = predict_all(
        model, args.data_dir_test, device,
        score_thresh=args.score_thresh)

    out_file = os.path.join(args.save_path, "detections.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {len(results)} cases processed")
    print(f"  Saved -> {out_file}")
    print(f"{'='*60}")
