"""
SegMamba-Det Prediction Script for ABUS 3D Bounding Box Detection.

Loads preprocessed 128^3 test volumes, runs inference, decodes boxes,
applies NMS, scales back to original resolution, and saves as JSON.

Usage:
    python abus_det_predict.py --model_path ./logs/segmamba_abus_det/model/best_model_0.XXXX.pt
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast

from model_segmamba.segmamba_det import SegMambaDet
from abus_det_utils import decode_detections, nms_3d

INPUT_SIZE = 128
STRIDES = [2, 4, 8, 16]


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
def predict_all(model, data_dir, device, score_thresh=0.05,
                nms_thresh=0.3, max_det=20):
    """Run detection on all cases in data_dir."""
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
            all_cls, all_reg, all_ctr = model(volume)

        boxes, scores = decode_detections(
            all_cls, all_reg, all_ctr, STRIDES, INPUT_SIZE,
            score_thresh=score_thresh)

        if len(boxes) > 0:
            keep = nms_3d(boxes, scores, iou_threshold=nms_thresh)
            boxes = boxes[keep][:max_det]
            scores = scores[keep][:max_det]

        # Scale to original resolution
        boxes_orig = scale_boxes_to_original(boxes, orig_shape)

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
        description="SegMamba-Det inference on ABUS test set")
    parser.add_argument("--data_dir_test", type=str,
                        default="./data/abus_det/test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str,
                        default="./prediction_results/segmamba_abus_det")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    model = SegMambaDet(in_chans=1, num_classes=1).to(device)
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f"Loaded checkpoint: {args.model_path}")

    results = predict_all(
        model, args.data_dir_test, device,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh)

    out_file = os.path.join(args.save_path, "detections.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {len(results)} cases processed")
    print(f"  Saved -> {out_file}")
    print(f"{'='*60}")
