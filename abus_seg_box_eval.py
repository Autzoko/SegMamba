"""
SegMamba Segmentation-to-Box Evaluation for ABUS.

Complete workflow for evaluating detection via segmentation:
1. Run SegMamba segmentation inference (optional, skip if predictions exist)
2. Extract bounding boxes from predicted masks via connected components
3. Extract bounding boxes from GT masks
4. Compute IoU and detection metrics

This provides a baseline for detection performance using only segmentation,
which can be compared against dedicated detection pipelines (FCOS, DETR, BoxHead).

Usage:
    # Full pipeline (inference + evaluation)
    python abus_seg_box_eval.py \
        --model_path ./logs/segmamba_abus/model/best_model.pt \
        --run_inference

    # Evaluation only (if predictions already exist)
    python abus_seg_box_eval.py \
        --pred_dir ./prediction_results/segmamba_abus

Output:
    - detections.json: Predicted boxes in standard format
    - seg_box_metrics.json: IoU metrics and per-case results
"""

import os
import glob
import argparse
import json
import numpy as np
import pickle
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

# Optional: for inference
try:
    import torch
    from torch.cuda.amp import autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_boxes_from_mask(mask):
    """Extract bounding boxes from a binary mask using connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (D, H, W)

    Returns
    -------
    boxes : np.ndarray (N, 6)
        Bounding boxes in CORNER format [z1, y1, x1, z2, y2, x2]
    volumes : list
        Volume of each connected component
    """
    if mask.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    # Label connected components
    labeled, num_features = ndimage.label(mask > 0)

    boxes = []
    volumes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        indices = np.where(component)

        if len(indices[0]) == 0:
            continue

        z1, y1, x1 = indices[0].min(), indices[1].min(), indices[2].min()
        z2, y2, x2 = indices[0].max() + 1, indices[1].max() + 1, indices[2].max() + 1

        boxes.append([z1, y1, x1, z2, y2, x2])
        volumes.append(int(component.sum()))

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    return np.array(boxes, dtype=np.float32), volumes


def compute_iou_3d(box1, box2):
    """Compute IoU between two boxes in corner format [z1,y1,x1,z2,y2,x2]."""
    # Intersection
    z1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    z2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    x2 = min(box1[5], box2[5])

    inter_z = max(0, z2 - z1)
    inter_y = max(0, y2 - y1)
    inter_x = max(0, x2 - x1)
    intersection = inter_z * inter_y * inter_x

    # Volumes
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU matrix between all pred and gt boxes.

    Returns
    -------
    iou_matrix : np.ndarray (N_pred, N_gt)
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou_3d(pb, gb)

    return iou_matrix


def run_inference(model_path, data_dir, output_dir, device="cuda:0"):
    """Run SegMamba segmentation inference."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install torch to run inference.")

    from model_segmamba.segmamba import SegMamba

    os.makedirs(output_dir, exist_ok=True)

    # Load model
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
    print(f"Loaded model from {model_path}")

    # Find test files
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Processing {len(npz_files)} cases...")

    PATCH_SIZE = (128, 128, 128)

    for npz_path in tqdm(npz_files, desc="Inference"):
        data = np.load(npz_path)
        volume = data['data'].astype(np.float32)  # (1, D, H, W)

        pkl_path = npz_path.replace('.npz', '.pkl')
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)

        case_name = props.get('name', os.path.basename(npz_path).replace('.npz', ''))
        volume_shape = volume.shape[1:]

        # Sliding window inference with Gaussian weighting
        positions = _sliding_window_positions(volume_shape, PATCH_SIZE, overlap=0.5)

        # Create Gaussian weight
        sigma = [s / 4 for s in PATCH_SIZE]
        zz, yy, xx = np.mgrid[:PATCH_SIZE[0], :PATCH_SIZE[1], :PATCH_SIZE[2]]
        center = [s / 2 for s in PATCH_SIZE]
        gaussian = np.exp(-((zz - center[0])**2 / (2*sigma[0]**2) +
                            (yy - center[1])**2 / (2*sigma[1]**2) +
                            (xx - center[2])**2 / (2*sigma[2]**2)))

        output = np.zeros((2, *volume_shape), dtype=np.float32)
        weight_sum = np.zeros(volume_shape, dtype=np.float32)

        with torch.no_grad():
            for pos in positions:
                z, y, x = pos
                patch = volume[:, z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]

                # Pad if necessary
                if patch.shape[1:] != PATCH_SIZE:
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

    print(f"Saved predictions to {output_dir}")


def _sliding_window_positions(volume_shape, patch_size, overlap=0.5):
    """Generate sliding window positions."""
    positions = []
    stride = [int(p * (1 - overlap)) for p in patch_size]

    for z in range(0, max(1, volume_shape[0] - patch_size[0] + 1), stride[0]):
        for y in range(0, max(1, volume_shape[1] - patch_size[1] + 1), stride[1]):
            for x in range(0, max(1, volume_shape[2] - patch_size[2] + 1), stride[2]):
                positions.append((z, y, x))

    if len(positions) == 0:
        positions.append((0, 0, 0))

    # Add corners
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


def evaluate_detections(pred_dir, abus_root, split="Test", min_volume=100):
    """Evaluate detection metrics from segmentation predictions.

    Parameters
    ----------
    pred_dir : str
        Directory with segmentation predictions (.nii.gz)
    abus_root : str
        Root directory of ABUS dataset
    split : str
        Dataset split (Train, Validation, Test)
    min_volume : int
        Minimum component volume to consider as detection

    Returns
    -------
    results : dict
        Evaluation metrics and per-case results
    """
    # Find prediction files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    if len(pred_files) == 0:
        print(f"No .nii.gz files found in {pred_dir}")
        return None

    gt_mask_dir = os.path.join(abus_root, "data", split, "MASK")
    if not os.path.exists(gt_mask_dir):
        print(f"GT mask directory not found: {gt_mask_dir}")
        return None

    print(f"Evaluating {len(pred_files)} predictions against GT masks...")

    all_detections = {}
    case_results = []

    # IoU thresholds for metrics
    iou_thresholds = [0.1, 0.25, 0.5, 0.75]

    for pf in tqdm(pred_files, desc="Evaluating"):
        case_name = pf.replace(".nii.gz", "")
        case_id = case_name.replace("ABUS_", "")

        # Load prediction
        pred_path = os.path.join(pred_dir, pf)
        pred_itk = sitk.ReadImage(pred_path)
        pred_mask = sitk.GetArrayFromImage(pred_itk).astype(np.uint8)

        # Extract predicted boxes
        pred_boxes, pred_volumes = extract_boxes_from_mask(pred_mask)

        # Filter small components
        if min_volume > 0 and len(pred_boxes) > 0:
            keep = [i for i, v in enumerate(pred_volumes) if v >= min_volume]
            if len(keep) > 0:
                pred_boxes = pred_boxes[keep]
                pred_volumes = [pred_volumes[i] for i in keep]
            else:
                pred_boxes = np.zeros((0, 6), dtype=np.float32)
                pred_volumes = []

        # Compute confidence scores based on volume
        pred_scores = []
        for vol in pred_volumes:
            score = min(1.0, vol / 10000.0)
            pred_scores.append(float(max(score, 0.01)))

        # Load GT mask
        gt_path = os.path.join(gt_mask_dir, f"MASK_{case_id}.nrrd")
        if not os.path.exists(gt_path):
            print(f"  Warning: GT not found for {case_name}, skipping")
            continue

        gt_itk = sitk.ReadImage(gt_path)
        gt_mask = sitk.GetArrayFromImage(gt_itk).astype(np.uint8)

        # Extract GT boxes
        gt_boxes, gt_volumes = extract_boxes_from_mask(gt_mask)

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Best IoU for each GT box
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            best_iou_per_gt = iou_matrix.max(axis=0)  # (N_gt,)
            best_iou_per_pred = iou_matrix.max(axis=1)  # (N_pred,)
            overall_best_iou = iou_matrix.max()
        else:
            best_iou_per_gt = np.zeros(len(gt_boxes))
            best_iou_per_pred = np.zeros(len(pred_boxes))
            overall_best_iou = 0.0

        # Mean IoU across GT boxes
        mean_gt_iou = float(best_iou_per_gt.mean()) if len(gt_boxes) > 0 else 0.0

        # Detection rates at different thresholds
        detected_at = {}
        for t in iou_thresholds:
            if len(gt_boxes) > 0:
                detected_at[t] = int((best_iou_per_gt >= t).sum())
            else:
                detected_at[t] = 0

        # Store results
        case_results.append({
            'case_id': case_id,
            'case_name': case_name,
            'n_pred': len(pred_boxes),
            'n_gt': len(gt_boxes),
            'best_iou': float(overall_best_iou),
            'mean_gt_iou': mean_gt_iou,
            'detected_at': detected_at,
            'pred_boxes': pred_boxes.tolist(),
            'gt_boxes': gt_boxes.tolist(),
        })

        # Store for JSON output
        all_detections[case_id] = {
            "boxes": pred_boxes.tolist(),
            "scores": pred_scores,
        }

    # Aggregate metrics
    total_gt = sum(r['n_gt'] for r in case_results)
    total_pred = sum(r['n_pred'] for r in case_results)

    # Detection rate (at least one GT detected per case)
    detection_rates = {}
    for t in iou_thresholds:
        detected_cases = sum(1 for r in case_results if r['detected_at'][t] > 0 and r['n_gt'] > 0)
        cases_with_gt = sum(1 for r in case_results if r['n_gt'] > 0)
        detection_rates[t] = detected_cases / max(cases_with_gt, 1)

    # Recall (fraction of GT boxes detected)
    recall_at = {}
    for t in iou_thresholds:
        detected_gt = sum(r['detected_at'][t] for r in case_results)
        recall_at[t] = detected_gt / max(total_gt, 1)

    # Mean best IoU
    mean_best_iou = np.mean([r['best_iou'] for r in case_results if r['n_gt'] > 0]) if case_results else 0.0
    mean_gt_iou_all = np.mean([r['mean_gt_iou'] for r in case_results if r['n_gt'] > 0]) if case_results else 0.0

    results = {
        'summary': {
            'n_cases': len(case_results),
            'total_pred_boxes': total_pred,
            'total_gt_boxes': total_gt,
            'mean_best_iou': float(mean_best_iou),
            'mean_gt_iou': float(mean_gt_iou_all),
        },
        'detection_rate': {str(t): detection_rates[t] for t in iou_thresholds},
        'recall': {str(t): recall_at[t] for t in iou_thresholds},
        'per_case': case_results,
    }

    return results, all_detections


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection via SegMamba segmentation")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to trained SegMamba checkpoint (for inference)")
    parser.add_argument("--data_dir", type=str, default="./data/abus/test",
                        help="Directory with preprocessed test data (for inference)")
    parser.add_argument("--pred_dir", type=str,
                        default="./prediction_results/segmamba_abus",
                        help="Directory with segmentation predictions (.nii.gz)")
    parser.add_argument("--abus_root", type=str, default="/Volumes/Autzoko/ABUS",
                        help="Root directory of ABUS dataset")
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Validation", "Test"])
    parser.add_argument("--run_inference", action="store_true",
                        help="Run segmentation inference first")
    parser.add_argument("--min_volume", type=int, default=100,
                        help="Minimum component volume to consider as detection")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Run inference if requested
    if args.run_inference:
        if not args.model_path:
            print("Error: --model_path required when using --run_inference")
            return
        run_inference(args.model_path, args.data_dir, args.pred_dir, args.device)

    # Evaluate
    results, detections = evaluate_detections(
        args.pred_dir, args.abus_root, args.split, args.min_volume)

    if results is None:
        return

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Segmentation-to-Box Detection Evaluation ({args.split} split)")
    print(f"{'='*70}")
    print(f"  Cases:           {results['summary']['n_cases']}")
    print(f"  Total Pred Boxes: {results['summary']['total_pred_boxes']}")
    print(f"  Total GT Boxes:   {results['summary']['total_gt_boxes']}")
    print(f"\n  Mean Best IoU:    {results['summary']['mean_best_iou']:.4f}")
    print(f"  Mean GT IoU:      {results['summary']['mean_gt_iou']:.4f}")
    print(f"\n  Detection Rate (case-level, at least one GT detected):")
    for t in ['0.1', '0.25', '0.5', '0.75']:
        print(f"    @IoU={t}:  {results['detection_rate'][t]:.4f}")
    print(f"\n  Recall (box-level, fraction of GT boxes detected):")
    for t in ['0.1', '0.25', '0.5', '0.75']:
        print(f"    @IoU={t}:  {results['recall'][t]:.4f}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(args.pred_dir, exist_ok=True)

    # Save detections JSON
    det_path = os.path.join(args.pred_dir, "detections.json")
    with open(det_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"\nSaved detections to: {det_path}")

    # Save metrics
    metrics_path = os.path.join(args.pred_dir, "seg_box_metrics.json")
    with open(metrics_path, 'w') as f:
        # Remove per-case boxes to reduce file size
        results_clean = {
            'summary': results['summary'],
            'detection_rate': results['detection_rate'],
            'recall': results['recall'],
            'per_case': [
                {k: v for k, v in r.items() if k not in ['pred_boxes', 'gt_boxes']}
                for r in results['per_case']
            ]
        }
        json.dump(results_clean, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Print per-case details for cases with low IoU
    print(f"\n  Per-case results (cases with best_IoU < 0.5):")
    low_iou_cases = [r for r in results['per_case'] if r['best_iou'] < 0.5 and r['n_gt'] > 0]
    for r in sorted(low_iou_cases, key=lambda x: x['best_iou'])[:10]:
        print(f"    {r['case_name']}: pred={r['n_pred']} gt={r['n_gt']} "
              f"best_IoU={r['best_iou']:.4f}")


if __name__ == "__main__":
    main()
