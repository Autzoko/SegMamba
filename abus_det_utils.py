"""
Detection utilities for SegMamba-Det ABUS pipeline.

Provides: focal loss, FCOS target assignment, prediction decoding,
3D NMS, and 3D IoU computation.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss (sum reduction).

    Parameters
    ----------
    pred   : (N,) logits
    target : (N,) binary labels {0, 1}
    """
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.sum()


# ---------------------------------------------------------------------------
# FCOS target assignment
# ---------------------------------------------------------------------------

def compute_fcos_targets(gt_boxes, strides, feature_sizes, input_size=128):
    """Compute FCOS targets for all FPN levels.

    Parameters
    ----------
    gt_boxes      : np.ndarray (N, 6)  [z1,y1,x1,z2,y2,x2] or empty (0,6)
    strides       : list of int, e.g. [2, 4, 8, 16]
    feature_sizes : list of (D,H,W) tuples
    input_size    : int (assumes cubic input)

    Returns
    -------
    list of dicts per level, each with keys 'cls', 'reg', 'ctr'
    """
    targets = []

    for stride, (D, H, W) in zip(strides, feature_sizes):
        cls_t = np.zeros((D, H, W), dtype=np.float32)
        reg_t = np.zeros((6, D, H, W), dtype=np.float32)
        ctr_t = np.zeros((D, H, W), dtype=np.float32)

        if len(gt_boxes) == 0:
            targets.append({'cls': cls_t, 'reg': reg_t, 'ctr': ctr_t})
            continue

        zz = np.arange(D) * stride + stride // 2
        yy = np.arange(H) * stride + stride // 2
        xx = np.arange(W) * stride + stride // 2
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')

        # Track best (smallest) box per voxel to resolve overlaps
        best_area = np.full((D, H, W), np.inf, dtype=np.float32)

        for box in gt_boxes:
            z1, y1, x1, z2, y2, x2 = box

            front  = Z - z1
            back   = z2 - Z
            top    = Y - y1
            bottom = y2 - Y
            left   = X - x1
            right  = x2 - X

            inside = (
                (front > 0) & (back > 0) &
                (top > 0) & (bottom > 0) &
                (left > 0) & (right > 0)
            )

            area = (z2 - z1) * (y2 - y1) * (x2 - x1)
            better = inside & (area < best_area)

            cls_t[better] = 1.0
            reg_t[0][better] = front[better]
            reg_t[1][better] = back[better]
            reg_t[2][better] = top[better]
            reg_t[3][better] = bottom[better]
            reg_t[4][better] = left[better]
            reg_t[5][better] = right[better]

            ctr = np.sqrt(
                (np.minimum(front, back) /
                 np.maximum(np.maximum(front, back), 1e-6)) *
                (np.minimum(top, bottom) /
                 np.maximum(np.maximum(top, bottom), 1e-6)) *
                (np.minimum(left, right) /
                 np.maximum(np.maximum(left, right), 1e-6))
            )
            ctr_t[better] = ctr[better]
            best_area[better] = area

        targets.append({'cls': cls_t, 'reg': reg_t, 'ctr': ctr_t})

    return targets


# ---------------------------------------------------------------------------
# Decoding predictions -> boxes
# ---------------------------------------------------------------------------

def decode_detections(all_cls, all_reg, all_ctr, strides, input_size=128,
                      score_thresh=0.05, topk_per_level=100):
    """Decode FCOS outputs into boxes + scores for a single sample.

    Parameters
    ----------
    all_cls : list of tensors (1, 1, D, H, W)  — raw logits
    all_reg : list of tensors (1, 6, D, H, W)  — positive distances
    all_ctr : list of tensors (1, 1, D, H, W)  — raw logits

    Returns
    -------
    boxes  : np.ndarray (K, 6)
    scores : np.ndarray (K,)
    """
    all_boxes, all_scores = [], []

    for cls_pred, reg_pred, ctr_pred, stride in zip(
            all_cls, all_reg, all_ctr, strides):
        cls_pred = cls_pred[0, 0]   # (D, H, W)
        reg_pred = reg_pred[0]      # (6, D, H, W)
        ctr_pred = ctr_pred[0, 0]   # (D, H, W)

        score = (torch.sigmoid(cls_pred) *
                 torch.sigmoid(ctr_pred)).cpu().numpy()
        reg_np = reg_pred.cpu().numpy()

        D, H, W = score.shape
        zz = np.arange(D) * stride + stride // 2
        yy = np.arange(H) * stride + stride // 2
        xx = np.arange(W) * stride + stride // 2
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')

        # Flatten
        score_flat = score.ravel()
        mask = score_flat > score_thresh
        if mask.sum() == 0:
            continue

        # Top-k
        indices = np.where(mask)[0]
        if len(indices) > topk_per_level:
            topk_idx = np.argpartition(
                score_flat[indices], -topk_per_level)[-topk_per_level:]
            indices = indices[topk_idx]

        s = score_flat[indices]
        reg_flat = reg_np.reshape(6, -1)[:, indices]
        Z_flat = Z.ravel()[indices]
        Y_flat = Y.ravel()[indices]
        X_flat = X.ravel()[indices]

        z1 = Z_flat - reg_flat[0]
        z2 = Z_flat + reg_flat[1]
        y1 = Y_flat - reg_flat[2]
        y2 = Y_flat + reg_flat[3]
        x1 = X_flat - reg_flat[4]
        x2 = X_flat + reg_flat[5]

        boxes = np.stack([z1, y1, x1, z2, y2, x2], axis=1)
        # Clip to input bounds
        boxes = np.clip(boxes, 0, input_size)

        all_boxes.append(boxes)
        all_scores.append(s)

    if len(all_boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return (np.concatenate(all_boxes).astype(np.float32),
            np.concatenate(all_scores).astype(np.float32))


# ---------------------------------------------------------------------------
# 3D IoU
# ---------------------------------------------------------------------------

def compute_iou_3d(boxes1, boxes2):
    """Compute 3D IoU matrix.

    Parameters
    ----------
    boxes1 : np.ndarray (N, 6)
    boxes2 : np.ndarray (M, 6)

    Returns
    -------
    iou : np.ndarray (N, M)
    """
    z1_1, y1_1, x1_1 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3]
    z2_1, y2_1, x2_1 = boxes1[:, 3:4], boxes1[:, 4:5], boxes1[:, 5:6]

    z1_2 = boxes2[:, 0:1].T
    y1_2 = boxes2[:, 1:2].T
    x1_2 = boxes2[:, 2:3].T
    z2_2 = boxes2[:, 3:4].T
    y2_2 = boxes2[:, 4:5].T
    x2_2 = boxes2[:, 5:6].T

    inter_z = np.maximum(0, np.minimum(z2_1, z2_2) - np.maximum(z1_1, z1_2))
    inter_y = np.maximum(0, np.minimum(y2_1, y2_2) - np.maximum(y1_1, y1_2))
    inter_x = np.maximum(0, np.minimum(x2_1, x2_2) - np.maximum(x1_1, x1_2))
    inter = inter_z * inter_y * inter_x

    vol1 = (z2_1 - z1_1) * (y2_1 - y1_1) * (x2_1 - x1_1)
    vol2 = (z2_2 - z1_2) * (y2_2 - y1_2) * (x2_2 - x1_2)
    iou = inter / np.maximum(vol1 + vol2 - inter, 1e-6)
    return iou


# ---------------------------------------------------------------------------
# 3D NMS
# ---------------------------------------------------------------------------

def nms_3d(boxes, scores, iou_threshold=0.3):
    """Greedy 3D NMS.

    Parameters
    ----------
    boxes : np.ndarray (N, 6)
    scores : np.ndarray (N,)

    Returns
    -------
    keep : list of int indices
    """
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1].copy()
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = compute_iou_3d(boxes[i:i + 1], boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

def compute_ap_for_dataset(all_pred_boxes, all_pred_scores, all_gt_boxes,
                           iou_threshold=0.25):
    """Compute Average Precision across an entire dataset.

    Parameters
    ----------
    all_pred_boxes  : list of (K_i, 6) arrays per case
    all_pred_scores : list of (K_i,) arrays per case
    all_gt_boxes    : list of (M_i, 6) arrays per case

    Returns
    -------
    ap : float
    """
    # Collect all predictions with case index
    preds = []
    for case_idx, (boxes, scores) in enumerate(
            zip(all_pred_boxes, all_pred_scores)):
        for j in range(len(scores)):
            preds.append((scores[j], case_idx, j))

    # Sort by score descending
    preds.sort(key=lambda x: -x[0])

    # Track which GT boxes have been matched
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in all_gt_boxes]
    total_gt = sum(len(gt) for gt in all_gt_boxes)

    if total_gt == 0:
        return 0.0

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for k, (score, case_idx, box_idx) in enumerate(preds):
        pred_box = all_pred_boxes[case_idx][box_idx:box_idx + 1]
        gt = all_gt_boxes[case_idx]

        if len(gt) == 0:
            fp[k] = 1
            continue

        ious = compute_iou_3d(pred_box, gt)[0]
        best_gt = ious.argmax()
        best_iou = ious[best_gt]

        if best_iou >= iou_threshold and not gt_matched[case_idx][best_gt]:
            tp[k] = 1
            gt_matched[case_idx][best_gt] = True
        else:
            fp[k] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # AP via 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11.0

    return ap


# ---------------------------------------------------------------------------
# 3D GIoU (single pair, numpy)
# ---------------------------------------------------------------------------

def compute_giou_3d_single(box1, box2):
    """Compute Generalized IoU between two single 3D boxes.

    Parameters
    ----------
    box1, box2 : np.ndarray (6,)  [z1, y1, x1, z2, y2, x2]

    Returns
    -------
    giou : float in [-1, 1]
    """
    inter_z = max(0, min(box1[3], box2[3]) - max(box1[0], box2[0]))
    inter_y = max(0, min(box1[4], box2[4]) - max(box1[1], box2[1]))
    inter_x = max(0, min(box1[5], box2[5]) - max(box1[2], box2[2]))
    inter = inter_z * inter_y * inter_x

    vol1 = max(0, box1[3] - box1[0]) * max(0, box1[4] - box1[1]) * max(0, box1[5] - box1[2])
    vol2 = max(0, box2[3] - box2[0]) * max(0, box2[4] - box2[1]) * max(0, box2[5] - box2[2])
    union = vol1 + vol2 - inter
    iou = inter / max(union, 1e-6)

    enc_z = max(box1[3], box2[3]) - min(box1[0], box2[0])
    enc_y = max(box1[4], box2[4]) - min(box1[1], box2[1])
    enc_x = max(box1[5], box2[5]) - min(box1[2], box2[2])
    enc_vol = enc_z * enc_y * enc_x

    return iou - (enc_vol - union) / max(enc_vol, 1e-6)


# ---------------------------------------------------------------------------
# Comprehensive detection metrics
# ---------------------------------------------------------------------------

def compute_detection_metrics(all_pred_boxes, all_pred_scores, all_gt_boxes):
    """Compute comprehensive detection metrics across the dataset.

    Returns
    -------
    dict with keys:
        AP@0.1, AP@0.25, AP@0.5, mAP,
        recall@0.1, recall@0.25, recall@0.5,
        mean_best_iou, mean_giou
    """
    metrics = {}

    # AP at multiple IoU thresholds
    for thresh in [0.1, 0.25, 0.5]:
        metrics[f'AP@{thresh}'] = compute_ap_for_dataset(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            iou_threshold=thresh)

    metrics['mAP'] = np.mean([metrics['AP@0.1'], metrics['AP@0.25'],
                              metrics['AP@0.5']])

    # Per-GT-box: recall, best IoU, GIoU with best-matching prediction
    total_gt = sum(len(gt) for gt in all_gt_boxes)
    matched = {0.1: 0, 0.25: 0, 0.5: 0}
    best_ious = []
    giou_values = []

    for pred_boxes, gt_boxes in zip(all_pred_boxes, all_gt_boxes):
        if len(gt_boxes) == 0:
            continue
        if len(pred_boxes) == 0:
            best_ious.extend([0.0] * len(gt_boxes))
            giou_values.extend([-1.0] * len(gt_boxes))
            continue

        ious = compute_iou_3d(pred_boxes, gt_boxes)      # (K, M)

        for j in range(len(gt_boxes)):
            best_pred = ious[:, j].argmax()
            best_iou = float(ious[best_pred, j])
            best_ious.append(best_iou)

            giou = compute_giou_3d_single(
                pred_boxes[best_pred], gt_boxes[j])
            giou_values.append(giou)

            for t in [0.1, 0.25, 0.5]:
                if best_iou >= t:
                    matched[t] += 1

    for t in [0.1, 0.25, 0.5]:
        metrics[f'recall@{t}'] = matched[t] / max(total_gt, 1)

    metrics['mean_best_iou'] = float(np.mean(best_ious)) if best_ious else 0.0
    metrics['mean_giou'] = float(np.mean(giou_values)) if giou_values else 0.0

    return metrics
