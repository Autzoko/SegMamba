"""
Training script for SegMamba-Retina (anchor-based 3D detection).

Features:
- Multi-task training: segmentation + detection
- Biased patch sampling: 50% patches with tumors
- Detection warmup: gradually increase detection loss weight
- Hard negative mining for classification balance

Usage:
    python abus_retina_train.py \
        --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
        --max_epoch 500 \
        --batch_size 2 \
        --fg_ratio 0.5
"""

import os
import sys
import argparse
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import glob

from monai.losses import DiceLoss

from model_segmamba.segmamba_retina import (
    SegMambaWithRetina,
    load_pretrained_segmamba,
)
from detection.atss_matcher import ATSSMatcher
from detection.sampler import HardNegativeSampler
from detection.losses import focal_loss, smooth_l1_loss, giou_loss_3d
from detection.box_coder import BoxCoder3D
from detection.retina_head import reshape_head_outputs

from light_training.utils.lr_scheduler import PolyLRScheduler


PATCH_SIZE = (128, 128, 128)


class ABUSRetinaDataset(Dataset):
    """
    Dataset for ABUS detection training with biased patch sampling.

    Samples patches with foreground (tumor) objects more frequently
    to address class imbalance.
    """

    def __init__(
        self,
        data_dir: str,
        fg_ratio: float = 0.5,
        patch_size: tuple = PATCH_SIZE,
        random_offset: int = 32,
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npz files
            fg_ratio: Fraction of patches that must contain tumor
            patch_size: Size of extracted patches
            random_offset: Random offset for foreground-centered patches
        """
        self.data_dir = data_dir
        self.fg_ratio = fg_ratio
        self.patch_size = patch_size
        self.random_offset = random_offset

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        print(f"Found {len(self.files)} training cases in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _get_boxes_from_mask(self, mask: np.ndarray) -> list:
        """Extract bounding boxes from segmentation mask."""
        from scipy import ndimage

        # Label connected components
        labeled, num_features = ndimage.label(mask > 0)

        boxes = []
        for i in range(1, num_features + 1):
            coords = np.where(labeled == i)
            if len(coords[0]) == 0:
                continue

            z1, z2 = coords[0].min(), coords[0].max() + 1
            y1, y2 = coords[1].min(), coords[1].max() + 1
            x1, x2 = coords[2].min(), coords[2].max() + 1

            # Box format: [x1, y1, x2, y2, z1, z2]
            boxes.append([x1, y1, x2, y2, z1, z2])

        return boxes

    def _sample_patch_position(
        self,
        volume_shape: tuple,
        gt_boxes: list,
    ) -> tuple:
        """Sample a patch position, biased toward foreground."""
        d, h, w = volume_shape
        pd, ph, pw = self.patch_size

        # Decide whether to sample foreground or background
        sample_fg = random.random() < self.fg_ratio and len(gt_boxes) > 0

        if sample_fg:
            # Sample around a random GT box
            box = random.choice(gt_boxes)
            x1, y1, x2, y2, z1, z2 = box

            # Center of box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cz = (z1 + z2) // 2

            # Add random offset
            offset = self.random_offset
            cx += random.randint(-offset, offset)
            cy += random.randint(-offset, offset)
            cz += random.randint(-offset, offset)

            # Compute start position (center patch on the box center)
            start_x = cx - pw // 2
            start_y = cy - ph // 2
            start_z = cz - pd // 2
        else:
            # Random position
            start_z = random.randint(0, max(0, d - pd))
            start_y = random.randint(0, max(0, h - ph))
            start_x = random.randint(0, max(0, w - pw))

        # Clip to valid range
        start_z = max(0, min(start_z, d - pd))
        start_y = max(0, min(start_y, h - ph))
        start_x = max(0, min(start_x, w - pw))

        return start_z, start_y, start_x

    def _extract_patch(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        start_pos: tuple,
    ) -> tuple:
        """Extract patch from volume and mask."""
        z, y, x = start_pos
        pd, ph, pw = self.patch_size

        patch_vol = volume[:, z:z+pd, y:y+ph, x:x+pw]
        patch_mask = mask[z:z+pd, y:y+ph, x:x+pw]

        # Pad if necessary
        if patch_vol.shape[1:] != self.patch_size:
            pad_d = pd - patch_vol.shape[1]
            pad_h = ph - patch_vol.shape[2]
            pad_w = pw - patch_vol.shape[3]
            patch_vol = np.pad(
                patch_vol,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant'
            )
            patch_mask = np.pad(
                patch_mask,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant'
            )

        return patch_vol, patch_mask

    def _transform_boxes_to_patch(
        self,
        gt_boxes: list,
        start_pos: tuple,
    ) -> list:
        """Transform global boxes to patch coordinates."""
        z, y, x = start_pos
        pd, ph, pw = self.patch_size

        patch_boxes = []
        for box in gt_boxes:
            x1, y1, x2, y2, z1, z2 = box

            # Transform to patch coordinates
            px1 = x1 - x
            py1 = y1 - y
            pz1 = z1 - z
            px2 = x2 - x
            py2 = y2 - y
            pz2 = z2 - z

            # Check if box overlaps with patch
            if (px2 <= 0 or py2 <= 0 or pz2 <= 0 or
                px1 >= pw or py1 >= ph or pz1 >= pd):
                continue  # No overlap

            # Clip to patch bounds
            px1 = max(0, px1)
            py1 = max(0, py1)
            pz1 = max(0, pz1)
            px2 = min(pw, px2)
            py2 = min(ph, py2)
            pz2 = min(pd, pz2)

            # Check minimum size
            if px2 - px1 > 2 and py2 - py1 > 2 and pz2 - pz1 > 2:
                patch_boxes.append([px1, py1, px2, py2, pz1, pz2])

        return patch_boxes

    def __getitem__(self, idx):
        # Load volume and mask
        npz_path = self.files[idx]
        data = np.load(npz_path)
        volume = data['data'].astype(np.float32)  # (1, D, H, W)
        mask = data['seg'].astype(np.float32)  # (1, D, H, W) or (D, H, W)

        # Handle both (1, D, H, W) and (D, H, W) formats
        if mask.ndim == 4:
            mask = mask[0]  # (D, H, W)

        volume_shape = mask.shape

        # Get GT boxes from mask
        gt_boxes = self._get_boxes_from_mask(mask)

        # Sample patch position
        start_pos = self._sample_patch_position(volume_shape, gt_boxes)

        # Extract patch
        patch_vol, patch_mask = self._extract_patch(volume, mask, start_pos)

        # Transform boxes to patch coordinates
        patch_boxes = self._transform_boxes_to_patch(gt_boxes, start_pos)

        # Convert to tensors
        patch_vol = torch.from_numpy(patch_vol)
        # Ensure mask is binary (0 or 1) before converting to long
        patch_mask = (patch_mask > 0).astype(np.int64)
        patch_mask = torch.from_numpy(patch_mask)

        # Pack boxes as tensor (pad to fixed size)
        max_boxes = 10
        if len(patch_boxes) > 0:
            boxes_tensor = torch.tensor(patch_boxes, dtype=torch.float32)
            if len(patch_boxes) > max_boxes:
                boxes_tensor = boxes_tensor[:max_boxes]
            elif len(patch_boxes) < max_boxes:
                padding = torch.zeros(max_boxes - len(patch_boxes), 6)
                boxes_tensor = torch.cat([boxes_tensor, padding], dim=0)
            num_boxes = min(len(patch_boxes), max_boxes)
        else:
            boxes_tensor = torch.zeros(max_boxes, 6)
            num_boxes = 0

        return {
            'image': patch_vol,
            'mask': patch_mask,
            'boxes': boxes_tensor,
            'num_boxes': num_boxes,
        }


def compute_detection_loss(
    cls_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    centerness: torch.Tensor,
    anchors: torch.Tensor,
    num_anchors_per_level: list,
    gt_boxes: torch.Tensor,
    num_boxes: torch.Tensor,
    matcher: ATSSMatcher,
    sampler: HardNegativeSampler,
    box_coder: BoxCoder3D,
    device: torch.device,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    l1_weight: float = 1.0,
    giou_weight: float = 2.0,
) -> tuple:
    """
    Compute detection losses for a batch.

    Returns:
        cls_loss: Classification loss
        reg_loss: Box regression loss (L1 + GIoU)
        num_pos: Number of positive samples
    """
    batch_size = gt_boxes.shape[0]
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_num_pos = 0

    # Process each sample in batch
    for b in range(batch_size):
        # Get GT boxes for this sample
        n_boxes = int(num_boxes[b].item())
        if n_boxes > 0:
            sample_gt_boxes = gt_boxes[b, :n_boxes]  # (n_boxes, 6)
        else:
            sample_gt_boxes = torch.zeros(0, 6, device=device)

        # Match anchors to GT
        matched_gt_idx, matched_iou, labels = matcher(
            sample_gt_boxes, anchors, num_anchors_per_level
        )

        # Sample positives and negatives
        cls_scores = torch.sigmoid(cls_logits[b].view(-1))
        pos_mask, neg_mask = sampler(labels, cls_scores)

        num_pos = pos_mask.sum().item()
        total_num_pos += num_pos

        # Classification loss
        sample_mask = pos_mask | neg_mask
        if sample_mask.sum() > 0:
            cls_loss = focal_loss(
                cls_logits[b].view(-1)[sample_mask],
                labels[sample_mask].float(),
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='sum'
            )
            total_cls_loss += cls_loss

        # Regression loss (on positives only)
        if num_pos > 0:
            pos_anchors = anchors[pos_mask]
            pos_deltas = box_deltas[b][pos_mask]
            pos_gt_idx = matched_gt_idx[pos_mask]
            pos_gt_boxes = sample_gt_boxes[pos_gt_idx]

            # Encode GT boxes
            target_deltas = box_coder.encode(pos_gt_boxes, pos_anchors)

            # Smooth L1 loss
            l1_loss = smooth_l1_loss(pos_deltas, target_deltas, reduction='sum')

            # GIoU loss (decode predictions first)
            pred_boxes = box_coder.decode(pos_deltas, pos_anchors)
            giou_loss = giou_loss_3d(pred_boxes, pos_gt_boxes, reduction='sum')

            total_reg_loss += l1_weight * l1_loss + giou_weight * giou_loss

    # Normalize by total positives
    num_pos_norm = max(total_num_pos, 1)
    total_cls_loss = total_cls_loss / num_pos_norm
    total_reg_loss = total_reg_loss / num_pos_norm

    return total_cls_loss, total_reg_loss, total_num_pos


def train_epoch(
    model: SegMambaWithRetina,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    matcher: ATSSMatcher,
    sampler: HardNegativeSampler,
    box_coder: BoxCoder3D,
    dice_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    det_warmup_epochs: int = 50,
    seg_weight: float = 1.0,
    det_weight: float = 1.0,
):
    """Train for one epoch."""
    model.train()

    # Detection loss warmup
    det_scale = min(1.0, epoch / det_warmup_epochs) if det_warmup_epochs > 0 else 1.0

    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_dice = 0.0
    total_pos = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        gt_boxes = batch['boxes'].to(device)
        num_boxes = batch['num_boxes'].to(device)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            outputs = model(images, return_seg=True, return_det=True)

            # Segmentation loss
            seg_logits = outputs['seg_logits']
            ce_loss = F.cross_entropy(seg_logits, masks)
            dice_loss = dice_loss_fn(seg_logits, masks.unsqueeze(1))
            seg_loss = ce_loss + dice_loss

            # Reshape detection outputs
            cls_flat, box_flat, ctr_flat, num_per_level = reshape_head_outputs(
                outputs['cls_logits'],
                outputs['box_deltas'],
                outputs['centerness'],
                model.num_anchors,
                model.num_classes,
            )
            anchors = outputs['anchors']

            # Detection loss
            cls_loss, reg_loss, num_pos = compute_detection_loss(
                cls_flat, box_flat, ctr_flat,
                anchors, num_per_level,
                gt_boxes, num_boxes,
                matcher, sampler, box_coder,
                device,
            )

            # Total loss with warmup
            det_loss = cls_loss + reg_loss
            loss = seg_weight * seg_loss + det_scale * det_weight * det_loss

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Logging
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        total_dice += (1 - dice_loss.item())
        total_pos += num_pos

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}',
            'cls': f'{cls_loss:.4f}' if isinstance(cls_loss, (int, float)) else f'{cls_loss.item():.4f}',
            'pos': num_pos,
        })

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'seg_loss': total_seg_loss / n,
        'cls_loss': total_cls_loss / n,
        'reg_loss': total_reg_loss / n,
        'dice': total_dice / n,
        'avg_pos': total_pos / n,
    }


def compute_iou_3d(box1, box2):
    """Compute 3D IoU between two boxes [x1, y1, x2, y2, z1, z2]."""
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    z1 = max(box1[4], box2[4])
    z2 = min(box1[5], box2[5])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_d = max(0, z2 - z1)
    inter_vol = inter_w * inter_h * inter_d

    # Volumes
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])

    union = vol1 + vol2 - inter_vol
    return inter_vol / (union + 1e-6)


def compute_ap_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.25):
    """
    Compute AP and Recall for a single sample.

    Args:
        pred_boxes: list of [x1, y1, x2, y2, z1, z2]
        pred_scores: list of scores
        gt_boxes: list of [x1, y1, x2, y2, z1, z2]
        iou_threshold: IoU threshold for matching

    Returns:
        ap: Average precision (simplified)
        recall: Recall (TP / num_gt)
        num_tp: True positives
        num_fp: False positives
        num_gt: Number of GT boxes
    """
    if len(gt_boxes) == 0:
        # No GT boxes - all predictions are FP
        return 0.0, 1.0, 0, len(pred_boxes), 0

    if len(pred_boxes) == 0:
        # No predictions - recall is 0
        return 0.0, 0.0, 0, 0, len(gt_boxes)

    # Sort predictions by score (descending)
    sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    precisions = []
    recalls = []

    for idx in sorted_indices:
        pred_box = pred_boxes[idx]

        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(gt_boxes)
        precisions.append(precision)
        recalls.append(recall)

    # Compute AP (area under precision-recall curve, simplified)
    if len(precisions) > 0:
        # Use all-point interpolation
        ap = 0.0
        for i in range(len(precisions)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i-1])
    else:
        ap = 0.0

    final_recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 1.0

    return ap, final_recall, tp, fp, len(gt_boxes)


def validate(
    model: SegMambaWithRetina,
    dataloader: DataLoader,
    dice_loss_fn: nn.Module,
    box_coder: BoxCoder3D,
    device: torch.device,
    score_threshold: float = 0.05,
    debug: bool = False,
):
    """Validate the model with both segmentation and detection metrics."""
    model.eval()

    total_dice = 0.0
    total_seg_loss = 0.0
    debug_printed = False

    # Detection metrics accumulators
    total_tp = {0.1: 0, 0.25: 0, 0.5: 0}
    total_fp = {0.1: 0, 0.25: 0, 0.5: 0}
    total_gt = 0
    total_ap = {0.1: 0.0, 0.25: 0.0, 0.5: 0.0}
    num_samples_with_gt = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            gt_boxes_batch = batch['boxes']  # (B, max_boxes, 6)
            num_boxes_batch = batch['num_boxes']  # (B,)

            with autocast():
                outputs = model(images, return_seg=True, return_det=True)
                seg_logits = outputs['seg_logits']

                ce_loss = F.cross_entropy(seg_logits, masks)
                dice_loss = dice_loss_fn(seg_logits, masks.unsqueeze(1))
                seg_loss = ce_loss + dice_loss

                # Compute Dice score
                pred = torch.argmax(seg_logits, dim=1)
                intersection = ((pred == 1) & (masks == 1)).sum()
                union = (pred == 1).sum() + (masks == 1).sum()
                dice = 2 * intersection / (union + 1e-6)

            # Debug: print first batch statistics
            if debug and not debug_printed:
                print(f"\n  DEBUG Validation:")
                print(f"    Input: shape={images.shape}, min={images.min():.3f}, max={images.max():.3f}")
                print(f"    Mask: shape={masks.shape}, unique={torch.unique(masks).tolist()}, sum={masks.sum().item()}")
                print(f"    Logits: shape={seg_logits.shape}, ch0 mean={seg_logits[:,0].mean():.3f}, ch1 mean={seg_logits[:,1].mean():.3f}")
                print(f"    Pred==1: {(pred == 1).sum().item()}, GT==1: {(masks == 1).sum().item()}")
                print(f"    Intersection: {intersection.item()}, Dice: {dice.item():.4f}")
                debug_printed = True

            total_seg_loss += seg_loss.item()
            total_dice += dice.item()

            # Detection evaluation
            cls_flat, box_flat, ctr_flat, num_per_level = reshape_head_outputs(
                outputs['cls_logits'],
                outputs['box_deltas'],
                outputs['centerness'],
                model.num_anchors,
                model.num_classes,
            )
            anchors = outputs['anchors']

            batch_size = images.shape[0]
            for b in range(batch_size):
                # Get predictions for this sample
                cls_scores = torch.sigmoid(cls_flat[b].view(-1))
                ctr_scores = torch.sigmoid(ctr_flat[b].view(-1))
                scores = cls_scores * ctr_scores

                # Filter by score
                keep = scores > score_threshold
                if keep.sum() > 0:
                    kept_scores = scores[keep].cpu().numpy()
                    kept_deltas = box_flat[b][keep]
                    kept_anchors = anchors[keep]

                    # Decode boxes
                    pred_boxes = box_coder.decode(kept_deltas, kept_anchors).cpu().numpy()
                    pred_boxes = pred_boxes.tolist()
                    pred_scores = kept_scores.tolist()
                else:
                    pred_boxes = []
                    pred_scores = []

                # Get GT boxes for this sample
                n_gt = int(num_boxes_batch[b].item())
                if n_gt > 0:
                    gt_boxes = gt_boxes_batch[b, :n_gt].numpy().tolist()
                    total_gt += n_gt
                    num_samples_with_gt += 1
                else:
                    gt_boxes = []

                # Compute metrics at different IoU thresholds
                for iou_thresh in [0.1, 0.25, 0.5]:
                    ap, recall, tp, fp, _ = compute_ap_recall(
                        pred_boxes, pred_scores, gt_boxes, iou_thresh
                    )
                    total_tp[iou_thresh] += tp
                    total_fp[iou_thresh] += fp
                    if n_gt > 0:
                        total_ap[iou_thresh] += ap

    n = len(dataloader)

    # Compute final detection metrics
    det_metrics = {}
    for iou_thresh in [0.1, 0.25, 0.5]:
        recall = total_tp[iou_thresh] / max(total_gt, 1)
        precision = total_tp[iou_thresh] / max(total_tp[iou_thresh] + total_fp[iou_thresh], 1)
        avg_ap = total_ap[iou_thresh] / max(num_samples_with_gt, 1)

        det_metrics[f'recall@{iou_thresh}'] = recall
        det_metrics[f'precision@{iou_thresh}'] = precision
        det_metrics[f'ap@{iou_thresh}'] = avg_ap

    return {
        'seg_loss': total_seg_loss / n,
        'dice': total_dice / n,
        **det_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SegMamba-Retina")

    # Data
    parser.add_argument("--data_dir_train", type=str, default="./data/abus/train")
    parser.add_argument("--data_dir_val", type=str, default="./data/abus/val")

    # Model
    parser.add_argument("--pretrained_seg", type=str, default="",
                        help="Path to pretrained SegMamba checkpoint")
    parser.add_argument("--fpn_channels", type=int, default=128)
    parser.add_argument("--num_head_convs", type=int, default=4)

    # Training
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Biased sampling
    parser.add_argument("--fg_ratio", type=float, default=0.5,
                        help="Fraction of patches with foreground")

    # Loss weights
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--det_weight", type=float, default=1.0)
    parser.add_argument("--det_warmup_epochs", type=int, default=50)

    # Detection hyperparameters
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--atss_candidates", type=int, default=9)
    parser.add_argument("--sampler_batch_size", type=int, default=256)
    parser.add_argument("--sampler_pos_fraction", type=float, default=0.25)

    # Misc
    parser.add_argument("--save_dir", type=str, default="./logs/segmamba_abus_retina")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.save_dir, "model"), exist_ok=True)

    # Create datasets
    train_dataset = ABUSRetinaDataset(
        args.data_dir_train,
        fg_ratio=args.fg_ratio,
    )
    val_dataset = ABUSRetinaDataset(
        args.data_dir_val,
        fg_ratio=0.5,  # Same as training to get meaningful detection metrics
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = SegMambaWithRetina(
        in_chans=1,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        fpn_channels=args.fpn_channels,
        num_head_convs=args.num_head_convs,
    ).to(device)

    # Load pretrained segmentation weights
    if args.pretrained_seg and os.path.exists(args.pretrained_seg):
        model = load_pretrained_segmamba(model, args.pretrained_seg)

    # Detection components
    matcher = ATSSMatcher(
        num_candidates=args.atss_candidates,
        center_in_gt=True,
    )
    sampler_det = HardNegativeSampler(
        batch_size=args.sampler_batch_size,
        positive_fraction=args.sampler_pos_fraction,
    )
    box_coder = BoxCoder3D()

    # Loss functions
    dice_loss_fn = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = PolyLRScheduler(
        optimizer,
        initial_lr=args.lr,
        max_steps=args.max_epoch,
    )

    scaler = GradScaler()

    # Training loop
    best_dice = 0.0

    for epoch in range(1, args.max_epoch + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            matcher, sampler_det, box_coder, dice_loss_fn,
            device, epoch,
            det_warmup_epochs=args.det_warmup_epochs,
            seg_weight=args.seg_weight,
            det_weight=args.det_weight,
        )

        scheduler.step()

        # Validation (debug=True for first epoch only)
        val_metrics = validate(model, val_loader, dice_loss_fn, box_coder, device,
                               debug=(epoch == 1))

        print(f"\nEpoch {epoch}/{args.max_epoch}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"seg={train_metrics['seg_loss']:.4f}, "
              f"cls={train_metrics['cls_loss']:.4f}, "
              f"reg={train_metrics['reg_loss']:.4f}, "
              f"dice={train_metrics['dice']:.4f}, "
              f"avg_pos={train_metrics['avg_pos']:.1f}")
        print(f"  Val: seg_loss={val_metrics['seg_loss']:.4f}, "
              f"dice={val_metrics['dice']:.4f}")
        print(f"  Det: recall@0.25={val_metrics['recall@0.25']:.4f}, "
              f"precision@0.25={val_metrics['precision@0.25']:.4f}, "
              f"AP@0.25={val_metrics['ap@0.25']:.4f}")
        print(f"       recall@0.5={val_metrics['recall@0.5']:.4f}, "
              f"precision@0.5={val_metrics['precision@0.5']:.4f}, "
              f"AP@0.5={val_metrics['ap@0.5']:.4f}")

        # Save best model (based on Dice)
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "model", f"best_model_dice{best_dice:.4f}.pt")
            )
            print(f"  Saved best model with Dice={best_dice:.4f}")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "model", f"checkpoint_epoch{epoch}.pt")
            )

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
