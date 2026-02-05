"""
SegMamba-BoxHead Training Script for ABUS 3D Multi-task Learning.

Multi-task: segmentation (Dice + CE) + attention-based box regression
(SmoothL1 + GIoU) from SegMamba decoder features.

Usage:
    python abus_boxhead_train.py --data_dir_train ./data/abus_boxhead/train \
                                  --data_dir_val ./data/abus_boxhead/val

Prerequisites:
    Run  abus_boxhead_preprocessing.py  first.
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from monai.losses.dice import DiceLoss

from model_segmamba.segmamba_boxhead import SegMambaWithBoxHead
from abus_detr_utils import (generalized_box_iou_3d, box_cxcyczdhwd_to_zyxzyx)
from abus_det_utils import compute_ap_for_dataset

INPUT_SIZE = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABUSBoxHeadDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files in {data_dir}. "
                "Run abus_boxhead_preprocessing.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        volume = d['data'].copy()           # (1, 128, 128, 128)
        seg = d['seg'].copy()               # (1, 128, 128, 128)
        boxes = d['boxes'].copy()           # (N, 6) [z1,y1,x1,z2,y2,x2]
        orig_shape = d['original_shape']

        if self.augment:
            volume, seg, boxes = self._augment(volume, seg, boxes)

        # Select target box: largest if >1, zeros if 0
        has_box = len(boxes) > 0
        if has_box:
            if len(boxes) > 1:
                vols = ((boxes[:, 3] - boxes[:, 0]) *
                        (boxes[:, 4] - boxes[:, 1]) *
                        (boxes[:, 5] - boxes[:, 2]))
                best_idx = vols.argmax()
                target_box = boxes[best_idx]
            else:
                target_box = boxes[0]

            # Convert to normalised [cz, cy, cx, dz, dy, dx]
            box_norm = np.array([
                (target_box[0] + target_box[3]) / 2.0 / INPUT_SIZE,
                (target_box[1] + target_box[4]) / 2.0 / INPUT_SIZE,
                (target_box[2] + target_box[5]) / 2.0 / INPUT_SIZE,
                (target_box[3] - target_box[0]) / INPUT_SIZE,
                (target_box[4] - target_box[1]) / INPUT_SIZE,
                (target_box[5] - target_box[2]) / INPUT_SIZE,
            ], dtype=np.float32)
        else:
            box_norm = np.zeros(6, dtype=np.float32)

        return {
            'data': torch.from_numpy(volume),
            'seg': torch.from_numpy(seg.astype(np.int64)),
            'box_target': torch.from_numpy(box_norm),
            'has_box': has_box,
            'boxes_corner': torch.from_numpy(boxes),
            'num_boxes': len(boxes),
            'name': os.path.basename(self.files[idx]).replace('.npz', ''),
            'original_shape': torch.from_numpy(orig_shape),
        }

    def _augment(self, volume, seg, boxes):
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis + 1).copy()
                seg = np.flip(seg, axis=axis + 1).copy()
                if len(boxes) > 0:
                    dim_size = volume.shape[axis + 1]
                    new_min = dim_size - boxes[:, axis + 3]
                    new_max = dim_size - boxes[:, axis]
                    boxes = boxes.copy()
                    boxes[:, axis] = new_min
                    boxes[:, axis + 3] = new_max

        if np.random.rand() > 0.5:
            volume = volume + np.random.normal(
                0, 0.1, volume.shape).astype(np.float32)
        if np.random.rand() > 0.5:
            volume = volume * np.random.uniform(0.9, 1.1)

        return volume, seg, boxes


def boxhead_collate_fn(batch):
    """Collate with variable-length boxes."""
    volumes = torch.stack([b['data'] for b in batch])
    segs = torch.stack([b['seg'] for b in batch])
    box_targets = torch.stack([b['box_target'] for b in batch])
    has_box = torch.tensor([b['has_box'] for b in batch], dtype=torch.bool)

    max_n = max(b['num_boxes'] for b in batch)
    max_n = max(max_n, 1)
    boxes_corner = torch.zeros(len(batch), max_n, 6)
    num_boxes = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        n = b['num_boxes']
        if n > 0:
            boxes_corner[i, :n] = b['boxes_corner']
        num_boxes[i] = n

    return {
        'data': volumes,
        'seg': segs,
        'box_target': box_targets,
        'has_box': has_box,
        'boxes_corner': boxes_corner,
        'num_boxes': num_boxes,
        'names': [b['name'] for b in batch],
        'original_shapes': torch.stack([b['original_shape'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_box_giou_loss(pred_box, target_box):
    """Compute 1 - GIoU for a single pair of normalised cxcycz boxes."""
    pred_corner = box_cxcyczdhwd_to_zyxzyx(pred_box.unsqueeze(0))
    tgt_corner = box_cxcyczdhwd_to_zyxzyx(target_box.unsqueeze(0))
    giou = generalized_box_iou_3d(pred_corner, tgt_corner)
    return 1 - giou[0, 0]


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler,
                    dice_loss_fn, ce_loss_fn, det_weight):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_det_loss = 0.0
    num_batches = 0

    for batch in loader:
        volume = batch['data'].to(device)
        seg_gt = batch['seg'][:, 0].to(device)          # (B, D, H, W) long
        box_target = batch['box_target'].to(device)      # (B, 6)
        has_box = batch['has_box'].to(device)            # (B,) bool

        optimizer.zero_grad()

        with autocast():
            seg_logits, box_pred = model(volume)

            # Seg loss: Dice + CE
            dice_l = dice_loss_fn(seg_logits, seg_gt.unsqueeze(1))
            ce_l = ce_loss_fn(seg_logits, seg_gt)
            seg_loss = dice_l + ce_l

            # Det loss: SmoothL1 + GIoU (only for samples with GT boxes)
            det_loss = torch.tensor(0.0, device=device)
            n_with_box = has_box.sum().item()

            if n_with_box > 0:
                pred_masked = box_pred[has_box]      # (K, 6)
                tgt_masked = box_target[has_box]     # (K, 6)

                l1_loss = F.smooth_l1_loss(pred_masked, tgt_masked)

                giou_loss = torch.tensor(0.0, device=device)
                for i in range(len(pred_masked)):
                    giou_loss = giou_loss + compute_box_giou_loss(
                        pred_masked[i], tgt_masked[i])
                giou_loss = giou_loss / n_with_box

                det_loss = l1_loss + giou_loss

            loss = seg_loss + det_weight * det_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_det_loss += det_loss.item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_seg_loss / n, total_det_loss / n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, iou_thresh=0.25):
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    dice_scores = []

    for batch in loader:
        volume = batch['data'].to(device)
        seg_gt = batch['seg'][:, 0].to(device)
        gt_corners = batch['boxes_corner'].numpy()
        gt_n = batch['num_boxes'].numpy()
        B = volume.shape[0]

        with autocast():
            seg_logits, box_pred = model(volume)

        # --- Seg dice ---
        seg_pred = seg_logits.argmax(dim=1)
        for b in range(B):
            pred_np = seg_pred[b].cpu().numpy()
            gt_np = seg_gt[b].cpu().numpy()
            if pred_np.sum() > 0 and gt_np.sum() > 0:
                intersection = (pred_np * gt_np).sum()
                d = 2.0 * intersection / (pred_np.sum() + gt_np.sum())
            elif pred_np.sum() == 0 and gt_np.sum() == 0:
                d = 1.0
            else:
                d = 0.0
            dice_scores.append(d)

        # --- Det AP ---
        for b in range(B):
            bp = box_pred[b].cpu()
            bp_corner = box_cxcyczdhwd_to_zyxzyx(bp.unsqueeze(0))
            bp_abs = (bp_corner * INPUT_SIZE).numpy()
            bp_abs = np.clip(bp_abs, 0, INPUT_SIZE)

            # Derive score from seg foreground fraction
            fg = float(seg_pred[b].sum().cpu())
            score = min(fg / (INPUT_SIZE ** 3) * 50.0, 1.0)
            score = max(score, 0.01)

            all_pred_boxes.append(bp_abs.astype(np.float32))
            all_pred_scores.append(np.array([score], dtype=np.float32))
            all_gt_boxes.append(gt_corners[b, :gt_n[b]])

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    ap = compute_ap_for_dataset(
        all_pred_boxes, all_pred_scores, all_gt_boxes,
        iou_threshold=iou_thresh)
    return mean_dice, ap


# ---------------------------------------------------------------------------
# Poly LR scheduler
# ---------------------------------------------------------------------------

class PolyLRScheduler:
    def __init__(self, optimizer, max_epochs, power=0.9):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        factor = (1 - epoch / self.max_epochs) ** self.power
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * factor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SegMamba-BoxHead on ABUS dataset")
    parser.add_argument("--data_dir_train", type=str,
                        default="./data/abus_boxhead/train")
    parser.add_argument("--data_dir_val", type=str,
                        default="./data/abus_boxhead/val")
    parser.add_argument("--logdir", type=str,
                        default="./logs/segmamba_abus_boxhead")
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--det_weight", type=float, default=5.0,
                        help="Weight for detection loss relative to seg loss")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_seg", type=str, default="",
                        help="Path to SegMamba segmentation checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_save_dir = os.path.join(args.logdir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Data
    train_ds = ABUSBoxHeadDataset(args.data_dir_train, augment=True)
    val_ds = ABUSBoxHeadDataset(args.data_dir_val, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=boxhead_collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=boxhead_collate_fn, pin_memory=True)

    # Model
    model = SegMambaWithBoxHead(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SegMamba-BoxHead parameters: {n_params:.2f}M")

    # Load pretrained segmentation weights if provided
    if args.pretrained_seg:
        sd = torch.load(args.pretrained_seg, map_location='cpu')
        if 'module' in sd:
            sd = sd['module']
        new_sd = {}
        for k, v in sd.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_sd[new_k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"  Loaded pretrained seg: {len(new_sd)} keys, "
              f"{len(missing)} missing (BoxHead), "
              f"{len(unexpected)} unexpected")

    # Optimiser: SGD with Poly LR (match segmentation pipeline)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        weight_decay=3e-5, momentum=0.99, nesterov=True)
    scheduler = PolyLRScheduler(optimizer, args.max_epoch)
    scaler = GradScaler()

    # Losses
    dice_loss_fn = DiceLoss(
        to_onehot_y=True, softmax=True, include_background=False)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_ap = 0.0

    for epoch in range(args.max_epoch):
        train_loss, seg_loss, det_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            dice_loss_fn, ce_loss_fn, args.det_weight)
        scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/seg_loss", seg_loss, epoch)
        writer.add_scalar("train/det_loss", det_loss, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"  [epoch {epoch:3d}]  loss={train_loss:.4f}  "
              f"seg={seg_loss:.4f}  det={det_loss:.4f}  lr={lr_now:.6f}")

        if (epoch + 1) % args.val_every == 0:
            mean_dice, ap25 = validate(model, val_loader, device)
            writer.add_scalar("val/dice", mean_dice, epoch)
            writer.add_scalar("val/AP@0.25", ap25, epoch)
            print(f"             val dice={mean_dice:.4f}  AP@0.25={ap25:.4f}")

            if ap25 > best_ap:
                best_ap = ap25
                path = os.path.join(
                    model_save_dir, f"best_model_{ap25:.4f}.pt")
                torch.save(model.state_dict(), path)
                for f in glob.glob(
                        os.path.join(model_save_dir, "best_model_*.pt")):
                    if f != path:
                        os.remove(f)
                print(f"             -> saved best model (AP={ap25:.4f})")

        torch.save(model.state_dict(),
                   os.path.join(model_save_dir, "latest.pt"))

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best AP@0.25 = {best_ap:.4f}")
    print(f"{'='*60}")
