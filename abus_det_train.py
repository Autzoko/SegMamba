"""
SegMamba-Det Training Script for ABUS 3D Bounding Box Detection.

Uses a custom training loop (independent of the segmentation Trainer)
with Focal Loss + Smooth-L1 regression + centerness BCE.

Usage:
    python abus_det_train.py --data_dir_train ./data/abus_det/train \
                             --data_dir_val ./data/abus_det/val

Prerequisites:
    Run  abus_det_preprocessing.py  first.
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

from model_segmamba.segmamba_det import SegMambaDet
from abus_det_utils import (sigmoid_focal_loss, compute_fcos_targets,
                            decode_detections, nms_3d, compute_ap_for_dataset)

INPUT_SIZE = 128
STRIDES = [2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABUSDetDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files in {data_dir}. "
                "Run abus_det_preprocessing.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        volume = d['data'].copy()          # (1, 128, 128, 128)
        boxes = d['boxes'].copy()           # (N, 6)
        orig_shape = d['original_shape']    # (3,)

        if self.augment:
            volume, boxes = self._augment(volume, boxes)

        return {
            'data': torch.from_numpy(volume),
            'boxes': torch.from_numpy(boxes),
            'num_boxes': len(boxes),
            'name': os.path.basename(self.files[idx]).replace('.npz', ''),
            'original_shape': torch.from_numpy(orig_shape),
        }

    def _augment(self, volume, boxes):
        # Random flip along each spatial axis
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis + 1).copy()
                if len(boxes) > 0:
                    dim_size = volume.shape[axis + 1]
                    new_min = dim_size - boxes[:, axis + 3]
                    new_max = dim_size - boxes[:, axis]
                    boxes = boxes.copy()
                    boxes[:, axis] = new_min
                    boxes[:, axis + 3] = new_max

        # Random intensity noise
        if np.random.rand() > 0.5:
            volume = volume + np.random.normal(0, 0.1,
                                               volume.shape).astype(np.float32)
        if np.random.rand() > 0.5:
            volume = volume * np.random.uniform(0.9, 1.1)

        return volume, boxes


def det_collate_fn(batch):
    """Collate with variable-length boxes (zero-padded)."""
    volumes = torch.stack([b['data'] for b in batch])
    max_n = max(b['num_boxes'] for b in batch)
    max_n = max(max_n, 1)

    boxes = torch.zeros(len(batch), max_n, 6)
    num_boxes = torch.zeros(len(batch), dtype=torch.long)
    names = []
    orig_shapes = []

    for i, b in enumerate(batch):
        n = b['num_boxes']
        if n > 0:
            boxes[i, :n] = b['boxes']
        num_boxes[i] = n
        names.append(b['name'])
        orig_shapes.append(b['original_shape'])

    return {
        'data': volumes,
        'boxes': boxes,
        'num_boxes': num_boxes,
        'names': names,
        'original_shapes': torch.stack(orig_shapes),
    }


# ---------------------------------------------------------------------------
# Feature sizes helper
# ---------------------------------------------------------------------------

def get_feature_sizes(input_size, strides):
    return [(input_size // s,) * 3 for s in strides]


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler, epoch):
    model.train()
    feat_sizes = get_feature_sizes(INPUT_SIZE, STRIDES)
    total_loss_sum = 0.0
    num_batches = 0

    for batch in loader:
        volume = batch['data'].to(device)
        gt_boxes = batch['boxes'].numpy()
        gt_n = batch['num_boxes'].numpy()
        B = volume.shape[0]

        optimizer.zero_grad()

        with autocast():
            all_cls, all_reg, all_ctr = model(volume)

            total_focal = torch.tensor(0.0, device=device)
            total_reg = torch.tensor(0.0, device=device)
            total_ctr = torch.tensor(0.0, device=device)
            num_pos = 0

            for b in range(B):
                boxes_b = gt_boxes[b, :gt_n[b]]
                targets = compute_fcos_targets(
                    boxes_b, STRIDES, feat_sizes, INPUT_SIZE)

                for lvl in range(len(STRIDES)):
                    cls_pred = all_cls[lvl][b, 0]
                    reg_pred = all_reg[lvl][b]
                    ctr_pred = all_ctr[lvl][b, 0]

                    cls_t = torch.from_numpy(
                        targets[lvl]['cls']).to(device)
                    reg_t = torch.from_numpy(
                        targets[lvl]['reg']).to(device)
                    ctr_t = torch.from_numpy(
                        targets[lvl]['ctr']).to(device)

                    # Focal loss on all voxels
                    total_focal += sigmoid_focal_loss(
                        cls_pred.reshape(-1),
                        cls_t.reshape(-1))

                    # Regression + centerness on positives only
                    pos = cls_t > 0
                    n_pos = pos.sum().item()
                    num_pos += n_pos

                    if n_pos > 0:
                        total_reg += F.smooth_l1_loss(
                            reg_pred[:, pos], reg_t[:, pos],
                            reduction='sum')
                        total_ctr += F.binary_cross_entropy_with_logits(
                            ctr_pred[pos], ctr_t[pos],
                            reduction='sum')

            num_pos = max(num_pos, 1)
            loss = (total_focal / max(B, 1)
                    + total_reg / num_pos
                    + total_ctr / num_pos)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item()
        num_batches += 1

    return total_loss_sum / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, iou_thresh=0.25):
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []

    for batch in loader:
        volume = batch['data'].to(device)
        gt = batch['boxes'].numpy()
        gt_n = batch['num_boxes'].numpy()
        B = volume.shape[0]

        with autocast():
            all_cls, all_reg, all_ctr = model(volume)

        for b in range(B):
            cls_b = [c[b:b + 1] for c in all_cls]
            reg_b = [r[b:b + 1] for r in all_reg]
            ctr_b = [c[b:b + 1] for c in all_ctr]

            boxes, scores = decode_detections(
                cls_b, reg_b, ctr_b, STRIDES, INPUT_SIZE,
                score_thresh=0.05)

            if len(boxes) > 0:
                keep = nms_3d(boxes, scores, iou_threshold=0.3)
                boxes = boxes[keep]
                scores = scores[keep]

            all_pred_boxes.append(boxes)
            all_pred_scores.append(scores)
            all_gt_boxes.append(gt[b, :gt_n[b]])

    ap = compute_ap_for_dataset(all_pred_boxes, all_pred_scores,
                                all_gt_boxes, iou_threshold=iou_thresh)
    return ap


# ---------------------------------------------------------------------------
# Pretrained backbone loading
# ---------------------------------------------------------------------------

def load_pretrained_backbone(model, ckpt_path):
    """Load MambaEncoder weights from a SegMamba segmentation checkpoint."""
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']

    backbone_sd = {}
    for k, v in sd.items():
        # SegMamba stores encoder as 'vit.*'
        clean = k[7:] if k.startswith('module.') else k
        if clean.startswith('vit.'):
            new_key = clean.replace('vit.', '', 1)
            backbone_sd[new_key] = v

    if len(backbone_sd) == 0:
        print("  Warning: no 'vit.*' keys found in checkpoint. "
              "Backbone not loaded.")
        return

    missing, unexpected = model.backbone.load_state_dict(
        backbone_sd, strict=False)
    print(f"  Loaded pretrained backbone: "
          f"{len(backbone_sd)} keys, {len(missing)} missing, "
          f"{len(unexpected)} unexpected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SegMamba-Det on ABUS dataset")
    parser.add_argument("--data_dir_train", type=str,
                        default="./data/abus_det/train")
    parser.add_argument("--data_dir_val", type=str,
                        default="./data/abus_det/val")
    parser.add_argument("--logdir", type=str,
                        default="./logs/segmamba_abus_det")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_backbone", type=str, default="",
                        help="Path to SegMamba segmentation checkpoint "
                             "(loads MambaEncoder weights)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_save_dir = os.path.join(args.logdir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Data
    train_ds = ABUSDetDataset(args.data_dir_train, augment=True)
    val_ds = ABUSDetDataset(args.data_dir_val, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=det_collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=det_collate_fn, pin_memory=True)

    # Model
    model = SegMambaDet(in_chans=1, num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SegMamba-Det parameters: {n_params:.2f}M")

    if args.pretrained_backbone:
        load_pretrained_backbone(model, args.pretrained_backbone)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch, eta_min=1e-6)
    scaler = GradScaler()

    best_ap = 0.0

    for epoch in range(args.max_epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"  [epoch {epoch:3d}]  loss={train_loss:.4f}  lr={lr_now:.6f}")

        if (epoch + 1) % args.val_every == 0:
            ap25 = validate(model, val_loader, device, iou_thresh=0.25)
            writer.add_scalar("val/AP@0.25", ap25, epoch)
            print(f"             val AP@0.25 = {ap25:.4f}")

            if ap25 > best_ap:
                best_ap = ap25
                path = os.path.join(
                    model_save_dir, f"best_model_{ap25:.4f}.pt")
                torch.save(model.state_dict(), path)
                # Remove previous best
                for f in glob.glob(
                        os.path.join(model_save_dir, "best_model_*.pt")):
                    if f != path:
                        os.remove(f)
                print(f"             -> saved best model ({ap25:.4f})")

        # Save latest every epoch
        torch.save(model.state_dict(),
                   os.path.join(model_save_dir, "latest.pt"))

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best AP@0.25 = {best_ap:.4f}")
    print(f"{'='*60}")
