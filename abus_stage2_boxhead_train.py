"""
Stage 2: Train BoxHead on Frozen SegMamba Features for ABUS Detection.

This script implements the second stage of the two-stage training strategy:
  1. Stage 1 (done separately): Train SegMamba for segmentation until convergence
  2. Stage 2 (this script): Freeze SegMamba, train only BoxHead for detection

The BoxHead is a lightweight attention-pooling module that converts mask features
into bounding box predictions. Patch-level predictions are fused into a global
box using soft-weighted aggregation based on objectness and quality scores.

Usage:
    python abus_stage2_boxhead_train.py \
        --pretrained_seg ./logs/segmamba_abus/model/best_model.pt \
        --data_dir_train ./data/abus/train \
        --data_dir_val ./data/abus/val

Prerequisites:
    - Trained SegMamba checkpoint from Stage 1
    - Preprocessed ABUS data (run abus_preprocessing.py)
"""

import os
import glob
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_segmamba.segmamba_patch_fusion import (
    SegMambaWithPatchFusion,
    transform_box_to_global,
    fuse_patch_boxes,
    compute_patch_targets,
    compute_giou_3d,
)

PATCH_SIZE = (128, 128, 128)


# ---------------------------------------------------------------------------
# Dataset: Same as patch fusion but optimized for detection-only training
# ---------------------------------------------------------------------------

class ABUSBoxHeadDataset(Dataset):
    """Dataset for BoxHead training with patch sampling biased toward GT boxes."""

    def __init__(self, data_dir, patches_per_volume=4, augment=False):
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.patch_size = PATCH_SIZE

        if len(self.npz_files) == 0:
            raise FileNotFoundError(
                f"No .npz files in {data_dir}. Run abus_preprocessing.py first.")

        # Pre-load metadata
        self.metadata = []
        for npz_path in self.npz_files:
            pkl_path = npz_path.replace('.npz', '.pkl')
            with open(pkl_path, 'rb') as f:
                props = pickle.load(f)
            self.metadata.append({
                'npz_path': npz_path,
                'shape': props.get('shape_after_cropping', props.get('original_shape')),
                'name': props.get('name', os.path.basename(npz_path).replace('.npz', '')),
            })

    def __len__(self):
        return len(self.npz_files)

    def _extract_gt_box(self, seg):
        """Extract global GT bounding box from segmentation mask."""
        if seg.sum() == 0:
            return None

        indices = np.where(seg > 0)
        z1, y1, x1 = indices[0].min(), indices[1].min(), indices[2].min()
        z2, y2, x2 = indices[0].max() + 1, indices[1].max() + 1, indices[2].max() + 1
        return np.array([z1, y1, x1, z2, y2, x2], dtype=np.float32)

    def _sample_patch_positions(self, volume_shape, gt_box):
        """Sample patch positions biased toward GT box overlap."""
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size

        positions = []
        n = self.patches_per_volume

        if gt_box is not None:
            gz1, gy1, gx1, gz2, gy2, gx2 = gt_box

            # Valid range for patch start to guarantee overlap with GT
            z_min = max(0, int(gz1) - pd + 1)
            z_max = min(D - pd, int(gz2) - 1)
            y_min = max(0, int(gy1) - ph + 1)
            y_max = min(H - ph, int(gy2) - 1)
            x_min = max(0, int(gx1) - pw + 1)
            x_max = min(W - pw, int(gx2) - 1)

            # Center position (fallback)
            z_center = max(0, min(D - pd, int((gz1 + gz2) / 2 - pd / 2)))
            y_center = max(0, min(H - ph, int((gy1 + gy2) / 2 - ph / 2)))
            x_center = max(0, min(W - pw, int((gx1 + gx2) / 2 - pw / 2)))

            # At least half should overlap GT
            n_overlap = max(1, (n + 1) // 2)
            for i in range(n_overlap):
                if i == 0:
                    # First patch centered on tumor
                    z, y, x = z_center, y_center, x_center
                elif z_max >= z_min and y_max >= y_min and x_max >= x_min:
                    z = np.random.randint(z_min, z_max + 1)
                    y = np.random.randint(y_min, y_max + 1)
                    x = np.random.randint(x_min, x_max + 1)
                else:
                    jitter = 10
                    z = max(0, min(D - pd, z_center + np.random.randint(-jitter, jitter + 1)))
                    y = max(0, min(H - ph, y_center + np.random.randint(-jitter, jitter + 1)))
                    x = max(0, min(W - pw, x_center + np.random.randint(-jitter, jitter + 1)))

                positions.append((z, y, x))

        # Fill remaining with random positions
        while len(positions) < n:
            z = np.random.randint(0, max(1, D - pd + 1))
            y = np.random.randint(0, max(1, H - ph + 1))
            x = np.random.randint(0, max(1, W - pw + 1))
            positions.append((z, y, x))

        return positions[:n]

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        data = np.load(meta['npz_path'])

        volume = data['data'].astype(np.float32)  # (1, D, H, W)
        seg = data['seg'].astype(np.int64)        # (1, D, H, W)
        seg = np.clip(seg, 0, 1)

        if self.augment:
            volume, seg = self._augment(volume, seg)

        # Extract global GT box
        gt_box = self._extract_gt_box(seg[0])
        has_gt_box = gt_box is not None
        if not has_gt_box:
            gt_box = np.zeros(6, dtype=np.float32)

        volume_shape = volume.shape[1:]

        # Sample patch positions
        positions = self._sample_patch_positions(volume_shape, gt_box if has_gt_box else None)

        # Extract patches
        patches = []
        for z, y, x in positions:
            patch = volume[:, z:z+self.patch_size[0],
                             y:y+self.patch_size[1],
                             x:x+self.patch_size[2]]

            if patch.shape[1:] != self.patch_size:
                pad_d = self.patch_size[0] - patch.shape[1]
                pad_h = self.patch_size[1] - patch.shape[2]
                pad_w = self.patch_size[2] - patch.shape[3]
                patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))

            patches.append(patch)

        return {
            'patches': torch.from_numpy(np.stack(patches)),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'gt_box': torch.from_numpy(gt_box),
            'has_gt_box': has_gt_box,
            'volume_shape': torch.tensor(volume_shape, dtype=torch.float32),
            'name': meta['name'],
        }

    def _augment(self, volume, seg):
        # Random flip along each axis
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis + 1).copy()
                seg = np.flip(seg, axis=axis + 1).copy()
        return volume, seg


def collate_fn(batch):
    """Return single item (batch_size must be 1)."""
    return batch[0]


# ---------------------------------------------------------------------------
# Loss computation (detection only)
# ---------------------------------------------------------------------------

def compute_detection_losses(boxes_local, objectness, quality,
                              positions, gt_box, has_gt_box, volume_shape, device):
    """Compute detection losses for BoxHead training.

    Returns:
        losses: dict with det_loss, obj_loss, qual_loss, total
        metrics: dict with monitoring values
    """
    N = boxes_local.shape[0]

    if not has_gt_box:
        # No GT box: no detection loss, but still supervise objectness to predict 0
        obj_targets = torch.zeros(N, 1, device=device)
        objectness_clamped = objectness.float().clamp(1e-4, 1 - 1e-4)
        objectness_logits = torch.log(objectness_clamped / (1 - objectness_clamped))
        obj_loss = F.binary_cross_entropy_with_logits(objectness_logits, obj_targets)

        return {
            'det_loss': torch.tensor(0.0, device=device),
            'obj_loss': obj_loss,
            'qual_loss': torch.tensor(0.0, device=device),
            'total': obj_loss,
        }, {
            'objectness_mean': objectness.mean().item(),
            'quality_mean': quality.mean().item(),
        }

    # Transform boxes to global coordinates
    boxes_global = transform_box_to_global(
        boxes_local, positions, PATCH_SIZE, volume_shape.tolist())

    # Fuse patch boxes
    fused_box, _ = fuse_patch_boxes(boxes_global, objectness, quality)

    # Compute per-patch targets for auxiliary losses
    gt_box_np = gt_box.cpu().numpy()
    obj_targets = []
    qual_targets = []

    for i in range(N):
        pos = positions[i].cpu().numpy().astype(int)
        obj_gt, qual_gt, _ = compute_patch_targets(
            pos, PATCH_SIZE, gt_box_np, volume_shape.cpu().numpy().astype(int))
        obj_targets.append(obj_gt)
        qual_targets.append(qual_gt)

    obj_targets = torch.tensor(obj_targets, device=device, dtype=torch.float32).unsqueeze(1)
    qual_targets = torch.tensor(qual_targets, device=device, dtype=torch.float32).unsqueeze(1)

    # Detection loss on fused box
    gt_cz = (gt_box[0] + gt_box[3]) / 2
    gt_cy = (gt_box[1] + gt_box[4]) / 2
    gt_cx = (gt_box[2] + gt_box[5]) / 2
    gt_dz = gt_box[3] - gt_box[0]
    gt_dy = gt_box[4] - gt_box[1]
    gt_dx = gt_box[5] - gt_box[2]
    gt_box_center = torch.stack([gt_cz, gt_cy, gt_cx, gt_dz, gt_dy, gt_dx])

    fused_box_f32 = fused_box.float()
    gt_box_center_f32 = gt_box_center.float()

    # L1 loss
    l1_loss = F.smooth_l1_loss(fused_box_f32, gt_box_center_f32)

    # GIoU loss
    giou = compute_giou_3d(fused_box_f32, gt_box_center_f32)
    giou_loss = 1 - giou

    det_loss = (l1_loss + giou_loss).clamp(max=10.0)

    # Objectness BCE
    objectness_clamped = objectness.float().clamp(1e-4, 1 - 1e-4)
    objectness_logits = torch.log(objectness_clamped / (1 - objectness_clamped))
    obj_loss = F.binary_cross_entropy_with_logits(objectness_logits, obj_targets.float())

    # Quality loss (only for patches with GT overlap)
    has_overlap = obj_targets > 0.5
    if has_overlap.any():
        qual_loss = F.mse_loss(quality[has_overlap].float(), qual_targets[has_overlap].float())
    else:
        qual_loss = torch.tensor(0.0, device=device)

    total = det_loss + obj_loss + qual_loss

    return {
        'det_loss': det_loss,
        'obj_loss': obj_loss,
        'qual_loss': qual_loss,
        'total': total,
    }, {
        'objectness_mean': objectness.mean().item(),
        'quality_mean': quality.mean().item(),
        'giou': giou.item(),
        'l1': l1_loss.item(),
        'n_overlap_patches': has_overlap.sum().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    """Train one epoch (BoxHead only, backbone frozen).

    Note: AMP is disabled for Stage 2 because the BoxHead is small (~0.5M params)
    and fp16 precision issues can cause NaN gradients.
    """
    # Keep model in eval mode for frozen backbone, but BoxHead will still train
    model.eval()

    total_losses = {'det_loss': 0, 'obj_loss': 0, 'qual_loss': 0, 'total': 0}
    num_samples = 0
    nan_batches = 0

    for batch in tqdm(loader, desc="Training"):
        patches = batch['patches'].to(device)
        positions = batch['positions'].to(device)
        gt_box = batch['gt_box'].to(device)
        has_gt_box = batch['has_gt_box']
        volume_shape = batch['volume_shape'].to(device)

        optimizer.zero_grad()

        # Forward pass using Stage 2 method - backbone no_grad, BoxHead with grad
        # No AMP to avoid fp16 precision issues
        # Pass patch positions and volume shape for positional encoding
        seg_logits, boxes_local, objectness, quality = model.forward_boxhead_only(
            patches, patch_pos=positions, volume_shape=volume_shape)

        # Check for NaN in predictions
        if torch.isnan(boxes_local).any() or torch.isnan(objectness).any() or torch.isnan(quality).any():
            nan_batches += 1
            continue

        losses, metrics = compute_detection_losses(
            boxes_local, objectness, quality,
            positions, gt_box, has_gt_box, volume_shape, device)

        # Check for NaN in loss
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            nan_batches += 1
            continue

        losses['total'].backward()

        # Check for NaN gradients
        has_nan_grad = False
        for p in model.patch_box_head.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break

        if has_nan_grad:
            nan_batches += 1
            optimizer.zero_grad()
            continue

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.patch_box_head.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        num_samples += 1

    if nan_batches > 0:
        print(f"  Skipped {nan_batches} batches due to NaN")

    return {k: v / max(num_samples, 1) for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device):
    """Validate detection performance."""
    model.eval()

    giou_list = []
    iou_list = []

    for batch in tqdm(loader, desc="Validating"):
        patches = batch['patches'].to(device)
        positions = batch['positions'].to(device)
        gt_box = batch['gt_box'].to(device)
        has_gt_box = batch['has_gt_box']
        volume_shape = batch['volume_shape'].to(device)

        if not has_gt_box:
            continue

        # Use forward_boxhead_only for consistency with training
        seg_logits, boxes_local, objectness, quality = model.forward_boxhead_only(
            patches, patch_pos=positions, volume_shape=volume_shape)

        boxes_global = transform_box_to_global(
            boxes_local, positions, PATCH_SIZE, volume_shape.tolist())
        fused_box, _ = fuse_patch_boxes(boxes_global, objectness, quality)

        # Convert GT to center format
        gt_cz = (gt_box[0] + gt_box[3]) / 2
        gt_cy = (gt_box[1] + gt_box[4]) / 2
        gt_cx = (gt_box[2] + gt_box[5]) / 2
        gt_dz = gt_box[3] - gt_box[0]
        gt_dy = gt_box[4] - gt_box[1]
        gt_dx = gt_box[5] - gt_box[2]
        gt_box_center = torch.stack([gt_cz, gt_cy, gt_cx, gt_dz, gt_dy, gt_dx])

        if gt_dz > 0 and gt_dy > 0 and gt_dx > 0:
            giou = compute_giou_3d(fused_box, gt_box_center)
            giou_val = giou.item()

            if not np.isnan(giou_val):
                giou_list.append(giou_val)
                iou = (giou_val + 1) / 2
                iou_list.append(max(0, min(1, iou)))

    return {
        'giou': np.mean(giou_list) if giou_list else 0.0,
        'iou': np.mean(iou_list) if iou_list else 0.0,
    }


# ---------------------------------------------------------------------------
# Learning rate scheduler
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
        description="Stage 2: Train BoxHead on frozen SegMamba for detection")
    parser.add_argument("--pretrained_seg", type=str, required=True,
                        help="Path to pretrained SegMamba checkpoint (Stage 1)")
    parser.add_argument("--data_dir_train", type=str, default="./data/abus/train")
    parser.add_argument("--data_dir_val", type=str, default="./data/abus/val")
    parser.add_argument("--logdir", type=str, default="./logs/segmamba_abus_stage2_boxhead")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--patches_per_volume", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for BoxHead (lower than Stage 1)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_save_dir = os.path.join(args.logdir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Data
    train_ds = ABUSBoxHeadDataset(
        args.data_dir_train,
        patches_per_volume=args.patches_per_volume,
        augment=True)
    val_ds = ABUSBoxHeadDataset(
        args.data_dir_val,
        patches_per_volume=args.patches_per_volume,
        augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"Train: {len(train_ds)} volumes, {args.patches_per_volume} patches each")
    print(f"Val: {len(val_ds)} volumes")

    # Model
    model = SegMambaWithPatchFusion(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    # Load pretrained SegMamba weights
    print(f"\nLoading pretrained SegMamba from: {args.pretrained_seg}")
    sd = torch.load(args.pretrained_seg, map_location='cpu')
    if 'module' in sd:
        sd = sd['module']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Loaded {len(new_sd)} keys")
    print(f"  Missing (BoxHead, expected): {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    # Freeze backbone (everything except BoxHead)
    print("\nFreezing SegMamba backbone...")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'patch_box_head' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen parameters: {frozen_count / 1e6:.2f}M")
    print(f"  Trainable parameters (BoxHead): {trainable_count / 1e6:.2f}M")

    # Optimizer - only for BoxHead parameters
    optimizer = torch.optim.AdamW(
        model.patch_box_head.parameters(),
        lr=args.lr,
        weight_decay=1e-4)
    scheduler = PolyLRScheduler(optimizer, args.max_epoch)

    best_metric = -1.0

    print(f"\n{'='*60}")
    print(f"  Stage 2: Training BoxHead on frozen SegMamba features")
    print(f"  (AMP disabled for numerical stability)")
    print(f"{'='*60}\n")

    for epoch in range(args.max_epoch):
        losses = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]['lr']

        for k, v in losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"  [epoch {epoch:3d}]  total={losses['total']:.4f}  "
              f"det={losses['det_loss']:.4f}  obj={losses['obj_loss']:.4f}  "
              f"qual={losses['qual_loss']:.4f}  lr={lr_now:.6f}")

        if (epoch + 1) % args.val_every == 0:
            metrics = validate(model, val_loader, device)

            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            print(f"             GIoU={metrics['giou']:.4f}  IoU={metrics['iou']:.4f}")

            if metrics['giou'] > best_metric:
                best_metric = metrics['giou']
                path = os.path.join(model_save_dir, f"best_model_giou{metrics['giou']:.4f}.pt")
                # Save only BoxHead weights for efficiency
                torch.save({
                    'patch_box_head': model.patch_box_head.state_dict(),
                    'full_model': model.state_dict(),
                    'epoch': epoch,
                    'giou': metrics['giou'],
                }, path)
                for f in glob.glob(os.path.join(model_save_dir, "best_model_giou*.pt")):
                    if f != path:
                        os.remove(f)
                print(f"             -> saved best model (GIoU={metrics['giou']:.4f})")

        # Save latest
        torch.save({
            'patch_box_head': model.patch_box_head.state_dict(),
            'epoch': epoch,
        }, os.path.join(model_save_dir, "latest.pt"))

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Stage 2 complete.  Best GIoU = {best_metric:.4f}")
    print(f"{'='*60}")
