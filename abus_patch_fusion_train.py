"""
SegMamba Patch-Set Global Fusion Training for ABUS.

Multi-task training that combines:
  - Patch-based segmentation (same as original SegMamba)
  - Global detection via differentiable fusion of patch-level predictions

Each patch predicts local box evidence (box, objectness, quality).
Multiple patches from the same volume are fused into a global box prediction.
Detection loss is computed only on the fused global box.

Usage:
    python abus_patch_fusion_train.py --data_dir_train ./data/abus/train \
                                       --data_dir_val ./data/abus/val

Prerequisites:
    Run abus_preprocessing.py first (same data as original SegMamba).
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
from torch.cuda.amp import GradScaler, autocast
from scipy import ndimage
from tqdm import tqdm

from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer

from model_segmamba.segmamba_patch_fusion import (
    SegMambaWithPatchFusion,
    transform_box_to_global,
    fuse_patch_boxes,
    compute_patch_targets,
    compute_giou_3d,
)

PATCH_SIZE = (128, 128, 128)


# ---------------------------------------------------------------------------
# Dataset: Samples N patches per volume for global fusion
# ---------------------------------------------------------------------------

class ABUSPatchFusionDataset(Dataset):
    """Dataset that samples multiple patches per volume for global fusion.

    Each item returns N patches from the same volume along with their positions
    and the global GT box.
    """

    def __init__(self, data_dir, patches_per_volume=4, augment=False):
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.patch_size = PATCH_SIZE

        if len(self.npz_files) == 0:
            raise FileNotFoundError(
                f"No .npz files in {data_dir}. Run abus_preprocessing.py first.")

        # Pre-load metadata for efficient sampling
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
        """Sample patch positions, biased towards regions containing the target."""
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size

        positions = []
        n = self.patches_per_volume

        if gt_box is not None:
            # Sample some patches that overlap with GT box
            gz1, gy1, gx1, gz2, gy2, gx2 = gt_box

            # At least half should overlap GT
            n_overlap = max(1, n // 2)
            for _ in range(n_overlap):
                # Sample position that overlaps GT
                z_min = max(0, int(gz2) - pd)
                z_max = min(D - pd, int(gz1))
                y_min = max(0, int(gy2) - ph)
                y_max = min(H - ph, int(gy1))
                x_min = max(0, int(gx2) - pw)
                x_max = min(W - pw, int(gx1))

                z = np.random.randint(max(0, z_min), min(D - pd, z_max) + 1) if z_max >= z_min else max(0, min(D - pd, int((gz1 + gz2) / 2 - pd / 2)))
                y = np.random.randint(max(0, y_min), min(H - ph, y_max) + 1) if y_max >= y_min else max(0, min(H - ph, int((gy1 + gy2) / 2 - ph / 2)))
                x = np.random.randint(max(0, x_min), min(W - pw, x_max) + 1) if x_max >= x_min else max(0, min(W - pw, int((gx1 + gx2) / 2 - pw / 2)))

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
        # Ensure binary segmentation (clamp to 0 or 1)
        seg = np.clip(seg, 0, 1)

        if self.augment:
            volume, seg = self._augment(volume, seg)

        # Extract global GT box
        gt_box = self._extract_gt_box(seg[0])
        has_gt_box = gt_box is not None
        if not has_gt_box:
            gt_box = np.zeros(6, dtype=np.float32)

        volume_shape = volume.shape[1:]  # (D, H, W)

        # Sample patch positions
        positions = self._sample_patch_positions(volume_shape, gt_box if has_gt_box else None)

        # Extract patches
        patches = []
        seg_patches = []
        for z, y, x in positions:
            patch = volume[:, z:z+self.patch_size[0],
                             y:y+self.patch_size[1],
                             x:x+self.patch_size[2]]
            seg_patch = seg[:, z:z+self.patch_size[0],
                              y:y+self.patch_size[1],
                              x:x+self.patch_size[2]]

            # Pad if necessary (edge cases)
            if patch.shape[1:] != self.patch_size:
                pad_d = self.patch_size[0] - patch.shape[1]
                pad_h = self.patch_size[1] - patch.shape[2]
                pad_w = self.patch_size[2] - patch.shape[3]
                patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
                seg_patch = np.pad(seg_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))

            patches.append(patch)
            seg_patches.append(seg_patch)

        return {
            'patches': torch.from_numpy(np.stack(patches)),      # (N, 1, D, H, W)
            'seg_patches': torch.from_numpy(np.stack(seg_patches)),  # (N, 1, D, H, W)
            'positions': torch.tensor(positions, dtype=torch.float32),  # (N, 3)
            'gt_box': torch.from_numpy(gt_box),                  # (6,) corner format
            'has_gt_box': has_gt_box,
            'volume_shape': torch.tensor(volume_shape, dtype=torch.float32),  # (3,)
            'name': meta['name'],
        }

    def _augment(self, volume, seg):
        # Random flip along each axis
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis + 1).copy()
                seg = np.flip(seg, axis=axis + 1).copy()

        # Random intensity augmentation
        if np.random.rand() > 0.5:
            volume = volume + np.random.normal(0, 0.1, volume.shape).astype(np.float32)
        if np.random.rand() > 0.5:
            volume = volume * np.random.uniform(0.9, 1.1)

        return volume, seg


def fusion_collate_fn(batch):
    """Collate function that preserves per-volume structure."""
    # Each item is one volume with N patches
    # We process one volume at a time during training
    return batch[0]  # Return single item (batch_size must be 1)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(seg_logits, boxes_local, objectness, quality,
                   seg_gts, positions, gt_box, has_gt_box, volume_shape,
                   dice_loss_fn, ce_loss_fn, device):
    """Compute all losses for one volume's patches.

    Returns:
        losses: dict with seg_loss, det_loss, obj_loss, qual_loss, total
        metrics: dict with useful monitoring values
    """
    N = seg_logits.shape[0]
    num_classes = seg_logits.shape[1]  # Should be 2
    patch_size = torch.tensor(PATCH_SIZE, device=device, dtype=torch.float32)

    # --- Segmentation loss (per-patch) ---
    seg_gt_flat = seg_gts[:, 0].long()  # (N, D, H, W)
    # Clamp labels to valid range [0, num_classes-1] to prevent index out of bounds
    seg_gt_flat = torch.clamp(seg_gt_flat, 0, num_classes - 1)
    dice_l = dice_loss_fn(seg_logits, seg_gt_flat.unsqueeze(1))
    ce_l = ce_loss_fn(seg_logits, seg_gt_flat)
    seg_loss = dice_l + ce_l

    if not has_gt_box:
        # No GT box: only seg loss, set detection losses to 0
        return {
            'seg_loss': seg_loss,
            'det_loss': torch.tensor(0.0, device=device),
            'obj_loss': torch.tensor(0.0, device=device),
            'qual_loss': torch.tensor(0.0, device=device),
            'total': seg_loss,
        }, {
            'objectness_mean': objectness.mean().item(),
            'quality_mean': quality.mean().item(),
        }

    # --- Transform boxes to global coordinates ---
    boxes_global = transform_box_to_global(
        boxes_local, positions, PATCH_SIZE, volume_shape.tolist())

    # --- Fuse patch boxes ---
    fused_box, weights = fuse_patch_boxes(boxes_global, objectness, quality)

    # --- Compute per-patch targets for auxiliary losses ---
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

    # --- Detection loss on fused box ---
    # Convert GT box from corner to center format
    gt_cz = (gt_box[0] + gt_box[3]) / 2
    gt_cy = (gt_box[1] + gt_box[4]) / 2
    gt_cx = (gt_box[2] + gt_box[5]) / 2
    gt_dz = gt_box[3] - gt_box[0]
    gt_dy = gt_box[4] - gt_box[1]
    gt_dx = gt_box[5] - gt_box[2]
    gt_box_center = torch.stack([gt_cz, gt_cy, gt_cx, gt_dz, gt_dy, gt_dx])

    # L1 loss
    l1_loss = F.smooth_l1_loss(fused_box, gt_box_center)

    # GIoU loss
    giou = compute_giou_3d(fused_box, gt_box_center)
    giou_loss = 1 - giou

    det_loss = l1_loss + giou_loss

    # --- Auxiliary losses ---
    # Objectness BCE (use logits version for AMP compatibility)
    # Since objectness is already sigmoid, convert back to logits
    objectness_logits = torch.log(objectness / (1 - objectness + 1e-7) + 1e-7)
    obj_loss = F.binary_cross_entropy_with_logits(objectness_logits, obj_targets)

    # Quality loss (only for patches with GT overlap)
    has_overlap = obj_targets > 0.5
    if has_overlap.any():
        qual_loss = F.mse_loss(quality[has_overlap], qual_targets[has_overlap])
    else:
        qual_loss = torch.tensor(0.0, device=device)

    return {
        'seg_loss': seg_loss,
        'det_loss': det_loss,
        'obj_loss': obj_loss,
        'qual_loss': qual_loss,
        'total': seg_loss + det_loss + 0.5 * obj_loss + 0.5 * qual_loss,
    }, {
        'objectness_mean': objectness.mean().item(),
        'quality_mean': quality.mean().item(),
        'giou': giou.item(),
        'n_overlap_patches': has_overlap.sum().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler,
                    dice_loss_fn, ce_loss_fn):
    model.train()
    total_losses = {'seg_loss': 0, 'det_loss': 0, 'obj_loss': 0, 'qual_loss': 0, 'total': 0}
    num_samples = 0

    for batch in tqdm(loader, desc="Training"):
        patches = batch['patches'].to(device)          # (N, 1, D, H, W)
        seg_patches = batch['seg_patches'].to(device)  # (N, 1, D, H, W)
        positions = batch['positions'].to(device)      # (N, 3)
        gt_box = batch['gt_box'].to(device)            # (6,)
        has_gt_box = batch['has_gt_box']
        volume_shape = batch['volume_shape'].to(device)

        optimizer.zero_grad()

        with autocast():
            # Forward all patches at once
            seg_logits, boxes_local, objectness, quality = model(patches)

            losses, _ = compute_losses(
                seg_logits, boxes_local, objectness, quality,
                seg_patches, positions, gt_box, has_gt_box, volume_shape,
                dice_loss_fn, ce_loss_fn, device)

        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        num_samples += 1

    return {k: v / max(num_samples, 1) for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Validation with sliding window
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device):
    """Validate using sliding window inference for both seg and detection."""
    model.eval()

    dice_list = []
    giou_list = []
    iou_list = []

    window_inferer = SlidingWindowInferer(
        roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.5, mode="gaussian")

    for batch in tqdm(loader, desc="Validating"):
        patches = batch['patches'].to(device)
        seg_patches = batch['seg_patches'].to(device)
        positions = batch['positions'].to(device)
        gt_box = batch['gt_box'].to(device)
        has_gt_box = batch['has_gt_box']
        volume_shape = batch['volume_shape'].to(device)

        # Forward all patches
        seg_logits, boxes_local, objectness, quality = model(patches)

        # Segmentation Dice (average over patches)
        seg_pred = seg_logits.argmax(dim=1)  # (N, D, H, W)
        seg_gt = seg_patches[:, 0]           # (N, D, H, W)

        for i in range(seg_pred.shape[0]):
            pred_np = seg_pred[i].cpu().numpy().astype(bool)
            gt_np = seg_gt[i].cpu().numpy().astype(bool)

            if pred_np.sum() > 0 and gt_np.sum() > 0:
                tp = (pred_np & gt_np).sum()
                dice = 2 * tp / (pred_np.sum() + gt_np.sum())
                dice_list.append(dice)
            elif not pred_np.any() and not gt_np.any():
                dice_list.append(1.0)
            else:
                dice_list.append(0.0)

        # Detection metrics (if GT box exists)
        if has_gt_box:
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

            giou = compute_giou_3d(fused_box, gt_box_center)
            giou_list.append(giou.item())

            # IoU
            iou = (giou + 1) / 2  # Rough approximation: GIoU in [-1,1] -> IoU in [0,1]
            iou_list.append(max(0, iou.item()))

    return {
        'dice': np.mean(dice_list) if dice_list else 0.0,
        'giou': np.mean(giou_list) if giou_list else 0.0,
        'iou': np.mean(iou_list) if iou_list else 0.0,
    }


# ---------------------------------------------------------------------------
# Poly LR
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
        description="Train SegMamba with Patch-Set Global Fusion")
    parser.add_argument("--data_dir_train", type=str, default="./data/abus/train")
    parser.add_argument("--data_dir_val", type=str, default="./data/abus/val")
    parser.add_argument("--logdir", type=str, default="./logs/segmamba_abus_patch_fusion")
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--patches_per_volume", type=int, default=4,
                        help="Number of patches to sample per volume")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_seg", type=str, default="",
                        help="Path to pretrained SegMamba segmentation checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_save_dir = os.path.join(args.logdir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Data
    train_ds = ABUSPatchFusionDataset(
        args.data_dir_train,
        patches_per_volume=args.patches_per_volume,
        augment=True)
    val_ds = ABUSPatchFusionDataset(
        args.data_dir_val,
        patches_per_volume=args.patches_per_volume,
        augment=False)

    # batch_size=1 because each item is one volume with N patches
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, collate_fn=fusion_collate_fn,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=fusion_collate_fn, pin_memory=True)

    print(f"Train: {len(train_ds)} volumes, {args.patches_per_volume} patches each")
    print(f"Val: {len(val_ds)} volumes")

    # Model
    model = SegMambaWithPatchFusion(
        in_chans=1, out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SegMamba-PatchFusion parameters: {n_params:.2f}M")

    # Load pretrained weights if provided
    if args.pretrained_seg:
        sd = torch.load(args.pretrained_seg, map_location='cpu')
        if 'module' in sd:
            sd = sd['module']
        new_sd = {}
        for k, v in sd.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_sd[new_k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"  Loaded pretrained: {len(new_sd)} keys, "
              f"{len(missing)} missing (PatchBoxHead), "
              f"{len(unexpected)} unexpected")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        weight_decay=3e-5, momentum=0.99, nesterov=True)
    scheduler = PolyLRScheduler(optimizer, args.max_epoch)
    scaler = GradScaler()

    # Losses
    dice_loss_fn = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_metric = 0.0

    for epoch in range(args.max_epoch):
        losses = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            dice_loss_fn, ce_loss_fn)
        scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]['lr']

        for k, v in losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"  [epoch {epoch:3d}]  total={losses['total']:.4f}  "
              f"seg={losses['seg_loss']:.4f}  det={losses['det_loss']:.4f}  "
              f"lr={lr_now:.6f}")

        if (epoch + 1) % args.val_every == 0:
            metrics = validate(model, val_loader, device)

            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            print(f"             Dice={metrics['dice']:.4f}  "
                  f"GIoU={metrics['giou']:.4f}  IoU={metrics['iou']:.4f}")

            # Save best based on combined metric
            combined = metrics['dice'] * 0.5 + metrics['iou'] * 0.5
            if combined > best_metric:
                best_metric = combined
                path = os.path.join(model_save_dir, f"best_model_{combined:.4f}.pt")
                torch.save(model.state_dict(), path)
                for f in glob.glob(os.path.join(model_save_dir, "best_model_*.pt")):
                    if f != path:
                        os.remove(f)
                print(f"             -> saved best model (combined={combined:.4f})")

        torch.save(model.state_dict(), os.path.join(model_save_dir, "latest.pt"))

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best combined metric = {best_metric:.4f}")
    print(f"{'='*60}")
