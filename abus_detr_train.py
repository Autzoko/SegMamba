"""
SegMamba-DETR Training Script for ABUS 3D Object Detection.

DETR-style transformer decoder on MambaEncoder backbone.
Uses Hungarian matching + set prediction loss (CE + L1 + GIoU).

Usage:
    python abus_detr_train.py --data_dir_train ./data/abus_det/train \
                               --data_dir_val ./data/abus_det/val

Prerequisites:
    Run  abus_det_preprocessing.py  first (same data format).
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from model_segmamba.segmamba_detr import SegMambaDETR
from abus_detr_utils import (HungarianMatcher, SetCriterion,
                              prepare_detr_targets,
                              box_cxcyczdhwd_to_zyxzyx)
from abus_det_train import ABUSDetDataset, det_collate_fn, load_pretrained_backbone
from abus_det_utils import compute_detection_metrics

INPUT_SIZE = 128


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        volume = batch['data'].to(device)
        gt_boxes = batch['boxes']
        gt_n = batch['num_boxes']

        targets = prepare_detr_targets(gt_boxes, gt_n, INPUT_SIZE)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            outputs = model(volume)
            losses = criterion(outputs, targets)
            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, score_thresh=0.05):
    """Compute comprehensive detection metrics on the validation set.

    Returns dict with AP@0.1/0.25/0.5, mAP, recall, mean IoU, mean GIoU.
    """
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []

    for batch in loader:
        volume = batch['data'].to(device)
        gt = batch['boxes'].numpy()
        gt_n = batch['num_boxes'].numpy()
        B = volume.shape[0]

        with autocast():
            outputs = model(volume)

        pred_logits = outputs['pred_logits']   # (B, N, 2)
        pred_boxes = outputs['pred_boxes']     # (B, N, 6)

        for b in range(B):
            probs = pred_logits[b].softmax(-1)            # (N, 2)
            scores = probs[:, 0].cpu().numpy()             # tumour prob

            boxes_norm = pred_boxes[b].cpu()               # (N, 6) cxcycz
            boxes_corner = box_cxcyczdhwd_to_zyxzyx(boxes_norm)
            boxes_abs = (boxes_corner * INPUT_SIZE).numpy()
            boxes_abs = np.clip(boxes_abs, 0, INPUT_SIZE)

            mask = scores > score_thresh
            all_pred_boxes.append(boxes_abs[mask].astype(np.float32))
            all_pred_scores.append(scores[mask].astype(np.float32))
            all_gt_boxes.append(gt[b, :gt_n[b]])

    return compute_detection_metrics(
        all_pred_boxes, all_pred_scores, all_gt_boxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SegMamba-DETR on ABUS dataset")
    parser.add_argument("--data_dir_train", type=str,
                        default="./data/abus_det/train")
    parser.add_argument("--data_dir_val", type=str,
                        default="./data/abus_det/val")
    parser.add_argument("--logdir", type=str,
                        default="./logs/segmamba_abus_detr")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_backbone", type=str, default="",
                        help="Path to SegMamba segmentation checkpoint "
                             "(loads MambaEncoder weights)")
    # DETR-specific
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_save_dir = os.path.join(args.logdir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Data (reuse existing detection dataset)
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
    model = SegMambaDETR(
        in_chans=1, num_classes=1,
        num_queries=args.num_queries,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        aux_loss=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SegMamba-DETR parameters: {n_params:.2f}M")

    if args.pretrained_backbone:
        load_pretrained_backbone(model, args.pretrained_backbone)

    # Differential learning rate: backbone 10x lower
    backbone_ids = set(id(p) for p in model.backbone.parameters())
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.1)
    scaler = GradScaler()

    # Loss
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(
        num_classes=1, matcher=matcher, eos_coef=0.1).to(device)

    best_ap = 0.0

    for epoch in range(args.max_epoch):
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, scaler)
        scheduler.step()

        lr_now = optimizer.param_groups[1]['lr']  # transformer LR
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"  [epoch {epoch:3d}]  loss={train_loss:.4f}  lr={lr_now:.6f}")

        if (epoch + 1) % args.val_every == 0:
            metrics = validate(model, val_loader, device)

            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            ap25 = metrics['AP@0.25']
            print(f"             AP@0.1={metrics['AP@0.1']:.4f}  "
                  f"AP@0.25={ap25:.4f}  "
                  f"AP@0.5={metrics['AP@0.5']:.4f}  "
                  f"mAP={metrics['mAP']:.4f}")
            print(f"             recall@0.25={metrics['recall@0.25']:.4f}  "
                  f"mean_IoU={metrics['mean_best_iou']:.4f}  "
                  f"mean_GIoU={metrics['mean_giou']:.4f}")

            if ap25 > best_ap:
                best_ap = ap25
                path = os.path.join(
                    model_save_dir, f"best_model_{ap25:.4f}.pt")
                torch.save(model.state_dict(), path)
                for f in glob.glob(
                        os.path.join(model_save_dir, "best_model_*.pt")):
                    if f != path:
                        os.remove(f)
                print(f"             -> saved best model ({ap25:.4f})")

        torch.save(model.state_dict(),
                   os.path.join(model_save_dir, "latest.pt"))

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best AP@0.25 = {best_ap:.4f}")
    print(f"{'='*60}")
