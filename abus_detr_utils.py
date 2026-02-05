"""
Utilities for SegMamba-DETR: box conversions, 3D GIoU, Hungarian matching,
and DETR set-prediction loss.

All box formats use (z, y, x) ordering to match the rest of the ABUS pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Box format conversions
# ---------------------------------------------------------------------------

def box_cxcyczdhwd_to_zyxzyx(boxes):
    """Convert [cz, cy, cx, dz, dy, dx] to [z1, y1, x1, z2, y2, x2]."""
    cz, cy, cx, dz, dy, dx = boxes.unbind(-1)
    return torch.stack([
        cz - dz / 2, cy - dy / 2, cx - dx / 2,
        cz + dz / 2, cy + dy / 2, cx + dx / 2,
    ], dim=-1)


def box_zyxzyx_to_cxcyczdhwd(boxes):
    """Convert [z1, y1, x1, z2, y2, x2] to [cz, cy, cx, dz, dy, dx]."""
    z1, y1, x1, z2, y2, x2 = boxes.unbind(-1)
    return torch.stack([
        (z1 + z2) / 2, (y1 + y2) / 2, (x1 + x2) / 2,
        z2 - z1, y2 - y1, x2 - x1,
    ], dim=-1)


# ---------------------------------------------------------------------------
# 3D IoU and Generalized IoU (differentiable)
# ---------------------------------------------------------------------------

def box_iou_3d(boxes1, boxes2):
    """Compute pairwise IoU between two sets of 3D boxes.

    Parameters
    ----------
    boxes1 : (N, 6) [z1, y1, x1, z2, y2, x2]
    boxes2 : (M, 6) [z1, y1, x1, z2, y2, x2]

    Returns
    -------
    iou : (N, M) IoU matrix
    """
    vol1 = ((boxes1[:, 3] - boxes1[:, 0]) *
            (boxes1[:, 4] - boxes1[:, 1]) *
            (boxes1[:, 5] - boxes1[:, 2])).clamp(min=0)
    vol2 = ((boxes2[:, 3] - boxes2[:, 0]) *
            (boxes2[:, 4] - boxes2[:, 1]) *
            (boxes2[:, 5] - boxes2[:, 2])).clamp(min=0)

    inter_z1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x1 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_z2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_y2 = torch.min(boxes1[:, None, 4], boxes2[None, :, 4])
    inter_x2 = torch.min(boxes1[:, None, 5], boxes2[None, :, 5])

    inter = ((inter_z2 - inter_z1).clamp(min=0) *
             (inter_y2 - inter_y1).clamp(min=0) *
             (inter_x2 - inter_x1).clamp(min=0))

    union = vol1[:, None] + vol2[None, :] - inter
    return inter / (union + 1e-6)


def generalized_box_iou_3d(boxes1, boxes2):
    """Compute pairwise Generalized IoU for 3D boxes.

    Parameters / Returns same as box_iou_3d but values in [-1, 1].
    """
    vol1 = ((boxes1[:, 3] - boxes1[:, 0]) *
            (boxes1[:, 4] - boxes1[:, 1]) *
            (boxes1[:, 5] - boxes1[:, 2])).clamp(min=0)
    vol2 = ((boxes2[:, 3] - boxes2[:, 0]) *
            (boxes2[:, 4] - boxes2[:, 1]) *
            (boxes2[:, 5] - boxes2[:, 2])).clamp(min=0)

    inter_z1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x1 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_z2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_y2 = torch.min(boxes1[:, None, 4], boxes2[None, :, 4])
    inter_x2 = torch.min(boxes1[:, None, 5], boxes2[None, :, 5])

    inter = ((inter_z2 - inter_z1).clamp(min=0) *
             (inter_y2 - inter_y1).clamp(min=0) *
             (inter_x2 - inter_x1).clamp(min=0))

    union = vol1[:, None] + vol2[None, :] - inter
    iou = inter / (union + 1e-6)

    # Enclosing box
    enc_z1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x1 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_z2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_y2 = torch.max(boxes1[:, None, 4], boxes2[None, :, 4])
    enc_x2 = torch.max(boxes1[:, None, 5], boxes2[None, :, 5])

    enc_vol = ((enc_z2 - enc_z1) * (enc_y2 - enc_y1) * (enc_x2 - enc_x1))

    return iou - (enc_vol - union) / (enc_vol + 1e-6)


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth boxes.

    Cost = cost_class * (-softmax_prob) + cost_bbox * L1 + cost_giou * (-GIoU)
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Parameters
        ----------
        outputs : dict
            pred_logits (B, N, num_classes+1), pred_boxes (B, N, 6)
        targets : list of B dicts
            Each with 'labels' (M,) and 'boxes' (M, 6) in normalised cxcycz

        Returns
        -------
        list of (pred_indices, gt_indices) tuples per batch element
        """
        B, N = outputs['pred_logits'].shape[:2]
        indices = []

        for b in range(B):
            out_prob = outputs['pred_logits'][b].softmax(-1)  # (N, C+1)
            out_bbox = outputs['pred_boxes'][b]                # (N, 6)

            tgt_labels = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            if len(tgt_labels) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)))
                continue

            # Classification cost: negative probability of correct class
            cost_class = -out_prob[:, tgt_labels]              # (N, M)

            # L1 cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (N, M)

            # GIoU cost
            cost_giou = -generalized_box_iou_3d(
                box_cxcyczdhwd_to_zyxzyx(out_bbox),
                box_cxcyczdhwd_to_zyxzyx(tgt_bbox))            # (N, M)

            C = (self.cost_class * cost_class +
                 self.cost_bbox * cost_bbox +
                 self.cost_giou * cost_giou)

            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.tensor(row, dtype=torch.int64),
                torch.tensor(col, dtype=torch.int64)))

        return indices


# ---------------------------------------------------------------------------
# Set Prediction Loss
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """DETR set-prediction loss: CE + L1 + GIoU with auxiliary decoder losses.

    Parameters
    ----------
    num_classes : int
        Number of foreground classes (1 for binary tumour detection).
    matcher : HungarianMatcher
    eos_coef : float
        CE weight for the "no-object" class (down-weighted).
    loss_ce_w, loss_bbox_w, loss_giou_w : float
        Relative loss weights.
    """

    def __init__(self, num_classes, matcher, eos_coef=0.1,
                 loss_ce_w=1.0, loss_bbox_w=5.0, loss_giou_w=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.loss_ce_w = loss_ce_w
        self.loss_bbox_w = loss_bbox_w
        self.loss_giou_w = loss_giou_w

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    # ----- classification -----

    def loss_labels(self, outputs, targets, indices):
        pred_logits = outputs['pred_logits']          # (B, N, C+1)
        B, N, _ = pred_logits.shape

        target_classes = torch.full(
            (B, N), self.num_classes,
            dtype=torch.int64, device=pred_logits.device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][gt_idx].to(
                    pred_logits.device)

        return F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes, self.empty_weight)

    # ----- box regression -----

    def loss_boxes(self, outputs, targets, indices):
        device = outputs['pred_boxes'].device
        src_list, tgt_list = [], []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                src_list.append(outputs['pred_boxes'][b, pred_idx])
                tgt_list.append(targets[b]['boxes'][gt_idx].to(device))

        if len(src_list) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero

        src_boxes = torch.cat(src_list)    # (K, 6)
        tgt_boxes = torch.cat(tgt_list)    # (K, 6)

        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='mean')

        src_corner = box_cxcyczdhwd_to_zyxzyx(src_boxes)
        tgt_corner = box_cxcyczdhwd_to_zyxzyx(tgt_boxes)
        giou = generalized_box_iou_3d(src_corner, tgt_corner)
        loss_giou = (1 - giou.diag()).mean()

        return loss_l1, loss_giou

    # ----- combined -----

    def _compute_loss(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        l_ce = self.loss_labels(outputs, targets, indices)
        l_bbox, l_giou = self.loss_boxes(outputs, targets, indices)
        return (self.loss_ce_w * l_ce +
                self.loss_bbox_w * l_bbox +
                self.loss_giou_w * l_giou), l_ce, l_bbox, l_giou

    def forward(self, outputs, targets):
        total, l_ce, l_bbox, l_giou = self._compute_loss(outputs, targets)

        losses = {
            'loss_ce': l_ce,
            'loss_bbox': l_bbox,
            'loss_giou': l_giou,
        }

        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_total, aux_ce, aux_bb, aux_gi = self._compute_loss(
                    aux, targets)
                total = total + aux_total
                losses[f'loss_ce_aux{i}'] = aux_ce
                losses[f'loss_bbox_aux{i}'] = aux_bb
                losses[f'loss_giou_aux{i}'] = aux_gi

        losses['total'] = total
        return losses


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def prepare_detr_targets(batch_boxes, batch_num_boxes, input_size=128):
    """Convert collated batch boxes to DETR target format.

    Parameters
    ----------
    batch_boxes : (B, max_n, 6)  [z1,y1,x1,z2,y2,x2] in 128^3 space
    batch_num_boxes : (B,)       valid box counts

    Returns
    -------
    list of B dicts with:
        labels : (M,) long — all 0 (tumour class)
        boxes  : (M, 6) float — [cz,cy,cx,dz,dy,dx] normalised to [0,1]
    """
    targets = []
    B = batch_boxes.shape[0]
    for b in range(B):
        n = batch_num_boxes[b].item()
        if n == 0:
            targets.append({
                'labels': torch.zeros(0, dtype=torch.int64),
                'boxes': torch.zeros(0, 6, dtype=torch.float32),
            })
            continue

        boxes = batch_boxes[b, :n].float()            # (n, 6)
        boxes_norm = boxes / input_size               # normalise to [0, 1]
        boxes_cxcycz = box_zyxzyx_to_cxcyczdhwd(boxes_norm)
        labels = torch.zeros(n, dtype=torch.int64)

        targets.append({'labels': labels, 'boxes': boxes_cxcycz})
    return targets
