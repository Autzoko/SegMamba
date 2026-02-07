"""
Detection Losses for anchor-based 3D detection.

Implements:
- Focal Loss for classification (handles class imbalance)
- Smooth L1 Loss for box regression
- 3D GIoU Loss for box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        inputs: (N,) or (N, 1) logits
        targets: (N,) binary targets (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        reduction: 'none', 'mean', or 'sum'

    Returns:
        loss: Focal loss value
    """
    inputs = inputs.view(-1)
    targets = targets.view(-1).float()

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t).pow(gamma)

    loss = focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def focal_loss_multiclass(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for multi-class classification.

    Args:
        inputs: (N, C) logits
        targets: (N,) class indices
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: 'none', 'mean', or 'sum'

    Returns:
        loss: Focal loss value
    """
    num_classes = inputs.shape[1]
    p = F.softmax(inputs, dim=1)

    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes).float()

    ce_loss = F.cross_entropy(inputs, targets, reduction='none')

    p_t = (p * targets_one_hot).sum(dim=1)
    focal_weight = alpha * (1 - p_t).pow(gamma)

    loss = focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def smooth_l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Smooth L1 Loss (Huber Loss).

    loss = 0.5 * x^2 / beta     if |x| < beta
         = |x| - 0.5 * beta     otherwise

    Args:
        inputs: (N, D) predictions
        targets: (N, D) targets
        beta: Threshold for switching between L1 and L2
        reduction: 'none', 'mean', or 'sum'

    Returns:
        loss: Smooth L1 loss
    """
    diff = inputs - targets
    abs_diff = diff.abs()

    loss = torch.where(
        abs_diff < beta,
        0.5 * diff.pow(2) / beta,
        abs_diff - 0.5 * beta
    )

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def giou_loss_3d(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Generalized IoU Loss for 3D boxes.

    GIoU = IoU - (C - U) / C
    where C is the smallest enclosing box, U is the union.

    Args:
        pred_boxes: (N, 6) predicted boxes [x1, y1, x2, y2, z1, z2]
        target_boxes: (N, 6) target boxes
        reduction: 'none', 'mean', or 'sum'

    Returns:
        loss: 1 - GIoU
    """
    giou = compute_giou_3d(pred_boxes, target_boxes)
    loss = 1 - giou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def compute_giou_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 3D Generalized IoU.

    Args:
        boxes1: (N, 6) boxes [x1, y1, x2, y2, z1, z2]
        boxes2: (N, 6) boxes

    Returns:
        giou: (N,) GIoU values in [-1, 1]
    """
    # Ensure valid boxes (x1 < x2, etc.)
    x1_1 = torch.min(boxes1[:, 0], boxes1[:, 2])
    x2_1 = torch.max(boxes1[:, 0], boxes1[:, 2])
    y1_1 = torch.min(boxes1[:, 1], boxes1[:, 3])
    y2_1 = torch.max(boxes1[:, 1], boxes1[:, 3])
    z1_1 = torch.min(boxes1[:, 4], boxes1[:, 5])
    z2_1 = torch.max(boxes1[:, 4], boxes1[:, 5])

    x1_2 = torch.min(boxes2[:, 0], boxes2[:, 2])
    x2_2 = torch.max(boxes2[:, 0], boxes2[:, 2])
    y1_2 = torch.min(boxes2[:, 1], boxes2[:, 3])
    y2_2 = torch.max(boxes2[:, 1], boxes2[:, 3])
    z1_2 = torch.min(boxes2[:, 4], boxes2[:, 5])
    z2_2 = torch.max(boxes2[:, 4], boxes2[:, 5])

    # Volumes
    vol1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
    vol2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)

    # Intersection
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_z1 = torch.max(z1_1, z1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_z2 = torch.min(z2_1, z2_2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_d = torch.clamp(inter_z2 - inter_z1, min=0)
    inter_vol = inter_w * inter_h * inter_d

    # Union
    union = vol1 + vol2 - inter_vol

    # IoU
    iou = inter_vol / (union + 1e-6)

    # Enclosing box
    enclose_x1 = torch.min(x1_1, x1_2)
    enclose_y1 = torch.min(y1_1, y1_2)
    enclose_z1 = torch.min(z1_1, z1_2)
    enclose_x2 = torch.max(x2_1, x2_2)
    enclose_y2 = torch.max(y2_1, y2_2)
    enclose_z2 = torch.max(z2_1, z2_2)

    enclose_vol = (
        (enclose_x2 - enclose_x1) *
        (enclose_y2 - enclose_y1) *
        (enclose_z2 - enclose_z1)
    )

    # GIoU
    giou = iou - (enclose_vol - union) / (enclose_vol + 1e-6)

    return giou


def diou_loss_3d(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Distance IoU Loss for 3D boxes.

    DIoU = IoU - rho^2(b, b_gt) / c^2
    where rho is the Euclidean distance between centers,
    c is the diagonal of the enclosing box.

    Args:
        pred_boxes: (N, 6) predicted boxes
        target_boxes: (N, 6) target boxes
        reduction: 'none', 'mean', or 'sum'

    Returns:
        loss: 1 - DIoU
    """
    diou = compute_diou_3d(pred_boxes, target_boxes)
    loss = 1 - diou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def compute_diou_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Compute 3D Distance IoU."""
    # Centers
    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    cz1 = (boxes1[:, 4] + boxes1[:, 5]) / 2

    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
    cz2 = (boxes2[:, 4] + boxes2[:, 5]) / 2

    # Center distance squared
    rho2 = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2) + (cz1 - cz2).pow(2)

    # Enclosing box diagonal squared
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_z1 = torch.min(boxes1[:, 4], boxes2[:, 4])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclose_z2 = torch.max(boxes1[:, 5], boxes2[:, 5])

    c2 = (
        (enclose_x2 - enclose_x1).pow(2) +
        (enclose_y2 - enclose_y1).pow(2) +
        (enclose_z2 - enclose_z1).pow(2)
    )

    # IoU
    giou = compute_giou_3d(boxes1, boxes2)
    # DIoU = IoU - rho2/c2 (but we need IoU, not GIoU)
    # Recompute IoU
    x1_1 = torch.min(boxes1[:, 0], boxes1[:, 2])
    x2_1 = torch.max(boxes1[:, 0], boxes1[:, 2])
    y1_1 = torch.min(boxes1[:, 1], boxes1[:, 3])
    y2_1 = torch.max(boxes1[:, 1], boxes1[:, 3])
    z1_1 = torch.min(boxes1[:, 4], boxes1[:, 5])
    z2_1 = torch.max(boxes1[:, 4], boxes1[:, 5])

    x1_2 = torch.min(boxes2[:, 0], boxes2[:, 2])
    x2_2 = torch.max(boxes2[:, 0], boxes2[:, 2])
    y1_2 = torch.min(boxes2[:, 1], boxes2[:, 3])
    y2_2 = torch.max(boxes2[:, 1], boxes2[:, 3])
    z1_2 = torch.min(boxes2[:, 4], boxes2[:, 5])
    z2_2 = torch.max(boxes2[:, 4], boxes2[:, 5])

    vol1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
    vol2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_z1 = torch.max(z1_1, z1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_z2 = torch.min(z2_1, z2_2)

    inter_vol = (
        torch.clamp(inter_x2 - inter_x1, min=0) *
        torch.clamp(inter_y2 - inter_y1, min=0) *
        torch.clamp(inter_z2 - inter_z1, min=0)
    )

    union = vol1 + vol2 - inter_vol
    iou = inter_vol / (union + 1e-6)

    diou = iou - rho2 / (c2 + 1e-6)

    return diou


class RetinaLoss(nn.Module):
    """
    Combined loss for RetinaNet-style detection.

    total_loss = focal_loss(cls) + l1_weight * smooth_l1(box) + giou_weight * giou_loss(box)

    Args:
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        l1_weight: Weight for smooth L1 loss
        giou_weight: Weight for GIoU loss
        l1_beta: Beta for smooth L1 loss
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        l1_weight: float = 1.0,
        giou_weight: float = 2.0,
        l1_beta: float = 1.0 / 9.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.l1_beta = l1_beta

    def forward(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        anchors: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        box_coder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute detection losses.

        Args:
            cls_logits: (N,) or (N, 1) classification logits
            box_deltas: (N, 6) predicted box deltas
            anchors: (N, 6) anchor boxes
            targets: (N, 6) target box deltas (encoded)
            labels: (N,) labels (1=pos, 0=neg)
            pos_mask: (N,) boolean mask for positives
            neg_mask: (N,) boolean mask for negatives
            box_coder: Optional box coder for GIoU loss (decodes boxes)

        Returns:
            cls_loss: Classification loss
            l1_loss: Smooth L1 loss on box deltas
            giou_loss: GIoU loss on decoded boxes
        """
        # Classification loss on sampled anchors
        sample_mask = pos_mask | neg_mask
        if sample_mask.sum() == 0:
            return (
                torch.tensor(0.0, device=cls_logits.device),
                torch.tensor(0.0, device=cls_logits.device),
                torch.tensor(0.0, device=cls_logits.device),
            )

        cls_logits_sampled = cls_logits[sample_mask]
        labels_sampled = labels[sample_mask].float()

        cls_loss = focal_loss(
            cls_logits_sampled,
            labels_sampled,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction='sum'
        )

        # Normalize by number of positives
        num_pos = max(pos_mask.sum().item(), 1)
        cls_loss = cls_loss / num_pos

        # Box regression loss on positives only
        if pos_mask.sum() == 0:
            l1_loss = torch.tensor(0.0, device=cls_logits.device)
            giou_loss = torch.tensor(0.0, device=cls_logits.device)
        else:
            box_deltas_pos = box_deltas[pos_mask]
            targets_pos = targets[pos_mask]

            l1_loss = smooth_l1_loss(
                box_deltas_pos,
                targets_pos,
                beta=self.l1_beta,
                reduction='sum'
            ) / num_pos

            # GIoU loss requires decoding boxes
            if box_coder is not None and self.giou_weight > 0:
                anchors_pos = anchors[pos_mask]
                pred_boxes = box_coder.decode(box_deltas_pos, anchors_pos)
                target_boxes = box_coder.decode(targets_pos, anchors_pos)

                giou_loss = giou_loss_3d(
                    pred_boxes,
                    target_boxes,
                    reduction='sum'
                ) / num_pos
            else:
                giou_loss = torch.tensor(0.0, device=cls_logits.device)

        return cls_loss, self.l1_weight * l1_loss, self.giou_weight * giou_loss


def nms_3d(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Non-Maximum Suppression for 3D boxes.

    Args:
        boxes: (N, 6) boxes [x1, y1, x2, y2, z1, z2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: (K,) indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort by score
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = compute_iou_single(boxes[i], boxes[remaining])

        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def compute_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one box and multiple boxes."""
    x1, y1, x2, y2, z1, z2 = box.unbind(-1)
    x1_b, y1_b, x2_b, y2_b, z1_b, z2_b = boxes.unbind(-1)

    vol = (x2 - x1) * (y2 - y1) * (z2 - z1)
    vol_b = (x2_b - x1_b) * (y2_b - y1_b) * (z2_b - z1_b)

    inter_x1 = torch.max(x1, x1_b)
    inter_y1 = torch.max(y1, y1_b)
    inter_z1 = torch.max(z1, z1_b)
    inter_x2 = torch.min(x2, x2_b)
    inter_y2 = torch.min(y2, y2_b)
    inter_z2 = torch.min(z2, z2_b)

    inter_vol = (
        torch.clamp(inter_x2 - inter_x1, min=0) *
        torch.clamp(inter_y2 - inter_y1, min=0) *
        torch.clamp(inter_z2 - inter_z1, min=0)
    )

    union = vol + vol_b - inter_vol
    return inter_vol / (union + 1e-6)
