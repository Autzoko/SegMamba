"""
3D Box Encoder/Decoder for anchor-based detection.

Encodes ground truth boxes relative to anchors as (dx, dy, dw, dh, dz, dd).
Decodes predicted deltas back to absolute box coordinates.

Box format: [x1, y1, x2, y2, z1, z2]
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class BoxCoder3D(nn.Module):
    """
    Encode/decode 3D boxes relative to anchors.

    Encoding:
        dx = (gt_cx - anchor_cx) / anchor_w
        dy = (gt_cy - anchor_cy) / anchor_h
        dz = (gt_cz - anchor_cz) / anchor_d
        dw = log(gt_w / anchor_w)
        dh = log(gt_h / anchor_h)
        dd = log(gt_d / anchor_d)

    Args:
        weights: Scaling factors for (dx, dy, dz, dw, dh, dd)
        clip: Maximum value for exp in decode (prevents overflow)
    """

    def __init__(
        self,
        weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        clip: float = math.log(1000. / 16),
    ):
        super().__init__()
        self.weights = weights
        self.clip = clip

    def encode(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode ground truth boxes relative to anchors.

        Args:
            gt_boxes: (N, 6) ground truth boxes [x1, y1, x2, y2, z1, z2]
            anchors: (N, 6) anchor boxes [x1, y1, x2, y2, z1, z2]

        Returns:
            deltas: (N, 6) encoded deltas [dx, dy, dw, dh, dz, dd]
        """
        # Convert to center format
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_cz = (gt_boxes[:, 4] + gt_boxes[:, 5]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_d = gt_boxes[:, 5] - gt_boxes[:, 4]

        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_cz = (anchors[:, 4] + anchors[:, 5]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_d = anchors[:, 5] - anchors[:, 4]

        # Prevent division by zero
        anchor_w = torch.clamp(anchor_w, min=1.0)
        anchor_h = torch.clamp(anchor_h, min=1.0)
        anchor_d = torch.clamp(anchor_d, min=1.0)
        gt_w = torch.clamp(gt_w, min=1.0)
        gt_h = torch.clamp(gt_h, min=1.0)
        gt_d = torch.clamp(gt_d, min=1.0)

        # Encode
        dx = self.weights[0] * (gt_cx - anchor_cx) / anchor_w
        dy = self.weights[1] * (gt_cy - anchor_cy) / anchor_h
        dw = self.weights[3] * torch.log(gt_w / anchor_w)
        dh = self.weights[4] * torch.log(gt_h / anchor_h)
        dz = self.weights[2] * (gt_cz - anchor_cz) / anchor_d
        dd = self.weights[5] * torch.log(gt_d / anchor_d)

        deltas = torch.stack([dx, dy, dw, dh, dz, dd], dim=1)
        return deltas

    def decode(
        self,
        deltas: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode predicted deltas to boxes.

        Args:
            deltas: (N, 6) predicted deltas [dx, dy, dw, dh, dz, dd]
            anchors: (N, 6) anchor boxes [x1, y1, x2, y2, z1, z2]

        Returns:
            boxes: (N, 6) decoded boxes [x1, y1, x2, y2, z1, z2]
        """
        # Unpack deltas
        dx = deltas[:, 0] / self.weights[0]
        dy = deltas[:, 1] / self.weights[1]
        dw = deltas[:, 2] / self.weights[3]
        dh = deltas[:, 3] / self.weights[4]
        dz = deltas[:, 4] / self.weights[2]
        dd = deltas[:, 5] / self.weights[5]

        # Clip to prevent overflow
        dw = torch.clamp(dw, max=self.clip)
        dh = torch.clamp(dh, max=self.clip)
        dd = torch.clamp(dd, max=self.clip)

        # Convert anchors to center format
        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_cz = (anchors[:, 4] + anchors[:, 5]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_d = anchors[:, 5] - anchors[:, 4]

        # Decode
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_cz = dz * anchor_d + anchor_cz
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h
        pred_d = torch.exp(dd) * anchor_d

        # Convert back to corner format
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        z1 = pred_cz - pred_d / 2
        z2 = pred_cz + pred_d / 2

        boxes = torch.stack([x1, y1, x2, y2, z1, z2], dim=1)
        return boxes

    def decode_batch(
        self,
        deltas: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode predicted deltas to boxes for batched input.

        Args:
            deltas: (B, N, 6) predicted deltas
            anchors: (N, 6) anchor boxes (same for all batch elements)

        Returns:
            boxes: (B, N, 6) decoded boxes
        """
        batch_size = deltas.shape[0]

        # Expand anchors for batch
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)

        # Unpack deltas
        dx = deltas[..., 0] / self.weights[0]
        dy = deltas[..., 1] / self.weights[1]
        dw = deltas[..., 2] / self.weights[3]
        dh = deltas[..., 3] / self.weights[4]
        dz = deltas[..., 4] / self.weights[2]
        dd = deltas[..., 5] / self.weights[5]

        # Clip
        dw = torch.clamp(dw, max=self.clip)
        dh = torch.clamp(dh, max=self.clip)
        dd = torch.clamp(dd, max=self.clip)

        # Anchor centers
        anchor_cx = (anchors[..., 0] + anchors[..., 2]) / 2
        anchor_cy = (anchors[..., 1] + anchors[..., 3]) / 2
        anchor_cz = (anchors[..., 4] + anchors[..., 5]) / 2
        anchor_w = anchors[..., 2] - anchors[..., 0]
        anchor_h = anchors[..., 3] - anchors[..., 1]
        anchor_d = anchors[..., 5] - anchors[..., 4]

        # Decode
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_cz = dz * anchor_d + anchor_cz
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h
        pred_d = torch.exp(dd) * anchor_d

        # Corner format
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        z1 = pred_cz - pred_d / 2
        z2 = pred_cz + pred_d / 2

        boxes = torch.stack([x1, y1, x2, y2, z1, z2], dim=-1)
        return boxes


def clip_boxes_to_volume(boxes: torch.Tensor, volume_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Clip boxes to volume boundaries.

    Args:
        boxes: (N, 6) boxes [x1, y1, x2, y2, z1, z2]
        volume_shape: (D, H, W) volume dimensions

    Returns:
        clipped_boxes: (N, 6) clipped boxes
    """
    d, h, w = volume_shape

    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)  # x1
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)  # y1
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)  # x2
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)  # y2
    boxes[:, 4] = boxes[:, 4].clamp(min=0, max=d)  # z1
    boxes[:, 5] = boxes[:, 5].clamp(min=0, max=d)  # z2

    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float = 1.0) -> torch.Tensor:
    """
    Remove boxes smaller than min_size in any dimension.

    Args:
        boxes: (N, 6) boxes [x1, y1, x2, y2, z1, z2]
        min_size: Minimum size in each dimension

    Returns:
        keep: (N,) boolean mask of boxes to keep
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    d = boxes[:, 5] - boxes[:, 4]

    keep = (w >= min_size) & (h >= min_size) & (d >= min_size)
    return keep
