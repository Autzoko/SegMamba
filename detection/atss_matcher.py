"""
Adaptive Training Sample Selection (ATSS) Matcher for 3D detection.

ATSS automatically selects positive/negative samples based on statistical
properties of IoU distribution, eliminating the need for manual threshold tuning.

Reference: Bridging the Gap Between Anchor-based and Anchor-free Detection
           via Adaptive Training Sample Selection (CVPR 2020)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ATSSMatcher(nn.Module):
    """
    Adaptive Training Sample Selection (ATSS) for 3D anchor matching.

    For each GT box:
    1. Select top-k anchors per FPN level by center distance
    2. Compute IoU for candidate anchors only
    3. Compute adaptive threshold: mean(IoU) + std(IoU)
    4. Filter: keep anchors with IoU > threshold and center inside GT
    5. Resolve conflicts: assign anchor to GT with highest IoU

    Args:
        num_candidates: Number of candidate anchors per FPN level (k)
        center_in_gt: Require anchor center to be inside GT box
        min_iou: Minimum IoU threshold (fallback)
    """

    def __init__(
        self,
        num_candidates: int = 9,
        center_in_gt: bool = True,
        min_iou: float = 0.0,
    ):
        super().__init__()
        self.num_candidates = num_candidates
        self.center_in_gt = center_in_gt
        self.min_iou = min_iou

    @torch.no_grad()
    def forward(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
        num_anchors_per_level: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match anchors to ground truth boxes using ATSS.

        Args:
            gt_boxes: (M, 6) ground truth boxes [x1, y1, x2, y2, z1, z2]
            anchors: (N, 6) anchor boxes [x1, y1, x2, y2, z1, z2]
            num_anchors_per_level: Number of anchors at each FPN level

        Returns:
            matched_gt_indices: (N,) index of matched GT for each anchor (-1 if unmatched)
            matched_iou: (N,) IoU with matched GT (0 if unmatched)
            labels: (N,) 1 for positive, 0 for negative, -1 for ignore
        """
        device = anchors.device
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            # No ground truth: all anchors are negative
            return (
                torch.full((num_anchors,), -1, dtype=torch.long, device=device),
                torch.zeros(num_anchors, device=device),
                torch.zeros(num_anchors, dtype=torch.long, device=device),
            )

        # Compute anchor centers
        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_cz = (anchors[:, 4] + anchors[:, 5]) / 2
        anchor_centers = torch.stack([anchor_cx, anchor_cy, anchor_cz], dim=1)  # (N, 3)

        # Compute GT centers
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_cz = (gt_boxes[:, 4] + gt_boxes[:, 5]) / 2
        gt_centers = torch.stack([gt_cx, gt_cy, gt_cz], dim=1)  # (M, 3)

        # Compute center distances: (N, M)
        distances = torch.cdist(anchor_centers, gt_centers, p=2)

        # Initialize outputs
        matched_gt_indices = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        matched_iou = torch.zeros(num_anchors, device=device)
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)

        # Process each GT box
        candidate_mask = torch.zeros(num_anchors, num_gt, dtype=torch.bool, device=device)

        # Split anchors by level
        level_starts = [0] + list(torch.cumsum(
            torch.tensor(num_anchors_per_level), dim=0
        ).tolist())

        for gt_idx in range(num_gt):
            gt_distances = distances[:, gt_idx]

            # Select top-k candidates per level
            for level_idx, (start, end) in enumerate(zip(level_starts[:-1], level_starts[1:])):
                level_distances = gt_distances[start:end]
                num_level_anchors = end - start

                k = min(self.num_candidates, num_level_anchors)
                _, topk_indices = level_distances.topk(k, largest=False)

                # Mark as candidates
                candidate_mask[start + topk_indices, gt_idx] = True

        # Compute IoU only for candidates (for efficiency)
        iou_matrix = self._compute_iou(anchors, gt_boxes)  # (N, M)

        # For each GT, compute adaptive threshold from candidates
        for gt_idx in range(num_gt):
            candidates = candidate_mask[:, gt_idx]
            if candidates.sum() == 0:
                continue

            candidate_ious = iou_matrix[candidates, gt_idx]

            # Adaptive threshold: mean + std
            iou_mean = candidate_ious.mean()
            iou_std = candidate_ious.std()
            threshold = iou_mean + iou_std
            threshold = max(threshold.item(), self.min_iou)

            # Select positives: IoU > threshold
            positive_mask = (iou_matrix[:, gt_idx] >= threshold) & candidates

            # Optional: require anchor center inside GT
            if self.center_in_gt:
                inside = self._center_in_box(anchor_centers, gt_boxes[gt_idx])
                positive_mask = positive_mask & inside

            # Update matched indices (handle conflicts later)
            new_positives = positive_mask & (matched_gt_indices == -1)
            matched_gt_indices[new_positives] = gt_idx
            matched_iou[new_positives] = iou_matrix[new_positives, gt_idx]

            # Handle conflicts: keep assignment with higher IoU
            conflicts = positive_mask & (matched_gt_indices != -1) & (matched_gt_indices != gt_idx)
            if conflicts.any():
                conflict_indices = conflicts.nonzero(as_tuple=True)[0]
                for idx in conflict_indices:
                    old_gt = matched_gt_indices[idx].item()
                    old_iou = iou_matrix[idx, old_gt]
                    new_iou = iou_matrix[idx, gt_idx]
                    if new_iou > old_iou:
                        matched_gt_indices[idx] = gt_idx
                        matched_iou[idx] = new_iou

        # Set labels: 1 for positives, 0 for negatives
        labels[matched_gt_indices >= 0] = 1

        return matched_gt_indices, matched_iou, labels

    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 3D IoU between two sets of boxes."""
        x1_1, y1_1, x2_1, y2_1, z1_1, z2_1 = boxes1.unbind(-1)
        x1_2, y1_2, x2_2, y2_2, z1_2, z2_2 = boxes2.unbind(-1)

        # Volumes
        vol1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
        vol2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)

        # Intersection
        inter_x1 = torch.max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
        inter_y1 = torch.max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
        inter_z1 = torch.max(z1_1.unsqueeze(1), z1_2.unsqueeze(0))
        inter_x2 = torch.min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
        inter_y2 = torch.min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))
        inter_z2 = torch.min(z2_1.unsqueeze(1), z2_2.unsqueeze(0))

        inter_vol = (
            torch.clamp(inter_x2 - inter_x1, min=0) *
            torch.clamp(inter_y2 - inter_y1, min=0) *
            torch.clamp(inter_z2 - inter_z1, min=0)
        )

        # Union
        union = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol

        return inter_vol / (union + 1e-6)

    def _center_in_box(
        self,
        centers: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """Check if centers are inside a single box."""
        x1, y1, x2, y2, z1, z2 = box.unbind(-1)

        inside_x = (centers[:, 0] >= x1) & (centers[:, 0] <= x2)
        inside_y = (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
        inside_z = (centers[:, 2] >= z1) & (centers[:, 2] <= z2)

        return inside_x & inside_y & inside_z


class IoUMatcher(nn.Module):
    """
    Simple IoU-based matcher with fixed thresholds.

    Args:
        high_threshold: IoU threshold for positive matches
        low_threshold: IoU threshold below which anchors are negative
        allow_low_quality: Allow matches below threshold if they're the best for a GT
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.4,
        allow_low_quality: bool = True,
    ):
        super().__init__()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality = allow_low_quality

    @torch.no_grad()
    def forward(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
        num_anchors_per_level: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match anchors to GT using IoU thresholds.

        Args:
            gt_boxes: (M, 6) ground truth boxes
            anchors: (N, 6) anchor boxes
            num_anchors_per_level: (unused, for API compatibility)

        Returns:
            matched_gt_indices: (N,) matched GT index (-1 if unmatched)
            matched_iou: (N,) IoU with matched GT
            labels: (N,) 1=positive, 0=negative, -1=ignore
        """
        device = anchors.device
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            return (
                torch.full((num_anchors,), -1, dtype=torch.long, device=device),
                torch.zeros(num_anchors, device=device),
                torch.zeros(num_anchors, dtype=torch.long, device=device),
            )

        # Compute IoU matrix
        iou_matrix = self._compute_iou(anchors, gt_boxes)  # (N, M)

        # For each anchor, find best GT
        max_iou, matched_gt = iou_matrix.max(dim=1)

        # Initialize labels
        labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

        # Negatives: IoU < low_threshold
        labels[max_iou < self.low_threshold] = 0

        # Positives: IoU >= high_threshold
        labels[max_iou >= self.high_threshold] = 1

        # Allow low-quality matches: best anchor for each GT
        if self.allow_low_quality:
            best_anchor_per_gt = iou_matrix.argmax(dim=0)  # (M,)
            for gt_idx in range(num_gt):
                anchor_idx = best_anchor_per_gt[gt_idx].item()
                if labels[anchor_idx] != 1:  # Not already positive
                    labels[anchor_idx] = 1
                    matched_gt[anchor_idx] = gt_idx

        # Set matched_gt to -1 for non-positives
        matched_gt_indices = matched_gt.clone()
        matched_gt_indices[labels != 1] = -1

        return matched_gt_indices, max_iou, labels

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute 3D IoU."""
        x1_1, y1_1, x2_1, y2_1, z1_1, z2_1 = boxes1.unbind(-1)
        x1_2, y1_2, x2_2, y2_2, z1_2, z2_2 = boxes2.unbind(-1)

        vol1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
        vol2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)

        inter_x1 = torch.max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
        inter_y1 = torch.max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
        inter_z1 = torch.max(z1_1.unsqueeze(1), z1_2.unsqueeze(0))
        inter_x2 = torch.min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
        inter_y2 = torch.min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))
        inter_z2 = torch.min(z2_1.unsqueeze(1), z2_2.unsqueeze(0))

        inter_vol = (
            torch.clamp(inter_x2 - inter_x1, min=0) *
            torch.clamp(inter_y2 - inter_y1, min=0) *
            torch.clamp(inter_z2 - inter_z1, min=0)
        )

        union = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol

        return inter_vol / (union + 1e-6)
