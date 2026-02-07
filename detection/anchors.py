"""
3D Anchor Generation for nnDetection-style detection.

Generates multi-scale anchors at each FPN level with various sizes and aspect ratios.
Box format: [x1, y1, x2, y2, z1, z2] (nnDetection convention)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math


class AnchorGenerator3D(nn.Module):
    """
    Generate 3D anchors for multi-scale feature maps.

    For each FPN level, generates anchors with different sizes and aspect ratios.
    Anchors are defined by (size_xy, aspect_ratio_hw, size_z).

    Args:
        sizes_per_level: Base sizes for each FPN level
        aspect_ratios: Height/Width aspect ratios (applied to x-y plane)
        z_ratios: Depth ratios relative to xy size
        strides: Feature map strides for each level
    """

    def __init__(
        self,
        sizes_per_level: Tuple[Tuple[int, ...], ...] = (
            (8, 16, 32),      # P4 (stride 4)
            (16, 32, 64),     # P8 (stride 8)
            (32, 64, 128),    # P16 (stride 16)
            (64, 128, 256),   # P32 (stride 32)
        ),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        z_ratios: Tuple[float, ...] = (0.5, 1.0),
        strides: Tuple[int, ...] = (4, 8, 16, 32),
    ):
        super().__init__()
        self.sizes_per_level = sizes_per_level
        self.aspect_ratios = aspect_ratios
        self.z_ratios = z_ratios
        self.strides = strides

        # Precompute base anchors for each level
        self.num_anchors_per_location = []
        self.register_buffer('_dummy', torch.empty(0))  # For device tracking

        # Cache base anchors
        self._base_anchors_cache = {}

    @property
    def device(self):
        return self._dummy.device

    def _get_base_anchors(self, level: int) -> torch.Tensor:
        """
        Generate base anchors for a given FPN level.

        Returns:
            Tensor of shape (num_anchors, 6) with [x1, y1, x2, y2, z1, z2]
        """
        if level in self._base_anchors_cache:
            cached = self._base_anchors_cache[level]
            if cached.device == self.device:
                return cached

        sizes = self.sizes_per_level[level]
        anchors = []

        for size in sizes:
            for ar in self.aspect_ratios:
                # ar = h/w, so h = size * sqrt(ar), w = size / sqrt(ar)
                w = size / math.sqrt(ar)
                h = size * math.sqrt(ar)

                for zr in self.z_ratios:
                    d = size * zr  # depth

                    # Anchor centered at origin: [-w/2, -h/2, w/2, h/2, -d/2, d/2]
                    anchor = torch.tensor([
                        -w / 2, -h / 2, w / 2, h / 2, -d / 2, d / 2
                    ], dtype=torch.float32, device=self.device)
                    anchors.append(anchor)

        base_anchors = torch.stack(anchors, dim=0)
        self._base_anchors_cache[level] = base_anchors
        return base_anchors

    def num_anchors_at_level(self, level: int) -> int:
        """Return number of anchors per location at a given level."""
        return (len(self.sizes_per_level[level]) *
                len(self.aspect_ratios) *
                len(self.z_ratios))

    @property
    def total_anchors_per_location(self) -> List[int]:
        """Return list of anchors per location for each level."""
        return [self.num_anchors_at_level(i) for i in range(len(self.strides))]

    def forward(
        self,
        feature_maps: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate anchors for all FPN levels.

        Args:
            feature_maps: List of feature maps from FPN, each (B, C, D, H, W)

        Returns:
            anchors: Tensor of shape (total_anchors, 6) with [x1, y1, x2, y2, z1, z2]
            num_anchors_per_level: List of anchor counts per level
        """
        all_anchors = []
        num_anchors_per_level = []

        for level, feat in enumerate(feature_maps):
            stride = self.strides[level]
            _, _, d, h, w = feat.shape

            # Generate grid centers
            # Centers are at stride * (i + 0.5) for i in [0, size-1]
            shift_x = (torch.arange(w, device=self.device) + 0.5) * stride
            shift_y = (torch.arange(h, device=self.device) + 0.5) * stride
            shift_z = (torch.arange(d, device=self.device) + 0.5) * stride

            # Create meshgrid
            shift_z, shift_y, shift_x = torch.meshgrid(
                shift_z, shift_y, shift_x, indexing='ij'
            )

            # Flatten and create shifts: [x, y, x, y, z, z]
            shifts = torch.stack([
                shift_x.flatten(),
                shift_y.flatten(),
                shift_x.flatten(),
                shift_y.flatten(),
                shift_z.flatten(),
                shift_z.flatten(),
            ], dim=1)  # (D*H*W, 6)

            # Get base anchors for this level
            base_anchors = self._get_base_anchors(level)  # (A, 6)

            # Broadcast add: (D*H*W, 1, 6) + (1, A, 6) -> (D*H*W, A, 6)
            anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
            anchors = anchors.reshape(-1, 6)  # (D*H*W*A, 6)

            all_anchors.append(anchors)
            num_anchors_per_level.append(anchors.shape[0])

        # Concatenate all anchors
        all_anchors = torch.cat(all_anchors, dim=0)

        return all_anchors, num_anchors_per_level

    def generate_for_shape(
        self,
        input_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate anchors for a given input shape without feature maps.

        Args:
            input_shape: (D, H, W) of the input volume

        Returns:
            anchors: Tensor of shape (total_anchors, 6)
            num_anchors_per_level: List of anchor counts per level
        """
        d, h, w = input_shape
        all_anchors = []
        num_anchors_per_level = []

        for level, stride in enumerate(self.strides):
            # Feature map size at this level
            fd = math.ceil(d / stride)
            fh = math.ceil(h / stride)
            fw = math.ceil(w / stride)

            # Generate grid centers
            shift_x = (torch.arange(fw, device=self.device) + 0.5) * stride
            shift_y = (torch.arange(fh, device=self.device) + 0.5) * stride
            shift_z = (torch.arange(fd, device=self.device) + 0.5) * stride

            shift_z, shift_y, shift_x = torch.meshgrid(
                shift_z, shift_y, shift_x, indexing='ij'
            )

            shifts = torch.stack([
                shift_x.flatten(),
                shift_y.flatten(),
                shift_x.flatten(),
                shift_y.flatten(),
                shift_z.flatten(),
                shift_z.flatten(),
            ], dim=1)

            base_anchors = self._get_base_anchors(level)
            anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
            anchors = anchors.reshape(-1, 6)

            all_anchors.append(anchors)
            num_anchors_per_level.append(anchors.shape[0])

        all_anchors = torch.cat(all_anchors, dim=0)
        return all_anchors, num_anchors_per_level


def compute_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D IoU between two sets of boxes.

    Args:
        boxes1: (N, 6) with [x1, y1, x2, y2, z1, z2]
        boxes2: (M, 6) with [x1, y1, x2, y2, z1, z2]

    Returns:
        iou: (N, M) IoU matrix
    """
    # Get corners
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

    iou = inter_vol / (union + 1e-6)
    return iou


def box_center_3d(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute box centers from corner format.

    Args:
        boxes: (N, 6) with [x1, y1, x2, y2, z1, z2]

    Returns:
        centers: (N, 3) with [cx, cy, cz]
    """
    x1, y1, x2, y2, z1, z2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    return torch.stack([cx, cy, cz], dim=-1)


def point_in_box_3d(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Check if points are inside boxes.

    Args:
        points: (N, 3) with [x, y, z]
        boxes: (M, 6) with [x1, y1, x2, y2, z1, z2]

    Returns:
        inside: (N, M) boolean tensor
    """
    x, y, z = points.unbind(-1)
    x1, y1, x2, y2, z1, z2 = boxes.unbind(-1)

    inside_x = (x.unsqueeze(1) >= x1.unsqueeze(0)) & (x.unsqueeze(1) <= x2.unsqueeze(0))
    inside_y = (y.unsqueeze(1) >= y1.unsqueeze(0)) & (y.unsqueeze(1) <= y2.unsqueeze(0))
    inside_z = (z.unsqueeze(1) >= z1.unsqueeze(0)) & (z.unsqueeze(1) <= z2.unsqueeze(0))

    return inside_x & inside_y & inside_z
