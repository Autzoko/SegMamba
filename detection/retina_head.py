"""
RetinaNet-style 3D Detection Head.

Uses shared convolutional towers for classification and box regression,
with proper weight initialization for focal loss stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class Retina3DHead(nn.Module):
    """
    RetinaNet-style 3D detection head.

    Consists of two parallel towers:
    - Classification tower: predicts objectness/class scores
    - Regression tower: predicts box deltas

    Both towers use shared convolutions across all FPN levels.

    Args:
        in_channels: Input channels from FPN
        num_classes: Number of object classes (1 for single-class detection)
        num_anchors: Number of anchors per spatial location
        num_convs: Number of convolutions in each tower
        prior_prob: Prior probability for focal loss initialization
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        num_anchors: int = 18,  # 3 sizes * 3 ratios * 2 z_ratios
        num_convs: int = 4,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification tower
        cls_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.SiLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Regression tower
        reg_tower = []
        for i in range(num_convs):
            reg_tower.append(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.SiLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Classification output: num_anchors * num_classes
        self.cls_logits = nn.Conv3d(
            in_channels, num_anchors * num_classes,
            kernel_size=3, padding=1
        )

        # Box regression output: num_anchors * 6 (dx, dy, dw, dh, dz, dd)
        self.box_reg = nn.Conv3d(
            in_channels, num_anchors * 6,
            kernel_size=3, padding=1
        )

        # Optional: centerness prediction (like FCOS)
        self.centerness = nn.Conv3d(
            in_channels, num_anchors,
            kernel_size=3, padding=1
        )

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        """Initialize weights with proper bias for focal loss."""
        for module in [self.cls_tower, self.reg_tower]:
            for m in module.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Classification head initialization
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        # Bias initialization for focal loss: -log((1-prior)/prior)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        # Regression head initialization
        nn.init.normal_(self.box_reg.weight, std=0.01)
        nn.init.constant_(self.box_reg.bias, 0)

        # Centerness initialization
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            features: List of FPN feature maps, each (B, C, D, H, W)

        Returns:
            cls_logits: List of classification logits, each (B, A*C, D, H, W)
            box_deltas: List of box deltas, each (B, A*6, D, H, W)
            centerness: List of centerness predictions, each (B, A, D, H, W)
        """
        cls_logits = []
        box_deltas = []
        centerness = []

        for feat in features:
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_logits.append(self.cls_logits(cls_feat))
            box_deltas.append(self.box_reg(reg_feat))
            centerness.append(self.centerness(reg_feat))

        return cls_logits, box_deltas, centerness

    def forward_single(
        self,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single feature level.

        Args:
            feature: Single FPN feature map (B, C, D, H, W)

        Returns:
            cls_logits: (B, A*num_classes, D, H, W)
            box_deltas: (B, A*6, D, H, W)
            centerness: (B, A, D, H, W)
        """
        cls_feat = self.cls_tower(feature)
        reg_feat = self.reg_tower(feature)

        return (
            self.cls_logits(cls_feat),
            self.box_reg(reg_feat),
            self.centerness(reg_feat),
        )


def reshape_head_outputs(
    cls_logits: List[torch.Tensor],
    box_deltas: List[torch.Tensor],
    centerness: List[torch.Tensor],
    num_anchors: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Reshape and concatenate head outputs across all FPN levels.

    Args:
        cls_logits: List of (B, A*C, D, H, W) tensors
        box_deltas: List of (B, A*6, D, H, W) tensors
        centerness: List of (B, A, D, H, W) tensors
        num_anchors: Number of anchors per location
        num_classes: Number of classes

    Returns:
        cls_flat: (B, N, C) classification logits
        box_flat: (B, N, 6) box deltas
        ctr_flat: (B, N, 1) centerness
        num_anchors_per_level: List of anchor counts per level
    """
    batch_size = cls_logits[0].shape[0]
    num_anchors_per_level = []

    cls_list = []
    box_list = []
    ctr_list = []

    for cls, box, ctr in zip(cls_logits, box_deltas, centerness):
        # cls: (B, A*C, D, H, W) -> (B, D*H*W*A, C)
        B, _, D, H, W = cls.shape
        num_anchors_level = D * H * W * num_anchors
        num_anchors_per_level.append(num_anchors_level)

        # Reshape: (B, A*C, D, H, W) -> (B, A, C, D, H, W) -> (B, D, H, W, A, C) -> (B, N, C)
        cls = cls.view(B, num_anchors, num_classes, D, H, W)
        cls = cls.permute(0, 3, 4, 5, 1, 2).reshape(B, num_anchors_level, num_classes)
        cls_list.append(cls)

        # box: (B, A*6, D, H, W) -> (B, N, 6)
        box = box.view(B, num_anchors, 6, D, H, W)
        box = box.permute(0, 3, 4, 5, 1, 2).reshape(B, num_anchors_level, 6)
        box_list.append(box)

        # ctr: (B, A, D, H, W) -> (B, N, 1)
        ctr = ctr.view(B, num_anchors, D, H, W)
        ctr = ctr.permute(0, 2, 3, 4, 1).reshape(B, num_anchors_level, 1)
        ctr_list.append(ctr)

    cls_flat = torch.cat(cls_list, dim=1)
    box_flat = torch.cat(box_list, dim=1)
    ctr_flat = torch.cat(ctr_list, dim=1)

    return cls_flat, box_flat, ctr_flat, num_anchors_per_level


class SharedRetina3DHead(nn.Module):
    """
    Alternative head design with truly shared weights across levels.

    Uses scale-specific learnable parameters to adjust for different
    feature scales while sharing the core conv weights.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        num_anchors: int = 18,
        num_convs: int = 4,
        num_levels: int = 4,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_levels = num_levels

        # Shared classification tower
        self.cls_tower = nn.ModuleList()
        for i in range(num_convs):
            self.cls_tower.append(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            self.cls_tower.append(nn.GroupNorm(32, in_channels))
            self.cls_tower.append(nn.SiLU(inplace=True))

        # Shared regression tower
        self.reg_tower = nn.ModuleList()
        for i in range(num_convs):
            self.reg_tower.append(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            self.reg_tower.append(nn.GroupNorm(32, in_channels))
            self.reg_tower.append(nn.SiLU(inplace=True))

        # Scale-specific regression scales (like FCOS)
        self.scales = nn.Parameter(torch.ones(num_levels))

        # Output layers
        self.cls_logits = nn.Conv3d(
            in_channels, num_anchors * num_classes,
            kernel_size=3, padding=1
        )
        self.box_reg = nn.Conv3d(
            in_channels, num_anchors * 6,
            kernel_size=3, padding=1
        )

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_logits = []
        box_deltas = []

        for level, feat in enumerate(features):
            # Classification tower
            cls_feat = feat
            for layer in self.cls_tower:
                cls_feat = layer(cls_feat)

            # Regression tower
            reg_feat = feat
            for layer in self.reg_tower:
                reg_feat = layer(reg_feat)

            # Outputs
            cls = self.cls_logits(cls_feat)
            box = self.box_reg(reg_feat) * self.scales[level]

            cls_logits.append(cls)
            box_deltas.append(box)

        return cls_logits, box_deltas
