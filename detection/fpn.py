"""
3D Feature Pyramid Network (FPN) for multi-scale detection.

Takes multi-scale features from MambaEncoder and builds a feature pyramid
with top-down pathway and lateral connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock3D(nn.Module):
    """3D Conv + GroupNorm + SiLU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # Ensure groups divides out_channels
        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups //= 2
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FPN3D(nn.Module):
    """
    3D Feature Pyramid Network.

    Builds a multi-scale feature pyramid from backbone features using:
    - 1x1x1 lateral convolutions to unify channel dimensions
    - Top-down pathway with trilinear upsampling
    - 3x3x3 output convolutions with GroupNorm + SiLU

    Args:
        in_channels: List of input channels for each level (bottom-up)
        out_channels: Output channels for all FPN levels
        num_outs: Number of output levels (can add extra levels via pooling)
        extra_convs_on_inputs: If True, add extra levels from inputs; else from outputs
    """

    def __init__(
        self,
        in_channels: List[int] = [96, 192, 384, 384],
        out_channels: int = 128,
        num_outs: int = 4,
        extra_convs_on_inputs: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        # Lateral convolutions (1x1x1) to reduce channels
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            self.lateral_convs.append(
                nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False)
            )

        # Output convolutions (3x3x3 + GN + SiLU)
        self.output_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.output_convs.append(
                ConvBlock3D(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Extra convolutions for additional levels (if num_outs > num_ins)
        self.extra_convs = nn.ModuleList()
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                if i == 0 and extra_convs_on_inputs:
                    in_ch = in_channels[-1]
                else:
                    in_ch = out_channels
                self.extra_convs.append(
                    ConvBlock3D(in_ch, out_channels, kernel_size=3, stride=2, padding=1)
                )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs: List of feature maps from backbone, ordered from
                    high-resolution to low-resolution (P4, P8, P16, P32)

        Returns:
            outs: List of FPN feature maps, same order as inputs
        """
        assert len(inputs) == self.num_ins, \
            f"Expected {self.num_ins} inputs, got {len(inputs)}"

        # Build lateral features
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(self.num_ins - 2, -1, -1):
            # Upsample higher-level features
            up = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='trilinear',
                align_corners=False,
            )
            laterals[i] = laterals[i] + up

        # Output convolutions
        outs = [
            self.output_convs[i](laterals[i])
            for i in range(self.num_ins)
        ]

        # Extra levels
        if self.num_outs > self.num_ins:
            # P6 from P5 (or inputs[-1])
            if len(self.extra_convs) > 0:
                outs.append(self.extra_convs[0](inputs[-1]))
            # P7+
            for i in range(1, len(self.extra_convs)):
                outs.append(self.extra_convs[i](outs[-1]))

        return outs


class BiFPN3D(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN) for 3D.

    Adds bottom-up pathway after the top-down pathway for better feature fusion.
    Uses weighted feature fusion (fast normalized fusion).

    Args:
        in_channels: List of input channels for each level
        out_channels: Output channels for all levels
        num_repeats: Number of BiFPN repeat blocks
    """

    def __init__(
        self,
        in_channels: List[int] = [96, 192, 384, 384],
        out_channels: int = 128,
        num_repeats: int = 2,
    ):
        super().__init__()
        self.num_levels = len(in_channels)

        # Initial lateral convs (only for first repeat)
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        # BiFPN repeat blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels, self.num_levels)
            for _ in range(num_repeats)
        ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Apply lateral convs
        features = [
            self.lateral_convs[i](inputs[i])
            for i in range(self.num_levels)
        ]

        # Apply BiFPN blocks
        for block in self.bifpn_blocks:
            features = block(features)

        return features


class BiFPNBlock(nn.Module):
    """Single BiFPN block with top-down and bottom-up pathways."""

    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels

        # Top-down fusion weights and convs
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2))
            for _ in range(num_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            ConvBlock3D(channels, channels)
            for _ in range(num_levels - 1)
        ])

        # Bottom-up fusion weights and convs
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i > 0 else 2))
            for i in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            ConvBlock3D(channels, channels)
            for _ in range(num_levels - 1)
        ])

        self.eps = 1e-4

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down pathway
        td_features = [None] * self.num_levels
        td_features[-1] = inputs[-1]

        for i in range(self.num_levels - 2, -1, -1):
            # Fast normalized fusion
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.eps)

            up = F.interpolate(
                td_features[i + 1],
                size=inputs[i].shape[2:],
                mode='trilinear',
                align_corners=False,
            )
            td_features[i] = self.td_convs[i](
                w[0] * inputs[i] + w[1] * up
            )

        # Bottom-up pathway
        outputs = [None] * self.num_levels
        outputs[0] = td_features[0]

        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.eps)

            down = F.interpolate(
                outputs[i - 1],
                size=td_features[i].shape[2:],
                mode='trilinear',
                align_corners=False,
            )

            if i < self.num_levels - 1:
                # Intermediate levels: fuse input, td, and bu
                outputs[i] = self.bu_convs[i - 1](
                    w[0] * inputs[i] + w[1] * td_features[i] + w[2] * down
                )
            else:
                # Last level: fuse td and bu only
                outputs[i] = self.bu_convs[i - 1](
                    w[0] * td_features[i] + w[1] * down
                )

        return outputs
