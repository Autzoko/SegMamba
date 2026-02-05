"""
SegMamba-Det: 3D Bounding Box Detection using MambaEncoder backbone.

Architecture:
    MambaEncoder (reused from segmamba.py) -> FPN3D -> FCOS-style 3D detection head

Input:  (B, 1, 128, 128, 128)
Output: per-level classification, regression, centerness predictions

Does NOT modify any original SegMamba code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_segmamba.segmamba import MambaEncoder


class FPN3D(nn.Module):
    """Feature Pyramid Network for 3D multi-scale feature fusion."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv3d(c, out_channels, 1) for c in in_channels_list])
        self.smooth_convs = nn.ModuleList(
            [nn.Conv3d(out_channels, out_channels, 3, padding=1)
             for _ in in_channels_list])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Parameters
        ----------
        features : tuple of 4 tensors from MambaEncoder
            [0] (B, 48, 64,64,64)  stride 2
            [1] (B, 96, 32,32,32)  stride 4
            [2] (B,192, 16,16,16)  stride 8
            [3] (B,384,  8, 8, 8)  stride 16
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='trilinear', align_corners=False)

        return [s(l) for s, l in zip(self.smooth_convs, laterals)]


class FCOS3DHead(nn.Module):
    """Anchor-free 3D detection head (shared across FPN levels)."""

    def __init__(self, in_channels, num_classes=1, num_convs=4):
        super().__init__()

        def _make_tower(n):
            layers = []
            for _ in range(n):
                layers.extend([
                    nn.Conv3d(in_channels, in_channels, 3, padding=1,
                              bias=False),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ])
            return nn.Sequential(*layers)

        self.cls_tower = _make_tower(num_convs)
        self.reg_tower = _make_tower(num_convs)

        self.cls_logits = nn.Conv3d(in_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv3d(in_channels, 6, 3, padding=1)
        self.ctr_pred = nn.Conv3d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_tower, self.reg_tower]:
            for layer in modules:
                if isinstance(layer, nn.Conv3d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -4.6)  # sigmoid ≈ 0.01
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
        nn.init.normal_(self.ctr_pred.weight, std=0.01)
        nn.init.constant_(self.ctr_pred.bias, 0)

    def forward(self, x):
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        cls_score = self.cls_logits(cls_feat)
        reg_dist = F.relu(self.reg_pred(reg_feat))
        ctr_score = self.ctr_pred(cls_feat)
        return cls_score, reg_dist, ctr_score


class SegMambaDet(nn.Module):
    """SegMamba-Det: 3D object detection with MambaEncoder backbone.

    Input must be (B, in_chans, 128, 128, 128) to match the hardcoded
    num_slices_list=[64,32,16,8] in MambaEncoder.
    """

    STRIDES = [2, 4, 8, 16]

    def __init__(self, in_chans=1, num_classes=1,
                 depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
                 fpn_channels=128, num_head_convs=4):
        super().__init__()

        self.backbone = MambaEncoder(
            in_chans=in_chans, depths=depths, dims=feat_size)

        self.fpn = FPN3D(feat_size, fpn_channels)
        self.head = FCOS3DHead(fpn_channels, num_classes, num_head_convs)

        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(len(feat_size))])

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)

        all_cls, all_reg, all_ctr = [], [], []
        for i, feat in enumerate(fpn_features):
            cls_score, reg_dist, ctr_score = self.head(feat)
            all_cls.append(cls_score)
            all_reg.append(reg_dist * self.scales[i])
            all_ctr.append(ctr_score)

        return all_cls, all_reg, all_ctr
