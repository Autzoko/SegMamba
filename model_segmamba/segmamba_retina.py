"""
SegMamba with RetinaNet-style 3D Detection Head.

Combines SegMamba segmentation backbone with nnDetection-style anchor-based
detection using FPN and RetinaNet head.

Architecture:
    MambaEncoder (shared backbone)
        ├── Segmentation Decoder (UNETR-style, unchanged)
        │   └── Seg output (Dice + CE)
        └── Detection Pathway
            ├── FPN3D (multi-scale features)
            └── Retina3DHead (focal + L1 + GIoU)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .segmamba import MambaEncoder

# Import detection components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.fpn import FPN3D
from detection.retina_head import Retina3DHead, reshape_head_outputs
from detection.anchors import AnchorGenerator3D
from detection.box_coder import BoxCoder3D
from detection.atss_matcher import ATSSMatcher
from detection.sampler import HardNegativeSampler
from detection.losses import RetinaLoss, focal_loss, smooth_l1_loss, giou_loss_3d, nms_3d


class SegMambaWithRetina(nn.Module):
    """
    SegMamba with RetinaNet-style 3D object detection.

    Combines the SegMamba segmentation backbone with an anchor-based detection
    pathway using FPN and RetinaNet-style heads.

    Args:
        in_chans: Input channels (1 for grayscale)
        out_chans: Segmentation output channels (2 for binary)
        depths: Number of Mamba layers at each stage
        feat_size: Feature dimensions at each stage
        fpn_channels: FPN output channels
        num_anchors: Anchors per spatial location
        num_classes: Detection classes (1 for tumor)
        num_head_convs: Conv layers in detection head
        hidden_size: Size for encoder5
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 2,
        depths: List[int] = [2, 2, 2, 2],
        feat_size: List[int] = [48, 96, 192, 384],
        fpn_channels: int = 128,
        num_anchors: int = 18,  # 3 sizes * 3 ratios * 2 z_ratios
        num_classes: int = 1,
        num_head_convs: int = 4,
        hidden_size: int = 768,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # ============== Shared Backbone ==============
        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        # ============== Segmentation Pathway (unchanged from SegMamba) ==============
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.seg_out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=out_chans,
        )

        # ============== Detection Pathway ==============
        # FPN takes features from MambaEncoder levels 1-3 plus pooled level 3
        # Input channels: [96, 192, 384, 384] for strides [4, 8, 16, 32]
        self.fpn = FPN3D(
            in_channels=[feat_size[1], feat_size[2], feat_size[3], feat_size[3]],
            out_channels=fpn_channels,
            num_outs=4,
        )

        # Detection head
        self.retina_head = Retina3DHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=num_head_convs,
        )

        # Anchor generator
        self.anchor_generator = AnchorGenerator3D(
            sizes_per_level=(
                (8, 16, 32),      # P4 (stride 4)
                (16, 32, 64),     # P8 (stride 8)
                (32, 64, 128),    # P16 (stride 16)
                (64, 128, 256),   # P32 (stride 32)
            ),
            aspect_ratios=(0.5, 1.0, 2.0),
            z_ratios=(0.5, 1.0),
            strides=(4, 8, 16, 32),
        )

        # Box coder
        self.box_coder = BoxCoder3D()

    def forward(
        self,
        x: torch.Tensor,
        return_seg: bool = True,
        return_det: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            return_seg: Whether to compute segmentation output
            return_det: Whether to compute detection output

        Returns:
            Dictionary containing:
            - 'seg_logits': (B, out_chans, D, H, W) segmentation logits
            - 'cls_logits': List of (B, A*C, D, H, W) classification logits
            - 'box_deltas': List of (B, A*6, D, H, W) box deltas
            - 'centerness': List of (B, A, D, H, W) centerness predictions
            - 'anchors': (N, 6) anchor boxes
            - 'num_anchors_per_level': List of anchor counts per level
        """
        results = {}

        # Shared backbone forward
        outs = self.vit(x)
        # outs[0]: (B, 48, D/2, H/2, W/2) - stride 2
        # outs[1]: (B, 96, D/4, H/4, W/4) - stride 4
        # outs[2]: (B, 192, D/8, H/8, W/8) - stride 8
        # outs[3]: (B, 384, D/16, H/16, W/16) - stride 16

        # Segmentation pathway
        if return_seg:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(outs[0])
            enc3 = self.encoder3(outs[1])
            enc4 = self.encoder4(outs[2])
            enc_hidden = self.encoder5(outs[3])

            dec3 = self.decoder5(enc_hidden, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0)

            results['seg_logits'] = self.seg_out(out)

        # Detection pathway
        if return_det:
            # Build FPN input: [P4, P8, P16, P32]
            # P4 = outs[1], P8 = outs[2], P16 = outs[3], P32 = pooled(outs[3])
            p32 = F.max_pool3d(outs[3], kernel_size=2, stride=2)
            fpn_inputs = [outs[1], outs[2], outs[3], p32]

            # FPN forward
            fpn_features = self.fpn(fpn_inputs)

            # Detection head forward
            cls_logits, box_deltas, centerness = self.retina_head(fpn_features)

            # Generate anchors
            anchors, num_anchors_per_level = self.anchor_generator(fpn_features)

            results['cls_logits'] = cls_logits
            results['box_deltas'] = box_deltas
            results['centerness'] = centerness
            results['anchors'] = anchors
            results['num_anchors_per_level'] = num_anchors_per_level

        return results

    def forward_seg_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation only (for compatibility)."""
        results = self.forward(x, return_seg=True, return_det=False)
        return results['seg_logits']

    def forward_det_only(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Forward pass for detection only."""
        results = self.forward(x, return_seg=False, return_det=True)

        # Reshape outputs
        cls_flat, box_flat, ctr_flat, num_per_level = reshape_head_outputs(
            results['cls_logits'],
            results['box_deltas'],
            results['centerness'],
            self.num_anchors,
            self.num_classes,
        )

        return cls_flat, box_flat, ctr_flat, results['anchors'], num_per_level


def load_pretrained_segmamba(
    model: SegMambaWithRetina,
    checkpoint_path: str,
    strict: bool = False,
) -> SegMambaWithRetina:
    """
    Load pretrained SegMamba weights into SegMambaWithRetina.

    The detection components (FPN, retina_head, anchor_generator) will be
    randomly initialized while the shared backbone and segmentation decoder
    will be loaded from the checkpoint.

    Args:
        model: SegMambaWithRetina model
        checkpoint_path: Path to SegMamba checkpoint
        strict: Whether to require exact key match

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }

    # Map 'out' to 'seg_out' if needed
    if 'out.conv.conv.weight' in state_dict and 'seg_out.conv.conv.weight' not in state_dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('out.'):
                new_k = 'seg_out.' + k[4:]
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Load with strict=False to ignore detection components
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded pretrained SegMamba from {checkpoint_path}")
    print(f"  Missing keys (detection components): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    return model


class DetectionPostProcessor:
    """
    Post-processor for detection outputs.

    Handles:
    - Box decoding
    - Score thresholding
    - NMS
    """

    def __init__(
        self,
        box_coder: BoxCoder3D,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        self.box_coder = box_coder
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    @torch.no_grad()
    def __call__(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        anchors: torch.Tensor,
        centerness: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process detection outputs.

        Args:
            cls_logits: (N,) or (N, 1) classification logits
            box_deltas: (N, 6) box deltas
            anchors: (N, 6) anchor boxes
            centerness: (N,) optional centerness scores

        Returns:
            boxes: (K, 6) detected boxes [x1, y1, x2, y2, z1, z2]
            scores: (K,) detection scores
        """
        # Compute scores
        scores = torch.sigmoid(cls_logits.view(-1))

        # Apply centerness if provided
        if centerness is not None:
            scores = scores * torch.sigmoid(centerness.view(-1))

        # Filter by score threshold
        keep = scores > self.score_threshold
        scores = scores[keep]
        box_deltas = box_deltas[keep]
        anchors = anchors[keep]

        if len(scores) == 0:
            return torch.zeros(0, 6, device=cls_logits.device), torch.zeros(0, device=cls_logits.device)

        # Decode boxes
        boxes = self.box_coder.decode(box_deltas, anchors)

        # NMS
        keep = nms_3d(boxes, scores, self.nms_threshold)
        keep = keep[:self.max_detections]

        return boxes[keep], scores[keep]
