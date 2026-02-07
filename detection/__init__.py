"""
Detection module for SegMamba-Retina.

nnDetection-style anchor-based 3D object detection with:
- 3D FPN for multi-scale features
- RetinaNet-style detection head
- ATSS matcher for adaptive sample selection
- Focal loss + L1 + GIoU losses
"""

from .anchors import AnchorGenerator3D
from .box_coder import BoxCoder3D
from .fpn import FPN3D
from .retina_head import Retina3DHead
from .atss_matcher import ATSSMatcher
from .sampler import HardNegativeSampler
from .losses import RetinaLoss, focal_loss, smooth_l1_loss, giou_loss_3d

__all__ = [
    'AnchorGenerator3D',
    'BoxCoder3D',
    'FPN3D',
    'Retina3DHead',
    'ATSSMatcher',
    'HardNegativeSampler',
    'RetinaLoss',
    'focal_loss',
    'smooth_l1_loss',
    'giou_loss_3d',
]
