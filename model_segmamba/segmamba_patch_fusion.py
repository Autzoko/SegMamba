"""
SegMamba with Patch-Set Global Fusion for Multi-task Segmentation and Detection.

Each 128³ patch produces:
  - Segmentation logits (as usual)
  - Local box prediction (mapped to global coordinates)
  - Objectness score (is target present in patch?)
  - Quality score (how reliable is this patch's prediction?)

Multiple patches from the same volume are fused via differentiable soft-weighted
aggregation to produce a single global bounding box prediction.

Loss is computed on the fused global box, allowing the network to learn which
patches provide reliable evidence even when individual patches only partially
contain the target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmamba import SegMamba


class PatchBoxHead(nn.Module):
    """Predicts box, objectness, and quality from patch features.

    For each patch, outputs:
      - box: (B, 6) normalized [cz, cy, cx, dz, dy, dx] in local patch coords
      - objectness: (B, 1) probability that patch contains (part of) target
      - quality: (B, 1) prediction reliability score

    Coordinate Convention:
      - Center (cz, cy, cx): Normalized to patch size. 0.5 = patch center.
        Range [-0.5, 1.5] allows predicting boxes extending beyond patch.
      - Dimensions (dz, dy, dx): Normalized to patch size. 1.0 = full patch size (128).
        Range (0, 2] ensures positive dimensions, max 2x patch size.

    Positional Encoding:
      - Patch position in global volume is injected as additional features
      - Helps network understand global context from local patch view
    """

    def __init__(self, in_channels=48, hidden_dim=64):
        super().__init__()

        # Feature compression
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(hidden_dim)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(hidden_dim)
        self.conv3 = nn.Conv3d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm3d(hidden_dim)

        # Global pooling will give us (B, hidden_dim) features
        # After 2 stride-2 convs on 128³: 128 -> 64 -> 32

        # Position encoding: 6 values (patch_start normalized + patch_end normalized)
        self.pos_embed = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, hidden_dim),
        )

        # Prediction heads - input is hidden_dim * 2 (features + position)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat, patch_pos=None, volume_shape=None):
        """
        Args:
            feat: (B, C, D, H, W) decoder features, typically (B, 48, 128, 128, 128)
            patch_pos: (B, 3) or None, patch start position [z, y, x] in voxels
            volume_shape: (3,) or None, full volume shape [D, H, W] for normalization

        Returns:
            box: (B, 6) normalized box params [cz, cy, cx, dz, dy, dx]
            objectness: (B, 1) sigmoid probability
            quality: (B, 1) sigmoid probability
        """
        B = feat.shape[0]
        patch_size = feat.shape[2:]  # (D, H, W) = (128, 128, 128)

        x = F.relu(self.norm1(self.conv1(feat)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        # Global average pooling
        x = x.mean(dim=[2, 3, 4])  # (B, hidden_dim)

        # Position encoding
        if patch_pos is not None and volume_shape is not None:
            # Normalize positions to [0, 1] based on volume shape
            volume_shape = torch.as_tensor(volume_shape, dtype=feat.dtype, device=feat.device)
            patch_size_t = torch.tensor(patch_size, dtype=feat.dtype, device=feat.device)

            # Compute normalized start and end positions
            pos_start = patch_pos / volume_shape  # (B, 3) in [0, 1]
            pos_end = (patch_pos + patch_size_t) / volume_shape  # (B, 3)
            pos_end = pos_end.clamp(max=1.0)  # Clamp for edge patches

            pos_info = torch.cat([pos_start, pos_end], dim=1)  # (B, 6)
            pos_feat = self.pos_embed(pos_info)  # (B, hidden_dim)
        else:
            # No position info - use zeros (for backward compatibility)
            pos_feat = torch.zeros(B, x.shape[1], device=x.device, dtype=x.dtype)

        # Concatenate features and position encoding
        x = torch.cat([x, pos_feat], dim=1)  # (B, hidden_dim * 2)

        # Predictions
        box_raw = self.box_head(x)  # (B, 6)

        # Apply constraints to box parameters
        # Center (first 3): sigmoid * 2 - 0.5 → range [-0.5, 1.5]
        #   Allows predicting centers outside patch for partially visible targets
        center = torch.sigmoid(box_raw[:, :3]) * 2 - 0.5

        # Dimensions (last 3): sigmoid * 2 + eps → range (0, 2]
        #   Must be positive, max 2x patch size (256 voxels)
        dims = torch.sigmoid(box_raw[:, 3:]) * 2 + 0.01  # eps=0.01 ensures positive

        box = torch.cat([center, dims], dim=1)  # (B, 6)

        objectness = torch.sigmoid(self.objectness_head(x))  # (B, 1)
        quality = torch.sigmoid(self.quality_head(x))        # (B, 1)

        return box, objectness, quality


class SegMambaWithPatchFusion(SegMamba):
    """SegMamba extended with Patch-Set Global Fusion for detection.

    Inherits full segmentation capability from SegMamba and adds a PatchBoxHead
    for multi-task learning. The fusion of patch predictions happens outside
    this module (in the training loop) to support differentiable aggregation.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ):
        super().__init__(
            in_chans=in_chans,
            out_chans=out_chans,
            depths=depths,
            feat_size=feat_size,
        )

        # Add patch box head branching from decoder1 output
        self.patch_box_head = PatchBoxHead(in_channels=feat_size[0])

    def forward(self, x_in, patch_pos=None, volume_shape=None):
        """Forward pass returning both segmentation and detection outputs.

        Args:
            x_in: (B, 1, D, H, W) input volume patch
            patch_pos: (B, 3) or None, patch start positions for positional encoding
            volume_shape: (3,) or None, full volume shape for normalization

        Returns:
            seg_logits: (B, out_chans, D, H, W) segmentation logits
            box: (B, 6) local box prediction
            objectness: (B, 1) objectness score
            quality: (B, 1) quality score
        """
        # Encoder (same as parent SegMamba)
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        # Decoder (same as parent SegMamba)
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)  # (B, 48, D, H, W)

        # Segmentation head
        seg_logits = self.out(out)  # (B, out_chans, D, H, W)

        # Detection head (operates on decoder1 output features)
        box, objectness, quality = self.patch_box_head(out, patch_pos, volume_shape)

        return seg_logits, box, objectness, quality

    def forward_boxhead_only(self, x_in, patch_pos=None, volume_shape=None):
        """Forward pass for Stage 2 training with frozen backbone.

        Uses torch.no_grad() for backbone to prevent gradient issues,
        then enables gradients only for BoxHead.

        Args:
            x_in: (B, 1, D, H, W) input volume patch
            patch_pos: (B, 3) or None, patch start positions for positional encoding
            volume_shape: (3,) or None, full volume shape for normalization

        Returns:
            seg_logits: (B, out_chans, D, H, W) segmentation logits (no grad)
            box: (B, 6) local box prediction (with grad)
            objectness: (B, 1) objectness score (with grad)
            quality: (B, 1) quality score (with grad)
        """
        # Run backbone without gradients
        with torch.no_grad():
            outs = self.vit(x_in)
            enc1 = self.encoder1(x_in)
            x2 = outs[0]
            enc2 = self.encoder2(x2)
            x3 = outs[1]
            enc3 = self.encoder3(x3)
            x4 = outs[2]
            enc4 = self.encoder4(x4)
            enc_hidden = self.encoder5(outs[3])

            dec3 = self.decoder5(enc_hidden, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0)  # (B, 48, D, H, W)

            seg_logits = self.out(out)  # (B, out_chans, D, H, W)

        # BoxHead with gradients - clone to create new tensor with grad enabled
        out_for_boxhead = out.clone().detach().requires_grad_(True)
        box, objectness, quality = self.patch_box_head(out_for_boxhead, patch_pos, volume_shape)

        return seg_logits, box, objectness, quality


# ---------------------------------------------------------------------------
# Fusion utilities
# ---------------------------------------------------------------------------

def transform_box_to_global(box_local, patch_start, patch_size, volume_shape):
    """Transform local patch box to global volume coordinates.

    Args:
        box_local: (B, 6) or (6,) normalized [cz, cy, cx, dz, dy, dx]
        patch_start: (3,) or (B, 3) [z_start, y_start, x_start] in voxels
        patch_size: (3,) patch dimensions [D, H, W]
        volume_shape: (3,) full volume dimensions [D, H, W]

    Returns:
        box_global: same shape as input, in global voxel coordinates
    """
    patch_size = torch.as_tensor(patch_size, dtype=box_local.dtype, device=box_local.device)
    volume_shape = torch.as_tensor(volume_shape, dtype=box_local.dtype, device=box_local.device)

    if box_local.dim() == 1:
        box_local = box_local.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if not isinstance(patch_start, torch.Tensor):
        patch_start = torch.as_tensor(patch_start, dtype=box_local.dtype, device=box_local.device)

    if patch_start.dim() == 1:
        patch_start = patch_start.unsqueeze(0).expand(box_local.shape[0], -1)

    # Unpack local box
    cz_local = box_local[:, 0]
    cy_local = box_local[:, 1]
    cx_local = box_local[:, 2]
    dz = box_local[:, 3]
    dy = box_local[:, 4]
    dx = box_local[:, 5]

    # Transform center to global (denormalize then add offset)
    cz_global = patch_start[:, 0] + cz_local * patch_size[0]
    cy_global = patch_start[:, 1] + cy_local * patch_size[1]
    cx_global = patch_start[:, 2] + cx_local * patch_size[2]

    # Dimensions stay the same but denormalize
    dz_global = dz * patch_size[0]
    dy_global = dy * patch_size[1]
    dx_global = dx * patch_size[2]

    box_global = torch.stack([
        cz_global, cy_global, cx_global,
        dz_global, dy_global, dx_global
    ], dim=1)

    if squeeze_output:
        box_global = box_global.squeeze(0)

    return box_global


def fuse_patch_boxes(boxes_global, objectness, quality, eps=1e-6):
    """Fuse multiple patch box predictions via soft-weighted aggregation.

    Args:
        boxes_global: (N, 6) boxes in global coordinates [cz, cy, cx, dz, dy, dx]
        objectness: (N, 1) objectness scores
        quality: (N, 1) quality scores

    Returns:
        fused_box: (6,) weighted average box
        weights: (N,) normalized fusion weights
    """
    # Ensure float32 for numerical stability
    boxes_global = boxes_global.float()
    objectness = objectness.float()
    quality = quality.float()

    # Compute fusion weights
    weights = (objectness * quality).squeeze(-1)  # (N,)

    # Normalize weights (softmax-like but preserves relative magnitude)
    weights_sum = weights.sum() + eps
    weights_norm = weights / weights_sum  # (N,)

    # Weighted average of boxes
    fused_box = (weights_norm.unsqueeze(-1) * boxes_global).sum(dim=0)  # (6,)

    # Ensure dimensions are positive (clamp to minimum 1.0 voxel)
    # Use torch.cat to avoid in-place operation that breaks autograd
    fused_box = torch.cat([fused_box[:3], fused_box[3:].clamp(min=1.0)], dim=0)

    return fused_box, weights_norm


def compute_patch_targets(patch_start, patch_size, gt_box_global, volume_shape):
    """Compute supervision targets for a single patch.

    Args:
        patch_start: (3,) [z_start, y_start, x_start] in voxels
        patch_size: (3,) [D, H, W] patch dimensions
        gt_box_global: (6,) [z1, y1, x1, z2, y2, x2] corner format in global coords
        volume_shape: (3,) full volume shape

    Returns:
        objectness_gt: float, 1 if patch overlaps GT, 0 otherwise
        quality_gt: float, centerness score (0-1)
        box_gt_local: (6,) box in normalized local coords [cz, cy, cx, dz, dy, dx]
    """
    import numpy as np

    pz1, py1, px1 = patch_start
    pz2 = pz1 + patch_size[0]
    py2 = py1 + patch_size[1]
    px2 = px1 + patch_size[2]

    gz1, gy1, gx1, gz2, gy2, gx2 = gt_box_global

    # Check overlap
    inter_z = max(0, min(pz2, gz2) - max(pz1, gz1))
    inter_y = max(0, min(py2, gy2) - max(py1, gy1))
    inter_x = max(0, min(px2, gx2) - max(px1, gx1))

    has_overlap = (inter_z > 0) and (inter_y > 0) and (inter_x > 0)
    objectness_gt = 1.0 if has_overlap else 0.0

    if not has_overlap:
        return objectness_gt, 0.0, np.zeros(6, dtype=np.float32)

    # Compute centerness (how centered is the visible GT within the patch)
    # Clip GT to patch boundaries
    clipped_z1 = max(gz1, pz1) - pz1
    clipped_y1 = max(gy1, py1) - py1
    clipped_x1 = max(gx1, px1) - px1
    clipped_z2 = min(gz2, pz2) - pz1
    clipped_y2 = min(gy2, py2) - py1
    clipped_x2 = min(gx2, px2) - px1

    # Center of clipped (visible) box in local coords
    cz_vis = (clipped_z1 + clipped_z2) / 2
    cy_vis = (clipped_y1 + clipped_y2) / 2
    cx_vis = (clipped_x1 + clipped_x2) / 2

    # Centerness: how centered is the visible part within the patch
    dz_left, dz_right = cz_vis, patch_size[0] - cz_vis
    dy_left, dy_right = cy_vis, patch_size[1] - cy_vis
    dx_left, dx_right = cx_vis, patch_size[2] - cx_vis

    ctr_z = min(dz_left, dz_right) / max(dz_left, dz_right, 1e-6)
    ctr_y = min(dy_left, dy_right) / max(dy_left, dy_right, 1e-6)
    ctr_x = min(dx_left, dx_right) / max(dx_left, dx_right, 1e-6)

    # Also consider coverage: what fraction of GT is visible
    gt_vol = (gz2 - gz1) * (gy2 - gy1) * (gx2 - gx1)
    vis_vol = inter_z * inter_y * inter_x
    coverage = vis_vol / max(gt_vol, 1e-6)

    # Quality combines centerness and coverage
    quality_gt = float(np.sqrt(ctr_z * ctr_y * ctr_x * coverage))

    # Box target: FULL GT box in local normalized coordinates
    # (allows network to predict beyond patch boundaries)
    gt_cz = ((gz1 + gz2) / 2 - pz1) / patch_size[0]
    gt_cy = ((gy1 + gy2) / 2 - py1) / patch_size[1]
    gt_cx = ((gx1 + gx2) / 2 - px1) / patch_size[2]
    gt_dz = (gz2 - gz1) / patch_size[0]
    gt_dy = (gy2 - gy1) / patch_size[1]
    gt_dx = (gx2 - gx1) / patch_size[2]

    box_gt_local = np.array([gt_cz, gt_cy, gt_cx, gt_dz, gt_dy, gt_dx], dtype=np.float32)

    return objectness_gt, quality_gt, box_gt_local


def box_cxcyczdhwd_to_corners(box):
    """Convert [cz, cy, cx, dz, dy, dx] to [z1, y1, x1, z2, y2, x2]."""
    cz, cy, cx, dz, dy, dx = box.unbind(-1)
    z1 = cz - dz / 2
    y1 = cy - dy / 2
    x1 = cx - dx / 2
    z2 = cz + dz / 2
    y2 = cy + dy / 2
    x2 = cx + dx / 2
    return torch.stack([z1, y1, x1, z2, y2, x2], dim=-1)


def compute_giou_3d(box1, box2):
    """Compute 3D Generalized IoU between two boxes.

    Args:
        box1, box2: (..., 6) in [cz, cy, cx, dz, dy, dx] format

    Returns:
        giou: (...,) GIoU values in [-1, 1]
    """
    # Ensure dimensions are positive (clamp to small positive value)
    eps = 1e-6
    box1_safe = box1.clone()
    box2_safe = box2.clone()
    box1_safe[..., 3:] = box1[..., 3:].clamp(min=eps)
    box2_safe[..., 3:] = box2[..., 3:].clamp(min=eps)

    # Convert to corners
    b1 = box_cxcyczdhwd_to_corners(box1_safe)
    b2 = box_cxcyczdhwd_to_corners(box2_safe)

    # Intersection
    inter_z = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) - torch.max(b1[..., 0], b2[..., 0]), min=0)
    inter_y = torch.clamp(torch.min(b1[..., 4], b2[..., 4]) - torch.max(b1[..., 1], b2[..., 1]), min=0)
    inter_x = torch.clamp(torch.min(b1[..., 5], b2[..., 5]) - torch.max(b1[..., 2], b2[..., 2]), min=0)
    inter = inter_z * inter_y * inter_x

    # Volumes (guaranteed positive due to clamping above)
    vol1 = box1_safe[..., 3] * box1_safe[..., 4] * box1_safe[..., 5]
    vol2 = box2_safe[..., 3] * box2_safe[..., 4] * box2_safe[..., 5]
    union = vol1 + vol2 - inter

    iou = inter / union.clamp(min=eps)

    # Enclosing box
    enc_z = torch.max(b1[..., 3], b2[..., 3]) - torch.min(b1[..., 0], b2[..., 0])
    enc_y = torch.max(b1[..., 4], b2[..., 4]) - torch.min(b1[..., 1], b2[..., 1])
    enc_x = torch.max(b1[..., 5], b2[..., 5]) - torch.min(b1[..., 2], b2[..., 2])
    enc_vol = (enc_z * enc_y * enc_x).clamp(min=eps)

    giou = iou - (enc_vol - union) / enc_vol

    # Clamp output to valid range and handle NaN
    giou = torch.clamp(giou, min=-1.0, max=1.0)
    giou = torch.nan_to_num(giou, nan=0.0)

    return giou
