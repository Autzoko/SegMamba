"""
SegMamba-BoxHead: Multi-task segmentation + attention-based box regression.

Subclasses SegMamba to add a lightweight BoxHead branch from the decoder's
full-resolution feature map.  The BoxHead learns a spatial attention map
over all voxels, aggregates features into a global vector, and regresses
a single 3D bounding box via MLP.

Architecture:
    SegMamba encoder-decoder (unchanged)
      -> decoder1 output (B, 48, D, H, W)
         ├─> UnetOutBlock -> seg logits (B, out_chans, D, H, W)
         └─> BoxHead -> box prediction (B, 6)

Input:  (B, 1, 128, 128, 128)
Output: (seg_logits, box_pred)

Does NOT modify segmamba.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_segmamba.segmamba import SegMamba


# ---------------------------------------------------------------------------
# BoxHead — attention-based spatial aggregation + MLP regression
# ---------------------------------------------------------------------------

class BoxHead(nn.Module):
    """Lightweight box regression head with learned spatial attention.

    1. Two 3×3×3 convs compress decoder features for attention computation.
    2. A 1×1×1 conv produces single-channel attention logits.
    3. Softmax over the full voxel space yields a probability map.
    4. Weighted sum of the *original* features → global vector.
    5. MLP regresses normalised box parameters.

    Parameters
    ----------
    in_channels : int
        Channels of the input feature map (48 for SegMamba decoder1).
    hidden_channels : int
        Intermediate conv channels for attention computation.
    mlp_hidden : int
        MLP intermediate dimension.
    """

    def __init__(self, in_channels=48, hidden_channels=32, mlp_hidden=128):
        super().__init__()

        # Feature compression for attention
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1,
                      bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True))

        # Single-channel attention logits
        self.attn_conv = nn.Conv3d(hidden_channels, 1, 1)

        # MLP for box regression
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 6))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        """
        feat : (B, 48, D, H, W) — full-resolution decoder features.

        Returns
        -------
        box : (B, 6) — [cz, cy, cx, dz, dy, dx] normalised to [0, 1].
        attn_map : (B, 1, D, H, W) — attention weights (for visualisation).
        """
        B, C, D, H, W = feat.shape

        # Compressed features for attention
        x = self.conv1(feat)
        x = self.conv2(x)

        # Attention map: softmax over full voxel space
        attn = self.attn_conv(x)                          # (B, 1, D, H, W)
        attn = attn.view(B, 1, -1)                        # (B, 1, D*H*W)
        attn = F.softmax(attn, dim=-1)                    # (B, 1, D*H*W)
        attn = attn.view(B, 1, D, H, W)                   # (B, 1, D, H, W)

        # Weighted aggregation of original features
        global_feat = (feat * attn).sum(dim=[2, 3, 4])     # (B, C)

        # Regress box
        box = self.mlp(global_feat).sigmoid()              # (B, 6)

        return box, attn


# ---------------------------------------------------------------------------
# SegMamba with BoxHead
# ---------------------------------------------------------------------------

class SegMambaWithBoxHead(SegMamba):
    """SegMamba extended with an attention-based BoxHead for multi-task
    segmentation + bounding-box regression.

    Subclasses SegMamba — the encoder-decoder is identical.  The only
    addition is a :class:`BoxHead` that branches off ``decoder1``'s output.

    Parameters
    ----------
    Inherits all SegMamba parameters, plus:

    box_hidden_channels : int
        BoxHead intermediate conv channels.
    box_mlp_hidden : int
        BoxHead MLP intermediate dimension.
    """

    def __init__(self, in_chans=1, out_chans=2, depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384], drop_path_rate=0,
                 layer_scale_init_value=1e-6, hidden_size=768,
                 norm_name="instance", conv_block=True, res_block=True,
                 spatial_dims=3,
                 box_hidden_channels=32, box_mlp_hidden=128):
        super().__init__(
            in_chans=in_chans, out_chans=out_chans, depths=depths,
            feat_size=feat_size, drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            hidden_size=hidden_size, norm_name=norm_name,
            conv_block=conv_block, res_block=res_block,
            spatial_dims=spatial_dims)

        self.box_head = BoxHead(
            in_channels=feat_size[0],
            hidden_channels=box_hidden_channels,
            mlp_hidden=box_mlp_hidden)

    def forward(self, x_in):
        """
        Returns
        -------
        seg_logits : (B, out_chans, D, H, W)
        box_pred   : (B, 6) — [cz, cy, cx, dz, dy, dx] in [0, 1]
        """
        # ---------- encoder ----------
        outs = self.vit(x_in)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])

        # ---------- decoder ----------
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)           # (B, 48, D, H, W)

        # ---------- heads ----------
        seg_logits = self.out(out)          # (B, out_chans, D, H, W)
        box_pred, _ = self.box_head(out)    # (B, 6)

        return seg_logits, box_pred
