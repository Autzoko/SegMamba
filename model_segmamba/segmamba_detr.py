"""
SegMamba-DETR: 3D Object Detection with DETR-style Transformer Decoder.

Architecture:
    MambaEncoder (level 3 only, 8^3 = 512 tokens)
      -> 1x1x1 Conv projection (384 -> d_model)
      -> 3D Sinusoidal Positional Encoding
      -> Transformer Encoder (self-attention with pos injection)
      -> Transformer Decoder (cross-attention with learned object queries)
      -> Classification head (Linear -> num_classes + 1)
      -> BBox head (MLP -> 6D sigmoid)

Input:  (B, 1, 128, 128, 128)
Output: dict with 'pred_logits', 'pred_boxes', optional 'aux_outputs'

Does NOT modify any original SegMamba code.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_segmamba.segmamba import MambaEncoder


# ---------------------------------------------------------------------------
# 3D Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding3D(nn.Module):
    """3D sinusoidal positional encoding for volumetric feature maps.

    Splits d_model channels across z, y, x axes.  Each axis gets
    interleaved sin/cos at geometrically-spaced frequencies.
    """

    def __init__(self, d_model=256, temperature=10000, normalize=True):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        """x: (B, C, D, H, W) — only spatial dims are used."""
        B, _, D, H, W = x.shape
        device = x.device

        nf_z = self.d_model // 3
        nf_y = self.d_model // 3
        nf_x = self.d_model - nf_z - nf_y  # absorbs remainder

        z = torch.arange(D, device=device, dtype=torch.float32)
        y = torch.arange(H, device=device, dtype=torch.float32)
        xc = torch.arange(W, device=device, dtype=torch.float32)

        if self.normalize:
            z = z / max(D - 1, 1) * self.scale
            y = y / max(H - 1, 1) * self.scale
            xc = xc / max(W - 1, 1) * self.scale

        def _encode(coord_1d, num_feats):
            dim_t = torch.arange(num_feats, device=device, dtype=torch.float32)
            dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)
            pos = coord_1d[..., None] / dim_t          # (..., nf)
            out = torch.empty_like(pos)
            out[..., 0::2] = pos[..., 0::2].sin()
            out[..., 1::2] = pos[..., 1::2].cos()
            return out

        Z, Y, X = torch.meshgrid(z, y, xc, indexing='ij')  # (D,H,W)

        pos_z = _encode(Z, nf_z)   # (D,H,W, nf_z)
        pos_y = _encode(Y, nf_y)   # (D,H,W, nf_y)
        pos_x = _encode(X, nf_x)   # (D,H,W, nf_x)

        pos = torch.cat([pos_z, pos_y, pos_x], dim=-1)   # (D,H,W, d_model)
        pos = pos.permute(3, 0, 1, 2)                     # (d_model, D,H,W)
        return pos.unsqueeze(0).expand(B, -1, -1, -1, -1) # (B, d_model, D,H,W)


# ---------------------------------------------------------------------------
# Transformer (custom layers with DETR-style pos injection)
# ---------------------------------------------------------------------------

class DETREncoderLayer(nn.Module):
    """Transformer encoder layer — adds pos to Q and K at every layer."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos):
        q = k = src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DETRDecoderLayer(nn.Module):
    """Transformer decoder layer — self-attn + cross-attn with pos injection."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, pos, query_pos):
        # Self-attention among queries
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention: queries attend to encoder memory
        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            key=memory + pos,
            value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DETRTransformer(nn.Module):
    """Full DETR transformer: encoder + decoder with intermediate outputs."""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_layers = nn.ModuleList([
            DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            DETRDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos, query_embed):
        """
        src:         (S, B, d_model)   S = D*H*W tokens
        pos:         (S, B, d_model)   spatial positional encoding
        query_embed: (N, d_model)      learned query embeddings

        Returns
        -------
        hs:     (num_decoder_layers, N, B, d_model)
        memory: (S, B, d_model)
        """
        # Encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, pos)
        memory = self.encoder_norm(memory)

        # Decoder
        B = src.shape[1]
        N = query_embed.shape[0]
        tgt = torch.zeros(N, B, self.d_model, device=src.device)
        query_pos = query_embed.unsqueeze(1).expand(-1, B, -1)

        intermediate = []
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, pos, query_pos)
            intermediate.append(self.decoder_norm(tgt))

        return torch.stack(intermediate), memory


# ---------------------------------------------------------------------------
# MLP (from DETR)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron for bounding-box regression."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# SegMamba-DETR
# ---------------------------------------------------------------------------

class SegMambaDETR(nn.Module):
    """MambaEncoder backbone + DETR transformer for 3D object detection.

    Uses only the deepest backbone level (8^3 = 512 tokens).

    Parameters
    ----------
    in_chans : int
        Input volume channels (1 for ABUS).
    num_classes : int
        Number of foreground classes (1 = tumor).
    num_queries : int
        Number of learned object queries.
    d_model : int
        Transformer hidden dimension.
    nhead : int
        Number of attention heads.
    num_encoder_layers, num_decoder_layers : int
        Transformer depth.
    dim_feedforward : int
        FFN intermediate dimension.
    dropout : float
        Dropout rate in transformer.
    aux_loss : bool
        If True, return intermediate decoder outputs for auxiliary losses.
    """

    def __init__(self, in_chans=1, num_classes=1, num_queries=20,
                 depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
                 d_model=256, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=1024,
                 dropout=0.1, aux_loss=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        # Backbone (reused from segmamba.py)
        self.backbone = MambaEncoder(
            in_chans=in_chans, depths=depths, dims=feat_size)

        # Project level-3 features to d_model
        self.input_proj = nn.Conv3d(feat_size[-1], d_model, kernel_size=1)

        # Positional encoding
        self.pos_encoder = PositionalEncoding3D(d_model=d_model)

        # Transformer
        self.transformer = DETRTransformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads (shared across decoder layers)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 6, 3)

        self._init_heads()

    def _init_heads(self):
        nn.init.constant_(self.class_head.bias, 0)
        # Bias the no-object class to high prior so initially most
        # queries predict no-object (like DETR's prior_prob=0.01).
        # For 2-class softmax this isn't as critical as sigmoid, but
        # we still zero-init for clean start.
        nn.init.xavier_uniform_(self.class_head.weight)
        for layer in self.bbox_head.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        x: (B, in_chans, 128, 128, 128)

        Returns dict:
            pred_logits: (B, num_queries, num_classes + 1)
            pred_boxes:  (B, num_queries, 6)  sigmoid [cz,cy,cx,dz,dy,dx]
            aux_outputs: list of dicts (one per decoder layer except last)
        """
        # Backbone — only use deepest level
        features = self.backbone(x)
        feat = features[-1]                    # (B, 384, 8, 8, 8)

        # Project to d_model
        src = self.input_proj(feat)            # (B, d_model, 8, 8, 8)

        # Positional encoding
        pos = self.pos_encoder(src)            # (B, d_model, 8, 8, 8)

        # Flatten spatial dims → sequence
        src_flat = src.flatten(2).permute(2, 0, 1)   # (512, B, d_model)
        pos_flat = pos.flatten(2).permute(2, 0, 1)   # (512, B, d_model)

        # Transformer
        hs, _ = self.transformer(
            src_flat, pos_flat, self.query_embed.weight)
        # hs: (num_dec_layers, num_queries, B, d_model)

        hs = hs.permute(0, 2, 1, 3)
        # hs: (num_dec_layers, B, num_queries, d_model)

        # Prediction heads
        outputs_class = self.class_head(hs)              # (L, B, N, C+1)
        outputs_coord = self.bbox_head(hs).sigmoid()     # (L, B, N, 6)

        out = {
            'pred_logits': outputs_class[-1],   # (B, N, C+1)
            'pred_boxes': outputs_coord[-1],    # (B, N, 6)
        }

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': outputs_class[i],
                 'pred_boxes': outputs_coord[i]}
                for i in range(len(outputs_class) - 1)
            ]

        return out
