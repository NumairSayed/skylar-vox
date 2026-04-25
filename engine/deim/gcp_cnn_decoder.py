"""
GCPCNNDecoder — lightweight dense CNN head to replace the transformer decoder.

Takes the 3 FPN feature maps from HybridEncoder and runs a small 3×3-conv head
on the stride-8 (P3) map, which gives an 80×80 grid for 640-px input.

A GCP marker is ~12 px in the 640-px frame → ~1.5 cells at stride 8.
A 3×3 kernel covers 24 px of receptive field, fully enclosing the marker.

Each of the 6 400 grid cells independently predicts:
  pred_logits  [B, 6400, num_classes]   — raw logits (VFL/focal loss)
  pred_boxes   [B, 6400, 4]             — cxcywh normalised via sigmoid

The output dict is identical to DEIMTransformer's, so PostProcessor and
DEIMCriterion (with losses=['vfl','boxes']) work without modification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


def _conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


@register()
class GCPCNNDecoder(nn.Module):
    """Dense CNN detection head operating on the stride-8 FPN feature map."""

    __share__ = ["num_classes"]

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3,
        num_stem_convs: int = 3,
        dropout: float = 0.0,
        p3_index: int = 0,          # which FPN level is stride-8 (P3)
    ):
        super().__init__()
        self.p3_index = p3_index
        self.num_classes = num_classes

        # shared stem: N× (3×3 conv + BN + SiLU) with optional dropout
        stem = []
        for i in range(num_stem_convs):
            stem.append(_conv_bn_act(in_channels if i == 0 else hidden_dim, hidden_dim))
            if dropout > 0.0:
                stem.append(nn.Dropout2d(p=dropout))
        self.stem = nn.Sequential(*stem)

        # classification head
        self.cls_head = nn.Sequential(
            _conv_bn_act(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

        # box regression head  → outputs sigmoid(x) = cxcywh normalised
        self.reg_head = nn.Sequential(
            _conv_bn_act(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # bias init for cls head: low initial probability → stable early training
        nn.init.constant_(self.cls_head[-1].bias, -4.0)
        nn.init.constant_(self.reg_head[-1].bias, 0.0)

    def forward(self, feats: list[torch.Tensor], targets=None) -> dict:
        p3 = feats[self.p3_index]          # [B, C, 80, 80] for 640-px input
        B, _, H, W = p3.shape

        x = self.stem(p3)

        logits = self.cls_head(x)          # [B, num_classes, H, W]
        boxes_raw = self.reg_head(x)       # [B, 4, H, W]

        # --- anchor boxes from grid centres ----------------------------
        # Build grid of cell centres in normalised [0,1] coords
        # so the model starts predicting near the right spatial location.
        device = p3.device
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # cell-centre anchors, shape [1, 1, H, W]
        anchor_cx = ((gx + 0.5) / W).unsqueeze(0).unsqueeze(0)
        anchor_cy = ((gy + 0.5) / H).unsqueeze(0).unsqueeze(0)

        # cx/cy: anchor + bounded offset (tanh → ±0.5 cell shift)
        pred_cx = anchor_cx + torch.tanh(boxes_raw[:, 0:1]) * (0.5 / W)
        pred_cy = anchor_cy + torch.tanh(boxes_raw[:, 1:2]) * (0.5 / H)
        # w/h: sigmoid so they stay in (0, 1)
        pred_w  = torch.sigmoid(boxes_raw[:, 2:3])
        pred_h  = torch.sigmoid(boxes_raw[:, 3:4])

        boxes = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=1)  # [B, 4, H, W]

        # flatten spatial dims → sequence of locations
        logits = logits.flatten(2).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        boxes  = boxes.flatten(2).permute(0, 2, 1).contiguous()   # [B, H*W, 4]

        return {"pred_logits": logits, "pred_boxes": boxes}

    def deploy(self):
        self.eval()
        return self
