"""
High-resolution refinement without a second network:

Run the U-Net at `net_size`, then upsample predicted ab* to `full_h x full_w`,
concatenate with full-resolution L* from the original image, and decode to RGB.

This keeps global coherence from the net while aligning chroma to fine luminance detail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color_space import lab_to_rgb, rgb_to_lab


@torch.no_grad()
def refine_ab_to_fullres(
    L_full: torch.Tensor,
    ab_low: torch.Tensor,
) -> torch.Tensor:
    """
    L_full: (1,1,H,W) normalized L at full resolution
    ab_low: (1,2,h,w) predicted ab at network resolution (h<=H, w<=W)
    returns RGB (1,3,H,W) in [0,1]
    """
    ab_up = F.interpolate(ab_low, size=L_full.shape[-2:], mode="bilinear", align_corners=False)
    lab = torch.cat([L_full, ab_up], dim=1)
    return lab_to_rgb(lab)


def rgb_pil_to_L_tensor(rgb_01: torch.Tensor) -> torch.Tensor:
    """rgb_01 (1,3,H,W) -> L (1,1,H,W) normalized."""
    lab = rgb_to_lab(rgb_01)
    return lab[:, 0:1]
