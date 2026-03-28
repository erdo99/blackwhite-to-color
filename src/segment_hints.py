"""
Object-level hints (integration stub): given a label map (H,W int) and a color per label,
fill hint mask + hint_ab inside each region with constant chroma.

Use with outputs from SAM, Mask2Former, etc. Convert chosen RGB to normalized ab* offline.
"""

from __future__ import annotations

import torch

from .color_space import rgb_to_lab


def hints_from_label_map(
    rgb_01: torch.Tensor,
    labels: torch.Tensor,
    label_to_rgb: dict[int, tuple[float, float, float]],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rgb_01: (1,3,H,W) in [0,1] — used only to spatially align labels
    labels: (1,1,H,W) long tensor, 0 = background / ignore
    label_to_rgb: maps label id -> (r,g,b) in [0,1]

    Returns mask (1,1,H,W), hint_ab (1,2,H,W) in the same normalized space as training.
    """
    if device is None:
        device = rgb_01.device
    rgb_01 = rgb_01.to(device)
    labels = labels.long().to(device)
    _, _, h, w = rgb_01.shape
    mask = torch.zeros(1, 1, h, w, device=device)
    hint_ab = torch.zeros(1, 2, h, w, device=device)

    for lid, (r, g, b) in label_to_rgb.items():
        if lid == 0:
            continue
        m = (labels == lid).float()
        if m.sum() == 0:
            continue
        patch = torch.tensor([[[[r]], [[g]], [[b]]]], device=device)
        lab = rgb_to_lab(patch)
        ab = lab[:, 1:3].view(1, 2, 1, 1)
        mask = torch.maximum(mask, m)
        hint_ab = hint_ab + m * ab

    return mask, hint_ab
