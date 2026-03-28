"""
Hint simulation (training) and extraction from user UI (inference).

Training: random sparse pixels / small discs copy ground-truth AB from the image.
The U-Net sees L everywhere, mask + hint_ab only on strokes; it learns to propagate
chrominance while respecting L structure (edges) and global semantics.

Inference: compare composite vs background to find painted pixels; read AB from RGB at those sites.
"""

from __future__ import annotations

import torch

from .color_space import rgb_to_lab


def simulate_hints(
    lab: torch.Tensor,
    min_points: int = 4,
    max_points: int = 24,
    min_patch_radius: int = 1,
    max_patch_radius: int = 4,
    patch_prob: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    lab: (B, 3, H, W) normalized L*, a*, b*
    Returns:
      mask: (B, 1, H, W) float in {0,1}
      hint_ab: (B, 2, H, W) — AB copied at hint locations, 0 elsewhere
    """
    device = lab.device
    b, _, h, w = lab.shape
    mask = torch.zeros(b, 1, h, w, device=device)
    hint_ab = torch.zeros(b, 2, h, w, device=device)
    gt_ab = lab[:, 1:3]

    for i in range(b):
        n = torch.randint(min_points, max_points + 1, (1,), device=device).item()
        ys = torch.randint(0, h, (n,), device=device)
        xs = torch.randint(0, w, (n,), device=device)

        for y, x in zip(ys.tolist(), xs.tolist()):
            if torch.rand(1, device=device).item() < patch_prob:
                r = torch.randint(min_patch_radius, max_patch_radius + 1, (1,), device=device).item()
                yy, xx = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="ij",
                )
                dist = torch.sqrt((yy - y).float() ** 2 + (xx - x).float() ** 2)
                m = (dist <= r).float()
            else:
                m = torch.zeros(h, w, device=device)
                m[y, x] = 1.0

            mask[i, 0] = torch.maximum(mask[i, 0], m)
            for c in range(2):
                hint_ab[i, c] = torch.where(m > 0, gt_ab[i, c], hint_ab[i, c])

    mask = (mask > 0).float()
    return mask, hint_ab


def extract_hints_from_rgb_pair(
    background_rgb: torch.Tensor,
    composite_rgb: torch.Tensor,
    diff_thresh: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    background_rgb, composite_rgb: (1, 3, H, W) in [0,1]
    Finds pixels where composite visibly differs from grayscale background.
    Returns mask (1,1,H,W), hint_ab (1,2,H,W) normalized like training.
    """
    bg = background_rgb.clamp(0, 1)
    comp = composite_rgb.clamp(0, 1)
    gray = bg.mean(dim=1, keepdim=True)
    comp_gray = comp.mean(dim=1, keepdim=True)
    chroma = (comp - comp_gray).abs().sum(dim=1, keepdim=True) / 3.0
    delta = (comp - bg).abs().sum(dim=1, keepdim=True) / 3.0
    mask = ((delta > diff_thresh) | (chroma > diff_thresh)).float()

    lab = rgb_to_lab(comp)
    hint_ab = lab[:, 1:3] * mask
    return mask, hint_ab


def build_model_input(
    L: torch.Tensor,
    mask: torch.Tensor,
    hint_ab: torch.Tensor,
) -> torch.Tensor:
    """Concatenate channels for U-Net: (B, 4, H, W)."""
    return torch.cat([L, mask, hint_ab], dim=1)
