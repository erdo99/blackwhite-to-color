"""
RGB ↔ LAB conversions in PyTorch (D65, sRGB).

Network uses normalized tensors:
  L* in [0, 1]   (L / 100)
  a*, b* in approximately [-1, 1]  (a/128, b/128; clamp after decode if needed)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """rgb in [0,1], shape (..., 3)."""
    a = 0.055
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + a) / (1 + a), 2.4),
    )


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        (1 + a) * torch.pow(rgb.clamp(min=1e-12), 1 / 2.4) - a,
    )


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: (B, 3, H, W) in [0, 1]
    returns: (B, 3, H, W) — [L_norm, a_norm, b_norm] with L_norm=L/100, a_norm=a/128, b_norm=b/128
    """
    b, _, h, w = rgb.shape
    rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    lin = _srgb_to_linear(rgb_flat.clamp(0, 1))

    # sRGB D65 -> XYZ (matrix from Bruce Lindbloom / IEC 61966-2-1)
    m = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    xyz_ref = rgb.new_tensor([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / xyz_ref

    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, torch.pow(t.clamp(min=1e-12), 1 / 3), (kappa * t + 16) / 116)

    fx, fy, fz = f(xyz_n[:, 0]), f(xyz_n[:, 1]), f(xyz_n[:, 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    lab_b = 200 * (fy - fz)

    L = L.clamp(0, 100)
    a = a.clamp(-128, 127)
    lab_b = lab_b.clamp(-128, 127)

    Ln = (L / 100.0).view(b, h, w, 1)
    an = (a / 128.0).view(b, h, w, 1)
    bn = (lab_b / 128.0).view(b, h, w, 1)
    lab = torch.cat([Ln, an, bn], dim=-1).permute(0, 3, 1, 2)
    return lab


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    lab: (B, 3, H, W) with L_norm, a_norm, b_norm as produced by rgb_to_lab.
    returns: (B, 3, H, W) in [0, 1]
    """
    b, _, h, w = lab.shape
    L = (lab[:, 0:1] * 100.0).clamp(0, 100)
    a = (lab[:, 1:2] * 128.0).clamp(-128, 127)
    bch = (lab[:, 2:3] * 128.0).clamp(-128, 127)

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - bch / 200

    eps = 216 / 24389
    kappa = 24389 / 27

    def finv(t: torch.Tensor) -> torch.Tensor:
        t3 = t**3
        return torch.where(t3 > eps, t3, (116 * t - 16) / kappa)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    xyz_ref = lab.new_tensor([0.95047, 1.00000, 1.08883])
    xyz = torch.cat([xr, yr, zr], dim=1) * xyz_ref.view(1, 3, 1, 1)

    # XYZ -> linear sRGB
    m_inv = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    lin = torch.einsum("bchw,dc->bdhw", xyz, m_inv)
    lin = lin.clamp(0, 1)
    srgb = _linear_to_srgb(lin.permute(0, 2, 3, 1).reshape(-1, 3)).reshape(b, h, w, 3)
    srgb = srgb.permute(0, 3, 1, 2).clamp(0, 1)
    return srgb


class LabToRgbNoGrad(nn.Module):
    """Module wrapper for use inside loss (detach-friendly)."""

    def forward(self, lab: torch.Tensor) -> torch.Tensor:
        return lab_to_rgb(lab)
