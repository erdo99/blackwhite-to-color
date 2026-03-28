"""PatchGAN-style discriminator on RGB (for optional adversarial loss)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    C=3 input (RGB in [0,1] space — caller applies ImageNet norm if desired).
    Outputs NxN patch logits.
    """

    def __init__(self, in_channels: int = 3, base: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            spectral_norm(nn.Conv2d(in_channels, base, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                spectral_norm(
                    nn.Conv2d(ch, next_ch, 4, stride=2 if i < n_layers - 1 else 1, padding=1, bias=False)
                ),
                nn.BatchNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers.append(spectral_norm(nn.Conv2d(ch, 1, 4, stride=1, padding=1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
