"""
U-Net: 4-channel input (L, hint mask, hint_a, hint_b) -> 2-channel AB (normalized).

Input: (B, 4, H, W). Output: (B, 2, H, W). Skip connections preserve spatial detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dh, dw = x2.shape[2] - x1.shape[2], x2.shape[3] - x1.shape[3]
        if dh != 0 or dw != 0:
            x1 = nn.functional.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


def _make_funnel(base: int, num_down: int) -> list[int]:
    """Channel width at each encoder depth: [base, 2*base, ...]."""
    return [min(base * (2**i), 1024) for i in range(num_down)]


class HintGuidedUNet(nn.Module):
    """
    Encoder-decoder with skip connections.

    num_down: number of encoder levels (each halves H,W). E.g. num_down=5 and H=W=256
    yields bottleneck spatial size 256/2^(num_down-1) = 16 after (num_down-1) pools from first level.
    Here: inc (no pool) + (num_down-1) Down modules = num_down-1 halvings after inc's resolution... 

    Actually: inc keeps size. Down1 halves once. So after k Down modules, size is H/2^k.
    If we want 5 levels of features before bottleneck: inc + 4 downs -> 4 halvings: 256->128->64->32->16.

    So num_down = 5 means chs length 5 and downs = 4.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        base_channels: int = 64,
        num_down: int = 5,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        chs = _make_funnel(base_channels, num_down)
        self.inc = DoubleConv(in_channels, chs[0])

        self.downs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(Down(chs[i], chs[i + 1]))

        self.bot = DoubleConv(chs[-1], chs[-1])

        self.ups = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            in_ch = chs[i] + chs[i - 1]
            out_ch = chs[i - 1]
            self.ups.append(Up(in_ch, out_ch, bilinear=bilinear))

        self.outc = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        skips = [x0]
        xi = x0
        for d in self.downs:
            xi = d(xi)
            skips.append(xi)

        xi = self.bot(xi)

        for j, up in enumerate(self.ups):
            skip = skips[-(j + 2)]
            xi = up(xi, skip)

        return torch.tanh(self.outc(xi))
