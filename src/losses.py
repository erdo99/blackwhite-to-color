"""L1, perceptual (VGG), and optional GAN losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def chroma_weighted_l1(
    pred_ab: torch.Tensor,
    target_ab: torch.Tensor,
    scale: float = 2.0,
) -> torch.Tensor:
    """
    Doygun bölgelerde (|a|+|b| ortalaması yüksek) ab hatasını daha ağır cezalandırır;
    düz L1'e göre gri/ortalama renge (regression to the mean) kaymayı azaltmaya yardımcı olur.
    pred_ab, target_ab: (B, 2, H, W) normalized ab.
    """
    w = target_ab.abs().mean(dim=1, keepdim=True) * scale + 1.0
    return (pred_ab - target_ab).abs().mul(w).mean()


def imagenet_norm_rgb(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1]"""
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class VGGPerceptualLoss(nn.Module):
    """Multi-layer feature L1 on pretrained VGG16 (RGB inputs in [0,1])."""

    def __init__(self, layer_indices: list[int]):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.slice1 = nn.Sequential(*list(vgg.features[: layer_indices[0] + 1]))
        self.slice2 = nn.Sequential(*list(vgg.features[layer_indices[0] + 1 : layer_indices[1] + 1]))
        self.slice3 = nn.Sequential(*list(vgg.features[layer_indices[1] + 1 : layer_indices[2] + 1]))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        p = imagenet_norm_rgb(pred_rgb.clamp(0, 1))
        t = imagenet_norm_rgb(target_rgb.clamp(0, 1))
        loss = 0.0
        x_p, x_t = p, t
        for block in (self.slice1, self.slice2, self.slice3):
            x_p = block(x_p)
            x_t = block(x_t)
            loss = loss + F.l1_loss(x_p, x_t)
        return loss


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - real_logits))
    loss_fake = torch.mean(F.relu(1.0 + fake_logits))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_logits)


def vgg_layer_indices_from_names(names: list[str] | None) -> list[int]:
    """Map friendly names to VGG16 feature indices (relu1_2, relu2_2, relu3_3)."""
    if names is None:
        return [3, 8, 15]
    key = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}
    return [key[n] for n in names]
