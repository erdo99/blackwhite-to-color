#!/usr/bin/env python3
"""Run inference on a single image + hint tensors. Used by app and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.color_space import lab_to_rgb, rgb_to_lab
from src.hints import build_model_input, extract_hints_from_rgb_pair
from src.models.unet import HintGuidedUNet
from src.refine import refine_ab_to_fullres, rgb_pil_to_L_tensor


def load_generator(ckpt_path: str, device: torch.device) -> tuple[HintGuidedUNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    mcfg = cfg.get("model", {})
    G = HintGuidedUNet(
        in_channels=mcfg.get("in_channels", 4),
        out_channels=mcfg.get("out_channels", 2),
        base_channels=mcfg.get("base_channels", 64),
        num_down=mcfg.get("num_down", 5),
    ).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, cfg


def pil_to_tensor_rgb(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    return tf(img).unsqueeze(0)


def tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


@torch.no_grad()
def colorize_auto(
    G: HintGuidedUNet,
    image_pil: Image.Image,
    image_size: int,
    device: torch.device,
    full_resolution: bool = True,
) -> Image.Image:
    """Otomatik renklendirme: sadece L (gri veya renkli yükleme → L kanalı). in_channels==1."""
    gray_rgb = image_pil.convert("L").convert("RGB")
    bg_small = pil_to_tensor_rgb(gray_rgb, image_size).to(device)
    lab = rgb_to_lab(bg_small)
    L = lab[:, 0:1]
    pred_ab = G(L)
    if full_resolution:
        full = transforms.ToTensor()(gray_rgb).unsqueeze(0).to(device)
        L_full = rgb_pil_to_L_tensor(full)
        rgb = refine_ab_to_fullres(L_full, pred_ab)
        return tensor_to_pil_rgb(rgb)
    out_lab = torch.cat([L, pred_ab], dim=1)
    return tensor_to_pil_rgb(lab_to_rgb(out_lab))


@torch.no_grad()
def colorize(
    G: HintGuidedUNet,
    background_pil: Image.Image,
    composite_pil: Image.Image,
    image_size: int,
    device: torch.device,
    diff_thresh: float = 0.06,
    full_resolution: bool = True,
) -> Image.Image:
    """
    1 kanallı model: colorize_auto ile aynı (composite yok sayılır).
    4 kanal: arka plan + boyali composite’ten ipucu çıkarılır.
    """
    if getattr(G, "in_channels", 4) == 1:
        return colorize_auto(G, background_pil, image_size, device, full_resolution)

    gray_rgb = background_pil.convert("L").convert("RGB")
    comp_rgb = composite_pil.convert("RGB")
    if gray_rgb.size != comp_rgb.size:
        comp_rgb = comp_rgb.resize(gray_rgb.size, Image.Resampling.BICUBIC)

    bg_small = pil_to_tensor_rgb(gray_rgb, image_size).to(device)
    comp_small = pil_to_tensor_rgb(comp_rgb, image_size).to(device)

    lab = rgb_to_lab(bg_small)
    L = lab[:, 0:1]
    mask, hint_ab = extract_hints_from_rgb_pair(bg_small, comp_small, diff_thresh=diff_thresh)
    x = build_model_input(L, mask, hint_ab)
    pred_ab = G(x)

    if full_resolution:
        full = transforms.ToTensor()(gray_rgb).unsqueeze(0).to(device)
        L_full = rgb_pil_to_L_tensor(full)
        rgb = refine_ab_to_fullres(L_full, pred_ab)
        return tensor_to_pil_rgb(rgb)

    out_lab = torch.cat([L, pred_ab], dim=1)
    rgb = lab_to_rgb(out_lab)
    return tensor_to_pil_rgb(rgb)


@torch.no_grad()
def colorize_from_ab_hint(
    G: HintGuidedUNet,
    gray_rgb_pil: Image.Image,
    mask_01: torch.Tensor,
    hint_ab: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> Image.Image:
    """Programmatic API: mask (1,1,H,W), hint_ab (1,2,H,W) on resized grid."""
    bg_t = pil_to_tensor_rgb(gray_rgb_pil, image_size).to(device)
    lab = rgb_to_lab(bg_t)
    L = lab[:, 0:1]
    x = build_model_input(L, mask_01.to(device), hint_ab.to(device))
    pred_ab = G(x)
    out_lab = torch.cat([L, pred_ab], dim=1)
    rgb = lab_to_rgb(out_lab)
    return tensor_to_pil_rgb(rgb)


@torch.no_grad()
def colorize_variants(
    G: HintGuidedUNet,
    background_pil: Image.Image,
    composite_pil: Image.Image,
    image_size: int,
    device: torch.device,
    diff_thresh: float = 0.06,
    n: int = 3,
    hint_noise_std: float = 0.03,
    full_resolution: bool = True,
) -> list[Image.Image]:
    """
    1 kanal: AB çıktısına hafif gürültü ile çeşitlilik.
    4 kanal: ipucu AB’ye gürültü.
    """
    if getattr(G, "in_channels", 4) == 1:
        gray_rgb = background_pil.convert("L").convert("RGB")
        bg_small = pil_to_tensor_rgb(gray_rgb, image_size).to(device)
        lab = rgb_to_lab(bg_small)
        L = lab[:, 0:1]
        outs: list[Image.Image] = []
        for _ in range(max(1, n)):
            pred_ab = G(L)
            pred_ab = (pred_ab + torch.randn_like(pred_ab) * hint_noise_std).clamp(-1, 1)
            if full_resolution:
                full = transforms.ToTensor()(gray_rgb).unsqueeze(0).to(device)
                L_full = rgb_pil_to_L_tensor(full)
                rgb = refine_ab_to_fullres(L_full, pred_ab)
            else:
                rgb = lab_to_rgb(torch.cat([L, pred_ab], dim=1))
            outs.append(tensor_to_pil_rgb(rgb))
        return outs

    gray_rgb = background_pil.convert("L").convert("RGB")
    comp_rgb = composite_pil.convert("RGB")
    if gray_rgb.size != comp_rgb.size:
        comp_rgb = comp_rgb.resize(gray_rgb.size, Image.Resampling.BICUBIC)

    bg_small = pil_to_tensor_rgb(gray_rgb, image_size).to(device)
    comp_small = pil_to_tensor_rgb(comp_rgb, image_size).to(device)
    lab = rgb_to_lab(bg_small)
    L = lab[:, 0:1]
    mask, hint_ab = extract_hints_from_rgb_pair(bg_small, comp_small, diff_thresh=diff_thresh)

    outs: list[Image.Image] = []
    for _ in range(max(1, n)):
        noise = torch.randn_like(hint_ab) * hint_noise_std
        hint_n = hint_ab + mask * noise
        x = build_model_input(L, mask, hint_n)
        pred_ab = G(x)
        if full_resolution:
            full = transforms.ToTensor()(gray_rgb).unsqueeze(0).to(device)
            L_full = rgb_pil_to_L_tensor(full)
            rgb = refine_ab_to_fullres(L_full, pred_ab)
        else:
            rgb = lab_to_rgb(torch.cat([L, pred_ab], dim=1))
        outs.append(tensor_to_pil_rgb(rgb))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="Colorize using checkpoint (expects pre-built 4ch tensor file optional)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--image", required=True, help="Grayscale or RGB image path")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg.get("infer", {})
    ckpt = args.checkpoint or icfg.get("checkpoint", "./checkpoints/best.pt")
    size = int(icfg.get("image_size", cfg["data"]["image_size"]))
    device_s = icfg.get("device", "cuda")
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")

    G, _ = load_generator(ckpt, device)
    img = Image.open(args.image).convert("RGB")
    if G.in_channels == 1:
        out = colorize_auto(G, img, size, device, full_resolution=True)
    else:
        bg_t = pil_to_tensor_rgb(img, size).to(device)
        lab = rgb_to_lab(bg_t)
        L = lab[:, 0:1]
        b, _, h, w = L.shape
        mask = torch.zeros(b, 1, h, w, device=device)
        hint_ab = torch.zeros(b, 2, h, w, device=device)
        x = build_model_input(L, mask, hint_ab)
        pred_ab = G(x)
        out = tensor_to_pil_rgb(lab_to_rgb(torch.cat([L, pred_ab], dim=1)))
    out_path = Path(args.image).with_suffix(".colorized.png")
    out.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
