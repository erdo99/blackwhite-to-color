#!/usr/bin/env python3
"""
Tek bir fotograf için: egitimdeki gibi on-isleme + (istege bagli) ag ciktisi vs hedef.

Ciktilar varsayilan olarak ./demo_out/ altina yazilir:
  - 01_resized_rgb.png     : aga gitmeden onceki renkli (kare resize)
  - 02_L_girdi.png         : sinir agina giden L* (gri, tek kanal mantigi)
  - 03_ab_hedef_a.png      : gercek a* (normalize, goruntuleme icin 0-255)
  - 04_ab_hedef_b.png      : gercek b*
  - 05_renk_hedef.png      : L + gercek ab -> RGB (kayip bununla kiyaslanir)
  - 06_tahmin_renk.png     : sadece --checkpoint verilirse (L -> tahmin ab -> RGB)

Kullanim (proje kokunden):
  python scripts/demo_training_io.py --image yol/foto.jpg
  python scripts/demo_training_io.py --image yol/foto.jpg --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# proje kokunu path'e ekle
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.color_space import lab_to_rgb, rgb_to_lab
from src.hints import build_model_input
from src.models.unet import HintGuidedUNet


def _resize_rgb(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    t = tf(img)
    arr = (t.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _tensor01_to_pil_gray(t: torch.Tensor) -> Image.Image:
    """(H,W) veya (1,H,W) in [0,1]"""
    if t.dim() == 3:
        t = t.squeeze(0)
    arr = (t.cpu().numpy() * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


def _ab_ch_to_gray_vis(ab_ch: torch.Tensor) -> Image.Image:
    """Normalize a veya b kanali ~[-1,1] -> 0-255 gri"""
    x = ab_ch.cpu().float()
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = (x.numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr, mode="L")


def _lab_to_pil_rgb(lab_13hw: torch.Tensor) -> Image.Image:
    rgb = lab_to_rgb(lab_13hw).squeeze(0).clamp(0, 1)
    arr = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def load_g_from_ckpt(path: str, device: torch.device) -> HintGuidedUNet:
    ckpt = torch.load(path, map_location=device)
    mcfg = ckpt.get("cfg", {}).get("model", {})
    G = HintGuidedUNet(
        in_channels=mcfg.get("in_channels", 1),
        out_channels=mcfg.get("out_channels", 2),
        base_channels=mcfg.get("base_channels", 64),
        num_down=mcfg.get("num_down", 5),
    ).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G


def main() -> None:
    ap = argparse.ArgumentParser(description="Egitim on-isleme + hedef cikti goster")
    ap.add_argument("--image", required=True, help="Giris fotografi")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--size", type=int, default=None, help="Kare boy (varsayilan: config data.image_size)")
    ap.add_argument("--out", default=str(ROOT / "demo_out"))
    ap.add_argument("--checkpoint", default=None, help="Varsa tahmin rengi de kaydedilir")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    size = args.size
    if size is None and cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        size = int(cfg.get("data", {}).get("image_size", 256))
    elif size is None:
        size = 256

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Dosya yok: {img_path.resolve()}", file=sys.stderr)
        print('Gercek bir yol ver, ornek: --image "C:\\Users\\erdem\\BlackWhitetoColor\\data\\train\\bir_foto.jpg"', file=sys.stderr)
        sys.exit(1)

    pil = Image.open(img_path).convert("RGB")
    resized = _resize_rgb(pil, size)
    resized.save(out_dir / "01_resized_rgb.png")

    rgb_t = transforms.ToTensor()(resized).unsqueeze(0)
    lab = rgb_to_lab(rgb_t)
    L = lab[:, 0:1]
    ab = lab[:, 1:3]

    _tensor01_to_pil_gray(L.squeeze(0)).save(out_dir / "02_L_girdi.png")
    _ab_ch_to_gray_vis(ab[0, 0]).save(out_dir / "03_ab_hedef_a.png")
    _ab_ch_to_gray_vis(ab[0, 1]).save(out_dir / "04_ab_hedef_b.png")
    _lab_to_pil_rgb(lab).save(out_dir / "05_renk_hedef.png")

    print("=== Boyutlar (batch=1) ===")
    print(f"  L girdi (ag):        {tuple(L.shape)}   # (1, 1, H, W)")
    print(f"  ab hedef (kayip):    {tuple(ab.shape)}   # (1, 2, H, W)")
    print(f"  Kare H=W:            {size}")
    print()
    print("=== Dosyalar ===")
    for name in sorted(out_dir.glob("0*.png")):
        print(f"  {name}")
    print()
    print("Kayip: tahmin_ab ile yukaridaki ab (L1) + (L+tahmin) vs (L+ab) perceptual (RGB).")

    if args.checkpoint and Path(args.checkpoint).is_file():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = load_g_from_ckpt(args.checkpoint, device)
        Lt = L.to(device)
        with torch.no_grad():
            if G.in_channels == 1:
                pred_ab = G(Lt).cpu()
            else:
                b, _, h, w = Lt.shape
                z = torch.zeros(b, 1, h, w, device=device)
                zz = torch.zeros(b, 2, h, w, device=device)
                pred_ab = G(build_model_input(Lt, z, zz)).cpu()
        pred_lab = torch.cat([L, pred_ab], dim=1)
        _lab_to_pil_rgb(pred_lab).save(out_dir / "06_tahmin_renk.png")
        print(f"  {out_dir / '06_tahmin_renk.png'}  (checkpoint: {args.checkpoint})")
        if G.in_channels != 1:
            print("  Not: 4 kanal model; ipucu sifir ile forward (otomatik moda yakin).")
    elif args.checkpoint:
        print(f"Checkpoint bulunamadi: {args.checkpoint}")


if __name__ == "__main__":
    main()
