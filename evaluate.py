#!/usr/bin/env python3
"""Evaluate a colorization checkpoint: SSIM, PSNR, MSE, speed, sample triplets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from infer import load_generator
from src.color_space import lab_to_rgb
from src.dataset import ColorizationDataset, list_images


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_splits(
    train_dir: str | Path,
    val_dir: str | Path | None,
    extensions: list[str],
    val_frac: float,
) -> tuple[list[Path], list[Path], str]:
    """Return (train_paths, val_paths, split_note)."""
    train_paths = list_images(train_dir, extensions)
    val_paths = list_images(val_dir, extensions) if val_dir else []

    if val_paths:
        return train_paths, val_paths, f"ayri val klasoru: {val_dir}"

    if not train_paths:
        raise SystemExit(f"Görüntü yok: {train_dir}")

    n_val = max(1, int(round(len(train_paths) * val_frac)))
    if n_val >= len(train_paths):
        n_val = max(1, len(train_paths) - 1) if len(train_paths) > 1 else 1
    val_paths = train_paths[-n_val:]
    train_paths = train_paths[:-n_val] if len(train_paths) > n_val else []
    return train_paths, val_paths, f"train son %{int(val_frac * 100)} ({len(val_paths)} görüntü)"


def lab_L_to_gray_rgb(L: torch.Tensor) -> np.ndarray:
    """L: (1, H, W) normalized -> (H, W, 3) float [0,1]."""
    g = L.squeeze(0).cpu().numpy()
    return np.stack([g, g, g], axis=-1)


def tensor_rgb_to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)


def save_triplet(gray_hwc: np.ndarray, pred_hwc: np.ndarray, gt_hwc: np.ndarray, path: Path) -> None:
    def u8(a: np.ndarray) -> np.ndarray:
        return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)

    g, p, t = u8(gray_hwc), u8(pred_hwc), u8(gt_hwc)
    h, w, _ = g.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = g
    canvas[:, w : 2 * w] = p
    canvas[:, 2 * w : 3 * w] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate colorization checkpoint")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None, help="Override infer.checkpoint")
    ap.add_argument("--out-dir", default="eval_samples", help="Side-by-side sample folder")
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Val yoksa train listesinin son bu oran (varsayilan 0.2)",
    )
    ap.add_argument("--num-samples", type=int, default=5, help="Kaydedilecek karsilastirma sayisi")
    ap.add_argument("--max-val", type=int, default=None, help="En fazla N val görüntü (hizli test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    icfg = cfg.get("infer", {})
    ckpt = args.checkpoint or icfg.get("checkpoint", "./checkpoints/best.pt")
    if not Path(ckpt).is_file():
        raise SystemExit(f"Checkpoint yok: {ckpt}")

    device_s = icfg.get("device", "cuda")
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")
    device_label = "GPU" if device.type == "cuda" else "CPU"

    extensions = data_cfg["extensions"]
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg.get("val_dir") or train_dir

    train_paths, val_paths, split_note = resolve_splits(
        train_dir, val_dir if Path(val_dir).is_dir() else None, extensions, args.val_frac
    )
    if not val_paths:
        raise SystemExit("Degerlendirme icin val görüntü yok.")

    if args.max_val is not None and args.max_val > 0:
        val_paths = val_paths[: args.max_val]

    use_hints = bool(data_cfg.get("use_hints", True))
    image_size = int(data_cfg.get("image_size", 256))

    val_ds = ColorizationDataset(
        None,
        image_size,
        extensions,
        cfg["hints"],
        is_train=False,
        use_hints=use_hints,
        image_paths=val_paths,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(4, int(data_cfg.get("num_workers", 0))),
        pin_memory=device.type == "cuda",
    )

    G, ckpt_cfg = load_generator(ckpt, device)
    mcfg = ckpt_cfg.get("model", cfg.get("model", {}))
    base_ch = mcfg.get("base_channels", 64)
    num_down = mcfg.get("num_down", 5)
    inch = getattr(G, "in_channels", mcfg.get("in_channels", 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_ssim = 0.0
    sum_psnr = 0.0
    sum_mse = 0.0
    sum_ms = 0.0
    n = 0
    saved = 0

    for batch in tqdm(val_loader, desc="eval"):
        x = batch["input"].to(device)
        L = batch["L"].to(device)
        ab = batch["ab"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_ab = G(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_lab = torch.cat([L, pred_ab], dim=1)
        gt_lab = torch.cat([L, ab], dim=1)
        pred_rgb = lab_to_rgb(pred_lab)
        gt_rgb = lab_to_rgb(gt_lab)

        mse = F.mse_loss(pred_rgb, gt_rgb).item()
        pred_hwc = tensor_rgb_to_hwc(pred_rgb)
        gt_hwc = tensor_rgb_to_hwc(gt_rgb)
        gray_hwc = lab_L_to_gray_rgb(L[0])

        ssim_v = float(ssim_metric(gt_hwc, pred_hwc, data_range=1.0, channel_axis=2))
        psnr_v = float(psnr_metric(gt_hwc, pred_hwc, data_range=1.0))

        sum_ssim += ssim_v
        sum_psnr += psnr_v
        sum_mse += mse
        sum_ms += elapsed_ms
        n += 1

        if saved < args.num_samples:
            stem = Path(batch["path"][0]).stem
            save_triplet(
                gray_hwc,
                pred_hwc,
                gt_hwc,
                out_dir / f"{saved + 1:02d}_{stem}.png",
            )
            saved += 1

    if n == 0:
        raise SystemExit("Val set bos.")

    mean_ssim = sum_ssim / n
    mean_psnr = sum_psnr / n
    mean_mse = sum_mse / n
    mean_ms = sum_ms / n

    print()
    print("=== COLORIZATION EVALUATION ===")
    print(f"Dataset: {len(train_paths) + len(val_paths)} images (train: {len(train_paths)}, val: {len(val_paths)})")
    print(f"Split: {split_note}")
    print(f"Checkpoint: {ckpt}")
    print(f"Architecture: U-Net (LAB color space, in={inch}, base={base_ch}, down={num_down})")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"PSNR: {mean_psnr:.1f} dB")
    print(f"Val Loss (MSE): {mean_mse:.6f}")
    print(f"Inference Speed: {mean_ms:.1f} ms/image ({device_label})")
    print(f"Samples saved: {out_dir.resolve()} ({saved} images)")
    print("================================")


if __name__ == "__main__":
    main()
