#!/usr/bin/env python3
"""Train LAB U-Net (otomatik L→ab veya ipuculu 4 kanal). Usage: python train.py --config config.yaml"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from src.color_space import lab_to_rgb
from src.dataset import ColorizationDataset, list_images
from src.losses import VGGPerceptualLoss, chroma_weighted_l1, hinge_d_loss, hinge_g_loss
from src.models.discriminator import PatchDiscriminator
from src.models.unet import HintGuidedUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(
    G: nn.Module,
    loader: DataLoader,
    perc: VGGPerceptualLoss,
    device: torch.device,
    lambda_l1: float,
    lambda_p: float,
    use_amp: bool,
    chroma_l1_scale: float,
) -> dict[str, float]:
    G.eval()
    tot_l1 = 0.0
    tot_p = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        L = batch["L"].to(device)
        ab = batch["ab"].to(device)
        with autocast("cuda", enabled=use_amp):
            pred = G(x)
            l1 = chroma_weighted_l1(pred, ab, chroma_l1_scale)
        with autocast("cuda", enabled=False):
            pred_lab = torch.cat([L.float(), pred.float()], dim=1)
            gt_lab = torch.cat([L.float(), ab.float()], dim=1)
            pr = lab_to_rgb(pred_lab)
            gr = lab_to_rgb(gt_lab)
            lp = perc(pr, gr)
        tot_l1 += l1.item() * x.size(0)
        tot_p += lp.item() * x.size(0)
        n += x.size(0)
    G.train()
    return {"l1": tot_l1 / max(n, 1), "perceptual": tot_p / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Önceki .pt (best veya epoch_N): G ağırlıkları + epoch/metric yüklenir; "
        "eğitim bir sonraki epoch'tan train.epochs'a kadar devam eder. Optimizer sıfırdan.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hint_cfg = cfg["hints"]
    data_cfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    use_hints = bool(data_cfg.get("use_hints", True))
    inch = int(mcfg.get("in_channels", 4))
    if use_hints and inch != 4:
        raise SystemExit("data.use_hints: true iken model.in_channels: 4 olmalı.")
    if not use_hints and inch != 1:
        raise SystemExit("data.use_hints: false iken model.in_channels: 1 olmalı (otomatik renklendirme).")

    max_train = data_cfg.get("max_train_samples")
    sequential_chunks = bool(data_cfg.get("train_sequential_chunks", False))
    train_dir = data_cfg["train_dir"]

    if sequential_chunks:
        if not isinstance(max_train, int) or max_train <= 0:
            raise SystemExit(
                "data.train_sequential_chunks: true iken data.max_train_samples pozitif bir tamsayı olmalı (blok boyutu)."
            )
        all_train_paths = list_images(train_dir, data_cfg["extensions"])
        if len(all_train_paths) == 0:
            raise SystemExit(
                f"No images in {train_dir}. Add RGB images to data/train (see README)."
            )
        tqdm.write(
            f"Train sıralı bloklar: {len(all_train_paths)} görüntü (alfabetik), epoch başına {max_train} adet; sona gelince başa döner."
        )
        chunk_cursor = 0
        chunk_size = max_train
    else:
        train_ds = ColorizationDataset(
            train_dir,
            data_cfg["image_size"],
            data_cfg["extensions"],
            hint_cfg,
            is_train=True,
            use_hints=use_hints,
            max_samples=max_train if isinstance(max_train, int) and max_train > 0 else None,
        )
        note = getattr(train_ds, "_truncation_note", None)
        if note:
            tqdm.write(note)
        if len(train_ds) == 0:
            raise SystemExit(
                f"No images in {train_dir}. Add RGB images to data/train (see README)."
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    val_dir = data_cfg.get("val_dir") or train_dir
    val_ds = ColorizationDataset(
        val_dir,
        data_cfg["image_size"],
        data_cfg["extensions"],
        hint_cfg,
        is_train=False,
        use_hints=use_hints,
    )

    if len(val_ds) == 0:
        tqdm.write(
            "UYARI: val klasörü boş — val L1/perc 0 görünür. "
            "data/val içine görüntü koy veya config'te val_dir: \"./data/train\" (küçük deneme)."
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=min(8, tcfg["batch_size"]),
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    G = HintGuidedUNet(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        base_channels=mcfg["base_channels"],
        num_down=mcfg["num_down"],
    ).to(device)

    resume_path = args.resume
    start_epoch = 0
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_file():
            raise SystemExit(f"--resume dosyası yok: {rp}")
        ckpt_resume = torch.load(rp, map_location=device, weights_only=False)
        if "G" not in ckpt_resume:
            raise SystemExit(f"Checkpoint'ta 'G' yok: {rp}")
        G.load_state_dict(ckpt_resume["G"], strict=True)
        start_epoch = int(ckpt_resume.get("epoch", 0))
        tqdm.write(f"Resume: {rp}  |  kayıtlı epoch={start_epoch}")
        if start_epoch >= int(tcfg["epochs"]):
            raise SystemExit(
                f"Kayıtlı epoch ({start_epoch}) >= train.epochs ({tcfg['epochs']}). "
                "config'te train.epochs değerini artır (örn. 100)."
            )

    use_gan = bool(tcfg.get("use_gan", False))
    D: PatchDiscriminator | None = None
    if use_gan:
        D = PatchDiscriminator(in_channels=3, base=64, n_layers=3).to(device)
        opt_d = torch.optim.Adam(D.parameters(), lr=tcfg["lr_d"], betas=(tcfg["beta1"], tcfg["beta2"]))

    opt_g = torch.optim.Adam(G.parameters(), lr=tcfg["lr_g"], betas=(tcfg["beta1"], tcfg["beta2"]))

    perc = VGGPerceptualLoss(tcfg["perceptual_layers"]).to(device)
    scaler = GradScaler("cuda", enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")
    use_amp = scaler.is_enabled()

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_tb = bool(tcfg.get("tensorboard", True)) and SummaryWriter is not None
    if bool(tcfg.get("tensorboard", True)) and SummaryWriter is None:
        tqdm.write("TensorBoard: paket yok. Kur: pip install tensorboard")
    tb_dir = Path(tcfg.get("tensorboard_dir", "./runs"))
    writer = None
    if use_tb:
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir / "colorization"))
        tqdm.write(f"TensorBoard log: {tb_dir / 'colorization'}  ->  tensorboard --logdir {tb_dir}")

    best_metric = float("inf")
    if resume_path:
        best_metric = float(ckpt_resume.get("metric", float("inf")))
        tqdm.write(f"Resume: best combined metric (önceki) = {best_metric}")

    global_step = 0
    lambda_l1 = float(tcfg["lambda_l1"])
    lambda_p = float(tcfg["lambda_perceptual"])
    lambda_gan = float(tcfg["lambda_gan"])
    chroma_l1_scale = float(tcfg.get("chroma_l1_scale", 2.0))
    gclip = tcfg.get("grad_clip_norm")
    grad_clip = float(gclip) if gclip is not None and float(gclip) > 0 else None

    bs = int(tcfg["batch_size"])
    for epoch in range(start_epoch + 1, tcfg["epochs"] + 1):
        if sequential_chunks:
            n_paths = len(all_train_paths)
            if chunk_cursor >= n_paths:
                chunk_cursor = 0
            start = chunk_cursor
            end = min(start + chunk_size, n_paths)
            chunk_paths = all_train_paths[start:end]
            chunk_cursor = end
            if chunk_cursor >= n_paths:
                chunk_cursor = 0
            tqdm.write(f"Epoch {epoch}: train indeks [{start}:{end}) → {len(chunk_paths)} görüntü")
            train_ds_epoch = ColorizationDataset(
                None,
                data_cfg["image_size"],
                data_cfg["extensions"],
                hint_cfg,
                is_train=True,
                use_hints=use_hints,
                image_paths=chunk_paths,
            )
            train_loader = DataLoader(
                train_ds_epoch,
                batch_size=bs,
                shuffle=True,
                num_workers=data_cfg["num_workers"],
                pin_memory=True,
                drop_last=len(chunk_paths) >= bs,
            )

        G.train()
        if D is not None:
            D.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tcfg['epochs']}")
        for step, batch in enumerate(pbar):
            x = batch["input"].to(device)
            L = batch["L"].to(device)
            ab = batch["ab"].to(device)

            with autocast("cuda", enabled=use_amp):
                pred = G(x)
                loss_l1 = chroma_weighted_l1(pred, ab, chroma_l1_scale)
            with autocast("cuda", enabled=False):
                pred_lab = torch.cat([L.float(), pred.float()], dim=1)
                gt_lab = torch.cat([L.float(), ab.float()], dim=1)
                pr = lab_to_rgb(pred_lab)
                gr = lab_to_rgb(gt_lab)
                loss_p = perc(pr, gr)
            loss_g = lambda_l1 * loss_l1 + lambda_p * loss_p
            if D is not None:
                with autocast("cuda", enabled=use_amp):
                    loss_g = loss_g + lambda_gan * hinge_g_loss(D(pr))

            if not torch.isfinite(loss_g):
                tqdm.write(f"Uyarı: loss nan/inf (epoch {epoch}, step {step}) — adım atlandı.")
                continue

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            if grad_clip is not None:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            scaler.step(opt_g)

            if D is not None:
                with autocast("cuda", enabled=use_amp):
                    real_logits = D(gr)
                    fake_logits = D(pr.detach())
                    loss_d = hinge_d_loss(real_logits, fake_logits)
                if not torch.isfinite(loss_d):
                    tqdm.write(f"Uyarı: loss_d nan/inf (epoch {epoch}, step {step}) — D adımı atlandı.")
                    scaler.update()
                    global_step += 1
                    continue
                opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
                scaler.step(opt_d)

            scaler.update()

            global_step += 1
            if step % tcfg["log_every"] == 0:
                if torch.isfinite(loss_l1) and torch.isfinite(loss_p):
                    msg = {"l1": f"{loss_l1.item():.4f}", "perc": f"{loss_p.item():.4f}"}
                    if D is not None and torch.isfinite(loss_d):
                        msg["d"] = f"{loss_d.item():.4f}"
                    pbar.set_postfix(msg)
                if writer is not None and torch.isfinite(loss_g):
                    writer.add_scalar("train/loss_l1", loss_l1.item(), global_step)
                    writer.add_scalar("train/loss_perceptual", loss_p.item(), global_step)
                    writer.add_scalar("train/loss_g_total", loss_g.item(), global_step)
                    if D is not None and torch.isfinite(loss_d):
                        writer.add_scalar("train/loss_d", loss_d.item(), global_step)

        metrics = validate(
            G, val_loader, perc, device, lambda_l1, lambda_p, use_amp, chroma_l1_scale
        )
        tqdm.write(f"val L1 {metrics['l1']:.5f}  val perc {metrics['perceptual']:.5f}")
        if writer is not None:
            writer.add_scalar("epoch/val_l1", metrics["l1"], epoch)
            writer.add_scalar("epoch/val_perceptual", metrics["perceptual"], epoch)

        combined = metrics["l1"] + 0.1 * metrics["perceptual"]
        if writer is not None:
            writer.add_scalar("epoch/val_combined_metric", combined, epoch)
        if combined < best_metric:
            best_metric = combined
            torch.save(
                {
                    "G": G.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metric": combined,
                },
                ckpt_dir / "best.pt",
            )

        if epoch % int(tcfg.get("save_every", 1)) == 0:
            torch.save(
                {"G": G.state_dict(), "cfg": cfg, "epoch": epoch},
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    if writer is not None:
        writer.close()
    tqdm.write(f"Done. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
