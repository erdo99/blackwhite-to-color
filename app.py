#!/usr/bin/env python3
"""Gradio UI: otomatik renklendirme veya ipuculu mod (config: data.use_hints)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

from infer import colorize, colorize_auto, load_generator


def _config() -> dict:
    p = Path("config.yaml")
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _editor_to_pils(editor_value):
    if editor_value is None:
        return None, None

    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        comp = editor_value.get("composite")
        if comp is None:
            comp = editor_value.get("image")
    else:
        bg = getattr(editor_value, "background", None)
        comp = getattr(editor_value, "composite", None) or getattr(editor_value, "image", None)

    def to_pil(x):
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            rgb = arr[..., :3].astype(np.float32)
            a = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = rgb * a + 255 * (1 - a)
            arr = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    return to_pil(bg), to_pil(comp)


def run_auto(image_pil, ckpt_path: str, image_size: int, full_resolution: bool):
    cfg = _config()
    icfg = cfg.get("infer", {})
    ckpt = ckpt_path or icfg.get("checkpoint", "./checkpoints/best.pt")
    if not Path(ckpt).is_file():
        return None, f"Checkpoint yok: {ckpt}. Önce `python train.py` çalıştır."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, _ = load_generator(ckpt, device)
    if image_pil is None:
        return None, "Bir görüntü yükleyin."
    if getattr(G, "in_channels", 1) != 1:
        return None, "Bu checkpoint 4 kanallı (ipuculu). config’te use_hints: true ve uygun .pt kullanın."

    out = colorize_auto(G, image_pil.convert("RGB"), image_size, device, full_resolution)
    return out, "Tamam."


def run_guided(editor_value, ckpt_path: str, image_size: int, diff_thresh: float, full_resolution: bool):
    cfg = _config()
    icfg = cfg.get("infer", {})
    ckpt = ckpt_path or icfg.get("checkpoint", "./checkpoints/best.pt")
    if not Path(ckpt).is_file():
        return None, f"Checkpoint yok: {ckpt}."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, _ = load_generator(ckpt, device)
    if getattr(G, "in_channels", 4) == 1:
        return None, "Bu checkpoint 1 kanallı (otomatik). Tek görüntü modunu kullanın veya otomatik eğitilmiş .pt yükleyin."

    bg, comp = _editor_to_pils(editor_value)
    if comp is None:
        return None, "Editörde görüntü yükleyin (gerekirse fırça ile ipucu çizin), sonra Çalıştır."

    if bg is None:
        gray = comp.convert("L").convert("RGB")
    else:
        gray = bg.convert("L").convert("RGB")

    if gray.size != comp.size:
        comp = comp.resize(gray.size, Image.Resampling.BICUBIC)

    out = colorize(
        G,
        gray,
        comp,
        image_size=image_size,
        device=device,
        diff_thresh=diff_thresh,
        full_resolution=full_resolution,
    )
    return out, "Tamam."


def build_demo(ckpt: str | None, port: int, host: str, share: bool):
    cfg = _config()
    icfg = cfg.get("infer", {})
    default_ckpt = ckpt or icfg.get("checkpoint", "./checkpoints/best.pt")
    image_size = int(icfg.get("image_size", cfg.get("data", {}).get("image_size", 256)))
    use_hints = bool(cfg.get("data", {}).get("use_hints", True))

    with gr.Blocks(title="LAB renklendirme") as demo:
        if use_hints:
            gr.Markdown(
                """
                ### İpuculu renklendirme (4 kanal model)
                1. Arka plana görüntü yükleyin, fırça ile renk ipuçları verin.
                2. **Renklendir** deyin.
                """
            )
            editor = gr.ImageEditor(
                type="numpy",
                label="Görüntü ve ipuçları",
                brush=gr.Brush(
                    default_size=24,
                    colors=["#c41e3a", "#228b22", "#1e90ff", "#ffd700", "#8b4513", "#ffffff"],
                ),
            )
            with gr.Row():
                ckpt_in = gr.Textbox(value=default_ckpt, label="Checkpoint (.pt)")
                diff = gr.Slider(0.02, 0.2, value=0.06, step=0.01, label="İpucu hassasiyeti")
            fullres = gr.Checkbox(value=True, label="Tam çözünürlük (AB yükseltme)")
            run_btn = gr.Button("Renklendir", variant="primary")
            out_img = gr.Image(type="pil", label="Sonuç")
            status = gr.Textbox(label="Durum", interactive=False)
            run_btn.click(
                fn=lambda ev, c, d, fr: run_guided(ev, c, image_size, d, fr),
                inputs=[editor, ckpt_in, diff, fullres],
                outputs=[out_img, status],
            )
        else:
            gr.Markdown(
                """
                ### Otomatik renklendirme (L → renk, **kullanıcı rengi yok**)
                Görüntü yükleyin; model parlaklıktan renk tahmin eder. **Yeni eğitim gerekir** (`data.use_hints: false`, `in_channels: 1`).
                Eski 4 kanallı `.pt` bu modda çalışmaz.
                """
            )
            inp = gr.Image(type="pil", label="Görüntü yükle (gri veya renkli)", image_mode="RGB")
            with gr.Row():
                ckpt_in = gr.Textbox(value=default_ckpt, label="Checkpoint (.pt)")
            fullres = gr.Checkbox(value=True, label="Tam çözünürlük (AB yükseltme)")
            run_btn = gr.Button("Renklendir", variant="primary")
            out_img = gr.Image(type="pil", label="Sonuç")
            status = gr.Textbox(label="Durum", interactive=False)
            run_btn.click(
                fn=lambda im, c, fr: run_auto(im, c, image_size, fr),
                inputs=[inp, ckpt_in, fullres],
                outputs=[out_img, status],
            )

    demo.launch(server_name=host, server_port=port, share=share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    cfg = _config()
    ui = cfg.get("ui", {})
    build_demo(
        args.checkpoint,
        port=int(args.port or ui.get("server_port", 7860)),
        host=args.host or ui.get("server_name", "0.0.0.0"),
        share=bool(args.share or ui.get("share", False)),
    )


if __name__ == "__main__":
    main()
