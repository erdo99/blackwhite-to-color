"""RGB image folder dataset -> LAB tensors; optional hint simulation for guided training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .color_space import rgb_to_lab
from .hints import build_model_input, simulate_hints


def list_images(root: str | Path, extensions: list[str]) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


class ColorizationDataset(Dataset):
    """
    Loads RGB images, converts to normalized LAB.

    use_hints=False: input is L only (1xHxW) — automatic colorization.
    use_hints=True: input is 4 channels (L, mask, hint_a, hint_b).
    """

    def __init__(
        self,
        root: str | None,
        image_size: int,
        extensions: list[str],
        hint_cfg: dict,
        is_train: bool = True,
        use_hints: bool = True,
        max_samples: int | None = None,
        image_paths: list[Path] | None = None,
    ):
        self._truncation_note: str | None = None
        if image_paths is not None:
            self.paths = [Path(p) for p in image_paths]
        elif root is not None:
            paths = list_images(root, extensions)
            if max_samples is not None and max_samples > 0:
                if len(paths) > max_samples:
                    self._truncation_note = (
                        f"Train: {len(paths)} görüntü bulundu, ilk {max_samples} kullanılıyor (max_train_samples)."
                    )
                self.paths = paths[:max_samples]
            else:
                self.paths = paths
        else:
            raise ValueError("ColorizationDataset: root veya image_paths gerekli.")
        self.image_size = image_size
        self.hint_cfg = hint_cfg
        self.is_train = is_train
        self.use_hints = use_hints

        aug = []
        if is_train:
            aug.append(transforms.RandomHorizontalFlip())
            aug.append(transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02))
        aug += [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
        self.tf = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        rgb = self.tf(img)
        lab = rgb_to_lab(rgb.unsqueeze(0)).squeeze(0)

        L = lab[0:1]
        ab = lab[1:3]

        if not self.use_hints:
            x_in = L
            mask = torch.zeros(1, *L.shape[1:])
            hint_ab = torch.zeros(2, *L.shape[1:])
        elif self.is_train:
            mask, hint_ab = simulate_hints(
                lab.unsqueeze(0),
                min_points=self.hint_cfg["min_points"],
                max_points=self.hint_cfg["max_points"],
                min_patch_radius=self.hint_cfg["min_patch_radius"],
                max_patch_radius=self.hint_cfg["max_patch_radius"],
                patch_prob=self.hint_cfg["patch_prob"],
            )
            mask = mask.squeeze(0)
            hint_ab = hint_ab.squeeze(0)
            x_in = build_model_input(L.unsqueeze(0), mask.unsqueeze(0), hint_ab.unsqueeze(0)).squeeze(0)
        else:
            g = torch.Generator().manual_seed(idx + 12345)
            n = 12
            h, w = L.shape[1], L.shape[2]
            mask = torch.zeros(1, h, w)
            hint_ab = torch.zeros(2, h, w)
            gt_ab = ab
            for _ in range(n):
                y = int(torch.randint(0, h, (1,), generator=g).item())
                x = int(torch.randint(0, w, (1,), generator=g).item())
                r = int(torch.randint(1, 4, (1,), generator=g).item())
                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                dist = ((yy - y).float() ** 2 + (xx - x).float() ** 2).sqrt()
                m = (dist <= r).float()
                mask[0] = torch.maximum(mask[0], m)
                for c in range(2):
                    hint_ab[c] = torch.where(m > 0, gt_ab[c], hint_ab[c])
            mask = (mask > 0).float()
            x_in = build_model_input(L.unsqueeze(0), mask.unsqueeze(0), hint_ab.unsqueeze(0)).squeeze(0)

        return {
            "input": x_in,
            "L": L,
            "ab": ab,
            "mask": mask,
            "hint_ab": hint_ab,
            "path": str(path),
        }
