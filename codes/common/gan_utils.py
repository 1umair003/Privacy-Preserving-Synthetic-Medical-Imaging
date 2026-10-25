"""PyTorch helpers shared by the SRGAN and SMGAN baselines."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import config


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_class_images(root: str | Path) -> tuple[list[Path], list[int], list[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((root / c).rglob("*")):
            if f.suffix.lower() in config.IMAGE_EXTENSIONS:
                paths.append(f)
                labels.append(i)
    if not paths:
        raise ValueError(f"No images found under {root}")
    return paths, labels, classes


class GrayscaleCXRDataset(Dataset):
    """Grayscale CXRs resized to `size` and scaled to [-1, 1]. Returns (image, class index)."""

    def __init__(self, paths, labels, size: int, hflip: bool = True):
        self.paths, self.labels, self.size, self.hflip = list(paths), list(labels), size, hflip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).unsqueeze(0)
        if self.hflip and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, self.labels[i]


def to_unit(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor (1, H, W) -> numpy (H, W) in [0, 1]."""
    return ((x.detach().float().cpu().squeeze(0).numpy() + 1.0) / 2.0).clip(0, 1)


def save_png(x: torch.Tensor, path: Path) -> None:
    Image.fromarray((to_unit(x) * 255).round().astype(np.uint8)).save(path)


def save_grid(images: torch.Tensor, path: Path, nrow: int) -> None:
    from torchvision.utils import save_image

    save_image((images.detach().cpu() + 1) / 2, path, nrow=nrow)


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
