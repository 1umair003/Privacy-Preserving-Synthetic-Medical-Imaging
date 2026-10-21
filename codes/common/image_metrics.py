"""Image-quality metrics used to compare synthetic and real CXRs (Sec. 3.10).

All functions take 2-D grayscale numpy arrays scaled to [0, 1].

* SSIM  - structural similarity (Eq. 33)
* PSNR  - peak signal-to-noise ratio in dB (Eq. 34)
* SNR   - signal-to-noise ratio in dB: 10*log10(sum(ref^2) / sum((ref - test)^2))
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize


def _match(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        test = resize(test, ref.shape, anti_aliasing=True, preserve_range=True)
    return np.clip(ref, 0, 1), np.clip(test, 0, 1)


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = _match(ref, test)
    return float(structural_similarity(ref, test, data_range=1.0))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = _match(ref, test)
    if np.allclose(ref, test):
        return float("inf")
    return float(peak_signal_noise_ratio(ref, test, data_range=1.0))


def snr(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = _match(ref, test)
    noise = np.sum((ref - test) ** 2)
    if noise == 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(ref ** 2) / noise))


def all_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    return {"SSIM": ssim(ref, test), "PSNR": psnr(ref, test), "SNR": snr(ref, test)}


def best_match_metrics(test: np.ndarray, references: list[np.ndarray]) -> dict:
    """Compare a generated image against its most similar real image (highest SSIM).

    Used for unpaired generation (SMGAN, LoRA-SDXL), where a generated image has
    no single ground-truth counterpart.
    """
    best = None
    for idx, ref in enumerate(references):
        m = all_metrics(ref, test)
        if best is None or m["SSIM"] > best["SSIM"]:
            best = {**m, "reference_index": idx}
    return best


def summarise(rows: list[dict]) -> dict:
    """Mean and std for each metric, ignoring infinite values."""
    out = {}
    for key in ("SSIM", "PSNR", "SNR"):
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])], dtype=float)
        out[key] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std()) if vals.size else float("nan"),
        }
    return out


def load_gray(path, size: int | None = None) -> np.ndarray:
    """Load an image file as grayscale float array in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0
