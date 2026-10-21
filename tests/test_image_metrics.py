"""Sanity checks for the SSIM / PSNR / SNR implementation in codes/common/image_metrics.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
from common import image_metrics as m  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    ref = rng.random((64, 64))
    noisy = np.clip(ref + rng.normal(0, 0.05, ref.shape), 0, 1)
    very_noisy = np.clip(ref + rng.normal(0, 0.2, ref.shape), 0, 1)

    # identical images
    assert abs(m.ssim(ref, ref) - 1.0) < 1e-9
    assert m.psnr(ref, ref) == float("inf")
    assert m.snr(ref, ref) == float("inf")

    # more noise -> lower scores
    assert m.ssim(ref, noisy) > m.ssim(ref, very_noisy)
    assert m.psnr(ref, noisy) > m.psnr(ref, very_noisy)
    assert m.snr(ref, noisy) > m.snr(ref, very_noisy)

    # PSNR matches its definition: 10 * log10(1 / MSE) for data range 1
    expected = 10 * np.log10(1.0 / np.mean((ref - noisy) ** 2))
    assert abs(m.psnr(ref, noisy) - expected) < 1e-6

    # images of different size are resized before comparison
    assert 0 < m.ssim(ref, noisy[:32, :32]) <= 1

    # best match picks the closest reference
    best = m.best_match_metrics(noisy, [very_noisy, ref])
    assert best["reference_index"] == 1

    print("All image-metric checks passed.")


if __name__ == "__main__":
    main()
