# Synthetic Image Generation

| Method | Folder | Role | Metrics |
|---|---|---|---|
| **LoRA-SDXL** | [`LoRA_SDXL/`](LoRA_SDXL) | Proposed method, 1024 × 1024 | SSIM, PSNR, SNR (+ SSIM ≥ 0.85 filter) |
| **SRGAN** | [`SRGAN/`](SRGAN) | GAN baseline (super-resolution) | SSIM, PSNR, SNR each epoch |
| **SMGAN** | [`SMGAN/`](SMGAN) | GAN baseline (style-modulated, class-conditional) | SSIM, PSNR, SNR each evaluation |
| Evaluation | [`image_quality_evaluation/`](image_quality_evaluation) | Scores any generator against real CXRs | SSIM, PSNR, SNR, optional FID |

## Paper results (Table 1)

| Metric | GAN (SMGAN/SRGAN) | LoRA-SDXL |
|---|---|---|
| SSIM ↑ | 0.62 | **0.92** |
| PSNR (dB) ↑ | 21.4 | **28.9** |
| FID ↓ | ~142 | **~38** |

## Metric definitions

- **SSIM**: structural similarity of luminance, contrast and structure (Eq. 33)
- **PSNR** = 10 · log₁₀(MAX² / MSE), in dB (Eq. 34)
- **SNR** = 10 · log₁₀(Σ ref² / Σ (ref − test)²), in dB

For unpaired generation (LoRA-SDXL, SMGAN), each synthetic image is compared with the **most similar real image of the same class**. SRGAN is paired, so each super-resolved image is compared with its own high-resolution original.
