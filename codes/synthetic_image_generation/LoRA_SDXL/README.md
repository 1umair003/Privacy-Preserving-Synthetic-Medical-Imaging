# LoRA-SDXL: Proposed Synthesis Pipeline

| Setting | Value |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` (+ fp16-fix VAE) |
| LoRA rank / alpha | 128 / 128 |
| Target layers | U-Net attention `to_q`, `to_k`, `to_v`, `to_out.0` |
| Epochs | 50 |
| Learning rate | 1 × 10⁻⁴, linear schedule, 500 warm-up steps |
| Batch size | 16 (4 per device × 4 gradient accumulation, configurable) |
| Optimiser | 8-bit Adam |
| Resolution | 1024 × 1024 |
| Sampler | DDIM, 200 steps, η = 0 |
| Quality filter | SSIM ≥ 0.85 against real images of the same class |
| Images per class | `--n_per_class` images are **kept** after filtering; generation stops after `n_per_class × --max_attempts_factor` (default 4) attempts |

## Run (GPU)

```bash
# 1. captioned dataset, TB and COVID-19 oversampled 2x
python prepare_dataset.py --real_dir /path/to/real_train --output_dir data/lora_train

# 2. LoRA fine-tuning (clones the official diffusers training script on first run)
bash train_lora_sdxl.sh data/lora_train outputs/lora_weights
#    adjust memory use: PER_DEVICE_BATCH=2 GRAD_ACCUM=8 bash train_lora_sdxl.sh ...

# 3. generation + SSIM filter + SSIM / PSNR / SNR log
python generate_images.py --lora_dir outputs/lora_weights --n_per_class 500 \
    --real_dir /path/to/real_train --output_dir outputs/synthetic
```

## Files

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Builds `metadata.jsonl` with class prompts, applies 2× oversampling |
| `train_lora_sdxl.sh` | Runs diffusers' `train_text_to_image_lora_sdxl.py` with the paper's settings |
| `generate_images.py` | DDIM generation per class, SSIM filtering, metrics CSV |

Class prompts live in `codes/common/config.py` (`CLASS_PROMPTS`).

> The official script applies random horizontal flips. The paper's additional geometric / photometric augmentation for LoRA training is not part of that script.
