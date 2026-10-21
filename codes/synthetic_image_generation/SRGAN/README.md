# SRGAN: GAN Baseline

Super-resolution GAN (Eqs. 5–7). Real CXRs are down-sampled by `--scale` (bicubic) and the generator learns to restore them to `--hr_size`.

| Component | Detail |
|---|---|
| Generator | SRResNet: 16 residual blocks + PixelShuffle up-sampling |
| Discriminator | 8 conv layers + dense head |
| Loss | VGG19 content loss (φ₅,₄) + 10⁻³ × adversarial loss |
| Evaluation | **SSIM, PSNR, SNR** vs the real high-resolution image, every epoch, plus a bicubic reference |

## Run

```bash
# train
python srgan.py train --data_dir /path/to/real_train --hr_size 512 --scale 4 --epochs 100

# super-resolve a folder with the best generator
python srgan.py infer --checkpoint outputs/checkpoints/generator_best.pt \
    --input_dir /path/to/low_res --output_dir outputs/generated
```

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `metrics_per_epoch.csv` | Losses + SSIM / PSNR / SNR (SRGAN and bicubic) per epoch |
| `best_metrics.json` | Mean and std of every metric at the best epoch |
| `checkpoints/` | `generator_best.pt`, `generator_last.pt`, `discriminator_last.pt` |
| `samples/` | Rows: bicubic · SRGAN · real |
