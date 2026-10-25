# SMGAN: GAN Baseline

Class-conditional style-modulated GAN (StyleGAN-style, Eqs. 2–4).

| Component | Detail |
|---|---|
| Mapping network | z + class embedding → 8 FC layers → w |
| Synthesis | Learned 4×4 constant → up-sampling blocks with noise injection and **AdaIN(w)** |
| Discriminator | Conv critic with projection class conditioning |
| D loss | WGAN-GP (λ = 10) + lazy R1 penalty |
| G loss | Adversarial + path-length regularisation |
| Evaluation | **SSIM, PSNR, SNR** vs the most similar real image of the same class |

## Run

```bash
# train (paper's GAN baseline used 512 x 512; 256 is faster)
python smgan.py train --data_dir /path/to/real_train --resolution 512 --epochs 200

# generate images per class
python smgan.py generate --checkpoint outputs/checkpoints/generator_last.pt \
    --n_per_class 500 --output_dir outputs/generated
```

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `metrics_per_epoch.csv` | Losses + SSIM / PSNR / SNR |
| `latest_quality_metrics.json` | Overall and per-class metrics at the last evaluation |
| `checkpoints/` | `generator_last.pt`, `discriminator_last.pt` |
| `samples/` | 4 samples per class at each evaluation |
