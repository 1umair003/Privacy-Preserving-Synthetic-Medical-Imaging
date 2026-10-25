# Image Quality Evaluation

Scores any folder of synthetic images (LoRA-SDXL, SMGAN or SRGAN) against real CXRs with **SSIM, PSNR, SNR** and optionally **FID**.

```bash
# from codes/
python synthetic_image_generation/image_quality_evaluation/evaluate_quality.py \
    --generated_dir synthetic_image_generation/LoRA_SDXL/outputs/synthetic \
    --real_dir /path/to/real_test --name LoRA_SDXL

python synthetic_image_generation/image_quality_evaluation/evaluate_quality.py \
    --generated_dir synthetic_image_generation/SMGAN/outputs/generated \
    --real_dir /path/to/real_test --name SMGAN

# add --fid for FID (pip install "torchmetrics[image]")
```

Outputs in `results/`: `<name>_per_image.csv` (with the matched real image), `<name>_per_class.csv` and `<name>_summary.json`.

> The per-image file also records which real image each synthetic image matched most closely. Very high SSIM against a single real image can indicate memorisation, which is useful for the privacy analysis.
