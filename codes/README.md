# Code

This folder holds every script used in the paper: synthetic image generation, CNN training (with and without pre-trained weights) and testing.

← [Back to paper overview](../README.md)

---

## Folder Structure

```
codes/
├── README.md                          ← you are here
├── requirements.txt
│
├── common/                            ← shared code (config, data, models, metrics)
│   ├── config.py                      ← all hyperparameters from the paper
│   ├── data.py                        ← data loading, augmentation, label encoding
│   ├── models.py                      ← backbones + shared classification head
│   ├── callbacks.py                   ← cosine annealing + plateau LR schedule
│   ├── trainer.py                     ← 5-fold stratified CV training loop
│   ├── evaluator.py                   ← test metrics, confusion matrix, ROC
│   ├── image_metrics.py               ← SSIM, PSNR, SNR
│   └── gan_utils.py                   ← PyTorch helpers for the GANs
│
├── synthetic_image_generation/
│   ├── LoRA_SDXL/                     ← proposed method
│   ├── SRGAN/                         ← GAN baseline (SSIM, PSNR, SNR)
│   ├── SMGAN/                         ← GAN baseline (SSIM, PSNR, SNR)
│   └── image_quality_evaluation/      ← compare any generator with real images
│
├── data_preparation/
│   └── make_hybrid_dataset.py         ← 50% real + 50% synthetic
│
├── code_with_weights/                 ← training, ImageNet pre-trained weights
│   ├── ResNet50/
│   ├── InceptionV3/
│   ├── Xception/
│   └── train_all.sh
│
├── code_without_weights/              ← training from scratch
│   ├── ResNet50/
│   ├── InceptionV3/
│   ├── Xception/
│   └── train_all.sh
│
└── Test/
    ├── test_with_weights/             ← tests models from code_with_weights
    │   ├── ResNet50/  InceptionV3/  Xception/  test_all.sh
    ├── test_without_weights/          ← tests models from code_without_weights
    │   ├── ResNet50/  InceptionV3/  Xception/  test_all.sh
    ├── statistical_tests.py           ← paired t-test + Wilcoxon
    └── summarize_results.py           ← Table 2-style summary
```

---

## 1. Setup

```bash
git clone https://github.com/1umair003/Privacy-Preserving-Synthetic-Medical-Imaging.git
cd Privacy-Preserving-Synthetic-Medical-Imaging/codes

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Part | Hardware |
|---|---|
| LoRA-SDXL training / generation | NVIDIA GPU, ≥ 24 GB VRAM recommended |
| SRGAN / SMGAN | NVIDIA GPU recommended (runs on CPU, slowly) |
| CNN training / testing | GPU recommended; CPU works for testing |

---

## 2. Prepare the Data

Every data folder (real, synthetic, hybrid, test) uses **one sub-folder per class**:

```
data/real_train/
├── Bacterial_Pneumonia/
├── COVID-19/
├── Normal/
├── Tuberculosis/
└── Viral_Pneumonia/
```

The real images come from the public Kaggle collections used in the paper: **NIH ChestX-ray14**, the **COVID-19 Radiography Database** and the **Montgomery County TB** set. Keep a separate **real held-out test set** (`data/real_test/`) that is never used for training or LoRA fine-tuning.

---

## 3. Generate Synthetic Images

### 3a. LoRA-SDXL (proposed method)

```bash
cd synthetic_image_generation/LoRA_SDXL

# build the captioned training set (TB and COVID-19 oversampled 2x)
python prepare_dataset.py --real_dir ../../data/real_train --output_dir data/lora_train

# fine-tune SDXL with LoRA (r = 128, alpha = 128, 50 epochs)
bash train_lora_sdxl.sh data/lora_train outputs/lora_weights

# generate 1024x1024 images per class, keep those with SSIM >= 0.85
python generate_images.py --lora_dir outputs/lora_weights --n_per_class 500 \
    --real_dir ../../data/real_train --output_dir ../../data/synthetic_train
```

### 3b. SRGAN baseline

```bash
cd synthetic_image_generation/SRGAN
python srgan.py train --data_dir ../../data/real_train --hr_size 512 --epochs 100
```

### 3c. SMGAN baseline

```bash
cd synthetic_image_generation/SMGAN
python smgan.py train --data_dir ../../data/real_train --resolution 512 --epochs 200
python smgan.py generate --checkpoint outputs/checkpoints/generator_last.pt --n_per_class 500
```

Both GAN scripts report **SSIM, PSNR and SNR** during training.

### 3d. Compare generators (Table 1)

```bash
cd codes
python synthetic_image_generation/image_quality_evaluation/evaluate_quality.py \
    --generated_dir data/synthetic_train --real_dir data/real_test --name LoRA_SDXL --fid
```

---

## 4. (Optional) Build the Hybrid Dataset

```bash
python data_preparation/make_hybrid_dataset.py \
    --real_dir data/real_train --synthetic_dir data/synthetic_train --output_dir data/hybrid_train
```

---

## 5. Train the Classifiers

| Paper condition | Folder | `--train_dir` |
|---|---|---|
| Hybrid (50% R + S) | `code_with_weights/` | `data/hybrid_train` |
| Synthetic only / no pre-train | `code_without_weights/` | `data/synthetic_train` |
| Synthetic only / with pre-train | `code_with_weights/` | `data/synthetic_train` |

Train one model:

```bash
python code_with_weights/Xception/train_xception.py --train_dir data/synthetic_train
python code_without_weights/Xception/train_xception.py --train_dir data/synthetic_train
```

Or all three models in one go:

```bash
bash code_with_weights/train_all.sh data/synthetic_train
bash code_without_weights/train_all.sh data/synthetic_train
```

**Training setup (shared):** 5-fold stratified CV · 30 epochs · batch 32 · Adam, η₀ = 1e-4 with cosine annealing and plateau drops (×0.2, patience 3) · class-weighted cross-entropy · augmentation (rotation ±30°, shift/shear/zoom 0.3, brightness 0.8–1.2, horizontal flip).

Each run saves `best_model.keras`, `label_encoder.pkl`, fold accuracies and accuracy / loss / k-fold plots to the model's `outputs/` folder.

> Tip: to train the hybrid and synthetic-only conditions with pre-trained weights side by side, pass `--output_dir` (for example `--output_dir code_with_weights/Xception/outputs_hybrid`) so one run doesn't overwrite the other.

---

## 6. Test the Models

Each test folder automatically loads the model from its matching training folder.

```bash
python Test/test_with_weights/Xception/test_xception.py --test_dir data/real_test
python Test/test_without_weights/Xception/test_xception.py --test_dir data/real_test

# or all models
bash Test/test_with_weights/test_all.sh data/real_test
bash Test/test_without_weights/test_all.sh data/real_test
```

Outputs per model (`results/`): accuracy, per-class precision / recall / F1, macro F1, **confusion matrix**, **ROC curves with AUC** and per-image predictions.

---

## 7. Statistics and Summary Table

```bash
# paired t-test + Wilcoxon signed-rank on fold accuracies
python Test/statistical_tests.py --condition with_weights
python Test/statistical_tests.py --condition without_weights

# collect all test results into one table (Table 2 format)
python Test/summarize_results.py
```

---

## Quick Reference

| Step | Command |
|---|---|
| Install | `pip install -r requirements.txt` |
| LoRA data | `python synthetic_image_generation/LoRA_SDXL/prepare_dataset.py --real_dir data/real_train` |
| LoRA train | `bash synthetic_image_generation/LoRA_SDXL/train_lora_sdxl.sh` |
| LoRA generate | `python synthetic_image_generation/LoRA_SDXL/generate_images.py --lora_dir ...` |
| SRGAN | `python synthetic_image_generation/SRGAN/srgan.py train --data_dir data/real_train` |
| SMGAN | `python synthetic_image_generation/SMGAN/smgan.py train --data_dir data/real_train` |
| Train (weights) | `bash code_with_weights/train_all.sh data/synthetic_train` |
| Train (no weights) | `bash code_without_weights/train_all.sh data/synthetic_train` |
| Test (weights) | `bash Test/test_with_weights/test_all.sh data/real_test` |
| Test (no weights) | `bash Test/test_without_weights/test_all.sh data/real_test` |
| Statistics | `python Test/statistical_tests.py --condition with_weights` |
| Summary table | `python Test/summarize_results.py` |

Every script supports `--help` for the full list of options.
