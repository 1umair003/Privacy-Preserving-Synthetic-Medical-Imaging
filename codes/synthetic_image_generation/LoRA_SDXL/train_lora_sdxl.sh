#!/usr/bin/env bash
# Fine-tune Stable Diffusion XL with LoRA on chest X-rays (Sec. 3.4).
#
# Paper configuration: rank r = 128, alpha = 128, 50 epochs, learning rate 1e-4
# with 500-step linear warm-up, batch size 16, 8-bit Adam, 1024 x 1024.
# LoRA adapters are injected into the U-Net query/key/value/output attention
# projections (to_q, to_k, to_v, to_out.0); the backbone stays frozen.
#
# Uses the official Hugging Face diffusers SDXL LoRA training script.
#
# Usage (from this folder, on a GPU machine):
#   bash train_lora_sdxl.sh data/lora_train outputs/lora_weights
set -euo pipefail

TRAIN_DIR="${1:-data/lora_train}"
OUTPUT_DIR="${2:-outputs/lora_weights}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Effective batch size 16 = per-device batch x gradient accumulation x GPUs.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# 1) Get the official training script (once).
if [ ! -d "$HERE/diffusers" ]; then
  git clone --depth 1 https://github.com/huggingface/diffusers "$HERE/diffusers"
  pip install -e "$HERE/diffusers"
  pip install -r "$HERE/diffusers/examples/text_to_image/requirements_sdxl.txt"
fi

# 2) Configure accelerate with defaults if it has not been configured yet.
accelerate config default >/dev/null 2>&1 || true

# 3) Train.
accelerate launch "$HERE/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --train_data_dir="$TRAIN_DIR" \
  --caption_column="text" \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --rank=128 \
  --train_batch_size="$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --gradient_checkpointing \
  --num_train_epochs=50 \
  --learning_rate=1e-4 \
  --lr_scheduler="linear" \
  --lr_warmup_steps=500 \
  --use_8bit_adam \
  --mixed_precision="fp16" \
  --checkpointing_steps=1000 \
  --validation_prompt="a frontal chest X-ray radiograph showing pulmonary tuberculosis, upper-lobe nodular infiltrates with cavitation, high resolution, grayscale" \
  --validation_epochs=10 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR"

echo "LoRA weights saved to $OUTPUT_DIR (pytorch_lora_weights.safetensors)"
