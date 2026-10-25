#!/usr/bin/env bash
# Train ResNet50, InceptionV3 and Xception with ImageNet pre-trained weights.
# Usage: bash train_all.sh /path/to/train_data
set -euo pipefail
TRAIN_DIR="${1:?Usage: bash train_all.sh /path/to/train_data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/ResNet50/train_resnet50.py" --train_dir "$TRAIN_DIR"
python "$HERE/InceptionV3/train_inceptionv3.py" --train_dir "$TRAIN_DIR"
python "$HERE/Xception/train_xception.py" --train_dir "$TRAIN_DIR"
