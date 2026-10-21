#!/usr/bin/env bash
# Test ResNet50, InceptionV3 and Xception with ImageNet pre-trained weights.
# Usage: bash test_all.sh /path/to/test_data
set -euo pipefail
TEST_DIR="${1:?Usage: bash test_all.sh /path/to/test_data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/ResNet50/test_resnet50.py" --test_dir "$TEST_DIR"
python "$HERE/InceptionV3/test_inceptionv3.py" --test_dir "$TEST_DIR"
python "$HERE/Xception/test_xception.py" --test_dir "$TEST_DIR"
