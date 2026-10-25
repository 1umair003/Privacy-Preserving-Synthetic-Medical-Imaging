"""Train ResNet50 without pre-trained weights (trained from scratch).

Random initialisation; every layer is trained from scratch.
Runs 5-fold stratified cross-validation, cosine-annealed learning rate
(eta_0 = 1e-4, 30 epochs) with plateau drops, and class-weighted cross-entropy.

Usage (from this folder):
    python train_resnet50.py --train_dir /path/to/synthetic_train
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.trainer import run_training, training_arg_parser  # noqa: E402

MODEL_NAME = "ResNet50"
PRETRAINED = False

if __name__ == "__main__":
    args = training_arg_parser(MODEL_NAME, PRETRAINED, HERE / "outputs").parse_args()
    run_training(MODEL_NAME, PRETRAINED, args)
