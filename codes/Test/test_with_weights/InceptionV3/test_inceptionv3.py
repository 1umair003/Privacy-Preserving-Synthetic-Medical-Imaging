"""Test InceptionV3 trained with ImageNet pre-trained weights.

Loads the model produced by codes/code_with_weights/InceptionV3/train_inceptionv3.py and reports
accuracy, per-class precision / recall / F1, macro F1, confusion matrix and
one-vs-rest ROC curves with AUC.

Usage (from this folder):
    python test_inceptionv3.py --test_dir /path/to/real_test
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parents[2]
sys.path.insert(0, str(CODES))

from common.evaluator import run_test, test_arg_parser  # noqa: E402

MODEL_NAME = "InceptionV3"
PRETRAINED = True
TRAIN_OUTPUT = CODES / "code_with_weights" / MODEL_NAME / "outputs"

if __name__ == "__main__":
    args = test_arg_parser(MODEL_NAME, PRETRAINED, TRAIN_OUTPUT, HERE / "results").parse_args()
    run_test(MODEL_NAME, PRETRAINED, args)
