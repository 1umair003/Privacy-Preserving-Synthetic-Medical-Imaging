"""Paired t-test and Wilcoxon signed-rank test on fold-wise accuracies (Sec. 4.9).

Reads `fold_results.json` written by the training scripts for each model and
compares every pair of architectures within a training condition.

Usage (from codes/):
    python Test/statistical_tests.py --condition with_weights
    python Test/statistical_tests.py --condition without_weights
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODES = Path(__file__).resolve().parents[1]
MODELS = ["ResNet50", "InceptionV3", "Xception"]
FOLDERS = {"with_weights": "code_with_weights", "without_weights": "code_without_weights"}


def load_fold_accuracies(condition: str) -> dict:
    accs = {}
    for m in MODELS:
        f = CODES / FOLDERS[condition] / m / "outputs" / "fold_results.json"
        if f.exists():
            accs[m] = np.array(json.loads(f.read_text())["fold_accuracies"])
        else:
            print(f"  skipping {m}: {f} not found (train it first)")
    return accs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=list(FOLDERS), required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "statistical_results"))
    args = p.parse_args()

    accs = load_fold_accuracies(args.condition)
    if len(accs) < 2:
        sys.exit("Need fold results for at least two models.")

    rows = []
    for a, b in itertools.combinations(accs, 2):
        x, y = accs[a], accs[b]
        if len(x) != len(y):
            print(f"  skipping {a} vs {b}: different number of folds")
            continue
        t_stat, t_p = stats.ttest_rel(x, y)
        try:
            w_stat, w_p = stats.wilcoxon(x, y)
        except ValueError:  # all differences zero
            w_stat, w_p = float("nan"), 1.0
        rows.append({
            "comparison": f"{a} vs {b}",
            "mean_acc_first": x.mean(), "mean_acc_second": y.mean(),
            "t_statistic": t_stat, "t_test_p": t_p,
            "wilcoxon_statistic": w_stat, "wilcoxon_p": w_p,
            "significant_t_test": bool(t_p < args.alpha),
            "significant_wilcoxon": bool(w_p < args.alpha),
        })

    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"statistical_tests_{args.condition}.csv", index=False)

    print(f"\nFold-wise accuracies ({args.condition}):")
    for m, v in accs.items():
        print(f"  {m:<12} {np.round(v, 4)}  mean={v.mean():.4f}")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {out / f'statistical_tests_{args.condition}.csv'}")
    if any(len(v) < 6 for v in accs.values()):
        print("Note: with 5 folds the smallest possible two-sided Wilcoxon p-value is 0.0625.")


if __name__ == "__main__":
    main()
