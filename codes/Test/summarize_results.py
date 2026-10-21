"""Collect every test_results.json into one table (the format of Table 2).

Usage (from codes/):
    python Test/summarize_results.py
"""

import json
from pathlib import Path

import pandas as pd

TEST_ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "test_with_weights": "Synthetic only / With pre-train",
    "test_without_weights": "Synthetic only / No pre-train",
}
MODELS = ["ResNet50", "InceptionV3", "Xception"]


def main():
    rows = []
    for folder, label in CONDITIONS.items():
        for m in MODELS:
            f = TEST_ROOT / folder / m / "results" / "test_results.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            row = {"Condition": label, "Model": m, "Accuracy (%)": round(100 * r["accuracy"], 2)}
            for c, v in r["per_class"].items():
                row[f"{c} F1"] = round(v["f1"], 2)
            row["Macro F1"] = round(r["macro_f1"], 3)
            row["Mean AUC"] = round(r["mean_auc"], 3)
            rows.append(row)

    if not rows:
        print("No test results found yet. Run the test scripts first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(TEST_ROOT / "results_summary.csv", index=False)
    (TEST_ROOT / "results_summary.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))
    print(f"\nSaved results_summary.csv and results_summary.md in {TEST_ROOT}")


if __name__ == "__main__":
    main()
