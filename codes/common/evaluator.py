"""Test-set evaluation shared by every test script (Sec. 3.10, Figs. 4-5, Table 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from common import config
from common.data import list_images, load_label_encoder, make_dataset


def test_arg_parser(model_name: str, pretrained: bool, train_output: Path, default_output: Path) -> argparse.ArgumentParser:
    mode = "with ImageNet pre-trained weights" if pretrained else "without pre-trained weights"
    p = argparse.ArgumentParser(description=f"Test {model_name} trained {mode}.")
    p.add_argument("--test_dir", required=True,
                   help="Held-out test folder with one sub-folder per class.")
    p.add_argument("--model_path", default=str(train_output / "best_model.keras"),
                   help="Trained model (.keras). Defaults to the matching training folder output.")
    p.add_argument("--label_encoder", default=str(train_output / "label_encoder.pkl"),
                   help="Label encoder saved during training.")
    p.add_argument("--output_dir", default=str(default_output))
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return p


def _plot_confusion(cm: np.ndarray, classes: list[str], acc: float, title: str, path: Path) -> None:
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
    plt.xlabel("Predicted"), plt.ylabel("Actual")
    plt.title(f"{title}\n(Accuracy: {acc * 100:.2f}%)")
    plt.xticks(rotation=45, ha="right"), plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc(y_true: np.ndarray, probs: np.ndarray, classes: list[str], title: str, path: Path) -> dict:
    aucs = {}
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(classes):
        positives = (y_true == i).astype(int)
        if positives.sum() in (0, len(positives)):
            aucs[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(positives, probs[:, i])
        aucs[name] = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {aucs[name]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
    plt.title(title), plt.legend(loc="lower right", fontsize=8), plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return aucs


def run_test(model_name: str, pretrained: bool, args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = "With Pre-trained Weights" if pretrained else "Without Pre-trained Weights"
    title = f"{model_name}: {condition}"

    for f in (args.model_path, args.label_encoder):
        if not Path(f).exists():
            raise FileNotFoundError(f"{f} not found. Train the model first or pass the path explicitly.")

    encoder = load_label_encoder(args.label_encoder)
    classes = [str(c) for c in encoder.classes_]
    paths, labels = list_images(args.test_dir)
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"Test classes {unknown} were not seen during training {classes}.")
    y_true = encoder.transform(labels)

    model = keras.models.load_model(args.model_path, compile=False)
    ds = make_dataset(paths, y_true, model_name, len(classes), args.batch_size, training=False)
    probs = model.predict(ds, verbose=1)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    labels_idx = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels_idx, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    _plot_confusion(cm, classes, acc, title, out_dir / "confusion_matrix.png")
    aucs = _plot_roc(y_true, probs, classes, f"ROC curves (one-vs-rest)\n{title}", out_dir / "roc_curves.png")

    per_class = {
        c: {
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
            "support": int(report[c]["support"]),
            "auc": aucs[c],
        }
        for c in classes
    }
    results = {
        "model": model_name,
        "condition": "with_pretrained_weights" if pretrained else "without_pretrained_weights",
        "test_dir": str(args.test_dir),
        "n_test_images": int(len(paths)),
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "mean_auc": float(np.nanmean(list(aucs.values()))),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(per_class).T.to_csv(out_dir / "per_class_metrics.csv")

    pred_df = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    pred_df.insert(0, "predicted", [classes[i] for i in y_pred])
    pred_df.insert(0, "actual", labels)
    pred_df.insert(0, "image", paths)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n{title}")
    print(f"Accuracy : {acc * 100:.2f}%   Macro F1: {results['macro_f1']:.3f}   Mean AUC: {results['mean_auc']:.3f}")
    print(classification_report(y_true, y_pred, labels=labels_idx, target_names=classes, zero_division=0))
    print(f"Results written to {out_dir}")
    return results
