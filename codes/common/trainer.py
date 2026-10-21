"""5-fold stratified cross-validation training loop shared by every training script."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from common import config
from common.callbacks import CosineAnnealingWithPlateau
from common.data import class_weights, fit_label_encoder, list_images, make_dataset
from common.models import build_model


def training_arg_parser(model_name: str, pretrained: bool, default_output: Path) -> argparse.ArgumentParser:
    mode = "ImageNet pre-trained weights" if pretrained else "random initialisation (from scratch)"
    p = argparse.ArgumentParser(description=f"Train {model_name} with {mode} using stratified k-fold CV.")
    p.add_argument("--train_dir", required=True,
                   help="Folder with one sub-folder per class (synthetic, real or hybrid data).")
    p.add_argument("--output_dir", default=str(default_output),
                   help="Where models, logs and plots are written.")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.INITIAL_LR)
    p.add_argument("--folds", type=int, default=config.N_FOLDS)
    p.add_argument("--seed", type=int, default=config.SEED)
    if pretrained:
        p.add_argument("--fine_tune_layers", type=int, default=config.FINE_TUNE_LAST_N_LAYERS,
                       help="Number of final backbone layers to unfreeze.")
    return p


def _plot_curves(histories: list[dict], out_dir: Path, model_name: str) -> None:
    n = min(len(h["accuracy"]) for h in histories)
    mean = lambda k: np.mean([h[k][:n] for h in histories], axis=0)
    epochs = np.arange(1, n + 1)
    for metric, ylabel, fname in (("accuracy", "Accuracy", "accuracy_curve.png"),
                                  ("loss", "Categorical cross-entropy", "loss_curve.png")):
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, mean(metric), label=f"Training {metric}", linewidth=2)
        plt.plot(epochs, mean(f"val_{metric}"), label=f"Validation {metric}", linewidth=2, linestyle="--")
        plt.xlabel("Epoch"), plt.ylabel(ylabel)
        plt.title(f"{model_name}: training vs validation {metric} (mean over folds)")
        plt.grid(alpha=0.3), plt.legend(), plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def _plot_folds(fold_acc: list[float], out_dir: Path, model_name: str) -> None:
    plt.figure(figsize=(7, 4.5))
    xs = [f"Fold {i}" for i in range(1, len(fold_acc) + 1)]
    bars = plt.bar(xs, fold_acc, color="#3b6ea5")
    plt.axhline(np.mean(fold_acc), color="#c0392b", linestyle="--", label=f"Mean = {np.mean(fold_acc):.4f}")
    for b, v in zip(bars, fold_acc):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.05), plt.ylabel("Validation accuracy")
    plt.title(f"{model_name}: {len(fold_acc)}-fold stratified cross-validation")
    plt.legend(), plt.tight_layout()
    plt.savefig(out_dir / "kfold_accuracy.png", dpi=150)
    plt.close()


def run_training(model_name: str, pretrained: bool, args: argparse.Namespace) -> dict:
    keras.utils.set_random_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = list_images(args.train_dir)
    encoder = fit_label_encoder(labels, out_dir / "label_encoder.pkl")
    y = encoder.transform(labels)
    num_classes = len(encoder.classes_)
    print(f"[{model_name}] {len(paths)} images, classes: {[str(c) for c in encoder.classes_]}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    histories, fold_acc = [], []

    for fold, (tr, va) in enumerate(skf.split(paths, y), start=1):
        print(f"\n========== {model_name} | fold {fold}/{args.folds} ==========")
        keras.backend.clear_session()
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        train_ds = make_dataset(paths[tr], y[tr], model_name, num_classes, args.batch_size, training=True, seed=args.seed)
        val_ds = make_dataset(paths[va], y[va], model_name, num_classes, args.batch_size, training=False)

        kwargs = {"fine_tune_last_n": args.fine_tune_layers} if pretrained else {}
        model = build_model(model_name, pretrained, num_classes, **kwargs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        best_path = fold_dir / "best_model.keras"
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights(y[tr], num_classes),
            callbacks=[
                CosineAnnealingWithPlateau(args.lr, args.epochs),
                keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True),
                keras.callbacks.CSVLogger(fold_dir / "history.csv"),
            ],
            verbose=2,
        )
        histories.append(history.history)

        model = keras.models.load_model(best_path, compile=False)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        _, acc = model.evaluate(val_ds, verbose=0)
        fold_acc.append(float(acc))
        print(f"[{model_name}] fold {fold} best validation accuracy: {acc:.4f}")

    best_fold = int(np.argmax(fold_acc)) + 1
    shutil.copy(out_dir / f"fold_{best_fold}" / "best_model.keras", out_dir / "best_model.keras")

    results = {
        "model": model_name,
        "condition": "with_pretrained_weights" if pretrained else "without_pretrained_weights",
        "train_dir": str(args.train_dir),
        "classes": [str(c) for c in encoder.classes_],
        "fold_accuracies": fold_acc,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "best_fold": best_fold,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("train_dir", "output_dir")},
    }
    (out_dir / "fold_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"fold": range(1, len(fold_acc) + 1), "val_accuracy": fold_acc}).to_csv(
        out_dir / "fold_results.csv", index=False)
    _plot_curves(histories, out_dir, model_name)
    _plot_folds(fold_acc, out_dir, model_name)

    print(f"\n[{model_name}] mean CV accuracy {results['mean_accuracy']:.4f} "
          f"± {results['std_accuracy']:.4f}; best fold {best_fold} saved to {out_dir / 'best_model.keras'}")
    return results
