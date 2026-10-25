"""Cosine-annealed learning rate (Eq. 28) with ReduceLROnPlateau-style drops.

    eta(t) = scale * eta_0 * 0.5 * (1 + cos(pi * t / T))

`scale` starts at 1 and is multiplied by `factor` (0.2) whenever the monitored
validation loss fails to improve for `patience` (3) epochs. This reproduces the
inflection points visible in the paper's loss curves (Fig. 2) without the two
schedulers overwriting each other.
"""

from __future__ import annotations

import math

import keras

from common import config


class CosineAnnealingWithPlateau(keras.callbacks.Callback):
    def __init__(
        self,
        initial_lr: float = config.INITIAL_LR,
        total_epochs: int = config.EPOCHS,
        factor: float = config.PLATEAU_FACTOR,
        patience: int = config.PLATEAU_PATIENCE,
        monitor: str = "val_loss",
        min_lr: float = 1e-7,
        verbose: int = 1,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = max(total_epochs, 1)
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.min_lr = min_lr
        self.verbose = verbose
        self.scale = 1.0
        self.best = math.inf
        self.wait = 0

    def _lr_for(self, epoch: int) -> float:
        cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / self.total_epochs))
        return max(self.initial_lr * self.scale * cosine, self.min_lr)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self._lr_for(epoch)
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs["lr"] = float(self.model.optimizer.learning_rate.numpy())
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best = current
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.scale *= self.factor
            self.wait = 0
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: plateau on {self.monitor}; LR scale -> {self.scale:.4g}")
