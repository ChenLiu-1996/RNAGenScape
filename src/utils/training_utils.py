"""Shared training helpers (seeding, early stopping, LR schedules)."""

from __future__ import annotations

import math
import os
import random
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for deterministic CUDA GEMM on newer drivers.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class EarlyStopping:
    """Stop when a monitored metric stops improving."""

    def __init__(self, *, mode: str = "max", patience: int = 20, min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"Unknown early-stop mode: {mode}")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, metric: float) -> bool:
        if metric != metric:  # NaN
            return True
        if self.best is None:
            self.best = metric
            return False
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta
        if improved:
            self.best = metric
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad >= self.patience


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup from ``warmup_start_lr`` to base lr, then cosine anneal to ``eta_min``.

    Call ``step()`` once per epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return list(self.base_lrs)
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        cosine_epochs = max(1, self.max_epochs - self.warmup_epochs)
        progress = min(max(self.last_epoch - self.warmup_epochs, 0), cosine_epochs)
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress / cosine_epochs))
            for base_lr in self.base_lrs
        ]
