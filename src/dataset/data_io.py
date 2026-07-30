"""Shared CSV loading, splits, label transforms, and DataLoader helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.metrics import TOKENS, to_token_ids
from utils.results import repo_root


DEFAULT_SEED = 1

# 8:1:1 train/val/test - val_ratio is of the post-test remainder (0.1 / 0.9).
SPLIT_8_1_1_TEST_RATIO = 0.1
SPLIT_8_1_1_VAL_RATIO = 0.1 / 0.9


def data_path(*parts: str) -> str:
    return os.path.join(repo_root(), "data", *parts)


def read_sequences_and_labels(
    csv_path: str,
    seq_key: str,
    label_key: str,
) -> Tuple[List[str], np.ndarray]:
    """Load sequence strings and float labels from a CSV."""
    df = pd.read_csv(csv_path)
    if seq_key not in df.columns:
        raise KeyError(f"Missing seq column '{seq_key}' in {csv_path}. Columns: {list(df.columns)}")
    if label_key not in df.columns:
        raise KeyError(f"Missing label column '{label_key}' in {csv_path}. Columns: {list(df.columns)}")
    seqs = df[seq_key].astype(str).tolist()
    labels = df[label_key].to_numpy(dtype=np.float64)
    return seqs, labels


@dataclass
class LabelTransform:
    """Fit on train labels only; apply the same transform to val/test."""

    mode: str = "none"  # "none" | "normal" | "minmax"
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0

    def fit(self, y: np.ndarray) -> "LabelTransform":
        y = np.asarray(y, dtype=np.float64)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if self.std == 0.0:
            self.std = 1.0
        self.min = float(np.min(y))
        self.max = float(np.max(y))
        if self.max == self.min:
            self.max = self.min + 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.mode in ("none", None, "None"):
            return y
        if self.mode == "normal":
            return (y - self.mean) / self.std
        if self.mode == "minmax":
            return -1.0 + 2.0 * (y - self.min) / (self.max - self.min)
        raise ValueError(f"Unknown label_norm mode: {self.mode}")

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.mode in ("none", None, "None"):
            return y
        if self.mode == "normal":
            return y * self.std + self.mean
        if self.mode == "minmax":
            return (y + 1.0) * 0.5 * (self.max - self.min) + self.min
        raise ValueError(f"Unknown label_norm mode: {self.mode}")

    def as_dict(self) -> Dict[str, float]:
        return {
            "label_mean": self.mean,
            "label_std": self.std,
            "label_min": self.min,
            "label_max": self.max,
        }


def random_split_indices(
    n: int,
    *,
    test_ratio: float = SPLIT_8_1_1_TEST_RATIO,
    val_ratio: float = SPLIT_8_1_1_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays. val_ratio is of the post-test remainder."""
    idx = np.arange(n)
    if test_ratio > 0:
        train_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=seed)
    else:
        train_idx, test_idx = idx, np.array([], dtype=np.int64)
    if val_ratio > 0 and len(train_idx) > 0:
        train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=seed)
    else:
        val_idx = np.array([], dtype=np.int64)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def take(
    seqs: Sequence[str],
    labels: np.ndarray,
    indices: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    return [seqs[i] for i in indices], labels[indices]


def sequences_to_representation(
    seqs: Sequence[str],
    representation: str,
) -> Union[List[str], torch.Tensor]:
    """Convert string sequences to the requested batch representation."""
    if representation == "string":
        return list(seqs)
    ids = to_token_ids(list(seqs))
    if representation == "token_ids":
        return ids
    if representation == "one_hot":
        eye = torch.eye(len(TOKENS), dtype=torch.float32)
        return eye[ids]
    raise ValueError(
        f"Unknown representation '{representation}'. "
        "Choose from: string, token_ids, one_hot"
    )


class StringLabelDataset(Dataset):
    """Dataset yielding (sequence_string, label_float)."""

    def __init__(self, sequences: Sequence[str], labels: np.ndarray) -> None:
        self.sequences = list(sequences)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def build_loader(
    seqs: Sequence[str],
    labels: np.ndarray,
    *,
    representation: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if len(seqs) == 0:
        empty = StringLabelDataset([], np.zeros((0,), dtype=np.float64))
        return DataLoader(empty, batch_size=batch_size, shuffle=False)

    if representation == "string":
        ds: Dataset = StringLabelDataset(seqs, labels)
    else:
        x = sequences_to_representation(seqs, representation)
        assert isinstance(x, torch.Tensor)
        y = torch.as_tensor(labels, dtype=torch.float32)
        ds = TensorDataset(x, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


@dataclass
class DataInfo:
    """Metadata returned alongside train/val/test loaders."""

    dataset: str
    seq_len: int
    seq_key: str
    label_key: str
    representation: str
    label_transform: LabelTransform
    n_train: int
    n_val: int
    n_test: int
    paths: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "dataset": self.dataset,
            "seq_len": self.seq_len,
            "seq_key": self.seq_key,
            "label_key": self.label_key,
            "representation": self.representation,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "paths": dict(self.paths),
        }
        out.update(self.label_transform.as_dict())
        return out


def make_loaders_from_splits(
    train_seqs: Sequence[str],
    train_y: np.ndarray,
    val_seqs: Sequence[str],
    val_y: np.ndarray,
    test_seqs: Sequence[str],
    test_y: np.ndarray,
    *,
    dataset: str,
    seq_len: int,
    seq_key: str,
    label_key: str,
    paths: Dict[str, str],
    representation: str = "string",
    label_norm: str = "normal",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataInfo]:
    """Fit label transform on train, transform all splits, build loaders."""
    transform = LabelTransform(mode=label_norm).fit(train_y)
    train_y_t = transform.transform(train_y)
    val_y_t = transform.transform(val_y) if len(val_seqs) else val_y
    test_y_t = transform.transform(test_y) if len(test_seqs) else test_y

    train_loader = build_loader(
        train_seqs,
        train_y_t,
        representation=representation,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_loader(
        val_seqs,
        val_y_t,
        representation=representation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = build_loader(
        test_seqs,
        test_y_t,
        representation=representation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    info = DataInfo(
        dataset=dataset,
        seq_len=seq_len,
        seq_key=seq_key,
        label_key=label_key,
        representation=representation,
        label_transform=transform,
        n_train=len(train_seqs),
        n_val=len(val_seqs),
        n_test=len(test_seqs),
        paths=paths,
    )
    return train_loader, val_loader, test_loader, info


def format_batch_x(x) -> str:
    """Return X batch shape as a tuple string, e.g. ``(8, 107)``."""
    if isinstance(x, (list, tuple)):
        lengths = [len(s) for s in x]
        if lengths and min(lengths) == max(lengths):
            return f"({len(x)}, {lengths[0]})"
        return f"(n={len(x)}, seq_len=[{min(lengths)}..{max(lengths)}])"
    return str(tuple(x.shape))


def run_loader_check(make_fn, name: str, *, batch_size: int = 8) -> None:
    """Load train/val/test and print split sizes plus one batch shape."""
    train, val, test, info = make_fn(batch_size=batch_size, representation="token_ids")
    print(f"=== {name} ===")
    print(f"n_train={info.n_train}, n_val={info.n_val}, n_test={info.n_test}")
    for split_name, loader in (("train", train), ("val", val), ("test", test)):
        x, y = next(iter(loader))
        print(f"  {split_name}: X shape={format_batch_x(x)}, Y shape={tuple(y.shape)}")
    print(f"{name} unit test finished")
