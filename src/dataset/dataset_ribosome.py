"""Ribosome loading (MRL) dataset - Sample library, 4.10 split."""

from __future__ import annotations

import os
import sys
from typing import Tuple

# Allow `python src/dataset/dataset_ribosome.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torch.utils.data import DataLoader

from dataset.data_io import (
    DEFAULT_SEED,
    DataInfo,
    data_path,
    make_loaders_from_splits,
    random_split_indices,
    read_sequences_and_labels,
    run_loader_check,
    take,
)

NAME = "RibosomeLoading"
SEQ_KEY = "utr"
LABEL_KEY = "rl"
SEQ_LEN = 50
# Fraction of the provider train file held out as validation.
VAL_RATIO = 0.15
_DIR = "RibosomeLoading/MRL_Random50Nuc_SynthesisLibrary_Sample"
RELATIVE_TRAIN_CSV = f"{_DIR}/4.10_train_data_GSM3130438_egfp_pseudo_2.csv"
RELATIVE_TEST_CSV = f"{_DIR}/4.10_test_data_GSM3130438_egfp_pseudo_2.csv"

# Kept for callers that only need the train file path (e.g. novelty reference).
DATASET_CONFIG = {
    "relative_csv": RELATIVE_TRAIN_CSV,
    "relative_train_csv": RELATIVE_TRAIN_CSV,
    "relative_test_csv": RELATIVE_TEST_CSV,
    "seq_key": SEQ_KEY,
    "label_key": LABEL_KEY,
    "seq_len": SEQ_LEN,
}


def make_dataloaders(
    *,
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
    representation: str = "string",
    label_norm: str = "normal",
    val_ratio: float = VAL_RATIO,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataInfo]:
    """Predefined train/test files; val is carved from train (val_ratio)."""
    train_path = data_path(RELATIVE_TRAIN_CSV)
    test_path = data_path(RELATIVE_TEST_CSV)
    train_seqs_all, train_y_all = read_sequences_and_labels(train_path, SEQ_KEY, LABEL_KEY)
    test_seqs, test_y = read_sequences_and_labels(test_path, SEQ_KEY, LABEL_KEY)

    train_idx, val_idx, _ = random_split_indices(
        len(train_seqs_all), test_ratio=0.0, val_ratio=val_ratio, seed=seed
    )
    train_seqs, train_y = take(train_seqs_all, train_y_all, train_idx)
    val_seqs, val_y = take(train_seqs_all, train_y_all, val_idx)

    return make_loaders_from_splits(
        train_seqs,
        train_y,
        val_seqs,
        val_y,
        test_seqs,
        test_y,
        dataset=NAME,
        seq_len=SEQ_LEN,
        seq_key=SEQ_KEY,
        label_key=LABEL_KEY,
        paths={"train_csv": train_path, "test_csv": test_path},
        representation=representation,
        label_norm=label_norm,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    run_loader_check(make_dataloaders, NAME)
