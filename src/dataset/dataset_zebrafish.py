"""Zebrafish 5'UTR MPRA translation dataset."""

from __future__ import annotations

import os
import sys
from typing import Tuple

# Allow `python src/dataset/dataset_zebrafish.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torch.utils.data import DataLoader

from dataset.data_io import (
    DEFAULT_SEED,
    SPLIT_8_1_1_TEST_RATIO,
    SPLIT_8_1_1_VAL_RATIO,
    DataInfo,
    data_path,
    make_loaders_from_splits,
    random_split_indices,
    read_sequences_and_labels,
    run_loader_check,
    take,
)

NAME = "Zebrafish"
SEQ_KEY = "sequence"
LABEL_KEY = "translation"
SEQ_LEN = 124
RELATIVE_CSV = "Zebrafish/MPRA_mean_translation_2hpf_pa_Fish5UTR.csv"

DATASET_CONFIG = {
    "relative_csv": RELATIVE_CSV,
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
    test_ratio: float = SPLIT_8_1_1_TEST_RATIO,
    val_ratio: float = SPLIT_8_1_1_VAL_RATIO,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataInfo]:
    csv_path = data_path(RELATIVE_CSV)
    seqs, labels = read_sequences_and_labels(csv_path, SEQ_KEY, LABEL_KEY)
    train_idx, val_idx, test_idx = random_split_indices(
        len(seqs), test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )
    train_seqs, train_y = take(seqs, labels, train_idx)
    val_seqs, val_y = take(seqs, labels, val_idx)
    test_seqs, test_y = take(seqs, labels, test_idx)
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
        paths={"csv": csv_path},
        representation=representation,
        label_norm=label_norm,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    run_loader_check(make_dataloaders, NAME)
