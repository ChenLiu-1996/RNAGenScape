"""Dataset registry and unified make_dataloaders entry point."""

from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader

from dataset import dataset_openvaccine, dataset_ribosome, dataset_zebrafish
from dataset.data_io import DEFAULT_SEED, DataInfo

_MODULES = {
    dataset_openvaccine.NAME: dataset_openvaccine,
    dataset_zebrafish.NAME: dataset_zebrafish,
    dataset_ribosome.NAME: dataset_ribosome,
}

DATASET_NAMES = tuple(_MODULES.keys())

# Flat config for callers that only need paths / columns / seq_len.
DATASET_CONFIG: Dict[str, dict] = {
    name: dict(mod.DATASET_CONFIG) for name, mod in _MODULES.items()
}


def make_dataloaders(
    dataset: str,
    *,
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
    representation: str = "string",
    label_norm: str = "normal",
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataInfo]:
    """Build train/val/test loaders for a registered dataset.

    Args:
        dataset: One of ``OpenVaccine``, ``Zebrafish``, ``Ribosome_loading``.
        representation: ``string`` | ``token_ids`` | ``one_hot``.
        label_norm: ``none`` | ``normal`` | ``minmax``.
        kwargs: Forwarded to the dataset module (e.g. test_ratio, val_ratio).
    """
    if dataset not in _MODULES:
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_NAMES)}")
    return _MODULES[dataset].make_dataloaders(
        batch_size=batch_size,
        seed=seed,
        representation=representation,
        label_norm=label_norm,
        num_workers=num_workers,
        **kwargs,
    )
