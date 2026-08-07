"""Shared start-sequence cache for comparable generation across methods."""

from __future__ import annotations

import hashlib
import os
from typing import Tuple, Union

import numpy as np
import torch


def starts_content_hash(token_ids) -> str:
    """SHA1 of CPU numpy bytes for a token-id array (order-sensitive)."""
    if isinstance(token_ids, torch.Tensor):
        arr = token_ids.detach().cpu().numpy()
    else:
        arr = np.asarray(token_ids)
    arr = np.ascontiguousarray(arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def save_starts_cache(
    path: str,
    sampled_x: Union[torch.Tensor, np.ndarray],
    sampled_y: Union[torch.Tensor, np.ndarray],
    sampled_indices: Union[torch.Tensor, np.ndarray],
) -> str:
    """Save start sequences / labels / indices to ``path`` (.pt). Returns path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(sampled_x, np.ndarray):
        sampled_x = torch.from_numpy(sampled_x)
    if isinstance(sampled_y, np.ndarray):
        sampled_y = torch.from_numpy(sampled_y.astype(np.float32))
    if isinstance(sampled_indices, np.ndarray):
        sampled_indices = torch.from_numpy(sampled_indices.astype(np.int64))
    payload = {
        "sampled_x": sampled_x.detach().cpu().long(),
        "sampled_y": sampled_y.detach().cpu().float().reshape(-1),
        "sampled_indices": sampled_indices.detach().cpu().long().reshape(-1),
        "starts_hash": starts_content_hash(sampled_x),
    }
    torch.save(payload, path)
    return path


def load_starts_cache(
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load ``(sampled_x, sampled_y, sampled_indices)`` from a starts cache."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Starts cache not found: {path}")
    payload = torch.load(path, map_location="cpu")
    sampled_x = payload["sampled_x"].long()
    sampled_y = payload["sampled_y"].float().reshape(-1)
    sampled_indices = np.asarray(payload["sampled_indices"], dtype=np.int64).reshape(-1)
    return sampled_x, sampled_y, sampled_indices
