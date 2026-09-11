"""CUDA-synchronised wall-clock timing for generation throughput."""

from __future__ import annotations

import time
from typing import Optional

import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaTimer:
    """Context manager: wall time with CUDA sync before/after.

    Usage::

        with CudaTimer() as t:
            ...
        elapsed_s = t.elapsed
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: Optional[float] = None

    def __enter__(self) -> "CudaTimer":
        cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        cuda_sync()
        self.elapsed = time.perf_counter() - float(self._t0 or 0.0)
        return False


def ms_per_sample(elapsed_s: float, n: int) -> float:
    return float(elapsed_s) * 1000.0 / max(1, int(n))
