"""Small helpers for aggregating numeric metrics across runs/seeds."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def as_float_list(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        x = float(v)
        if np.isfinite(x):
            out.append(x)
    return out


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], int]:
    """Return (mean, std, n). ``std`` is sample std (ddof=1) when n>1, else 0.0."""
    xs = as_float_list(values)
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
    return mean, std, n


def mean_pm_std(values: Sequence[float], *, digits: int = 3) -> str:
    """Format as ``mean \u00b1 std`` (or ``MISSING`` / single-value without pm)."""
    mean, std, n = mean_std(values)
    if n == 0:
        return "MISSING"
    if n == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"
