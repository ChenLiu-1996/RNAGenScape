"""Evaluate property-gradient stability by manifold region.

Example:
  python src/evaluation/eval_gradient.py \\
    --dataset Ribosome_loading \\
    --model OAE
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.oracle import DATASET_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate gradient stability by manifold region.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "eval_gradient is scaffolded; implement after eval_optimization. "
        f"dataset={args.dataset} model={args.model}"
    )


if __name__ == "__main__":
    main()
