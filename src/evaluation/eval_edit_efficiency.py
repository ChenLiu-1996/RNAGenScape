"""Evaluate property gain versus edit distance across experiment seeds.

Example:
  python src/evaluation/eval_edit_efficiency.py \\
    --dataset OpenVaccine \\
    --model OAE \\
    --experiment pos_samehyper_sugar1e0 \\
    --oracle UTRLM
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG
from utils.oracle import SUPPORTED_ORACLES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate edit efficiency across seeds.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--oracle", type=str, required=True, choices=sorted(SUPPORTED_ORACLES))
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "eval_edit_efficiency is scaffolded; implement next after eval_optimization. "
        f"dataset={args.dataset} model={args.model} experiment={args.experiment} oracle={args.oracle}"
    )


if __name__ == "__main__":
    main()
