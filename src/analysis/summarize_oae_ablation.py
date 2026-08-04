#!/usr/bin/env python3
"""Summarize OAE train ablations from saved ``metrics.json`` files (mean +/- std).

Example:
  python src/analysis/summarize_oae_ablation.py
  python src/analysis/summarize_oae_ablation.py --latent_dims 128 --recon_ws 1 5 10
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

# Allow `python src/analysis/summarize_oae_ablation.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analysis.metrics_io import read_metrics
from analysis.stats import mean_pm_std
from dataset import DATASET_NAMES
from utils.results import oae_seed_dir


DEFAULT_DATASETS = ("OpenVaccine", "Zebrafish", "RibosomeLoading")
DEFAULT_LATENT_DIMS = (64, 128)
DEFAULT_RECON_WS = (0.1, 0.5, 1.0, 5.0, 10.0)
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_METRICS = ("token_acc", "pearson", "spearman")


def _split_metrics(payload: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    block = payload.get(split)
    if isinstance(block, dict):
        return block
    return None


def collect_metric(
    *,
    dataset: str,
    latent_dim: int,
    recon_w: float,
    seeds: Sequence[int],
    metric: str,
    split: str,
    root: Optional[str],
) -> List[float]:
    values: List[float] = []
    for seed in seeds:
        run_dir = oae_seed_dir(
            dataset, int(seed), latent_dim=int(latent_dim), recon_w=float(recon_w), root=root
        )
        payload = read_metrics(run_dir)
        if payload is None:
            continue
        block = _split_metrics(payload, split)
        if block is None or metric not in block:
            continue
        try:
            values.append(float(block[metric]))
        except (TypeError, ValueError):
            continue
    return values


def format_table(
    *,
    datasets: Sequence[str],
    latent_dims: Sequence[int],
    recon_ws: Sequence[float],
    seeds: Sequence[int],
    metrics: Sequence[str],
    split: str,
    root: Optional[str],
    digits: int,
) -> str:
    """Markdown table: one row per (dataset, D, recon_w); cells are mean +/- std."""
    header_metrics = " | ".join(metrics)
    lines = [
        f"| dataset | D | recon_w | n | {header_metrics} |",
        "|---|---:|---:|---:|" + "|".join(["---:" for _ in metrics]) + "|",
    ]
    for dataset in datasets:
        for latent_dim in latent_dims:
            for recon_w in recon_ws:
                cols: List[str] = []
                n_used = 0
                for metric in metrics:
                    vals = collect_metric(
                        dataset=dataset,
                        latent_dim=int(latent_dim),
                        recon_w=float(recon_w),
                        seeds=seeds,
                        metric=metric,
                        split=split,
                        root=root,
                    )
                    n_used = max(n_used, len(vals))
                    cols.append(mean_pm_std(vals, digits=digits))
                lines.append(
                    f"| {dataset} | {latent_dim} | {recon_w:g} | {n_used} | "
                    + " | ".join(cols)
                    + " |"
                )
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Summarize OAE ablation metrics.json as mean +/- std.")
    p.add_argument("--datasets", type=str, nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(DATASET_NAMES))
    p.add_argument("--latent_dims", type=int, nargs="+", default=list(DEFAULT_LATENT_DIMS))
    p.add_argument("--recon_ws", type=float, nargs="+", default=list(DEFAULT_RECON_WS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--metrics", type=str, nargs="+", default=list(DEFAULT_METRICS), help="Keys under the split block in metrics.json.")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Which split block to aggregate.")
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--root", type=str, default=None, help="Repo root override (default: inferred).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = format_table(
        datasets=args.datasets,
        latent_dims=args.latent_dims,
        recon_ws=args.recon_ws,
        seeds=args.seeds,
        metrics=args.metrics,
        split=args.split,
        root=args.root,
        digits=args.digits,
    )
    print(f"# OAE ablation ({args.split} metrics, mean +/- std over seeds {list(args.seeds)})")
    print()
    print(table)


if __name__ == "__main__":
    main()
