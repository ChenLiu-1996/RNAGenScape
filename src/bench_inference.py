"""Light inference-throughput bench (CUDA-synced ms/sample).

Times only the generation kernel (baseline ``optimize`` / RNAGenScape Langevin+decode),
not dataloader / checkpoint load. Uses a small start pool so this is not a full experiment.

Example:
  python src/bench_inference.py --datasets OpenVaccine --n_samples 64 --seed 1
  python src/bench_inference.py  # all datasets × all methods, n=64

Writes:
  results/_bench_inference/<timestamp>/summary.json
  results/_bench_inference/<timestamp>/summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_NAMES, make_dataloaders
from modules.langevin import run_manifold_langevin
from run_generation import (
    apply_latent_normalization,
    collect_tokens_and_labels,
    decode_latents,
    encode_loader,
    load_baseline_model,
    load_label_stats,
    load_oae_model,
    load_projector,
    optimize_baseline_batch,
    resolve_start_indices,
)
from train_baseline import BASELINE_MODELS
from utils.oracle import resolve_device
from utils.results import results_root
from utils.timing import CudaTimer, ms_per_sample
from utils.training_utils import seed_everything


# Display order for the throughput figure (main paper method list).
BENCH_METHODS = (
    "DiffAb",
    "IgLM",
    "NOS_C",
    "NOS_D",
    "PCD",
    "EM",
    "MFM",
    "gg_dWJS",
    "MPGD",
    "RNAGenScape",
)

# Locked RNAGenScape generation settings (match main results).
RGS_LATENT_DIM = 128
RGS_RECON_W = 5.0
RGS_SUGAR_W = 1.0
RGS_PROJECTOR = "dae"
RGS_TEMPERATURE = 1e-3
RGS_NUM_STEPS = 100


def step_size_for(dataset: str) -> float:
    return 1e-2 if dataset == "OpenVaccine" else 1e-3


class _Args:
    """Minimal namespace for optimize_baseline_batch."""

    def __init__(self, direction: float):
        self.direction = float(direction)


def _warmup_baseline(model_name: str, model, starts: torch.Tensor, y: torch.Tensor, *, device: str, direction: float) -> None:
    n = min(8, int(starts.shape[0]))
    optimize_baseline_batch(
        model_name,
        model,
        starts[:n],
        args=_Args(direction),
        device=device,
        seed_labels=y[:n],
    )


def bench_baseline(
    *,
    model_name: str,
    dataset: str,
    seed: int,
    device: str,
    n_samples: int,
    batch_size: int,
    direction: float,
    subsample_seed: int,
    warmup: bool,
) -> Dict[str, Any]:
    model = load_baseline_model(model_name, dataset=dataset, seed=seed, device=device)
    _train, _val, test_loader, _info = make_dataloaders(
        dataset,
        batch_size=batch_size,
        seed=seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=0,
    )
    pool_x, pool_y = collect_tokens_and_labels(test_loader, device=device)
    indices, _ = resolve_start_indices(
        pool_x.shape[0],
        max_starts=n_samples,
        subsample_seed=subsample_seed,
        starts_cache=None,
    )
    idx = torch.from_numpy(indices.astype(np.int64))
    starts = pool_x[idx]
    labels = pool_y[idx]

    if warmup:
        _warmup_baseline(model_name, model, starts, labels, device=device, direction=direction)

    seed_everything(seed)
    outs: List[torch.Tensor] = []
    with CudaTimer() as timer:
        for i in range(0, starts.shape[0], batch_size):
            outs.append(
                optimize_baseline_batch(
                    model_name,
                    model,
                    starts[i : i + batch_size],
                    args=_Args(direction),
                    device=device,
                    seed_labels=labels[i : i + batch_size],
                )
            )
        new_sequences = torch.cat(outs, dim=0)
    n = int(new_sequences.shape[0])
    return {
        "method": model_name,
        "dataset": dataset,
        "seed": seed,
        "direction": direction,
        "n": n,
        "batch_size": batch_size,
        "elapsed_seconds": float(timer.elapsed),
        "ms_per_sample": ms_per_sample(timer.elapsed, n),
        "ok": True,
        "error": None,
    }


def bench_rnagenscape(
    *,
    dataset: str,
    seed: int,
    device: str,
    n_samples: int,
    batch_size: int,
    direction: float,
    subsample_seed: int,
    warmup: bool,
) -> Dict[str, Any]:
    oae = load_oae_model(
        dataset, seed, device, latent_dim=RGS_LATENT_DIM, recon_w=RGS_RECON_W
    )
    label_stats = load_label_stats(
        dataset, seed, latent_dim=RGS_LATENT_DIM, recon_w=RGS_RECON_W
    )
    del label_stats
    train_loader, _val, test_loader, _info = make_dataloaders(
        dataset,
        batch_size=batch_size,
        seed=seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=0,
    )
    train_latents, _ty, _tx = encode_loader(oae, train_loader, device=device)
    latent_stats = {
        "latent_mean": float(train_latents.mean().item()),
        "latent_std": float(train_latents.std().item()),
        "latent_min": float(train_latents.min().item()),
        "latent_max": float(train_latents.max().item()),
    }
    pool_latent, pool_y, pool_x = encode_loader(oae, test_loader, device=device)
    pool_latent = apply_latent_normalization(
        pool_latent, mode="none", stats=latent_stats
    )
    indices, _ = resolve_start_indices(
        pool_latent.shape[0],
        max_starts=n_samples,
        subsample_seed=subsample_seed,
        starts_cache=None,
    )
    idx = torch.from_numpy(indices.astype(np.int64))
    sampled_latent = pool_latent[idx]
    del pool_x, pool_y

    projector = load_projector(
        dataset=dataset,
        seed=seed,
        projector=RGS_PROJECTOR,
        sugar_w=RGS_SUGAR_W,
        latent_normalization="none",
        knn_k=1,
        latent_dim=sampled_latent.shape[1],
        recon_w=RGS_RECON_W,
        oae_latent_dim=RGS_LATENT_DIM,
        device=device,
    )
    step_size = step_size_for(dataset)
    fitness_fn = lambda z: oae.regress(z)

    def _one_pass(z0: torch.Tensor) -> torch.Tensor:
        z_gen, _ = run_manifold_langevin(
            z0.to(device),
            projector,
            fitness_fn=fitness_fn,
            direction=float(direction),
            num_steps=RGS_NUM_STEPS,
            step_size=step_size,
            temperature=RGS_TEMPERATURE,
            use_projector=True,
            batch_size=batch_size,
            return_history=False,
        )
        return decode_latents(oae, z_gen, device=device, batch_size=batch_size)

    if warmup:
        _one_pass(sampled_latent[: min(8, sampled_latent.shape[0])])

    seed_everything(seed)
    with CudaTimer() as timer:
        new_sequences = _one_pass(sampled_latent)
    n = int(new_sequences.shape[0])
    return {
        "method": "RNAGenScape",
        "dataset": dataset,
        "seed": seed,
        "direction": direction,
        "n": n,
        "batch_size": batch_size,
        "elapsed_seconds": float(timer.elapsed),
        "ms_per_sample": ms_per_sample(timer.elapsed, n),
        "ok": True,
        "error": None,
        "num_steps": RGS_NUM_STEPS,
        "step_size": step_size,
        "temperature": RGS_TEMPERATURE,
        "sugar_w": RGS_SUGAR_W,
        "projector": RGS_PROJECTOR,
    }


def run_one(
    method: str,
    *,
    dataset: str,
    seed: int,
    device: str,
    n_samples: int,
    batch_size: int,
    direction: float,
    subsample_seed: int,
    warmup: bool,
) -> Dict[str, Any]:
    try:
        if method == "RNAGenScape":
            return bench_rnagenscape(
                dataset=dataset,
                seed=seed,
                device=device,
                n_samples=n_samples,
                batch_size=batch_size,
                direction=direction,
                subsample_seed=subsample_seed,
                warmup=warmup,
            )
        if method not in BASELINE_MODELS:
            raise ValueError(f"Unknown method {method}")
        return bench_baseline(
            model_name=method,
            dataset=dataset,
            seed=seed,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
            direction=direction,
            subsample_seed=subsample_seed,
            warmup=warmup,
        )
    except Exception as exc:  # noqa: BLE001 - bench should continue on missing ckpts
        return {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "direction": direction,
            "n": 0,
            "batch_size": batch_size,
            "elapsed_seconds": None,
            "ms_per_sample": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean ms/sample over successful dataset cells, per method."""
    out: List[Dict[str, Any]] = []
    for method in BENCH_METHODS:
        vals = [
            float(r["ms_per_sample"])
            for r in rows
            if r.get("method") == method and r.get("ok") and r.get("ms_per_sample") is not None
        ]
        out.append(
            {
                "method": method,
                "n_cells": len(vals),
                "ms_per_sample_mean": float(np.mean(vals)) if vals else None,
                "ms_per_sample_std": float(np.std(vals)) if len(vals) > 1 else (0.0 if vals else None),
                "samples_per_ms_mean": (1.0 / float(np.mean(vals))) if vals else None,
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Light inference throughput bench.")
    p.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES), choices=sorted(DATASET_NAMES))
    p.add_argument("--methods", nargs="+", default=list(BENCH_METHODS))
    p.add_argument("--seed", type=int, default=1, help="Model / split seed (default: 1).")
    p.add_argument("--n_samples", type=int, default=64, help="Starts to time (default: 64).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument("--subsample_seed", type=int, default=42)
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output dir (default: results/_bench_inference/<utc_stamp>/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(results_root(), "_bench_inference", stamp)
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"device={device} n_samples={args.n_samples} batch_size={args.batch_size} "
        f"seed={args.seed} warmup={not args.no_warmup}"
    )
    print(f"methods={args.methods}")
    print(f"datasets={args.datasets}")
    print(f"out_dir={out_dir}")

    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        for method in args.methods:
            print(f"========== bench {method} on {dataset} ==========", flush=True)
            row = run_one(
                method,
                dataset=dataset,
                seed=args.seed,
                device=device,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                direction=args.direction,
                subsample_seed=args.subsample_seed,
                warmup=not args.no_warmup,
            )
            rows.append(row)
            if row["ok"]:
                print(
                    f"OK {method:12s} {dataset:16s} "
                    f"{row['ms_per_sample']:.3f} ms/sample "
                    f"({row['elapsed_seconds']:.2f}s, n={row['n']})",
                    flush=True,
                )
            else:
                print(f"FAIL {method:12s} {dataset:16s} {row['error']}", flush=True)

    summary = summarize(rows)
    payload = {
        "created_utc": stamp,
        "device": device,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "direction": args.direction,
        "warmup": not args.no_warmup,
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "cells": rows,
        "by_method": summary,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "cells.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "by_method.csv"), index=False)

    print("\n# Mean ms/sample (over successful dataset cells)")
    for s in summary:
        ms = s["ms_per_sample_mean"]
        if ms is None:
            print(f"  {s['method']:12s}  MISSING")
        else:
            thr = s["samples_per_ms_mean"]
            print(
                f"  {s['method']:12s}  {ms:8.3f} ms/sample  "
                f"({thr:.3f} samples/ms, n_cells={s['n_cells']})"
            )
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
