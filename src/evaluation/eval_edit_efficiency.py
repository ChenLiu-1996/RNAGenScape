"""Evaluate property gain versus Hamming distance across experiment seeds.

Example:
  python src/evaluation/eval_edit_efficiency.py \\
    --dataset OpenVaccine \\
    --model OAE \\
    --experiment pos_sugar0e0_dae_T1e-3_ss1e-2_ns100 \\
    --oracle UTRLM
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Allow `python src/evaluation/eval_edit_efficiency.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG
from utils.metrics import pairwise_hamming_distance
from utils.oracle import (
    SUPPORTED_ORACLES,
    load_oracle,
    resolve_device,
    score_sequences,
)
from utils.results import (
    discover_seed_runs,
    evaluation_dir,
    experiment_dir,
    load_generation_artifact,
    summarize_per_seed_csv,
    write_per_seed_csv,
)


def bin_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 20):
    """Mean ± std of y within equal-count bins of x (skips empty bins)."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    edges = np.linspace(0, len(x_s), n_bins + 1, dtype=int)
    xs, means, stds, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        xb, yb = x_s[lo:hi], y_s[lo:hi]
        xs.append(xb.mean())
        means.append(yb.mean())
        stds.append(yb.std())
        counts.append(len(yb))
    return (
        np.asarray(xs),
        np.asarray(means),
        np.asarray(stds),
        np.asarray(counts),
    )


def make_plot(
    hamming: np.ndarray,
    prop_gain: np.ndarray,
    title: str,
    ylabel: str,
    out: str,
    n_bins: int,
    seed: int,
    max_points: Optional[int],
) -> None:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(hamming))
    if max_points is not None and max_points < len(idx):
        idx = rng.choice(idx, size=max_points, replace=False)
    bin_x, bin_mean, bin_std, _ = bin_stats(
        hamming.astype(float), prop_gain, n_bins=n_bins
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        hamming[idx],
        prop_gain[idx],
        s=8,
        alpha=0.18,
        c="#4C78A8",
        linewidths=0,
        label="sequences",
    )
    ax.plot(bin_x, bin_mean, color="#E45756", linewidth=2.5, label="binned mean")
    ax.fill_between(
        bin_x,
        bin_mean - bin_std,
        bin_mean + bin_std,
        color="#E45756",
        alpha=0.2,
        label="binned ±1 std",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Hamming distance (mismatch count)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hamming efficiency across seeds.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--oracle", type=str, required=True, choices=sorted(SUPPORTED_ORACLES))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128, help="For OAE path tag.")
    p.add_argument("--recon_w", type=float, default=5.0, help="For OAE path tag.")
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--max_points", type=int, default=2000)
    return p.parse_args()


def evaluate_one_seed(
    *,
    seed: int,
    run_dir: str,
    oracle_model,
    oracle_name: str,
    dataset: str,
    batch_size: int,
    out_dir: str,
    n_bins: int,
    max_points: int,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    artifact = load_generation_artifact(run_dir)
    start = artifact["sampled_X"]
    generated = artifact["new_sequences"]
    direction = float(artifact["direction"])

    start_scores, _ = score_sequences(
        oracle_model,
        start,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=False,
    )
    gen_scores, _ = score_sequences(
        oracle_model,
        generated,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=False,
    )
    start_scores = np.asarray(start_scores, dtype=np.float64).reshape(-1)
    gen_scores = np.asarray(gen_scores, dtype=np.float64).reshape(-1)
    prop_gain = gen_scores - start_scores
    hamming = pairwise_hamming_distance(start, generated).astype(np.float64)

    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    npz_path = os.path.join(seed_dir, f"edit_efficiency_{oracle_name}.npz")
    np.savez_compressed(
        npz_path,
        hamming_distance=hamming,
        property_gain=prop_gain,
        start_fitness=start_scores,
        generated_fitness=gen_scores,
        direction=np.asarray(direction, dtype=np.float64),
    )
    png_path = os.path.join(seed_dir, f"property_gain_vs_hamming_{oracle_name}.png")
    title = (
        f"{dataset}\n{artifact['model_type']}  seed={seed}  "
        f"dir={'pos' if direction > 0 else 'neg'}"
    )
    make_plot(
        hamming,
        prop_gain,
        title,
        f"Property gain ({oracle_name})",
        png_path,
        n_bins,
        seed,
        max_points,
    )

    pearson = (
        float(np.corrcoef(hamming, prop_gain)[0, 1])
        if hamming.size > 1 and np.std(hamming) > 0 and np.std(prop_gain) > 0
        else float("nan")
    )
    row = {
        "seed": seed,
        "run_dir": run_dir,
        "dataset": dataset,
        "oracle": oracle_name,
        "direction": direction,
        "n_sequences": float(hamming.shape[0]),
        "mean_hamming_distance": float(np.mean(hamming)),
        "median_hamming_distance": float(np.median(hamming)),
        "mean_property_gain": float(np.mean(prop_gain)),
        "median_property_gain": float(np.median(prop_gain)),
        "pearson_hamming_prop_gain": pearson,
    }
    print(
        f"[seed {seed}] hamming mean={row['mean_hamming_distance']:.2f} "
        f"median={row['median_hamming_distance']:.1f} | "
        f"prop_gain mean={row['mean_property_gain']:.4f} "
        f"median={row['median_property_gain']:.4f} | "
        f"pearson={pearson:.4f}"
    )
    print(f"[seed {seed}] wrote {npz_path}")
    return row, hamming, prop_gain


def main():
    args = parse_args()
    device = resolve_device()
    exp_kwargs = {}
    if args.model == "OAE":
        exp_kwargs = {"oae_latent_dim": args.latent_dim, "oae_recon_w": args.recon_w}
    exp_dir = experiment_dir(args.dataset, args.model, args.experiment, **exp_kwargs)
    out_dir = os.path.join(evaluation_dir(exp_dir), "edit_efficiency")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Experiment: {exp_dir}")
    print(f"Oracle:     {args.oracle}")
    print(f"Device:     {device}")

    runs = discover_seed_runs(exp_dir)
    print(f"Found {len(runs)} seed run(s): {[s for s, _ in runs]}")

    oracle_model = load_oracle(args.oracle, args.dataset)
    rows: List[Dict] = []
    all_hamming: List[np.ndarray] = []
    all_gain: List[np.ndarray] = []
    for seed, run_dir in runs:
        print(f"[seed {seed}] scoring {run_dir} ...")
        row, hamming, prop_gain = evaluate_one_seed(
            seed=seed,
            run_dir=run_dir,
            oracle_model=oracle_model,
            oracle_name=args.oracle,
            dataset=args.dataset,
            batch_size=args.batch_size,
            out_dir=out_dir,
            n_bins=args.n_bins,
            max_points=args.max_points,
        )
        rows.append(row)
        all_hamming.append(hamming)
        all_gain.append(prop_gain)

    per_seed_path = os.path.join(out_dir, f"edit_efficiency_per_seed_{args.oracle}.csv")
    summary_path = os.path.join(out_dir, f"edit_efficiency_summary_{args.oracle}.csv")
    write_per_seed_csv(per_seed_path, rows)
    summarize_per_seed_csv(
        per_seed_path,
        summary_path,
        skip_cols={"run_dir", "dataset", "oracle", "direction"},
    )
    # No log summary table — paper metrics appear in eval_optimization.
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")

    if all_hamming:
        pooled_hamming = np.concatenate(all_hamming, axis=0)
        pooled_gain = np.concatenate(all_gain, axis=0)
        pooled_png = os.path.join(
            out_dir, f"property_gain_vs_hamming_pooled_{args.oracle}.png"
        )
        make_plot(
            pooled_hamming,
            pooled_gain,
            f"{args.dataset}\n{args.model} / {args.experiment} (pooled seeds)",
            f"Property gain ({args.oracle})",
            pooled_png,
            args.n_bins,
            seed=0,
            max_points=args.max_points,
        )


if __name__ == "__main__":
    main()
