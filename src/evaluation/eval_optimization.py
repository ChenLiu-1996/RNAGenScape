"""Evaluate sequence optimization performance across all seeds of an experiment.

Example:
  python src/evaluation/eval_optimization.py \\
    --dataset OpenVaccine \\
    --model OAE \\
    --experiment pos_samehyper_sugar1p0 \\
    --oracle UTRLM
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Allow `python src/evaluation/eval_optimization.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.metrics import optimization_metrics
from dataset import DATASET_CONFIG
from utils.oracle import (
    SUPPORTED_ORACLES,
    load_oracle,
    load_train_token_ids,
    resolve_device,
    score_sequences,
)
from utils.results import (
    discover_seed_runs,
    evaluation_dir,
    experiment_dir,
    format_summary_table,
    load_generation_artifact,
    summarize_per_seed_csv,
    write_per_seed_csv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate optimization across experiment seeds.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True, help="Method folder name under results/<dataset>/")
    p.add_argument("--experiment", type=str, required=True, help="Experiment id (fixed settings, no seed)")
    p.add_argument("--oracle", type=str, required=True, choices=sorted(SUPPORTED_ORACLES))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128, help="For OAE: path tag d{latent}_recon{w}.")
    p.add_argument("--recon_w", type=float, default=5.0, help="For OAE: path tag d{latent}_recon{w}.")
    return p.parse_args()


def evaluate_one_seed(
    *,
    seed: int,
    run_dir: str,
    oracle_model,
    oracle_name: str,
    dataset: str,
    batch_size: int,
    novelty_vs_train: torch.Tensor,
    reference_embeddings: np.ndarray,
) -> dict:
    artifact = load_generation_artifact(run_dir)
    start = artifact["sampled_X"]
    generated = artifact["new_sequences"]
    direction = float(artifact["direction"])
    novelty_vs_test = artifact["sampling_pool_X"]

    start_scores, _ = score_sequences(
        oracle_model,
        start,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=False,
    )
    gen_scores, gen_embeds = score_sequences(
        oracle_model,
        generated,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=True,
    )
    test_pool_scores, _ = score_sequences(
        oracle_model,
        novelty_vs_test,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=False,
    )

    row = optimization_metrics(
        start_tokens=start,
        generated_tokens=generated,
        start_scores=start_scores,
        generated_scores=gen_scores,
        generated_embeddings=gen_embeds,
        reference_embeddings=reference_embeddings,
        novelty_vs_train_reference=novelty_vs_train,
        novelty_vs_test_reference=novelty_vs_test,
        test_pool_tokens=novelty_vs_test,
        test_pool_scores=test_pool_scores,
        direction=direction,
        seed=seed,
    )
    row["seed"] = seed
    row["run_dir"] = run_dir
    row["dataset"] = dataset
    row["oracle"] = oracle_name
    row["direction"] = direction
    return row


def main():
    args = parse_args()
    device = resolve_device()
    exp_kwargs = {}
    if args.model == "OAE":
        exp_kwargs = {"oae_latent_dim": args.latent_dim, "oae_recon_w": args.recon_w}
    exp_dir = experiment_dir(args.dataset, args.model, args.experiment, **exp_kwargs)
    out_dir = evaluation_dir(exp_dir)

    print(f"Experiment: {exp_dir}")
    print(f"Oracle:     {args.oracle}")
    print(f"Device:     {device}")

    runs = discover_seed_runs(exp_dir)
    print(f"Found {len(runs)} seed run(s): {[s for s, _ in runs]}")

    oracle_model = load_oracle(args.oracle, args.dataset)
    novelty_vs_train = load_train_token_ids(args.dataset)

    # Reference embeddings: score a subsample of training sequences once.
    ref_cap = min(2000, novelty_vs_train.shape[0])
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(novelty_vs_train.shape[0], size=ref_cap, replace=False)
    _, reference_embeddings = score_sequences(
        oracle_model,
        novelty_vs_train[ref_idx],
        oracle_name=args.oracle,
        batch_size=args.batch_size,
        return_embeddings=True,
    )

    rows = []
    for seed, run_dir in runs:
        print(f"[seed {seed}] scoring {run_dir} ...")
        row = evaluate_one_seed(
            seed=seed,
            run_dir=run_dir,
            oracle_model=oracle_model,
            oracle_name=args.oracle,
            dataset=args.dataset,
            batch_size=args.batch_size,
            novelty_vs_train=novelty_vs_train,
            reference_embeddings=reference_embeddings,
        )
        rows.append(row)
        print(
            f"[seed {seed}] median_property_change={row['median_property_change']:.4f} "
            f"pct_improved={row['pct_improved']:.2f} "
            f"pct_worse={row['pct_worse']:.2f} "
            f"pct_identical_to_input={row['pct_identical_to_input']:.2f} "
            f"pct_edited_property_unchanged={row['pct_edited_property_unchanged']:.2f} "
            f"elite_nn_hamming_gen_mean={row['elite_nn_hamming_gen_mean']:.4f}"
        )

    per_seed_path = os.path.join(out_dir, f"optimization_per_seed_{args.oracle}.csv")
    summary_path = os.path.join(out_dir, f"optimization_summary_{args.oracle}.csv")
    write_per_seed_csv(per_seed_path, rows)
    summary = summarize_per_seed_csv(
        per_seed_path,
        summary_path,
        skip_cols={"run_dir", "dataset", "oracle", "direction"},
    )
    table = format_summary_table(summary)
    print()
    print(table)
    print()
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
