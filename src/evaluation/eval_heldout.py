"""Evaluate held-out set distances across all seeds of an experiment.

Paper metric: held-out pool = scores better than mean + direction * std.
Reports NN Hamming fractions and normalized Levenshtein distances (gen/start),
plus transport distances using each metric as the cost.

Example:
  python src/evaluation/eval_heldout.py \\
    --dataset OpenVaccine \\
    --model DiffAb \\
    --experiment pos_guided \\
    --oracle UTRLM
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow `python src/evaluation/eval_heldout.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG
from utils.metrics import heldout_distance_metrics
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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate held-out distances across seeds.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--oracle", type=str, required=True, choices=sorted(SUPPORTED_ORACLES))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128, help="For OAE path tag.")
    p.add_argument("--recon_w", type=float, default=5.0, help="For OAE path tag.")
    p.add_argument("--std_scale", type=float, default=1.0)
    p.add_argument("--histogram", action="store_true", help="Save pooled NN Hamming and edit-distance histograms.")
    p.add_argument("--bins", type=int, default=40, help="Histogram bins.")
    return p.parse_args()


def evaluate_one_seed(
    *,
    seed: int,
    run_dir: str,
    oracle_model,
    oracle_name: str,
    dataset: str,
    batch_size: int,
    std_scale: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    artifact = load_generation_artifact(run_dir)
    start = artifact["sampled_X"]
    generated = artifact["new_sequences"]
    direction = float(artifact["direction"])
    pool = artifact["sampling_pool_X"]

    pool_scores, _ = score_sequences(
        oracle_model,
        pool,
        oracle_name=oracle_name,
        batch_size=batch_size,
        return_embeddings=False,
    )
    held = heldout_distance_metrics(
        generated_tokens=generated,
        start_tokens=start,
        pool_tokens=pool,
        pool_scores=pool_scores,
        direction=direction,
        std_scale=std_scale,
        return_gen_nn=True,
        include_edit=True,
    )
    gen_nn = {
        metric: np.asarray(held.pop(f"heldout_nn_{metric}_gen"), dtype=np.float64)
        for metric in ("hamming", "edit")
    }
    row = dict(held)
    row["seed"] = seed
    row["run_dir"] = run_dir
    row["dataset"] = dataset
    row["oracle"] = oracle_name
    row["direction"] = direction
    return row, gen_nn


def main():
    args = parse_args()
    device = resolve_device()
    exp_kwargs = {}
    if args.model == "OAE":
        exp_kwargs = {"oae_latent_dim": args.latent_dim, "oae_recon_w": args.recon_w}
    exp_dir = experiment_dir(args.dataset, args.model, args.experiment, **exp_kwargs)
    out_dir = os.path.join(evaluation_dir(exp_dir), "heldout")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Experiment: {exp_dir}")
    print(f"Oracle:     {args.oracle}")
    print(f"Device:     {device}")

    runs = discover_seed_runs(exp_dir)
    print(f"Found {len(runs)} seed run(s): {[s for s, _ in runs]}")

    oracle_model = load_oracle(args.oracle, args.dataset)
    rows = []
    all_gen_nn: dict[str, list[np.ndarray]] = {"hamming": [], "edit": []}
    for seed, run_dir in runs:
        print(f"[seed {seed}] scoring held-out distances for {run_dir} ...")
        row, gen_nn = evaluate_one_seed(
            seed=seed,
            run_dir=run_dir,
            oracle_model=oracle_model,
            oracle_name=args.oracle,
            dataset=args.dataset,
            batch_size=args.batch_size,
            std_scale=args.std_scale,
        )
        rows.append(row)
        for metric, values in gen_nn.items():
            all_gen_nn[metric].append(values)
        print(
            f"[seed {seed}] heldout_n={int(row['heldout_n'])} "
            f"nn_hamming_gen={row['heldout_nn_hamming_gen_mean']:.4f} "
            f"nn_hamming_start={row['heldout_nn_hamming_start_mean']:.4f} "
            f"w2_hamming={row['heldout_w2_hamming']:.4f} "
            f"nn_edit_gen={row['heldout_nn_edit_gen_mean']:.4f} "
            f"nn_edit_start={row['heldout_nn_edit_start_mean']:.4f} "
            f"w2_edit={row['heldout_w2_edit']:.4f}"
        )

    per_seed_path = os.path.join(out_dir, f"heldout_per_seed_{args.oracle}.csv")
    summary_path = os.path.join(out_dir, f"heldout_summary_{args.oracle}.csv")
    write_per_seed_csv(per_seed_path, rows)
    summarize_per_seed_csv(
        per_seed_path,
        summary_path,
        skip_cols={"run_dir", "dataset", "oracle", "direction"},
    )
    # No log summary table — held-out metrics appear in eval_optimization.
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")

    for metric, batches in all_gen_nn.items():
        nonempty = [x for x in batches if x.size > 0]
        if not args.histogram or not nonempty:
            continue
        pooled = np.concatenate(nonempty, axis=0)
        if pooled.size > 0:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.hist(pooled, bins=args.bins, color="#4C78A8", alpha=0.85, edgecolor="white")
            ax.set_xlabel(
                "NN Hamming distance (mismatch fraction)" if metric == "hamming"
                else "NN normalized edit distance (Levenshtein)"
            )
            ax.set_ylabel("Count")
            ax.set_title(
                f"{args.dataset} / {args.model} / {args.experiment}\n"
                f"held-out NN distances ({args.oracle})"
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            hist_path = os.path.join(out_dir, f"heldout_nn_hist_{args.oracle}.png" if metric == "hamming"
                else f"heldout_nn_edit_hist_{args.oracle}.png")
            fig.savefig(hist_path, dpi=300)
            plt.close(fig)
            print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
