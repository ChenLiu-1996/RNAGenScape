"""Evaluate generation artifacts with RhoFold mean pLDDT (folding confidence).

Scores generated and start sequences with the same paired subsample indices
across methods/seeds.

Example:
  python src/evaluation/eval_rhofold.py \\
    --dataset OpenVaccine \\
    --model DiffAb \\
    --experiment pos_guided \\
    --n_seq 100
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow `python src/evaluation/eval_rhofold.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG
from utils.metrics import decode_token_ids, subsample_indices, to_token_ids
from utils.oracle import resolve_device
from utils.results import (
    discover_seed_runs,
    evaluation_dir,
    experiment_dir,
    load_generation_artifact,
    write_per_seed_csv,
)
from utils.rhofold_paths import (
    default_ckpt_path,
    default_rhofold_dir,
    ensure_rhofold_checkpoint,
    resolve_rhofold_dir,
)


def tokens_to_rna_strings(sequences) -> List[str]:
    """Decode token ids to RhoFold-ready A/U/G/C strings (pads / non-AUGC stripped)."""
    ids = np.asarray(to_token_ids(sequences).cpu().numpy())
    decoded = decode_token_ids(ids)
    if isinstance(decoded, str):
        decoded = [decoded]
    out: List[str] = []
    for seq in decoded:
        seq = str(seq).replace("T", "U")
        seq = "".join(c for c in seq if c in "AUGC")
        out.append(seq)
    return out


def load_rhofold(rhofold_dir: str | None, ckpt: str | None, device: str):
    rhofold_dir = resolve_rhofold_dir(rhofold_dir)
    ckpt = ensure_rhofold_checkpoint(ckpt)
    if rhofold_dir not in sys.path:
        sys.path.insert(0, rhofold_dir)
    from rhofold.rhofold import RhoFold
    from rhofold.config import rhofold_config

    model = RhoFold(rhofold_config)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded RhoFold from {rhofold_dir}")
    print(f"Loaded checkpoint {ckpt}")
    return model


@torch.no_grad()
def predict_plddt(
    model, sequence: str, device: str, work_dir: str
) -> Tuple[float, np.ndarray]:
    """Return (mean_plddt_0_100, per_residue_plddt_0_1)."""
    from rhofold.utils.alphabet import get_features

    os.makedirs(work_dir, exist_ok=True)
    fas = os.path.join(work_dir, "seq.fasta")
    with open(fas, "w", encoding="utf-8") as f:
        f.write(">target\n")
        f.write(sequence + "\n")

    data_dict = get_features(fas, fas)
    outputs = model(
        tokens=data_dict["tokens"].to(device),
        rna_fm_tokens=data_dict["rna_fm_tokens"].to(device),
        seq=data_dict["seq"],
    )
    output = outputs[-1]
    plddt = output["plddt"]
    if isinstance(plddt, (tuple, list)):
        local = plddt[0]
    else:
        local = plddt
    local_np = local[0].detach().float().cpu().numpy().reshape(-1)
    if local_np.max() > 1.5:
        mean_100 = float(local_np.mean())
        local_01 = local_np / 100.0
    else:
        mean_100 = float(local_np.mean() * 100.0)
        local_01 = local_np
    return mean_100, local_01


def evaluate_rhofold(
    sequences,
    *,
    indices: np.ndarray,
    model,
    device: str,
    success_threshold: float = 70.0,
) -> Dict[str, np.ndarray]:
    """Score ``sequences[indices]``; drop cleaned sequences with length < 4."""
    rna_seqs = tokens_to_rna_strings(sequences)
    keep_idx: List[int] = []
    keep_seqs: List[str] = []
    for i in np.asarray(indices, dtype=np.int64).reshape(-1):
        seq = rna_seqs[int(i)]
        if len(seq) >= 4:
            keep_idx.append(int(i))
            keep_seqs.append(seq)
    if not keep_seqs:
        raise RuntimeError("No valid A/U/G/C sequences after decoding (len>=4).")

    means: List[float] = []
    with tempfile.TemporaryDirectory(prefix="rhofold_eval_") as tmp:
        for j, seq in enumerate(keep_seqs):
            work = os.path.join(tmp, f"seq_{j}")
            mean_100, _local = predict_plddt(model, seq, device, work)
            means.append(mean_100)

    means_arr = np.asarray(means, dtype=np.float64)
    success = (means_arr >= success_threshold).astype(np.float64)
    return {
        "indices": np.asarray(keep_idx, dtype=np.int64),
        "mean_plddt": means_arr,
        "success": success,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RhoFold pLDDT across experiment seeds.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True, help="Method folder under results/<dataset>/")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--n_seq", type=int, default=100, help="Paired subsample size per seed.")
    p.add_argument("--success_threshold", type=float, default=70.0)
    p.add_argument(
        "--rhofold_dir",
        type=str,
        default=None,
        help=f"RhoFold repo root (default: $RHOFOLD_DIR or {default_rhofold_dir()}).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=f"Checkpoint path (default: $RHOFOLD_CKPT or {default_ckpt_path()}; auto-download).",
    )
    p.add_argument("--latent_dim", type=int, default=128, help="For OAE path tag.")
    p.add_argument("--recon_w", type=float, default=5.0, help="For OAE path tag.")
    p.add_argument("--force", action="store_true", help="Re-run even if npz exists.")
    p.add_argument(
        "--subsample_seed",
        type=int,
        default=42,
        help="Shared subsample seed so methods are paired.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device()
    exp_kwargs = {}
    if args.model == "OAE":
        exp_kwargs = {"oae_latent_dim": args.latent_dim, "oae_recon_w": args.recon_w}
    exp_dir = experiment_dir(args.dataset, args.model, args.experiment, **exp_kwargs)
    out_root = os.path.join(evaluation_dir(exp_dir), "rhofold")
    os.makedirs(out_root, exist_ok=True)

    print(f"Experiment: {exp_dir}")
    print(f"Device:     {device}")

    runs = discover_seed_runs(exp_dir)
    print(f"Found {len(runs)} seed run(s): {[s for s, _ in runs]}")

    model = load_rhofold(args.rhofold_dir, args.ckpt, device)
    rows = []
    for seed, run_dir in runs:
        seed_out = os.path.join(out_root, f"seed_{seed}")
        os.makedirs(seed_out, exist_ok=True)
        npz_path = os.path.join(seed_out, "rhofold_plddt.npz")
        if (not args.force) and os.path.isfile(npz_path):
            prev = np.load(npz_path, allow_pickle=True)
            prev_n = int(np.asarray(prev["gen_mean_plddt"]).reshape(-1).shape[0])
            if prev_n >= args.n_seq:
                gen_mean_plddt = float(np.mean(prev["gen_mean_plddt"]))
                start_mean_plddt = float(np.mean(prev["start_mean_plddt"]))
                print(
                    f"[seed {seed}] skip existing n={prev_n} >= n_seq={args.n_seq}: "
                    f"gen_mean_plddt={gen_mean_plddt:.2f} start_mean_plddt={start_mean_plddt:.2f}"
                )
                rows.append(
                    {
                        "seed": seed,
                        "run_dir": run_dir,
                        "dataset": args.dataset,
                        "n": prev_n,
                        # Per-run scalar: mean sequence pLDDT (aggregated across seeds later).
                        "gen_mean_plddt": gen_mean_plddt,
                        "start_mean_plddt": start_mean_plddt,
                    }
                )
                continue

        artifact = load_generation_artifact(run_dir)
        gen = artifact["new_sequences"]
        start = artifact["sampled_X"]
        n = int(to_token_ids(gen).shape[0])
        indices = subsample_indices(n, args.n_seq, args.subsample_seed)
        print(f"[seed {seed}] scoring {len(indices)} gen + start sequences ...")

        gen_res = evaluate_rhofold(
            gen,
            indices=indices,
            model=model,
            device=device,
            success_threshold=args.success_threshold,
        )
        # Re-score starts on the same kept indices (may drop short cleaned seqs).
        start_res = evaluate_rhofold(
            start,
            indices=gen_res["indices"],
            model=model,
            device=device,
            success_threshold=args.success_threshold,
        )
        # Align on intersection of kept indices.
        gen_map = {int(i): j for j, i in enumerate(gen_res["indices"])}
        start_map = {int(i): j for j, i in enumerate(start_res["indices"])}
        common = sorted(set(gen_map) & set(start_map))
        if not common:
            raise RuntimeError(f"[seed {seed}] no paired valid sequences after cleaning")
        common_idx = np.asarray(common, dtype=np.int64)
        gen_mean = np.asarray([gen_res["mean_plddt"][gen_map[i]] for i in common], dtype=np.float64)
        start_mean = np.asarray(
            [start_res["mean_plddt"][start_map[i]] for i in common], dtype=np.float64
        )
        gen_success = (gen_mean >= args.success_threshold).astype(np.float64)
        start_success = (start_mean >= args.success_threshold).astype(np.float64)

        np.savez_compressed(
            npz_path,
            indices=common_idx,
            gen_mean_plddt=gen_mean,
            start_mean_plddt=start_mean,
            gen_success=gen_success,
            start_success=start_success,
            success_threshold=np.asarray(args.success_threshold, dtype=np.float64),
            n_seq=np.asarray(args.n_seq, dtype=np.int64),
            subsample_seed=np.asarray(args.subsample_seed, dtype=np.int64),
        )
        row = {
            "seed": seed,
            "run_dir": run_dir,
            "dataset": args.dataset,
            "n": int(common_idx.shape[0]),
            # Per-run scalar: mean sequence pLDDT (aggregated across seeds later).
            "gen_mean_plddt": float(gen_mean.mean()),
            "start_mean_plddt": float(start_mean.mean()),
        }
        rows.append(row)
        print(
            f"[seed {seed}] gen_mean_plddt={row['gen_mean_plddt']:.2f} "
            f"start_mean_plddt={row['start_mean_plddt']:.2f}"
        )
        print(f"[seed {seed}] wrote {npz_path}")

    # Per-seed CSV only (no log summary table). Mean±std across seeds of
    # gen_mean_plddt / start_mean_plddt is reported in eval_optimization.
    per_seed_path = os.path.join(out_root, "rhofold_per_seed.csv")
    write_per_seed_csv(per_seed_path, rows)
    print(f"Wrote {per_seed_path}")


if __name__ == "__main__":
    main()
