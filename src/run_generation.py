"""Unified generation entrypoint.

Phase 1: RNAGenScape (OAE + manifold Langevin + projector).
Later: denovo / guided / simple_opt handlers behind ``--method``.

Example:
  python src/run_generation.py \\
    --dataset OpenVaccine --method rnagenscape --model OAE \\
    --experiment pos_samehyper_sugar0p0 --seed 1 --direction 1 \\
    --projector dae --sugar_w 0.0 --num_steps 100 --step_size 5e-3 --temperature 1e-3

``--seed`` selects the trained OAE / projector under
``results/<dataset>/OAE/d{latent}_recon{w}/seed_{seed}/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow `python src/run_generation.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from modules.langevin import run_manifold_langevin
from modules.manifold_projector_dae import load_manifold_projector_dae
from modules.manifold_projector_knn import ManifoldProjectorKNN
from modules.oae import OAE, load_oae
from utils.metrics import VOCAB_SIZE, to_token_ids
from utils.oracle import resolve_device
from utils.results import (
    experiment_dir,
    latent_trainset_path,
    load_generation_artifact,
    manifold_projector_dae_checkpoint_path,
    normalize_latent_norm_name,
    oae_checkpoint_path,
    save_generation_artifact,
)
from utils.training_utils import seed_everything

METHODS = ("rnagenscape", "denovo", "guided", "simple_opt")
DEFAULT_DAE_HIDDEN_DIMS = (32, 16, 32)


def load_oae_model(
    dataset: str,
    seed: int,
    device: str,
    *,
    latent_dim: int,
    recon_w: float,
) -> OAE:
    ckpt = oae_checkpoint_path(
        dataset, seed, latent_dim=latent_dim, recon_w=recon_w
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Missing OAE checkpoint: {ckpt}. Train with src/train_oae.py --seed {seed} first."
        )
    seq_len = int(DATASET_CONFIG[dataset]["seq_len"])
    return load_oae(ckpt, device=device, seq_len=seq_len, vocab_size=VOCAB_SIZE)


def load_label_stats(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
) -> Dict[str, float]:
    path = os.path.join(
        os.path.dirname(
            oae_checkpoint_path(
                dataset, seed, latent_dim=latent_dim, recon_w=recon_w
            )
        ),
        "label_stats.json",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing label stats: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "label_mean": float(raw["label_mean"]),
        "label_std": float(raw["label_std"]),
        "label_min": float(raw["label_min"]),
        "label_max": float(raw["label_max"]),
    }


def apply_latent_normalization(
    latents: torch.Tensor,
    *,
    mode: str,
    stats: Dict[str, float],
) -> torch.Tensor:
    mode = normalize_latent_norm_name(mode)
    if mode == "none":
        return latents
    if mode == "normal":
        std = stats["latent_std"] if stats["latent_std"] > 0 else 1.0
        return (latents - stats["latent_mean"]) / std
    if mode == "minmax":
        span = stats["latent_max"] - stats["latent_min"]
        span = span if span > 0 else 1.0
        return (latents - stats["latent_min"]) / span
    raise ValueError(f"Unknown latent_normalization '{mode}'")


@torch.no_grad()
def encode_loader(
    oae: OAE,
    loader,
    *,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (latents [N,D], labels [N], token_ids [N,L])."""
    latents: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    token_ids: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        z = oae.encode(x)
        latents.append(z.detach().cpu())
        labels.append(y.detach().cpu().float().reshape(-1))
        token_ids.append(to_token_ids(x).detach().cpu())
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0), torch.cat(token_ids, dim=0)


def load_projector(
    *,
    dataset: str,
    seed: int,
    projector: str,
    sugar_w: float,
    latent_normalization: str,
    knn_k: int,
    latent_dim: int,
    recon_w: float,
    oae_latent_dim: int,
    device: str,
    hidden_dims: Tuple[int, ...] = DEFAULT_DAE_HIDDEN_DIMS,
):
    if projector == "dae":
        path = manifold_projector_dae_checkpoint_path(
            dataset,
            seed,
            latent_dim=oae_latent_dim,
            recon_w=recon_w,
            sugar_w=sugar_w,
            latent_normalization=latent_normalization,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing DAE projector: {path}. "
                f"Train with src/train_manifold_projector.py --seed {seed} first."
            )
        return load_manifold_projector_dae(
            path, input_dim=latent_dim, hidden_dims=hidden_dims, device=device
        )
    if projector == "knn":
        path = latent_trainset_path(
            dataset,
            seed,
            latent_dim=oae_latent_dim,
            recon_w=recon_w,
            sugar_w=sugar_w,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing kNN latent trainset: {path}. "
                f"Run train_manifold_projector.py --projector knn --seed {seed} first."
            )
        return ManifoldProjectorKNN.from_latent_trainset(path, k=knn_k, device=device)
    raise ValueError(f"Unknown projector '{projector}'. Use dae|knn.")


@torch.no_grad()
def decode_latents(
    oae: OAE,
    latents: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Decode latent batches to token ids ``[N, L]``."""
    outs = []
    for i in range(0, latents.shape[0], batch_size):
        z = latents[i : i + batch_size].to(device)
        outs.append(oae.generate(z).detach().cpu())
    return torch.cat(outs, dim=0)


def subsample_starts(
    pool_latent: torch.Tensor,
    pool_y: torch.Tensor,
    pool_x: torch.Tensor,
    *,
    subsample_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    n = pool_latent.shape[0]
    sample_size = int(n * 0.25) if n >= 20000 else n
    rng = np.random.default_rng(seed=subsample_seed)
    indices = rng.choice(n, size=sample_size, replace=False)
    indices_t = torch.from_numpy(indices.astype(np.int64))
    return (
        pool_latent[indices_t],
        pool_y[indices_t],
        pool_x[indices_t],
        indices,
    )


def run_rnagenscape(args, device: str) -> str:
    if args.sugar_w > 0.0:
        raise NotImplementedError(
            "SUGAR augmentation is not ported yet. Use --sugar_w 0.0 for now."
        )
    if args.model != "OAE":
        raise ValueError("method=rnagenscape currently requires --model OAE")

    oae = load_oae_model(
        args.dataset,
        args.seed,
        device,
        latent_dim=args.latent_dim,
        recon_w=args.recon_w,
    )
    label_stats = load_label_stats(
        args.dataset,
        args.seed,
        latent_dim=args.latent_dim,
        recon_w=args.recon_w,
    )

    # Match the train/val/test split used when training this OAE / projector seed.
    train_loader, _val_loader, test_loader, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=args.num_workers,
    )
    print(
        f"dataset={args.dataset} n_train={info.n_train} n_test={info.n_test} "
        f"seed={args.seed} projector={args.projector} sugar_w={args.sugar_w}"
    )

    train_latents, _train_y, _train_x = encode_loader(oae, train_loader, device=device)
    latent_stats = {
        "latent_mean": float(train_latents.mean().item()),
        "latent_std": float(train_latents.std().item()),
        "latent_min": float(train_latents.min().item()),
        "latent_max": float(train_latents.max().item()),
    }
    train_stats = {**label_stats, **latent_stats}

    pool_latent, pool_y, pool_x = encode_loader(oae, test_loader, device=device)
    pool_latent = apply_latent_normalization(
        pool_latent, mode=args.latent_normalization, stats=latent_stats
    )

    seed_everything(args.subsample_seed)
    sampled_latent, sampled_y, sampled_x, sampled_indices = subsample_starts(
        pool_latent, pool_y, pool_x, subsample_seed=args.subsample_seed
    )
    print(
        f"start pool={pool_latent.shape[0]} starts={sampled_latent.shape[0]} "
        f"subsample_seed={args.subsample_seed}"
    )

    projector = load_projector(
        dataset=args.dataset,
        seed=args.seed,
        projector=args.projector,
        sugar_w=args.sugar_w,
        latent_normalization=args.latent_normalization,
        knn_k=args.knn_k,
        latent_dim=sampled_latent.shape[1],
        recon_w=args.recon_w,
        oae_latent_dim=args.latent_dim,
        device=device,
    )

    seed_everything(args.seed)
    fitness_fn = (lambda z: oae.regress(z)) if args.use_fitness else None

    print(
        f"Langevin steps={args.num_steps} step_size={args.step_size} "
        f"temperature={args.temperature} use_projector={args.use_projector} "
        f"direction={args.direction}"
    )
    z_gen, trajectories = run_manifold_langevin(
        sampled_latent.to(device),
        projector,
        fitness_fn=fitness_fn,
        direction=float(args.direction),
        num_steps=args.num_steps,
        step_size=args.step_size,
        temperature=args.temperature,
        step_size_rescale=args.step_size_rescale,
        use_projector=args.use_projector,
        projector_iters=args.projector_iters,
        annealed=args.annealed,
        batch_size=args.batch_size,
        return_history=True,
    )

    new_sequences = decode_latents(oae, z_gen, device=device, batch_size=args.batch_size)
    seq_trajectories = None
    if trajectories is not None and args.save_trajectories:
        print(f"decoding trajectories {tuple(trajectories.shape)} ...")
        steps = []
        for t in range(trajectories.shape[0]):
            steps.append(
                decode_latents(oae, trajectories[t], device=device, batch_size=args.batch_size)
            )
        seq_trajectories = torch.stack(steps, dim=0)

    out_dir = os.path.join(
        experiment_dir(
            args.dataset,
            args.model,
            args.experiment,
            oae_latent_dim=args.latent_dim,
            oae_recon_w=args.recon_w,
        ),
        f"seed_{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = save_generation_artifact(
        out_dir,
        new_sequences=new_sequences,
        sampled_X=sampled_x,
        sampled_Y=sampled_y,
        sampled_indices=sampled_indices,
        sampling_pool_X=pool_x,
        direction=float(args.direction),
        train_stats=train_stats,
        model_type=args.model,
        data=args.dataset,
        seed=args.seed,
        subsample_seed=args.subsample_seed,
        trajectories=seq_trajectories,
        trajectories_are_sequences=seq_trajectories is not None,
        sugar_w=float(args.sugar_w),
        extra_meta={
            "method": args.method,
            "projector": args.projector,
            "temperature": float(args.temperature),
            "num_steps": int(args.num_steps),
            "step_size": float(args.step_size),
            "use_fitness": bool(args.use_fitness),
            "use_projector": bool(args.use_projector),
            "latent_normalization": normalize_latent_norm_name(args.latent_normalization),
            "oae_latent_dim": int(args.latent_dim),
            "oae_recon_w": float(args.recon_w),
        },
    )
    print(f"wrote {path}")
    return path


def run_denovo(args, device: str) -> str:
    raise NotImplementedError("method=denovo is Phase 3 (not implemented yet).")


def run_guided(args, device: str) -> str:
    raise NotImplementedError("method=guided is Phase 3 (not implemented yet).")


def run_simple_opt(args, device: str) -> str:
    raise NotImplementedError("method=simple_opt is Phase 2 (not implemented yet).")


METHOD_FNS = {
    "rnagenscape": run_rnagenscape,
    "denovo": run_denovo,
    "guided": run_guided,
    "simple_opt": run_simple_opt,
}


def parse_args():
    p = argparse.ArgumentParser(description="Run sequence generation (RNAGenScape / baselines).")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--method", type=str, required=True, choices=METHODS)
    p.add_argument("--model", type=str, required=True, help="Results folder name (e.g. OAE).")
    p.add_argument("--experiment", type=str, required=True, help="Experiment id (no seed).")
    p.add_argument("--seed", type=int, default=1, help="Training seed; loads OAE/projector under d*_recon*/seed_{seed}/.")
    p.add_argument("--subsample_seed", type=int, default=42, help="RNG for start-pool subsample.")
    p.add_argument("--direction", type=float, default=1.0, help="+1 maximize / -1 minimize property.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)

    # OAE ablation folder (must match trained checkpoint)
    p.add_argument("--latent_dim", type=int, default=128, help="OAE latent dim; path OAE/d{latent}_recon{w}/.")
    p.add_argument("--recon_w", type=float, default=5.0, help="OAE recon weight; path OAE/d{latent}_recon{w}/.")

    # RNAGenScape / projector
    p.add_argument("--projector", type=str, default="dae", choices=["dae", "knn"])
    p.add_argument("--sugar_w", type=float, default=0.0)
    p.add_argument("--latent_normalization", type=str, default="none", choices=["none", "normal", "minmax"])
    p.add_argument("--knn_k", type=int, default=1)

    # Langevin
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--step_size", type=float, default=5e-3)
    p.add_argument("--temperature", type=float, default=1e-3)
    p.add_argument("--step_size_rescale", type=float, default=None)
    p.add_argument("--use_fitness", action="store_true", default=True)
    p.add_argument("--no_fitness", action="store_false", dest="use_fitness")
    p.add_argument("--use_projector", action="store_true", default=True)
    p.add_argument("--no_projector", action="store_false", dest="use_projector")
    p.add_argument("--projector_iters", type=int, default=1)
    p.add_argument("--annealed", action="store_true", default=False)
    p.add_argument("--save_trajectories", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(f"device={device} method={args.method} model={args.model} experiment={args.experiment}")
    path = METHOD_FNS[args.method](args, device)
    # Quick load check for rnagenscape.
    if args.method == "rnagenscape":
        art = load_generation_artifact(path)
        print(
            f"artifact OK new={tuple(art['new_sequences'].shape)} "
            f"starts={tuple(art['sampled_X'].shape)} direction={art['direction']}"
        )


if __name__ == "__main__":
    main()
