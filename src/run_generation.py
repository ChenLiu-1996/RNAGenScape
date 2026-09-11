"""Unified generation entrypoint.

Methods:
  * ``rnagenscape`` -- OAE + manifold Langevin + projector (RNAGenScape).
  * ``guided`` -- property-guided optimization of test starts (all baselines).
  * ``denovo`` -- unconditional generation (not implemented yet).

Examples:
  python src/run_generation.py \\
    --dataset OpenVaccine --method rnagenscape --model OAE \\
    --experiment pos_samehyper_sugar0e0 --seed 1 --direction 1 \\
    --projector dae --sugar_w 0.0 --num_steps 100 --step_size 5e-3 --temperature 1e-3

  python src/run_generation.py \\
    --dataset OpenVaccine --method guided --model DiffAb \\
    --experiment pos_guided --seed 1 --direction 1
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
from train_baseline import BASELINE_MODELS, build_model
from utils.metrics import VOCAB_SIZE, subsample_indices, to_token_ids
from utils.oracle import resolve_device
from utils.results import (
    baseline_checkpoint_path,
    experiment_dir,
    latent_trainset_path,
    load_generation_artifact,
    manifold_projector_dae_checkpoint_path,
    normalize_latent_norm_name,
    oae_checkpoint_path,
    save_generation_artifact,
)
from utils.starts_cache import (
    load_starts_cache,
    save_starts_cache,
    starts_content_hash,
)
from utils.training_utils import seed_everything
from utils.timing import CudaTimer, ms_per_sample

METHODS = ("rnagenscape", "guided", "denovo")
DEFAULT_DAE_HIDDEN_DIMS = (32, 16, 32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def direction_to_target(direction: float) -> str:
    return "increase" if float(direction) > 0 else "decrease"

def pad_mask(token_ids: torch.Tensor) -> torch.Tensor:
    return token_ids != 0


@torch.no_grad()
def collect_tokens_and_labels(loader, *, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (token_ids [N,L], labels [N]) from a dataloader."""
    tokens: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        tokens.append(to_token_ids(x).detach().cpu())
        labels.append(y.detach().cpu().float().reshape(-1))
    return torch.cat(tokens, dim=0), torch.cat(labels, dim=0)


def load_json_label_stats(path: str) -> Dict[str, float]:
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


def resolve_start_indices(
    pool_size: int,
    *,
    max_starts: int,
    subsample_seed: int,
    starts_cache: str | None,
) -> Tuple[np.ndarray, bool]:
    """Return ``(indices, from_cache)`` for start-pool selection."""
    cache_path = (starts_cache or "").strip()
    if cache_path and os.path.isfile(cache_path):
        _sampled_x, _sampled_y, sampled_indices = load_starts_cache(cache_path)
        indices = np.asarray(sampled_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"Empty starts cache: {cache_path}")
        if int(indices.max()) >= pool_size or int(indices.min()) < 0:
            raise ValueError(
                f"Starts cache indices out of range for pool_size={pool_size}: {cache_path}"
            )
        return indices, True
    seed_everything(subsample_seed)
    return subsample_indices(pool_size, max_starts, subsample_seed), False


def maybe_save_starts_cache(
    starts_cache: str | None,
    *,
    sampled_x: torch.Tensor,
    sampled_y: torch.Tensor,
    sampled_indices: np.ndarray,
    from_cache: bool,
) -> None:
    cache_path = (starts_cache or "").strip()
    if (not cache_path) or from_cache:
        return
    save_starts_cache(cache_path, sampled_x, sampled_y, sampled_indices)
    print(f"wrote starts_cache={cache_path}")


# ---------------------------------------------------------------------------
# RNAGenScape (OAE + Langevin + projector)
# ---------------------------------------------------------------------------


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
    return load_json_label_stats(path)


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
    dae_tag: Optional[str] = None,
):
    if projector == "dae":
        path = manifold_projector_dae_checkpoint_path(
            dataset,
            seed,
            latent_dim=oae_latent_dim,
            recon_w=recon_w,
            sugar_w=sugar_w,
            dae_tag=dae_tag,
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


def run_rnagenscape(args, device: str) -> str:
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

    indices, from_cache = resolve_start_indices(
        pool_latent.shape[0],
        max_starts=args.max_starts,
        subsample_seed=args.subsample_seed,
        starts_cache=args.starts_cache,
    )
    indices_t = torch.from_numpy(indices.astype(np.int64))
    sampled_latent = pool_latent[indices_t]
    sampled_y = pool_y[indices_t]
    sampled_x = pool_x[indices_t]
    sampled_indices = indices
    maybe_save_starts_cache(
        args.starts_cache,
        sampled_x=sampled_x,
        sampled_y=sampled_y,
        sampled_indices=sampled_indices,
        from_cache=from_cache,
    )
    print(
        f"starts_hash={starts_content_hash(sampled_x)} n_starts={sampled_x.shape[0]} "
        f"from_cache={from_cache} start_pool={pool_latent.shape[0]} "
        f"max_starts={args.max_starts} subsample_seed={args.subsample_seed}"
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
        dae_tag=(str(args.dae_tag).strip() or None),
    )

    seed_everything(args.seed)
    fitness_fn = (lambda z: oae.regress(z)) if args.use_fitness else None

    print(
        f"Langevin steps={args.num_steps} step_size={args.step_size} "
        f"temperature={args.temperature} use_projector={args.use_projector} "
        f"direction={args.direction}"
    )
    with CudaTimer() as timer:
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
            return_history=bool(args.save_trajectories),
        )
        new_sequences = decode_latents(oae, z_gen, device=device, batch_size=args.batch_size)
    n_gen = int(new_sequences.shape[0])
    gen_ms = ms_per_sample(timer.elapsed, n_gen)
    print(
        f"generation timing: {timer.elapsed:.3f}s for {n_gen} samples "
        f"= {gen_ms:.3f} ms/sample",
        flush=True,
    )

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
            "max_starts": int(args.max_starts),
            "elapsed_seconds": float(timer.elapsed),
            "ms_per_sample": float(gen_ms),
            "n_timed": n_gen,
        },
    )
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Baseline guided optimization (test starts -> model.optimize)
# ---------------------------------------------------------------------------


def load_baseline_model(model_name: str, *, dataset: str, seed: int, device: str) -> torch.nn.Module:
    ckpt = baseline_checkpoint_path(dataset, model_name, seed)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Missing baseline checkpoint: {ckpt}. "
            f"Train with src/train_baseline.py --model {model_name} --seed {seed}."
        )
    seq_len = int(DATASET_CONFIG[dataset]["seq_len"])
    model = build_model(model_name, seq_len=seq_len, device=device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    if hasattr(model, "device"):
        model.device = torch.device(device)
    model.eval()
    return model


def load_baseline_label_stats(dataset: str, model_name: str, seed: int) -> Dict[str, float]:
    path = os.path.join(
        os.path.dirname(baseline_checkpoint_path(dataset, model_name, seed)),
        "label_stats.json",
    )
    return load_json_label_stats(path)


def optimize_baseline_batch(
    model_name: str,
    model: torch.nn.Module,
    starts: torch.Tensor,
    *,
    args,
    device: str,
    seed_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one batch of seed-started property optimization. Returns tokens ``[B, L]``."""
    starts = starts.to(device)
    mask = pad_mask(starts)
    target = direction_to_target(args.direction)

    if model_name == "DiffAb":
        traj = model.optimize(starts, device=device, target_direction=target, mask=mask)
        return traj[-1].detach().cpu()

    if model_name == "IgLM":
        return model.optimize(starts, target_direction=target).detach().cpu()

    if model_name == "NOS_C":
        return model.optimize(starts, target_direction=target, mask=mask).detach().cpu()

    if model_name == "NOS_D":
        return model.optimize(starts, target_direction=target, mask=mask).detach().cpu()

    if model_name == "gg_dWJS":
        return model.optimize(starts, target_direction=target, pad_mask=mask).detach().cpu()

    if model_name == "EM":
        return model.optimize(starts, target_direction=target, pad_mask=mask).detach().cpu()

    if model_name == "MPGD":
        if seed_labels is None:
            raise ValueError("MPGD optimize requires seed_labels for y* = y_seed + direction*y_delta.")
        return model.optimize(
            starts,
            seed_labels=seed_labels.to(device),
            target_direction=target,
            pad_mask=mask,
        ).detach().cpu()

    if model_name == "MFM":
        return model.optimize(starts, target_direction=target, pad_mask=mask).detach().cpu()

    if model_name == "PCD":
        return model.optimize(starts, target_direction=target, pad_mask=mask).detach().cpu()

    raise ValueError(f"No guided optimize path for model '{model_name}'.")


def run_guided(args, device: str) -> str:
    if args.model not in BASELINE_MODELS:
        raise ValueError(
            f"method=guided requires a baseline --model in {BASELINE_MODELS}; "
            f"got '{args.model}'. For OAE use --method rnagenscape."
        )

    model = load_baseline_model(
        args.model, dataset=args.dataset, seed=args.seed, device=device
    )
    label_stats = load_baseline_label_stats(args.dataset, args.model, args.seed)

    _train_loader, _val_loader, test_loader, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=args.num_workers,
    )
    print(
        f"dataset={args.dataset} n_train={info.n_train} n_test={info.n_test} "
        f"model={args.model} seed={args.seed} direction={args.direction}"
    )

    pool_x, pool_y = collect_tokens_and_labels(test_loader, device=device)
    indices, from_cache = resolve_start_indices(
        pool_x.shape[0],
        max_starts=args.max_starts,
        subsample_seed=args.subsample_seed,
        starts_cache=args.starts_cache,
    )
    indices_t = torch.from_numpy(indices.astype(np.int64))
    sampled_x = pool_x[indices_t]
    sampled_y = pool_y[indices_t]
    sampled_indices = indices
    maybe_save_starts_cache(
        args.starts_cache,
        sampled_x=sampled_x,
        sampled_y=sampled_y,
        sampled_indices=sampled_indices,
        from_cache=from_cache,
    )
    print(
        f"starts_hash={starts_content_hash(sampled_x)} n_starts={sampled_x.shape[0]} "
        f"from_cache={from_cache} start_pool={pool_x.shape[0]} "
        f"max_starts={args.max_starts} subsample_seed={args.subsample_seed}"
    )

    seed_everything(args.seed)
    outs: List[torch.Tensor] = []
    with CudaTimer() as timer:
        for i in range(0, sampled_x.shape[0], args.batch_size):
            batch = sampled_x[i : i + args.batch_size]
            batch_y = sampled_y[i : i + args.batch_size]
            outs.append(
                optimize_baseline_batch(
                    args.model,
                    model,
                    batch,
                    args=args,
                    device=device,
                    seed_labels=batch_y,
                )
            )
        new_sequences = torch.cat(outs, dim=0)
    assert new_sequences.shape[0] == sampled_x.shape[0]
    n_gen = int(new_sequences.shape[0])
    gen_ms = ms_per_sample(timer.elapsed, n_gen)
    print(
        f"generation timing: {timer.elapsed:.3f}s for {n_gen} samples "
        f"= {gen_ms:.3f} ms/sample",
        flush=True,
    )

    out_dir = os.path.join(
        experiment_dir(args.dataset, args.model, args.experiment),
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
        train_stats=label_stats,
        model_type=args.model,
        data=args.dataset,
        seed=args.seed,
        subsample_seed=args.subsample_seed,
        trajectories=None,
        trajectories_are_sequences=False,
        sugar_w=0.0,
        extra_meta={
            "method": args.method,
            "max_starts": int(args.max_starts),
            "elapsed_seconds": float(timer.elapsed),
            "ms_per_sample": float(gen_ms),
            "n_timed": n_gen,
        },
    )
    print(f"wrote {path}")
    return path


def run_denovo(args, device: str) -> str:
    del args, device
    raise NotImplementedError("method=denovo is not implemented yet.")


METHOD_FNS = {
    "rnagenscape": run_rnagenscape,
    "guided": run_guided,
    "denovo": run_denovo,
}


def parse_args():
    p = argparse.ArgumentParser(description="Run sequence generation (RNAGenScape / baselines).")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--method", type=str, required=True, choices=METHODS)
    p.add_argument("--model", type=str, required=True, help="Results folder name (OAE or a baseline).")
    p.add_argument("--experiment", type=str, required=True, help="Experiment id (no seed).")
    p.add_argument("--seed", type=int, default=1, help="Training seed; loads checkpoint under seed_{seed}/.")
    p.add_argument("--subsample_seed", type=int, default=42, help="RNG for start-pool subsample.")
    p.add_argument("--max_starts", type=int, default=1000, help="Max start sequences sampled from the test pool.")
    p.add_argument("--starts_cache", type=str, default="", help="Optional .pt path for shared starts cache.")
    p.add_argument("--direction", type=float, default=1.0, help="+1 maximize / -1 minimize property.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)

    # OAE ablation folder (rnagenscape only)
    p.add_argument("--latent_dim", type=int, default=128, help="OAE latent dim; path OAE/d{latent}_recon{w}/.")
    p.add_argument("--recon_w", type=float, default=5.0, help="OAE recon weight; path OAE/d{latent}_recon{w}_reg1e0/.")

    # RNAGenScape / projector
    p.add_argument("--projector", type=str, default="dae", choices=["dae", "knn"])
    p.add_argument("--sugar_w", type=float, default=0.0)
    p.add_argument("--dae_tag", type=str, default="", help="Optional DAE path tag (e.g. steps1). Empty loads manifold_projector_dae_sugar{w}/.")
    p.add_argument("--latent_normalization", type=str, default="none", choices=["none", "normal", "minmax"])
    p.add_argument("--knn_k", type=int, default=1)

    # Shared / Langevin / OAE step controls
    p.add_argument("--num_steps", type=int, default=100, help="Langevin steps (OAE).")
    p.add_argument("--step_size", type=float, default=5e-3, help="Langevin step size (OAE).")
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
    art = load_generation_artifact(path)
    print(
        f"artifact OK new={tuple(art['new_sequences'].shape)} "
        f"starts={tuple(art['sampled_X'].shape)} direction={art['direction']}"
    )


if __name__ == "__main__":
    main()
