"""Build / train a manifold projector (DAE or kNN support artifacts).

Example:
  python src/train_manifold_projector.py --dataset OpenVaccine --projector dae --seed 1
  python src/train_manifold_projector.py --dataset OpenVaccine --projector knn --seed 1

* ``dae``: train ``ManifoldProjectorDAE`` on OAE train latents; save
  ``.../OAE/d{latent}_recon{w}/seed_{seed}/manifold_projector_dae_sugar{w}/model_latentnorm_{norm}.pt``
* ``knn``: **not trainable**. Only encodes the training set with the OAE and
  caches those latents as
  ``.../seed_{seed}/manifold_projector_knn_sugar{w}/latent_trainset.pt``. Choose ``k`` later
  at inference when constructing ``ManifoldProjectorKNN``.

Requires a trained OAE at
``results/<dataset>/OAE/d{latent}_recon{w}/seed_{seed}/model.pt``.
SUGAR augmentation (``sugar_w > 0``) is reserved for a later port.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch

# Allow `python src/train_manifold_projector.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from modules.manifold_projector_dae import (
    load_manifold_projector_dae,
    train_manifold_projector_dae,
)
from modules.oae import OAE, load_oae
from utils.metrics import VOCAB_SIZE
from utils.oracle import resolve_device
from utils.results import (
    latent_trainset_path,
    manifold_projector_dae_checkpoint_path,
    normalize_latent_norm_name,
    oae_checkpoint_path,
)
from utils.training_utils import seed_everything

PROJECTOR_CHOICES = ("dae", "knn")
DEFAULT_HIDDEN_DIMS = (32, 16, 32)
DEFAULT_NOISE_LEVELS = (0.5, 0.2, 0.1)


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


@torch.no_grad()
def encode_train_latents(
    oae: OAE,
    loader,
    *,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Encode the training set; return ``[N, D]`` and latent train stats."""
    latent_chunks = []
    for x, _y in loader:
        latent_chunks.append(oae.encode(x.to(device)).detach().cpu())
    latents = torch.cat(latent_chunks, dim=0)
    stats = {
        "latent_mean": float(latents.mean().item()),
        "latent_std": float(latents.std().item()),
        "latent_min": float(latents.min().item()),
        "latent_max": float(latents.max().item()),
    }
    return latents, stats


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
    raise ValueError(f"Unknown latent_normalization '{mode}'. Use none|normal|minmax.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DAE projector, or cache train latents for kNN (not trainable)."
    )
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--projector", type=str, required=True, choices=PROJECTOR_CHOICES)
    p.add_argument("--latent_dim", type=int, default=128, help="Must match OAE folder d{latent}_recon{w}.")
    p.add_argument("--recon_w", type=float, default=5.0, help="Must match OAE folder d{latent}_recon{w}.")
    p.add_argument("--sugar_w", type=float, default=0.0, help="SUGAR upsample weight (0 = off).")
    p.add_argument("--latent_normalization", type=str, default="none", choices=["none", "normal", "minmax"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128, help="OAE encode + DAE train batch size.")
    p.add_argument("--num_workers", type=int, default=0)

    # DAE-only
    p.add_argument("--dae_epochs", type=int, default=100, help="Epochs per noise stage.")
    p.add_argument("--dae_lr", type=float, default=1e-3)
    p.add_argument("--dae_patience", type=int, default=20)
    p.add_argument("--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    p.add_argument("--hidden_dims", type=int, nargs="+", default=list(DEFAULT_HIDDEN_DIMS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    print(f"device={device}")

    if args.sugar_w > 0.0:
        raise NotImplementedError(
            "SUGAR augmentation is not ported yet. Use --sugar_w 0.0 for now."
        )

    oae = load_oae_model(
        args.dataset,
        args.seed,
        device,
        latent_dim=args.latent_dim,
        recon_w=args.recon_w,
    )
    train_loader, _val_loader, _test_loader, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=args.num_workers,
    )
    print(
        f"dataset={args.dataset} projector={args.projector} seed={args.seed} "
        f"latent_dim={args.latent_dim} recon_w={args.recon_w} "
        f"n_train={info.n_train} sugar_w={args.sugar_w} "
        f"latent_normalization={args.latent_normalization}"
    )

    if args.projector == "knn":
        print(
            "NOTE: kNN is not a trainable projector. "
            "This step only encodes the OAE training-set latents and caches them "
            "for use at inference (choose k when constructing ManifoldProjectorKNN)."
        )

    latents, latent_stats = encode_train_latents(oae, train_loader, device=device)
    latents = apply_latent_normalization(
        latents, mode=args.latent_normalization, stats=latent_stats
    )
    print(f"train latents: shape={tuple(latents.shape)} stats={latent_stats}")

    meta = {
        "dataset": args.dataset,
        "projector": args.projector,
        "sugar_w": args.sugar_w,
        "latent_normalization": normalize_latent_norm_name(args.latent_normalization),
        "seed": args.seed,
        "oae_latent_dim": int(args.latent_dim),
        "oae_recon_w": float(args.recon_w),
        "latent_dim": int(latents.shape[1]),
        "n_manifold": int(latents.shape[0]),
        **latent_stats,
    }

    if args.projector == "dae":
        model, _histories, metrics = train_manifold_projector_dae(
            latents,
            noise_levels=args.noise_levels,
            hidden_dims=args.hidden_dims,
            num_epochs=args.dae_epochs,
            batch_size=args.batch_size,
            learning_rate=args.dae_lr,
            early_stop_patience=args.dae_patience,
            device=device,
            random_state=args.seed,
        )
        ckpt = manifold_projector_dae_checkpoint_path(
            args.dataset,
            args.seed,
            latent_dim=args.latent_dim,
            recon_w=args.recon_w,
            sugar_w=args.sugar_w,
            latent_normalization=args.latent_normalization,
        )
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        meta.update(
            {
                "hidden_dims": list(args.hidden_dims),
                "noise_levels": list(args.noise_levels),
                "dae_epochs": args.dae_epochs,
                "dae_lr": args.dae_lr,
                **metrics,
            }
        )
        with open(os.path.join(os.path.dirname(ckpt), "projector_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"saved ManifoldProjectorDAE -> {ckpt}")

        reloaded = load_manifold_projector_dae(
            ckpt,
            input_dim=latents.shape[1],
            hidden_dims=args.hidden_dims,
            device=device,
        )
        with torch.no_grad():
            err = (reloaded(latents[:8].to(device)) - model(latents[:8].to(device))).abs().max()
        print(f"reload max abs err={float(err):.2e}")

    else:
        # Cache train latents only; k is chosen later at inference.
        path = latent_trainset_path(
            args.dataset,
            args.seed,
            latent_dim=args.latent_dim,
            recon_w=args.recon_w,
            sugar_w=args.sugar_w,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "latents": latents.detach().cpu().float(),
                "latent_normalization": meta["latent_normalization"],
                **{key: latent_stats[key] for key in latent_stats},
            },
            path,
        )
        with open(os.path.join(os.path.dirname(path), "projector_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"saved train latents -> {path} "
            f"(shape={tuple(latents.shape)}; k is not stored - set it at inference)"
        )


if __name__ == "__main__":
    main()
