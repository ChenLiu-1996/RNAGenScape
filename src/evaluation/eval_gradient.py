"""Evaluate property-gradient stability by manifold region (OAE latent space).

Regions:
  dense_manifold  — top 30% local density on the training latent cloud
  sparse_manifold — bottom 30% local density (still near manifold)
  off_manifold    — points displaced by 5× median NN noise, filtered far

Example:
  python src/evaluation/eval_gradient.py \\
    --dataset RibosomeLoading \\
    --seed 1 \\
    --latent_dim 128 \\
    --recon_w 5.0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow `python src/evaluation/eval_gradient.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG, make_dataloaders
from modules.degree import GraphDegreeDensityEstimator
from modules.oae import load_oae
from utils.metrics import VOCAB_SIZE, to_token_ids
from utils.oracle import resolve_device
from utils.results import oae_checkpoint_path, oae_seed_dir
from utils.training_utils import seed_everything


@torch.no_grad()
def _mask_self_indices(
    dists: torch.Tensor,
    exclude_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    if exclude_idx is None:
        return dists
    exclude_idx = exclude_idx.to(device=dists.device, dtype=torch.long)
    dists = dists.clone()
    dists[torch.arange(dists.shape[0], device=dists.device), exclude_idx] = float("inf")
    return dists


@torch.no_grad()
def knn_indices(
    query: torch.Tensor,
    ref: torch.Tensor,
    k: int,
    exclude_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dists = _mask_self_indices(torch.cdist(query, ref), exclude_idx)
    max_k = ref.shape[0] - (1 if exclude_idx is not None else 0)
    return torch.topk(dists, k=min(k, max(max_k, 1)), largest=False).indices


@torch.no_grad()
def knn_distances(
    query: torch.Tensor,
    ref: torch.Tensor,
    k: int = 1,
    exclude_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dists = _mask_self_indices(torch.cdist(query, ref), exclude_idx)
    max_k = ref.shape[0] - (1 if exclude_idx is not None else 0)
    return torch.topk(dists, k=min(k, max(max_k, 1)), largest=False).values[:, -1]


def gradient_norms(oae, z: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """||∇_z f(z)||_2 for ``oae.regress``."""
    norms = []
    oae.eval()
    for i in range(0, z.shape[0], batch_size):
        zb = z[i : i + batch_size].detach().clone().requires_grad_(True)
        pred = oae.regress(zb)
        grads = torch.autograd.grad(pred.sum(), zb, create_graph=False)[0]
        norms.append(grads.flatten(1).norm(dim=1).detach().cpu().numpy())
    return np.concatenate(norms, axis=0)


@torch.no_grad()
def neighbor_pred_variance(
    oae,
    z: torch.Tensor,
    ref: torch.Tensor,
    k: int,
    batch_size: int = 256,
    exclude_idx: Optional[torch.Tensor] = None,
) -> np.ndarray:
    vars_: List[np.ndarray] = []
    max_k = ref.shape[0] - (1 if exclude_idx is not None else 0)
    k = min(k, max(max_k, 1))
    oae.eval()
    for i in range(0, z.shape[0], batch_size):
        zb = z[i : i + batch_size]
        ex = None if exclude_idx is None else exclude_idx[i : i + batch_size]
        idx = knn_indices(zb, ref, k, exclude_idx=ex)
        neigh = ref[idx]
        bsz, kk, dim = neigh.shape
        preds = oae.regress(neigh.reshape(bsz * kk, dim)).reshape(bsz, kk)
        vars_.append(preds.var(dim=1).cpu().numpy())
    return np.concatenate(vars_, axis=0)


def summarize(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x))


@torch.no_grad()
def encode_train_latents(oae, loader, *, device: str, batch_size: int) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    for x, _y in loader:
        x = x.to(device)
        # Dataloader may already yield one-hot; encode accepts [B,L,V].
        if x.dim() == 2:
            x = torch.nn.functional.one_hot(
                to_token_ids(x).to(device), num_classes=VOCAB_SIZE
            ).float()
        latents.append(oae.encode(x).detach().cpu())
    return torch.cat(latents, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate gradient stability by manifold region.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIG.keys()))
    p.add_argument("--model", type=str, default="OAE", help="Must be OAE for latent analysis.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--recon_w", type=float, default=5.0)
    p.add_argument("--ref_size", type=int, default=10000)
    p.add_argument("--n_per_region", type=int, default=500)
    p.add_argument("--dense_q", type=float, default=0.7, help="Dense = density >= this quantile.")
    p.add_argument("--sparse_q", type=float, default=0.3, help="Sparse = density <= this quantile.")
    p.add_argument(
        "--off_noise_scale",
        type=float,
        default=5.0,
        help="Off-manifold noise in units of median kNN dist.",
    )
    p.add_argument("--nn_k", type=int, default=10)
    p.add_argument("--density_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.model != "OAE":
        raise ValueError("eval_gradient currently requires --model OAE")

    seed_everything(args.seed)
    device = resolve_device()
    print(f"device={device} dataset={args.dataset} seed={args.seed}")

    ckpt = oae_checkpoint_path(
        args.dataset, args.seed, latent_dim=args.latent_dim, recon_w=args.recon_w
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing OAE checkpoint: {ckpt}")

    seq_len = int(DATASET_CONFIG[args.dataset]["seq_len"])
    oae = load_oae(ckpt, device=device, seq_len=seq_len, vocab_size=VOCAB_SIZE)
    oae.eval()

    train_loader, _val, _test, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation="one_hot",
        label_norm="normal",
        num_workers=args.num_workers,
    )
    print(f"n_train={info.n_train}")

    latent_train = encode_train_latents(
        oae, train_loader, device=device, batch_size=args.batch_size
    ).float()
    print(f"latent_train: {tuple(latent_train.shape)}")

    rng = np.random.default_rng(args.seed)
    n_ref = min(args.ref_size, latent_train.shape[0])
    ref_idx = rng.choice(latent_train.shape[0], size=n_ref, replace=False)
    ref = latent_train[ref_idx].to(device)
    print(f"reference cloud: {tuple(ref.shape)}")

    density_est = GraphDegreeDensityEstimator(
        bandwidth_type="adaptive",
        k=args.density_k,
        boundary_correction=False,
        device=device,
    )
    density_est.fit(ref)
    dens = density_est.transform(ref).detach().cpu().numpy().reshape(-1)
    dens_q_lo = float(np.quantile(dens, args.sparse_q))
    dens_q_hi = float(np.quantile(dens, args.dense_q))

    ref_self = torch.arange(ref.shape[0], device=device)
    nn1 = knn_distances(ref, ref, k=1, exclude_idx=ref_self).detach().cpu().numpy()
    median_nn = float(np.median(nn1))
    print(
        f"density quantiles: sparse<={dens_q_lo:.4g} dense>={dens_q_hi:.4g}; "
        f"median_nn={median_nn:.4g}"
    )

    dense_pool_idx = np.where(dens >= dens_q_hi)[0]
    sparse_pool_idx = np.where(dens <= dens_q_lo)[0]
    n = args.n_per_region

    def sample_ref_indices(pool_idx: np.ndarray, n_sample: int) -> np.ndarray:
        if pool_idx.size == 0:
            raise RuntimeError("empty region pool; adjust density quantiles")
        return rng.choice(
            pool_idx, size=min(n_sample, pool_idx.size), replace=pool_idx.size < n_sample
        )

    dense_idx = sample_ref_indices(dense_pool_idx, n)
    sparse_idx = sample_ref_indices(sparse_pool_idx, n)
    dense_z = ref[dense_idx]
    sparse_z = ref[sparse_idx]
    dense_exclude = torch.as_tensor(dense_idx, device=device, dtype=torch.long)
    sparse_exclude = torch.as_tensor(sparse_idx, device=device, dtype=torch.long)

    noise = torch.randn(n * 4, ref.shape[1], device=device) * (
        args.off_noise_scale * median_nn
    )
    base_idx = sample_ref_indices(np.arange(ref.shape[0]), n * 4)
    cand = ref[base_idx] + noise
    cand_nn = knn_distances(cand, ref, k=1).detach().cpu().numpy()
    far_thresh = float(np.quantile(nn1, 0.95) * max(args.off_noise_scale * 0.5, 1.0))
    far_mask = cand_nn >= far_thresh
    if int(far_mask.sum()) < n:
        keep = np.argsort(cand_nn)[-n:]
    else:
        keep = np.where(far_mask)[0][:n]
    off_z = cand[keep]
    print(
        f"region sizes: dense={dense_z.shape[0]} sparse={sparse_z.shape[0]} "
        f"off={off_z.shape[0]} (off nn-dist mean={cand_nn[keep].mean():.4g}, "
        f"thresh={far_thresh:.4g})"
    )

    regions = {
        "dense_manifold": (dense_z, dense_exclude),
        "sparse_manifold": (sparse_z, sparse_exclude),
        "off_manifold": (off_z, None),
    }

    rows: List[Dict] = []
    for name, (z, exclude_idx) in regions.items():
        dens_z = density_est.transform(z).detach().cpu().numpy().reshape(-1)
        nn_z = knn_distances(z, ref, k=1, exclude_idx=exclude_idx).detach().cpu().numpy()
        g = gradient_norms(oae, z, batch_size=args.batch_size)
        v = neighbor_pred_variance(
            oae,
            z,
            ref,
            k=args.nn_k,
            batch_size=args.batch_size,
            exclude_idx=exclude_idx,
        )
        g_m, g_s = summarize(g)
        v_m, v_s = summarize(v)
        d_m, d_s = summarize(dens_z)
        nn_m, nn_s = summarize(nn_z)
        rows.append(
            {
                "region": name,
                "n": int(z.shape[0]),
                "density_mean": d_m,
                "density_std": d_s,
                "nn_dist_mean": nn_m,
                "nn_dist_std": nn_s,
                "grad_norm_mean": g_m,
                "grad_norm_std": g_s,
                "pred_var_mean": v_m,
                "pred_var_std": v_s,
            }
        )

    out_dir = os.path.join(
        oae_seed_dir(
            args.dataset,
            args.seed,
            latent_dim=args.latent_dim,
            recon_w=args.recon_w,
        ),
        "evaluation",
        "gradient",
    )
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "manifold_region_stability.tsv")
    cols = list(rows[0].keys())
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"Saved {tsv_path}")

    md_lines = [
        "| region | n | density (mean±std) | NN dist (mean±std) | "
        "||∇f|| (mean±std) | neighbor pred-var (mean±std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['region']} | {r['n']} | "
            f"{r['density_mean']:.4g}±{r['density_std']:.4g} | "
            f"{r['nn_dist_mean']:.4g}±{r['nn_dist_std']:.4g} | "
            f"{r['grad_norm_mean']:.4g}±{r['grad_norm_std']:.4g} | "
            f"{r['pred_var_mean']:.4g}±{r['pred_var_std']:.4g} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    md_path = os.path.join(out_dir, "manifold_region_stability.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved {md_path}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
