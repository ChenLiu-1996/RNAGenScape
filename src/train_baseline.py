"""Train comparison baselines (DiffAb, IgLM, NOS_C, NOS_D, gg_dWJS, EM, MPGD).

Example:
  python src/train_baseline.py --dataset OpenVaccine --model DiffAb --seed 1

Saves ``results/<dataset>/<model>/seed_{seed}/model.pt`` (plus ``label_stats.json``,
``train_meta.json``, ``train.log``).

Optimization matches OAE training: AdamW + linear-warmup cosine annealing,
early stop on val combined loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Allow `python src/train_baseline.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from comparisons import DiffAb, EM, gg_dWJS, IgLM, MPGD, NOS_C, NOS_D
from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from utils.metrics import VOCAB_SIZE, to_token_ids
from utils.oracle import resolve_device
from utils.results import baseline_checkpoint_path
from utils.training_utils import EarlyStopping, LinearWarmupCosineAnnealingLR, seed_everything

BASELINE_MODELS = ("DiffAb", "IgLM", "NOS_C", "NOS_D", "gg_dWJS", "EM", "MPGD")
IGLM_SPECIAL_TOKENS = 3  # CLS, SEP, MASK

# Generative term name per model (property head is always ``property_mse``).
GEN_LOSS_NAME = {
    "DiffAb": "diffusion_kl",
    "IgLM": "infill_lm",
    "NOS_C": "noise_mse",
    "NOS_D": "masked_ce",
    "gg_dWJS": "denoise_mse",
    "EM": "energy_flow",
    "MPGD": "diffusion_mse",
}
PROP_LOSS_NAME = "property_mse"


def build_model(model_name: str, *, seq_len: int, device: str) -> torch.nn.Module:
    if model_name == "DiffAb":
        return DiffAb(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_properties=1,
            device=device,
        )
    if model_name == "IgLM":
        return IgLM(
            vocab_size=VOCAB_SIZE + IGLM_SPECIAL_TOKENS,
            seq_len=seq_len,
            n_embd=128,
            n_layer=2,
            n_head=4,
            num_properties=1,
            device=device,
        )
    if model_name == "NOS_C":
        return NOS_C(
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            num_properties=1,
            device=device,
        )
    if model_name == "NOS_D":
        return NOS_D(
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_properties=1,
            device=device,
        )
    if model_name == "gg_dWJS":
        return gg_dWJS(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            sigma=1.0,
            num_properties=1,
            device=device,
        )
    if model_name == "EM":
        return EM(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_properties=1,
            device=device,
        )
    if model_name == "MPGD":
        return MPGD(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_properties=1,
            device=device,
        )
    raise ValueError(f"Unknown baseline model '{model_name}'. Choose from {BASELINE_MODELS}.")


def _pad_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Bool mask with True on non-pad positions."""
    return token_ids != 0


def _batch_loss(
    model_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    recon_w: float,
    property_w: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total_loss, gen_loss, property_mse, y_hat [B]).

    ``gen_loss`` is the model-specific generative term; see ``GEN_LOSS_NAME``.
    """
    token_ids = to_token_ids(x)
    mask = _pad_mask(token_ids)
    targets = y.float().view(-1)

    if model_name == "DiffAb":
        losses, y_hat = model.compute_loss(token_ids, targets=targets, mask=mask)
        recon = losses["recon_loss"]
        prop = losses["property_loss"]
        total = recon_w * recon + property_w * prop
        return total, recon, prop, y_hat.view(-1)

    if model_name in ("NOS_C", "NOS_D", "gg_dWJS", "EM", "MPGD"):
        total, recon, prop, y_hat = model.compute_loss(
            token_ids,
            targets=targets.unsqueeze(-1),
            mask=mask,
            recon_weight=recon_w,
            property_weight=property_w,
        )
        return total, recon, prop, y_hat.view(-1)

    if model_name == "IgLM":
        seq_len = token_ids.shape[1]
        # Random contiguous span up to 15% of length (matches mRNA-translation trainer).
        mask_start = int(np.random.randint(0, max(seq_len, 1)))
        mask_len = int(np.random.randint(1, max(2, int(seq_len * 0.15) + 1)))
        mask_end = min(mask_start + mask_len, max(seq_len - 1, 1))
        total, recon, prop, y_hat = model.compute_loss(
            token_ids,
            targets,
            mask_start,
            mask_end,
            recon_weight=recon_w,
            property_weight=property_w,
        )
        return total, recon, prop, y_hat.view(-1)

    raise ValueError(f"Unknown baseline model '{model_name}'.")


@torch.no_grad()
def evaluate(
    model_name: str,
    model: torch.nn.Module,
    loader,
    *,
    device: str,
    recon_w: float,
    property_w: float,
) -> Dict[str, float]:
    model.eval()
    y_true: List[float] = []
    preds: List[float] = []
    total_loss = 0.0
    total_recon = 0.0
    total_prop = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss, recon, prop, y_hat = _batch_loss(
            model_name, model, x, y, recon_w=recon_w, property_w=property_w
        )
        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_prop += float(prop.item())
        n_batches += 1
        y_true.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        preds.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())

    n = max(n_batches, 1)
    pearson = float(pearsonr(y_true, preds)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(spearmanr(y_true, preds).correlation) if len(y_true) > 1 else float("nan")
    gen_name = GEN_LOSS_NAME[model_name]
    return {
        "loss": total_loss / n,
        gen_name: total_recon / n,
        PROP_LOSS_NAME: total_prop / n,
        "pearson": pearson,
        "spearman": spearman,
    }


def train(
    model_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    device: str,
    recon_w: float,
    property_w: float,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    max_epochs: int,
    patience: int,
    ckpt_path: str,
    log_path: str,
) -> Dict[str, float]:
    """Train until early stop on val combined loss. Saves best ``model.pt``."""
    model.to(device)
    stopper = EarlyStopping(mode="min", patience=patience)
    best_val_loss = float("inf")
    best_state = None
    gen_name = GEN_LOSS_NAME[model_name]
    prop_name = PROP_LOSS_NAME

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(
        f"epoch,lr,train_loss,train_{gen_name},train_{prop_name},train_pearson,train_spearman,"
        f"val_loss,val_{gen_name},val_{prop_name},val_pearson,val_spearman\n"
    )
    log_f.flush()

    try:
        for epoch in tqdm(range(1, max_epochs + 1), desc="epochs"):
            model.train()
            train_y_true: List[float] = []
            train_preds: List[float] = []
            train_loss = 0.0
            train_gen = 0.0
            train_prop = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, gen, prop, y_hat = _batch_loss(
                    model_name, model, x, y, recon_w=recon_w, property_w=property_w
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                train_gen += float(gen.item())
                train_prop += float(prop.item())
                n_batches += 1
                train_y_true.extend(y.detach().cpu().numpy().reshape(-1).tolist())
                train_preds.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())

            if lr_scheduler is not None:
                lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            n = max(n_batches, 1)
            train_pearson = (
                float(pearsonr(train_y_true, train_preds)[0])
                if len(train_y_true) > 1
                else float("nan")
            )
            train_spearman = (
                float(spearmanr(train_y_true, train_preds).correlation)
                if len(train_y_true) > 1
                else float("nan")
            )
            val = evaluate(
                model_name,
                model,
                val_loader,
                device=device,
                recon_w=recon_w,
                property_w=property_w,
            )

            line = (
                f"{epoch},{lr:.8e},"
                f"{train_loss / n:.6f},{train_gen / n:.6f},{train_prop / n:.6f},"
                f"{train_pearson:.6f},{train_spearman:.6f},"
                f"{val['loss']:.6f},{val[gen_name]:.6f},{val[prop_name]:.6f},"
                f"{val['pearson']:.6f},{val['spearman']:.6f}"
            )
            print(
                f"epoch {epoch}/{max_epochs}  lr={lr:.2e}  "
                f"train loss={train_loss / n:.4f} {gen_name}={train_gen / n:.4f} "
                f"{prop_name}={train_prop / n:.4f} "
                f"pearson={train_pearson:.4f} spearman={train_spearman:.4f}  "
                f"val loss={val['loss']:.4f} {gen_name}={val[gen_name]:.4f} "
                f"{prop_name}={val[prop_name]:.4f} "
                f"pearson={val['pearson']:.4f} spearman={val['spearman']:.4f}"
            )
            log_f.write(line + "\n")
            log_f.flush()

            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_path)
                print(f"  saved best checkpoint -> {ckpt_path} (val loss={best_val_loss:.4f})")

            if stopper.step(val["loss"]):
                print(f"early stopping at epoch {epoch} (patience={patience})")
                break
    finally:
        log_f.close()

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    return {"best_val_loss": best_val_loss}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DiffAb / IgLM / NOS_C / NOS_D / gg_dWJS / EM / MPGD baselines."
    )
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--model", type=str, required=True, choices=list(BASELINE_MODELS))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_w", type=float, default=1.0, help="Weight on generative / reconstruction loss.")
    p.add_argument("--property_w", type=float, default=1.0, help="Weight on property MSE loss.")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="Early stop patience on val loss.")
    p.add_argument("--label_norm", type=str, default="normal", choices=["none", "normal", "minmax"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    print(f"device={device}")

    seq_len = int(DATASET_CONFIG[args.dataset]["seq_len"])
    train_loader, val_loader, test_loader, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation="one_hot",
        label_norm=args.label_norm,
        num_workers=args.num_workers,
    )
    print(
        f"dataset={args.dataset} model={args.model} seed={args.seed} "
        f"n_train={info.n_train} n_val={info.n_val} n_test={info.n_test} "
        f"label_norm={args.label_norm} recon_w={args.recon_w} property_w={args.property_w}"
    )

    model = build_model(args.model, seq_len=seq_len, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_epochs = max(1, args.max_epochs // 10)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.lr * 0.01,
        eta_min=args.lr * 0.1,
    )
    print(
        f"optimizer=AdamW lr={args.lr}  "
        f"scheduler=LinearWarmupCosineAnnealingLR "
        f"warmup_epochs={warmup_epochs} start={args.lr * 0.01:.2e} end={args.lr * 0.1:.2e}"
    )

    ckpt_path = baseline_checkpoint_path(args.dataset, args.model, args.seed)
    out_dir = os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    stats_path = os.path.join(out_dir, "label_stats.json")
    meta_path = os.path.join(out_dir, "train_meta.json")

    gen_name = GEN_LOSS_NAME[args.model]
    meta: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "seq_len": seq_len,
        "recon_w": args.recon_w,
        "property_w": args.property_w,
        "gen_loss_name": gen_name,
        "property_loss_name": PROP_LOSS_NAME,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "label_norm": args.label_norm,
        "vocab_size": VOCAB_SIZE if args.model != "IgLM" else VOCAB_SIZE + IGLM_SPECIAL_TOKENS,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **info.label_transform.as_dict(),
                "label_norm": args.label_norm,
                "dataset": args.dataset,
                "model": args.model,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote label stats -> {stats_path}")
    print(f"wrote train meta  -> {meta_path}")

    train(
        args.model,
        model,
        train_loader,
        val_loader,
        device=device,
        recon_w=args.recon_w,
        property_w=args.property_w,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epochs=args.max_epochs,
        patience=args.patience,
        ckpt_path=ckpt_path,
        log_path=log_path,
    )

    test = evaluate(
        args.model,
        model,
        test_loader,
        device=device,
        recon_w=args.recon_w,
        property_w=args.property_w,
    )
    print(
        f"test loss={test['loss']:.4f} {gen_name}={test[gen_name]:.4f} "
        f"{PROP_LOSS_NAME}={test[PROP_LOSS_NAME]:.4f} "
        f"pearson={test['pearson']:.4f} spearman={test['spearman']:.4f}"
    )
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
