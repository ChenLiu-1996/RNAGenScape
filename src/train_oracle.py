"""Train a property oracle (per-dataset fine-tune).

Example:
  python src/train_oracle.py --dataset OpenVaccine --oracle UTRLM

Saves ``results/<dataset>/<oracle>/model.pt`` (and ``label_stats.json``).
Only trainable architectures are allowed (never UTRLM_TE / UTRLM_MRL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Allow `python src/train_oracle.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from utils.metrics import VOCAB_SIZE
from utils.oracle import TRAINABLE_ORACLES, oracle_checkpoint_path, resolve_device
from utils.training_utils import EarlyStopping, LinearWarmupCosineAnnealingLR, seed_everything


# Batch representation expected by each trainable oracle.
ORACLE_REPRESENTATION: Dict[str, str] = {
    "UTRLM": "string",
}


def build_oracle(
    oracle: str,
    *,
    device: str,
    seq_len: int,
    freeze_backbone: bool = False,
):
    if oracle == "UTRLM":
        from models.utrlm import UTRLM

        return UTRLM(
            device=device,
            seq_len=seq_len,
            vocab_size=VOCAB_SIZE,
            latent_dim=128,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(
        f"Unsupported trainable oracle '{oracle}'. "
        f"Choose from {sorted(TRAINABLE_ORACLES)}"
    )


def _move_batch(batch, device: str, representation: str):
    x, y = batch
    y = y.to(device)
    if representation == "string":
        # Default collate yields a tuple/list of strings.
        return list(x), y
    return x.to(device), y


@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    device: str,
    representation: str,
) -> Tuple[float, float, float]:
    """Return mean MSE, Pearson r, Spearman r on a loader."""
    model.eval()
    ys: List[float] = []
    preds: List[float] = []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x, y = _move_batch(batch, device, representation)
        y_pred = model(x)
        total_loss += model.regression_loss(y_pred, y).item()
        n_batches += 1
        ys.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        preds.extend(y_pred.detach().cpu().numpy().reshape(-1).tolist())
    mse = total_loss / max(n_batches, 1)
    if len(ys) < 2:
        return mse, float("nan"), float("nan")
    pearson = float(pearsonr(ys, preds)[0])
    spearman = float(spearmanr(ys, preds).correlation)
    return mse, pearson, spearman


def train(
    model,
    train_loader,
    val_loader,
    *,
    device: str,
    representation: str,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    max_epochs: int,
    patience: int,
    ckpt_path: str,
    log_path: str,
) -> Dict[str, float]:
    """Train until early stop on val Spearman. Saves best ``model.pt``."""
    model.to(device)
    stopper = EarlyStopping(mode="max", patience=patience)
    best_spearman = -float("inf")
    best_state = None
    history = {"best_val_spearman": best_spearman}

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(
        "epoch,lr,train_mse,train_pearson,train_spearman,val_mse,val_pearson,val_spearman\n"
    )
    log_f.flush()

    try:
        for epoch in tqdm(range(1, max_epochs + 1), desc="epochs"):
            model.train()
            train_ys: List[float] = []
            train_preds: List[float] = []
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x, y = _move_batch(batch, device, representation)
                y_pred = model(x)
                loss = model.regression_loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1
                train_ys.extend(y.detach().cpu().numpy().reshape(-1).tolist())
                train_preds.extend(y_pred.detach().cpu().numpy().reshape(-1).tolist())

            if lr_scheduler is not None:
                lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            train_mse = train_loss / max(n_batches, 1)
            train_pearson = float(pearsonr(train_ys, train_preds)[0]) if len(train_ys) > 1 else float("nan")
            train_spearman = (
                float(spearmanr(train_ys, train_preds).correlation) if len(train_ys) > 1 else float("nan")
            )
            val_mse, val_pearson, val_spearman = evaluate(
                model, val_loader, device=device, representation=representation
            )

            line = (
                f"{epoch},{lr:.8e},{train_mse:.6f},{train_pearson:.6f},{train_spearman:.6f},"
                f"{val_mse:.6f},{val_pearson:.6f},{val_spearman:.6f}"
            )
            print(
                f"epoch {epoch}/{max_epochs}  lr={lr:.2e}  "
                f"train mse={train_mse:.4f} P={train_pearson:.4f} S={train_spearman:.4f}  "
                f"val mse={val_mse:.4f} P={val_pearson:.4f} S={val_spearman:.4f}"
            )
            log_f.write(line + "\n")
            log_f.flush()

            if val_spearman == val_spearman and val_spearman > best_spearman:
                best_spearman = val_spearman
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_path)
                print(f"  saved best checkpoint -> {ckpt_path} (val Spearman={best_spearman:.4f})")

            if stopper.step(val_spearman if val_spearman == val_spearman else -float("inf")):
                print(f"early stopping at epoch {epoch} (patience={patience})")
                break
    finally:
        log_f.close()

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    history["best_val_spearman"] = best_spearman
    return history


def parse_args():
    p = argparse.ArgumentParser(description="Train a property oracle.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--oracle", type=str, required=True, choices=sorted(TRAINABLE_ORACLES))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="Early stop patience on val Spearman.")
    p.add_argument("--label_norm", type=str, default="normal", choices=["none", "normal", "minmax"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze UTR-LM backbone (linear probe only).")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    print(f"device={device}")

    if args.oracle not in TRAINABLE_ORACLES:
        raise ValueError(
            f"Oracle '{args.oracle}' is not trainable. "
            f"Frozen validators (UTRLM_TE / UTRLM_MRL) never enter train_oracle.py."
        )
    representation = ORACLE_REPRESENTATION[args.oracle]
    seq_len = int(DATASET_CONFIG[args.dataset]["seq_len"])

    train_loader, val_loader, test_loader, info = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        representation=representation,
        label_norm=args.label_norm,
        num_workers=args.num_workers,
    )
    print(
        f"dataset={args.dataset} oracle={args.oracle} "
        f"n_train={info.n_train} n_val={info.n_val} n_test={info.n_test} "
        f"representation={representation} label_norm={args.label_norm}"
    )

    model = build_oracle(
        args.oracle,
        device=device,
        seq_len=seq_len,
        freeze_backbone=args.freeze_backbone,
    )
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
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

    ckpt_path = oracle_checkpoint_path(args.dataset, args.oracle)
    out_dir = os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    stats_path = os.path.join(out_dir, "label_stats.json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **info.label_transform.as_dict(),
                "label_norm": args.label_norm,
                "dataset": args.dataset,
                "oracle": args.oracle,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"wrote label stats -> {stats_path}")

    train(
        model,
        train_loader,
        val_loader,
        device=device,
        representation=representation,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epochs=args.max_epochs,
        patience=args.patience,
        ckpt_path=ckpt_path,
        log_path=log_path,
    )

    test_mse, test_pearson, test_spearman = evaluate(
        model, test_loader, device=device, representation=representation
    )
    print(
        f"test mse={test_mse:.4f} pearson={test_pearson:.4f} spearman={test_spearman:.4f}"
    )
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
