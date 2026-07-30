"""Train the Organized Autoencoder (OAE).

Example:
  python src/train_oae.py --dataset OpenVaccine --lr 1e-2 --recon_w 1.0

Saves ``results/<dataset>/OAE/model.pt`` (and ``label_stats.json``, ``train.log``).

Loss: ``MSE + recon_w * CE`` (regression weight fixed at 1; tune via ``lr``).
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

# Allow `python src/train_oae.py` from repo root.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from modules.oae import OAE
from utils.metrics import VOCAB_SIZE
from utils.oracle import resolve_device
from utils.results import oae_checkpoint_path
from utils.training_utils import EarlyStopping, LinearWarmupCosineAnnealingLR, seed_everything


def _combined_loss(
    model: OAE,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    recon_w: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss, recon_loss, regression_loss, y_hat, logits)."""
    z = model.encode(x)
    logits = model.decode(z)
    y_hat = model.regress(z)
    recon_loss = model.reconstruction_loss(logits, x.argmax(dim=-1))
    regression_loss = model.regression_loss(y_hat, y)
    loss = regression_loss + recon_w * recon_loss
    return loss, recon_loss, regression_loss, y_hat, logits


def _recon_accuracy(logits: torch.Tensor, x: torch.Tensor) -> Tuple[float, float]:
    """Token accuracy and exact-sequence accuracy."""
    pred = logits.argmax(dim=-1)
    target = x.argmax(dim=-1)
    token_acc = (pred == target).float().mean().item()
    seq_acc = (pred == target).all(dim=-1).float().mean().item()
    return token_acc, seq_acc


@torch.no_grad()
def evaluate(
    model: OAE,
    loader,
    *,
    device: str,
    recon_w: float,
) -> Dict[str, float]:
    """Return val metrics including combined loss, recon/regression, correlations, accuracy."""
    model.eval()
    y_true: List[float] = []
    preds: List[float] = []
    total_loss = 0.0
    total_recon = 0.0
    total_regression = 0.0
    total_token_acc = 0.0
    total_seq_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss, recon_loss, regression_loss, y_hat, logits = _combined_loss(
            model, x, y, recon_w=recon_w
        )
        token_acc, seq_acc = _recon_accuracy(logits, x)
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_regression += regression_loss.item()
        total_token_acc += token_acc
        total_seq_acc += seq_acc
        n_batches += 1
        y_true.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        preds.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())

    n = max(n_batches, 1)
    pearson = float(pearsonr(y_true, preds)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(spearmanr(y_true, preds).correlation) if len(y_true) > 1 else float("nan")
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "regression": total_regression / n,
        "pearson": pearson,
        "spearman": spearman,
        "token_acc": total_token_acc / n,
        "seq_acc": total_seq_acc / n,
    }


def train(
    model: OAE,
    train_loader,
    val_loader,
    *,
    device: str,
    recon_w: float,
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
    history = {"best_val_loss": best_val_loss}

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(
        "epoch,lr,train_loss,train_recon,train_regression,train_pearson,train_spearman,"
        "train_token_acc,train_seq_acc,"
        "val_loss,val_recon,val_regression,val_pearson,val_spearman,val_token_acc,val_seq_acc\n"
    )
    log_f.flush()

    try:
        for epoch in tqdm(range(1, max_epochs + 1), desc="epochs"):
            model.train()
            train_y_true: List[float] = []
            train_preds: List[float] = []
            train_loss = 0.0
            train_recon = 0.0
            train_regression = 0.0
            train_token_acc = 0.0
            train_seq_acc = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, recon_loss, regression_loss, y_hat, logits = _combined_loss(
                    model, x, y, recon_w=recon_w
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                token_acc, seq_acc = _recon_accuracy(logits.detach(), x)
                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_regression += regression_loss.item()
                train_token_acc += token_acc
                train_seq_acc += seq_acc
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
            val = evaluate(model, val_loader, device=device, recon_w=recon_w)

            line = (
                f"{epoch},{lr:.8e},"
                f"{train_loss / n:.6f},{train_recon / n:.6f},{train_regression / n:.6f},"
                f"{train_pearson:.6f},{train_spearman:.6f},"
                f"{train_token_acc / n:.6f},{train_seq_acc / n:.6f},"
                f"{val['loss']:.6f},{val['recon']:.6f},{val['regression']:.6f},"
                f"{val['pearson']:.6f},{val['spearman']:.6f},"
                f"{val['token_acc']:.6f},{val['seq_acc']:.6f}"
            )
            print(
                f"epoch {epoch}/{max_epochs}  lr={lr:.2e}  "
                f"train loss={train_loss / n:.4f} recon={train_recon / n:.4f} "
                f"regression={train_regression / n:.4f} "
                f"pearson={train_pearson:.4f} spearman={train_spearman:.4f}  "
                f"val loss={val['loss']:.4f} recon={val['recon']:.4f} "
                f"regression={val['regression']:.4f} "
                f"pearson={val['pearson']:.4f} spearman={val['spearman']:.4f} "
                f"token_acc={val['token_acc']:.3f} seq_acc={val['seq_acc']:.3f}"
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
    history["best_val_loss"] = best_val_loss
    return history


def parse_args():
    p = argparse.ArgumentParser(description="Train the Organized Autoencoder (OAE).")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--recon_w", type=float, default=1.0, help="Weight on reconstruction CE.")
    p.add_argument("--max_epochs", type=int, default=100)
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
        f"dataset={args.dataset} model=OAE "
        f"n_train={info.n_train} n_val={info.n_val} n_test={info.n_test} "
        f"representation=one_hot label_norm={args.label_norm} recon_w={args.recon_w}"
    )

    model = OAE(
        device=device,
        seq_len=seq_len,
        vocab_size=VOCAB_SIZE,
        latent_dim=320,
    )
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

    ckpt_path = oae_checkpoint_path(args.dataset)
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
                "model": "OAE",
                "recon_w": args.recon_w,
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
        recon_w=args.recon_w,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epochs=args.max_epochs,
        patience=args.patience,
        ckpt_path=ckpt_path,
        log_path=log_path,
    )

    test = evaluate(model, test_loader, device=device, recon_w=args.recon_w)
    print(
        f"test loss={test['loss']:.4f} recon={test['recon']:.4f} "
        f"regression={test['regression']:.4f} "
        f"pearson={test['pearson']:.4f} spearman={test['spearman']:.4f} "
        f"token_acc={test['token_acc']:.3f} seq_acc={test['seq_acc']:.3f}"
    )
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
