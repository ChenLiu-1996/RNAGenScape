"""Train comparison baselines (DiffAb, IgLM, NOS_C, NOS_D, gg_dWJS, EM, MPGD, MFM, PCD).

Example:
  python src/train_baseline.py --dataset OpenVaccine --model DiffAb --seed 1

Saves ``results/<dataset>/<model>/seed_{seed}/model.pt`` (plus ``label_stats.json``,
``train_meta.json``, ``train.log``).

Loss: ``recon_w * gen + property_mse``.
Optimization: AdamW + linear-warmup cosine annealing, early stop on val loss.
MFM/EM use staged AE -> latent-pool -> generative stages without early stop.
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

from comparisons import DiffAb, EM, gg_dWJS, IgLM, MFM, MPGD, NOS_C, NOS_D, PCD
from dataset import DATASET_CONFIG, DATASET_NAMES, make_dataloaders
from utils.metrics import VOCAB_SIZE, to_token_ids
from utils.oracle import resolve_device
from utils.results import baseline_checkpoint_path
from utils.training_utils import EarlyStopping, LinearWarmupCosineAnnealingLR, seed_everything

BASELINE_MODELS = ("DiffAb", "IgLM", "NOS_C", "NOS_D", "gg_dWJS", "EM", "MPGD", "MFM", "PCD")
IGLM_SPECIAL_TOKENS = 3  # CLS, SEP, MASK

# Generative term name per model (property head is always ``property_mse``).
GEN_LOSS_NAME = {
    "DiffAb": "diffusion_kl",
    "IgLM": "infill_lm",
    "NOS_C": "token_ce",
    "NOS_D": "masked_ce",
    "gg_dWJS": "denoise_mse",
    "EM": "energy_flow",
    "MPGD": "diffusion_mse",
    "MFM": "metric_flow",
    "PCD": "pcd_contrastive",
}
PROP_LOSS_NAME = "property_mse"
# Models with no property head: corr vs labels is undefined (dummy zero preds).
NO_PROPERTY_HEAD = frozenset({"MFM"})


def _corr_or_na(model_name: str, y_true: List[float], preds: List[float]) -> Tuple[float, float]:
    """Pearson/Spearman, or (nan, nan) when the model has no property head."""
    if model_name in NO_PROPERTY_HEAD or len(y_true) <= 1:
        return float("nan"), float("nan")
    return float(pearsonr(y_true, preds)[0]), float(spearmanr(y_true, preds).correlation)


def _fmt_corr(model_name: str, pearson: float, spearman: float) -> str:
    if model_name in NO_PROPERTY_HEAD:
        return "pearson=n/a spearman=n/a (no property head)"
    return f"pearson={pearson:.4f} spearman={spearman:.4f}"


def _fmt_corr_csv(model_name: str, pearson: float, spearman: float) -> str:
    if model_name in NO_PROPERTY_HEAD:
        return "n/a,n/a"
    return f"{pearson:.6f},{spearman:.6f}"


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
            latent_dim=64,
            phase1_steps=10000,
            phase2_steps=1000,
            device=device,
        )
    if model_name == "MPGD":
        return MPGD(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            latent_dim=64,
            device=device,
        )
    if model_name == "MFM":
        return MFM(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            latent_dim=64,
            device=device,
        )
    if model_name == "PCD":
        return PCD(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            hidden_dim=256,
            n_gibbs=1,
            buffer_size=128,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total_loss, gen_loss, property_mse, y_hat [B]).

    Total is ``recon_w * gen + property_mse``.
    ``gen_loss`` is the model-specific generative term; see ``GEN_LOSS_NAME``.
    """
    token_ids = to_token_ids(x)
    mask = _pad_mask(token_ids)
    targets = y.float().view(-1)

    if model_name == "DiffAb":
        losses, y_hat = model.compute_loss(token_ids, targets=targets, mask=mask)
        recon = losses["recon_loss"]
        prop = losses["property_loss"]
        total = recon_w * recon + prop
        return total, recon, prop, y_hat.view(-1)

    if model_name in ("NOS_C", "NOS_D", "gg_dWJS", "EM", "MPGD", "MFM", "PCD"):
        total, recon, prop, y_hat = model.compute_loss(
            token_ids,
            targets=targets.unsqueeze(-1),
            mask=mask,
            recon_weight=recon_w,
        )
        return total, recon, prop, y_hat.view(-1)

    if model_name == "IgLM":
        seq_len = token_ids.shape[1]
        # Random contiguous span up to 15% of length.
        mask_start = int(np.random.randint(0, max(seq_len, 1)))
        mask_len = int(np.random.randint(1, max(2, int(seq_len * 0.15) + 1)))
        mask_end = min(mask_start + mask_len, max(seq_len - 1, 1))
        total, recon, prop, y_hat = model.compute_loss(
            token_ids,
            targets,
            mask_start,
            mask_end,
            recon_weight=recon_w,
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
            model_name, model, x, y, recon_w=recon_w
        )
        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_prop += float(prop.item())
        n_batches += 1
        if model_name not in NO_PROPERTY_HEAD:
            y_true.extend(y.detach().cpu().numpy().reshape(-1).tolist())
            preds.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())

    n = max(n_batches, 1)
    pearson, spearman = _corr_or_na(model_name, y_true, preds)
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
    if model_name in NO_PROPERTY_HEAD:
        print(
            f"note: {model_name} has no property head; "
            "property_mse stays 0 and pearson/spearman are n/a (not a training failure)."
        )
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
                    model_name, model, x, y, recon_w=recon_w
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                train_gen += float(gen.item())
                train_prop += float(prop.item())
                n_batches += 1
                if model_name not in NO_PROPERTY_HEAD:
                    train_y_true.extend(y.detach().cpu().numpy().reshape(-1).tolist())
                    train_preds.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())

            if lr_scheduler is not None:
                lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            n = max(n_batches, 1)
            train_pearson, train_spearman = _corr_or_na(model_name, train_y_true, train_preds)
            val = evaluate(
                model_name,
                model,
                val_loader,
                device=device,
                recon_w=recon_w,
            )

            line = (
                f"{epoch},{lr:.8e},"
                f"{train_loss / n:.6f},{train_gen / n:.6f},{train_prop / n:.6f},"
                f"{_fmt_corr_csv(model_name, train_pearson, train_spearman)},"
                f"{val['loss']:.6f},{val[gen_name]:.6f},{val[prop_name]:.6f},"
                f"{_fmt_corr_csv(model_name, val['pearson'], val['spearman'])}"
            )
            print(
                f"epoch {epoch}/{max_epochs}  lr={lr:.2e}  "
                f"train loss={train_loss / n:.4f} {gen_name}={train_gen / n:.4f} "
                f"{prop_name}={train_prop / n:.4f} "
                f"{_fmt_corr(model_name, train_pearson, train_spearman)}  "
                f"val loss={val['loss']:.4f} {gen_name}={val[gen_name]:.4f} "
                f"{prop_name}={val[prop_name]:.4f} "
                f"{_fmt_corr(model_name, val['pearson'], val['spearman'])}"
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


def train_mfm_d2d(
    model: MFM,
    train_loader,
    *,
    device: str,
    lr: float,
    batch_size: int,
    ae_epochs: int,
    geo_epochs: int,
    flow_epochs: int,
    ckpt_path: str,
    log_path: str,
) -> Dict[str, float]:
    """Staged MFM training: AE, tertile latent pool, geopath, then flow."""
    model.to(device)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("stage,epoch,loss,extra\n")
    log_f.flush()
    wd = 1e-4

    def _save() -> None:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, ckpt_path)

    # 1) AE.
    ae_opt = torch.optim.AdamW(model.ae.parameters(), lr=lr, weight_decay=wd)
    print(f"MFM stage=ae epochs={ae_epochs} lr={lr} wd={wd}")
    for ep in range(1, ae_epochs + 1):
        model.train()
        total, n, correct = 0.0, 0, 0
        for x, _y in train_loader:
            tokens = to_token_ids(x).to(device)
            loss = model.ae_loss(tokens)
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()
            with torch.no_grad():
                logits, _ = model.ae(tokens)
                correct += (logits.argmax(1) == tokens).float().mean().item() * tokens.size(0)
            total += float(loss.item()) * tokens.size(0)
            n += tokens.size(0)
        ce = total / max(n, 1)
        acc = correct / max(n, 1)
        print(f"[ae   ] epoch {ep}/{ae_epochs}  ce={ce:.4f}  recon_acc={acc:.4f}")
        log_f.write(f"ae,{ep},{ce:.6f},recon_acc={acc:.6f}\n")
        log_f.flush()
    for p in model.ae.parameters():
        p.requires_grad_(False)
    model.ae.eval()

    # 2) Encode train set; partition into property tertiles.
    pool_info = model.build_train_latent_pool(train_loader, to_tokens=lambda x: to_token_ids(x))
    print(
        f"[pool ] n_train={int(pool_info['n_train'])} "
        f"low/mid/high={int(pool_info['n_low'])}/{int(pool_info['n_mid'])}/{int(pool_info['n_high'])} "
        f"land_gamma={pool_info['land_gamma']:.4f}"
    )
    log_f.write(
        f"pool,0,{pool_info['n_train']:.0f},"
        f"low={pool_info['n_low']:.0f};mid={pool_info['n_mid']:.0f};"
        f"high={pool_info['n_high']:.0f};gamma={pool_info['land_gamma']:.6f}\n"
    )
    log_f.flush()

    n_frame = max(int(pool_info["n_low"]), int(pool_info["n_mid"]), int(pool_info["n_high"]))
    steps_per_epoch = max(1, n_frame // max(int(batch_size), 1))

    # 3) LAND geopath on tertile pool.
    if model.use_geopath and model.geopath_weight > 0.0:
        geo_params = list(model.geopath_pos.parameters()) + list(model.geopath_neg.parameters())
        geo_opt = torch.optim.AdamW(geo_params, lr=lr, weight_decay=wd)
        print(f"MFM stage=geo epochs={geo_epochs} steps/epoch={steps_per_epoch}")
        for ep in range(1, geo_epochs + 1):
            model.train()
            total, n = 0.0, 0
            for _ in range(steps_per_epoch):
                _flow, geo, total_loss = model.d2d_loss(
                    batch_size=batch_size, train_geopath=True, train_flow=False
                )
                geo_opt.zero_grad()
                total_loss.backward()
                geo_opt.step()
                total += float(geo.item()) * batch_size
                n += batch_size
            mean_geo = total / max(n, 1)
            print(f"[geo  ] epoch {ep}/{geo_epochs}  tang_v2={mean_geo:.4f}")
            log_f.write(f"geo,{ep},{mean_geo:.6f},\n")
            log_f.flush()
        for p in geo_params:
            p.requires_grad_(False)
        if model.geopath_pos is not None:
            model.geopath_pos.eval()
        if model.geopath_neg is not None:
            model.geopath_neg.eval()

    # 4) Conditional flow with frozen geopath.
    flow_params = list(model.flow_pos.parameters()) + list(model.flow_neg.parameters())
    flow_opt = torch.optim.AdamW(flow_params, lr=lr, weight_decay=wd)
    print(f"MFM stage=flow epochs={flow_epochs} steps/epoch={steps_per_epoch}")
    for ep in range(1, flow_epochs + 1):
        model.train()
        if model.geopath_pos is not None:
            model.geopath_pos.eval()
        if model.geopath_neg is not None:
            model.geopath_neg.eval()
        total, n = 0.0, 0
        for _ in range(steps_per_epoch):
            flow, _geo, total_loss = model.d2d_loss(
                batch_size=batch_size, train_geopath=False, train_flow=True
            )
            flow_opt.zero_grad()
            total_loss.backward()
            flow_opt.step()
            total += float(flow.item()) * batch_size
            n += batch_size
        mean_flow = total / max(n, 1)
        print(f"[flow ] epoch {ep}/{flow_epochs}  mse={mean_flow:.4f}")
        log_f.write(f"flow,{ep},{mean_flow:.6f},\n")
        log_f.flush()

    _save()
    log_f.close()
    print(f"saved MFM checkpoint -> {ckpt_path}")
    return {
        "n_train": pool_info["n_train"],
        "land_gamma": pool_info["land_gamma"],
        "ae_epochs": float(ae_epochs),
        "geo_epochs": float(geo_epochs),
        "flow_epochs": float(flow_epochs),
    }


def train_em_staged(
    model: EM,
    train_loader,
    *,
    device: str,
    lr: float,
    pot_lr: float,
    batch_size: int,
    ae_epochs: int,
    pred_epochs: int,
    phase1_steps: int,
    phase2_steps: int,
    pot_warmup: int,
    pot_grad_clip: float,
    ckpt_path: str,
    log_path: str,
) -> Dict[str, float]:
    """Staged EM training: AE, latent pool, predictor, then potential (OT then CD)."""
    model.to(device)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("stage,step,loss,extra\n")
    log_f.flush()
    wd = 1e-4

    def _save() -> None:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, ckpt_path)

    # 1) AE.
    ae_opt = torch.optim.AdamW(model.ae.parameters(), lr=lr, weight_decay=wd)
    print(f"EM stage=ae epochs={ae_epochs} lr={lr} wd={wd}")
    for ep in range(1, ae_epochs + 1):
        model.train()
        total, n, correct = 0.0, 0, 0
        for x, _y in train_loader:
            tokens = to_token_ids(x).to(device)
            loss = model.ae_loss(tokens)
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()
            with torch.no_grad():
                logits, _ = model.ae(tokens)
                correct += (logits.argmax(1) == tokens).float().mean().item() * tokens.size(0)
            total += float(loss.item()) * tokens.size(0)
            n += tokens.size(0)
        ce = total / max(n, 1)
        acc = correct / max(n, 1)
        print(f"[ae   ] epoch {ep}/{ae_epochs}  ce={ce:.4f}  recon_acc={acc:.4f}")
        log_f.write(f"ae,{ep},{ce:.6f},recon_acc={acc:.6f}\n")
        log_f.flush()
    for p in model.ae.parameters():
        p.requires_grad_(False)
    model.ae.eval()

    # 2) Encode train set into a fixed latent pool.
    pool_info = model.build_train_latent_pool(train_loader, to_tokens=lambda x: to_token_ids(x))
    print(
        f"[pool ] n_train={int(pool_info['n_train'])} "
        f"y_lo={pool_info['y_lo']:.4f} y_hi={pool_info['y_hi']:.4f}"
    )
    log_f.write(
        f"pool,0,{pool_info['n_train']:.0f},"
        f"y_lo={pool_info['y_lo']:.6f};y_hi={pool_info['y_hi']:.6f}\n"
    )
    log_f.flush()

    # 3) Property predictor.
    pred_opt = torch.optim.AdamW(model.predictor.parameters(), lr=lr, weight_decay=wd)
    print(f"EM stage=pred epochs={pred_epochs} lr={lr} wd={wd}")
    for ep in range(1, pred_epochs + 1):
        model.train()
        model.ae.eval()
        total, n = 0.0, 0
        for x, y in train_loader:
            tokens = to_token_ids(x).to(device)
            y = y.to(device)
            loss, _pred = model.predictor_loss(tokens, y)
            pred_opt.zero_grad()
            loss.backward()
            pred_opt.step()
            total += float(loss.item()) * tokens.size(0)
            n += tokens.size(0)
        mse = total / max(n, 1)
        print(f"[pred ] epoch {ep}/{pred_epochs}  mse={mse:.4f}")
        log_f.write(f"pred,{ep},{mse:.6f},\n")
        log_f.flush()
    for p in model.predictor.parameters():
        p.requires_grad_(False)
    model.predictor.eval()

    # 4) Potential: phase1 OT, then phase2 OT+CD.
    pot_opt = torch.optim.Adam(model.potential.parameters(), lr=pot_lr)
    warmup = max(int(pot_warmup), 1)
    pot_sched = torch.optim.lr_scheduler.LambdaLR(
        pot_opt, lr_lambda=lambda s: min(s + 1, warmup) / float(warmup)
    )
    total_steps = int(phase1_steps) + int(phase2_steps)
    log_every = max(1, total_steps // 40)
    print(
        f"EM stage=pot steps={total_steps} "
        f"(phase1={phase1_steps} phase2={phase2_steps}) pot_lr={pot_lr} warmup={warmup}"
    )
    for step in range(total_steps):
        phase2 = step >= int(phase1_steps)
        model.train()
        model.ae.eval()
        model.predictor.eval()
        ot, cd, loss = model.potential_loss(batch_size=batch_size, train_cd=phase2)
        pot_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.potential.parameters(), pot_grad_clip)
        pot_opt.step()
        pot_sched.step()
        model._ema_update(phase2=phase2)
        if step % log_every == 0 or step == total_steps - 1:
            lr_now = pot_sched.get_last_lr()[0]
            tag = "phase2" if phase2 else "phase1"
            print(
                f"[pot  ] step {step + 1}/{total_steps} ({tag})  "
                f"ot={float(ot.item()):.4f}  cd={float(cd.item()):.4f}  lr={lr_now:.2e}"
            )
            log_f.write(
                f"pot,{step + 1},{float(loss.item()):.6f},"
                f"{tag};ot={float(ot.item()):.6f};cd={float(cd.item()):.6f}\n"
            )
            log_f.flush()

    _save()
    log_f.close()
    print(f"saved EM checkpoint -> {ckpt_path}")
    return {
        "n_train": pool_info["n_train"],
        "ae_epochs": float(ae_epochs),
        "pred_epochs": float(pred_epochs),
        "phase1_steps": float(phase1_steps),
        "phase2_steps": float(phase2_steps),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train DiffAb / IgLM / NOS_C / NOS_D / gg_dWJS / EM / MPGD / MFM / PCD baselines.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--model", type=str, required=True, choices=list(BASELINE_MODELS))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_w", type=float, default=1.0, help="Weight on generative / reconstruction loss.")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="Early stop patience on val loss.")
    p.add_argument("--ae_epochs", type=int, default=200, help="MFM/EM AE stage epochs.")
    p.add_argument("--geo_epochs", type=int, default=100, help="MFM LAND geopath epochs.")
    p.add_argument("--flow_epochs", type=int, default=200, help="MFM flow epochs.")
    p.add_argument("--pred_epochs", type=int, default=50, help="EM predictor epochs.")
    p.add_argument("--phase1_steps", type=int, default=10000, help="EM potential OT steps.")
    p.add_argument("--phase2_steps", type=int, default=1000, help="EM potential CD steps.")
    p.add_argument("--pot_lr", type=float, default=1e-4, help="EM potential Adam lr.")
    p.add_argument("--pot_warmup", type=int, default=500, help="EM potential LR warmup steps.")
    p.add_argument("--pot_grad_clip", type=float, default=1.0, help="EM potential grad clip.")
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
        f"label_norm={args.label_norm} recon_w={args.recon_w}"
    )

    model = build_model(args.model, seq_len=seq_len, device=device)

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
        "gen_loss_name": gen_name,
        "property_loss_name": PROP_LOSS_NAME,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "label_norm": args.label_norm,
        "vocab_size": VOCAB_SIZE if args.model != "IgLM" else VOCAB_SIZE + IGLM_SPECIAL_TOKENS,
    }
    if args.model == "MFM":
        meta.update(
            {
                "ae_epochs": args.ae_epochs,
                "geo_epochs": args.geo_epochs,
                "flow_epochs": args.flow_epochs,
                "protocol": "mfm_d2d_staged_full_train_latent_pool",
            }
        )
    if args.model == "EM":
        meta.update(
            {
                "ae_epochs": args.ae_epochs,
                "pred_epochs": args.pred_epochs,
                "phase1_steps": args.phase1_steps,
                "phase2_steps": args.phase2_steps,
                "pot_lr": args.pot_lr,
                "protocol": "em_staged_full_train_latent_pool",
            }
        )
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

    if args.model == "MFM":
        print(
            f"optimizer=AdamW lr={args.lr} wd=1e-4  "
            f"MFM staged ae/geo/flow={args.ae_epochs}/{args.geo_epochs}/{args.flow_epochs}"
        )
        train_mfm_d2d(
            model,
            train_loader,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            ae_epochs=args.ae_epochs,
            geo_epochs=args.geo_epochs,
            flow_epochs=args.flow_epochs,
            ckpt_path=ckpt_path,
            log_path=log_path,
        )
    elif args.model == "EM":
        print(
            f"optimizer=AdamW(ae/pred) lr={args.lr} wd=1e-4; Adam(pot) lr={args.pot_lr}  "
            f"EM staged ae/pred/pot={args.ae_epochs}/{args.pred_epochs}/"
            f"{args.phase1_steps}+{args.phase2_steps}"
        )
        train_em_staged(
            model,
            train_loader,
            device=device,
            lr=args.lr,
            pot_lr=args.pot_lr,
            batch_size=args.batch_size,
            ae_epochs=args.ae_epochs,
            pred_epochs=args.pred_epochs,
            phase1_steps=args.phase1_steps,
            phase2_steps=args.phase2_steps,
            pot_warmup=args.pot_warmup,
            pot_grad_clip=args.pot_grad_clip,
            ckpt_path=ckpt_path,
            log_path=log_path,
        )
    else:
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
        train(
            args.model,
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

    test = evaluate(
        args.model,
        model,
        test_loader,
        device=device,
        recon_w=args.recon_w,
    )
    gen_name = GEN_LOSS_NAME[args.model]
    print(
        f"test loss={test['loss']:.4f} {gen_name}={test[gen_name]:.4f} "
        f"{PROP_LOSS_NAME}={test[PROP_LOSS_NAME]:.4f} "
        f"{_fmt_corr(args.model, test['pearson'], test['spearman'])}"
    )
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
