from __future__ import annotations

import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

try:
    from comparisons.nos_c import (
        _DiscreteCorruptionSchedule,
        _random_contig_infill_mask,
        _timestep_embedding,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from comparisons.nos_c import (
        _DiscreteCorruptionSchedule,
        _random_contig_infill_mask,
        _timestep_embedding,
    )


class NOS_D(nn.Module):
    """NOS-D guides discrete masked diffusion sampling with property gradients.

    Originally discrete (MASK) guided diffusion for protein / antibody design. This RNA adaptation
    applies the official NOS sampling and guidance recipe to nucleotide sequences.

    Paper: Protein Design with Guided Discrete Diffusion (NeurIPS 2023)
    Github: https://github.com/ngruver/NOS
    """

    def __init__(
        self,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_timesteps: int = 64,
        noise_schedule: str = "cosine",
        num_properties: int = 1,
        edit_frac: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.num_properties = int(num_properties)
        self.edit_frac = float(edit_frac)

        self.rna_vocab = {
            "PAD": 0,
            "A": 1,
            "G": 2,
            "C": 3,
            "T": 4,
            "U": 5,
            "N": 6,
            "[MASK]": 7,
        }
        self.idx_to_base = {v: k for k, v in self.rna_vocab.items()}
        self.mask_id = self.rna_vocab["[MASK]"]
        self.pad_id = 0
        self.vocab_size = len(self.rna_vocab)
        self.bad_word_ids = [self.pad_id, self.mask_id]

        self.schedule = _DiscreteCorruptionSchedule(
            mask_id=self.mask_id,
            timesteps=int(max_timesteps),
            noise_schedule=noise_schedule,
        )
        self.max_timesteps = self.schedule.timesteps

        self.embeddings = nn.Embedding(self.vocab_size, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Linear(hidden_dim, self.vocab_size)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_properties),
        )
        self.to(self.device)

    def forward(self, corrupted_ids, timesteps, attn_mask=None, token_embed=None):
        if token_embed is None:
            token_embed = self.embeddings(corrupted_ids)
        time_embed = self.time_embed(_timestep_embedding(timesteps, self.hidden_dim))
        time_embed = time_embed.unsqueeze(1).expand(-1, token_embed.size(1), -1)
        embed = self.drop(self.LayerNorm(token_embed + time_embed))
        key_pad = None if attn_mask is None else ~attn_mask.bool()
        sequence_output = self.encoder(embed, src_key_padding_mask=key_pad)
        logits = self.cls(sequence_output)
        return {
            "logits": logits,
            "sequence_output": sequence_output,
            "embeds": token_embed,
        }

    def get_labels(self, input_ids, timesteps, attn_mask=None, sequence_output=None):
        if sequence_output is None:
            sequence_output = self.forward(input_ids, timesteps, attn_mask)["sequence_output"]
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).float()
            pooled = (sequence_output * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = sequence_output.mean(1)
        return self.regression_head(pooled)

    def guidance_score(self, input_ids, timesteps, attn_mask=None, sequence_output=None):
        return self.get_labels(input_ids, timesteps, attn_mask, sequence_output).sum(-1)

    def compute_loss(self, x_0, targets=None, mask=None, recon_weight: float = 1.0):
        b = x_0.shape[0]
        t = torch.randint(0, self.max_timesteps, (b,), device=x_0.device)
        corrupt_mask = mask if mask is not None else torch.ones_like(x_0, dtype=torch.bool)
        corrupt_ids, used_mask = self.schedule.corrupt(x_0, t, corrupt_mask)
        out = self.forward(corrupt_ids, t, attn_mask=mask)
        logits = out["logits"]
        ce = F.cross_entropy(
            logits.view(-1, self.vocab_size), x_0.view(-1), reduction="none"
        ).view(b, -1)
        loss_mask = used_mask.float()
        if mask is not None:
            loss_mask = loss_mask * mask.float()
        denom = loss_mask.sum(dim=-1).clamp_min(1.0)
        recon = ((ce * loss_mask).sum(dim=-1) / denom).mean()

        prop_pred = self.get_labels(
            corrupt_ids, t, attn_mask=mask, sequence_output=out["sequence_output"]
        )
        prop_loss = torch.zeros((), device=x_0.device)
        if targets is not None:
            prop_loss = F.mse_loss(prop_pred, targets.float().view_as(prop_pred))
        total = recon_weight * recon + prop_loss
        return total, recon, prop_loss, prop_pred

    def guidance_steps(
        self,
        model_output,
        t,
        attn_mask,
        infill_mask,
        guidance_layer: str = "first",
        step_size: float = 0.1,
        stability_coef: float = 0.5,
        num_steps: int = 5,
        guidance_sign: float = 1.0,
    ):
        sign = float(guidance_sign)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        logits = model_output["logits"]
        if guidance_layer == "last":
            h = model_output["sequence_output"]
        elif guidance_layer == "first":
            h = model_output["embeds"]
        else:
            raise NotImplementedError(guidance_layer)

        delta = nn.Parameter(torch.zeros_like(h), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=step_size)
        inf = infill_mask.unsqueeze(-1).to(dtype=h.dtype)

        with torch.enable_grad():
            for _ in range(int(num_steps)):
                h_current = h + inf * delta
                if guidance_layer == "last":
                    target_loss = self.guidance_score(
                        None, t, attn_mask, sequence_output=h_current
                    ).sum()
                    new_logits = self.cls(h_current)
                else:
                    out = self.forward(None, t, attn_mask, token_embed=h_current)
                    target_loss = self.guidance_score(
                        None, t, attn_mask, sequence_output=out["sequence_output"]
                    ).sum()
                    new_logits = out["logits"]
                kl = kl_loss(
                    F.log_softmax(new_logits, dim=-1),
                    F.log_softmax(logits, dim=-1),
                )
                loss = -sign * target_loss + stability_coef * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if guidance_layer == "last":
            return self.cls(h + delta.data)
        out = self.forward(None, t, attn_mask, token_embed=(h + delta.data))
        return out["logits"]

    def sample(
        self,
        infill_seed: torch.Tensor,
        infill_mask: torch.Tensor,
        corrupt_mask: torch.Tensor,
        num_samples: int = 1,
        guidance_kwargs: Optional[dict] = None,
        bad_word_ids=None,
    ):
        device = self.device
        if bad_word_ids is None:
            bad_word_ids = self.bad_word_ids

        if infill_seed.dim() == 1:
            infill_seed = infill_seed.unsqueeze(0)
        if infill_mask.dim() == 1:
            infill_mask = infill_mask.unsqueeze(0)
        if corrupt_mask.dim() == 1:
            corrupt_mask = corrupt_mask.unsqueeze(0)

        b = infill_seed.shape[0]
        if num_samples != 1 and b != 1:
            raise ValueError("num_samples>1 only supported for a single seed row")
        if b == 1 and num_samples > 1:
            infill_seed = infill_seed.expand(num_samples, -1).clone()
            infill_mask = infill_mask.expand(num_samples, -1)
            corrupt_mask = corrupt_mask.expand(num_samples, -1)
            b = num_samples

        infill_m = infill_mask.bool()
        corrupt_m = corrupt_mask.bool()
        gt_vals = torch.where(infill_m, torch.full_like(infill_seed, self.mask_id), infill_seed)

        indices = list(range(self.schedule.timesteps))[::-1]
        t_top = torch.full((b,), indices[0], device=device, dtype=torch.long)
        noisy_gt, _ = self.schedule.corrupt(gt_vals, t_top, corrupt_m)
        noisy_gt = torch.where(corrupt_m, noisy_gt, gt_vals)

        x = self.schedule.sample_prior(gt_vals.shape, device)
        x = torch.where(infill_m, x, noisy_gt)
        attn_mask = (infill_seed != self.pad_id) | infill_m

        gkw = dict(guidance_kwargs) if guidance_kwargs is not None else None
        return_best = bool(gkw.pop("return_best", False)) if gkw is not None else False
        guidance_sign = float(gkw.pop("guidance_sign", 1.0)) if gkw is not None else 1.0

        traj_ids = []
        traj_scores = []
        for i in indices:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            with torch.no_grad():
                model_output = self.forward(x, t, attn_mask)
            logits = model_output["logits"]
            if gkw is not None:
                logits = self.guidance_steps(
                    model_output,
                    t,
                    attn_mask,
                    infill_m,
                    guidance_sign=guidance_sign,
                    **gkw,
                )

            logits = logits.clone()
            for wid in bad_word_ids:
                logits[:, :, wid] = -1e9

            x = Categorical(logits=logits).sample()
            clean_x = x.clone()

            if i != indices[-1]:
                x, _ = self.schedule.corrupt(x, t, infill_m)
                noise_t = torch.full((b,), max(i - 1, 0), device=device, dtype=torch.long)
                noisy_gt, _ = self.schedule.corrupt(gt_vals, noise_t, corrupt_m)
                noisy_gt = torch.where(corrupt_m, noisy_gt, gt_vals)
                x = torch.where(infill_m, x, noisy_gt)

            pred_ids = torch.where(infill_m, clean_x, infill_seed)
            pred_ids = torch.where(infill_seed == self.pad_id, infill_seed, pred_ids)
            traj_ids.append(pred_ids)
            if gkw is not None:
                scores = self.guidance_score(pred_ids, t, attn_mask)
                traj_scores.append(scores)

        if return_best and traj_scores:
            score_stack = torch.stack(traj_scores, dim=0)
            best_t = (guidance_sign * score_stack).argmax(dim=0)
            samples = torch.stack(
                [traj_ids[int(best_t[j])][j] for j in range(b)], dim=0
            )
        else:
            samples = traj_ids[-1]
        return samples

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        step_size: float = 0.1,
        stability_coef: float = 0.5,
        n_langevin: int = 10,
        guidance_layer: str = "first",
        return_best: bool = True,
        mask: Optional[torch.Tensor] = None,
        edit_frac: Optional[float] = None,
        **_unused,
    ) -> torch.Tensor:
        sequences = sequences.to(self.device)
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        if mask is None:
            mask = sequences != self.pad_id
        else:
            mask = mask.to(self.device).bool()

        frac = self.edit_frac if edit_frac is None else float(edit_frac)
        infill_mask = _random_contig_infill_mask(mask, edit_frac=frac)
        corrupt_mask = mask.clone()
        seed = sequences.clone()
        seed = torch.where(infill_mask, torch.full_like(seed, self.mask_id), seed)

        sign = 1.0 if target_direction == "increase" else -1.0
        guidance_kwargs = {
            "step_size": float(step_size),
            "stability_coef": float(stability_coef),
            "num_steps": int(n_langevin),
            "guidance_layer": guidance_layer,
            "return_best": bool(return_best),
            "guidance_sign": sign,
        }
        return self.sample(
            infill_seed=seed,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=1,
            guidance_kwargs=guidance_kwargs,
            bad_word_ids=self.bad_word_ids,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = NOS_D(seq_len=32, max_timesteps=8, device="cpu")
    x = torch.randint(1, 5, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    loss, *_ = model.compute_loss(x, targets=torch.zeros(2, 1), mask=mask)
    assert torch.isfinite(loss)
    out = model.optimize(x, target_direction="increase", n_langevin=2, return_best=True)
    assert out.shape == x.shape
    out_neg = model.optimize(x, target_direction="decrease", n_langevin=2, return_best=True)
    assert out_neg.shape == x.shape
    print("NOS_D unit tests passed.")
