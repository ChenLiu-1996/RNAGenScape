from __future__ import annotations

import copy
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NOS_C(nn.Module):
    """NOS-C guides continuous embedding-space diffusion with property gradients.

    Originally continuous (Gaussian) guided diffusion for protein / antibody design. This RNA
    adaptation applies the official NOS sampling and guidance recipe to nucleotide sequences.

    Paper: Protein Design with Guided Discrete Diffusion (NeurIPS 2023)
    Github: https://github.com/ngruver/NOS
    """

    def __init__(
        self,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        noise_scale: float = 10.0,
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
        self.in_channels = int(hidden_dim)

        self.rna_vocab = {"PAD": 0, "A": 1, "G": 2, "C": 3, "T": 4, "U": 5, "N": 6}
        self.idx_to_base = {v: k for k, v in self.rna_vocab.items()}
        self.vocab_size = len(self.rna_vocab)
        self.pad_id = 0
        self.bad_word_ids = [self.pad_id]

        self.schedule = _GaussianDiffusionSchedule(
            timesteps=int(max_timesteps),
            noise_schedule=noise_schedule,
            noise_scale=float(noise_scale),
        )
        self.max_timesteps = self.schedule.timesteps

        self.word_embedding = nn.Embedding(self.vocab_size, self.in_channels)
        self.time_embed_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.input_up_proj = nn.Linear(self.in_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position_embeddings = nn.Embedding(seq_len + 8, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, self.vocab_size)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_properties),
        )
        self.to(self.device)

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embedding(input_ids)
        normed = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return math.sqrt(self.in_channels) * normed

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, attn_mask=None):
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x = math.sqrt(self.in_channels) * x

        time_emb = self.time_embed(_timestep_embedding(timesteps, self.time_embed_dim))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        pos = self.position_embeddings(
            torch.arange(seq_length, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        emb_inputs = self.dropout(self.LayerNorm(pos + emb_x + time_emb.unsqueeze(1)))

        key_pad = None if attn_mask is None else ~attn_mask.bool()
        sequence_output = self.encoder(emb_inputs, src_key_padding_mask=key_pad)
        logits = self.cls(sequence_output)
        return {"logits": logits, "sequence_output": sequence_output}

    def pred_xstart(self, x, timesteps, attn_mask=None, sequence_output=None, bad_word_ids=None):
        if sequence_output is None:
            logits = self.forward(x, timesteps, attn_mask=attn_mask)["logits"]
        else:
            logits = self.cls(sequence_output)
        if bad_word_ids is not None:
            for wid in bad_word_ids:
                logits[:, :, wid] = -1e9
        probs = F.softmax(logits, dim=-1)
        all_embeds = self.word_embedding.weight
        all_embeds = all_embeds / all_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        all_embeds = math.sqrt(self.in_channels) * all_embeds
        xstart = probs @ all_embeds
        return {"probs": probs, "xstart": xstart}

    def posterior_sample(
        self,
        x,
        t,
        attn_mask=None,
        infill_mask=None,
        corrupt_mask=None,
        gt_vals=None,
        sequence_output=None,
        bad_word_ids=None,
    ):
        out = self.pred_xstart(x, t, attn_mask, sequence_output, bad_word_ids)
        mean, _, logvar = self.schedule.q_posterior_mean_variance(
            x_start=out["xstart"], x_t=x, t=t
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sigma = torch.exp(0.5 * logvar)
        x = mean + nonzero_mask * sigma * noise

        if gt_vals is not None and infill_mask is not None:
            noise_t = torch.maximum(t[:1] - 1, torch.zeros_like(t[:1]))
            noisy_gt = self.schedule.q_sample(gt_vals, noise_t)
            if corrupt_mask is not None:
                noisy_gt = torch.where((corrupt_mask * nonzero_mask).bool(), noisy_gt, gt_vals)
            x = torch.where(infill_mask, x, noisy_gt)

        return {"x": x, "probs": out["probs"], "mean": mean, "sigma": sigma}

    def get_labels(self, x, timesteps, attn_mask=None, sequence_output=None):
        if sequence_output is None:
            sequence_output = self.forward(x, timesteps, attn_mask)["sequence_output"]
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).float()
            pooled = (sequence_output * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = sequence_output.mean(1)
        return self.regression_head(pooled)

    def guidance_score(self, x, timesteps, attn_mask=None, sequence_output=None):
        return self.get_labels(x, timesteps, attn_mask, sequence_output).sum(-1)

    def compute_loss(self, x_0, targets=None, mask=None, recon_weight: float = 1.0):
        b = x_0.shape[0]
        t = torch.randint(0, self.max_timesteps, (b,), device=x_0.device)
        embeds = self.get_embeds(x_0)
        corrupt_mask = mask if mask is not None else torch.ones_like(x_0, dtype=torch.bool)
        x_t = self.schedule.q_sample(embeds, t)
        x_t = torch.where(corrupt_mask[..., None].bool(), x_t, embeds)

        out = self.forward(x_t, t, attn_mask=mask)
        logits = out["logits"]
        ce = F.cross_entropy(
            logits.view(-1, self.vocab_size), x_0.view(-1), reduction="none"
        ).view(b, -1)
        loss_mask = corrupt_mask.float()
        if mask is not None:
            loss_mask = loss_mask * mask.float()
        recon = (ce * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        prop_loss = torch.zeros((), device=x_0.device)
        prop_pred = self.get_labels(x_t, t, attn_mask=mask, sequence_output=out["sequence_output"])
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
        bad_word_ids,
        corrupt_mask=None,
        gt_vals=None,
        guidance_layer: str = "first",
        step_size: float = 0.1,
        stability_coef: float = 1.0,
        num_steps: int = 5,
        guidance_sign: float = 1.0,
    ):
        sign = float(guidance_sign)
        if guidance_layer == "last":
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            h = model_output["sequence_output"].detach()
            logits = model_output["logits"].detach()
            delta = nn.Parameter(torch.zeros_like(h), requires_grad=True)
        elif guidance_layer == "first":
            x = model_output["x"].detach()
            mean = model_output["mean"].detach()
            sigma = model_output["sigma"].detach().clamp_min(1e-8)
            delta = nn.Parameter(torch.zeros_like(x), requires_grad=True)
        else:
            raise NotImplementedError(guidance_layer)

        optimizer = torch.optim.Adagrad([delta], lr=step_size)
        with torch.enable_grad():
            for _ in range(int(num_steps)):
                if guidance_layer == "last":
                    h_current = h + infill_mask * delta
                    target_loss = self.guidance_score(
                        None, t, attn_mask, sequence_output=h_current
                    ).sum()
                    new_logits = self.cls(h_current)
                    kl = kl_loss(F.log_softmax(new_logits, dim=-1), F.log_softmax(logits, dim=-1))
                    loss = -sign * target_loss + stability_coef * kl
                else:
                    x_current = x + infill_mask * delta
                    target_loss = self.guidance_score(x_current, t, attn_mask).sum()
                    nll = ((x_current - mean) ** 2 / sigma).sum()
                    loss = -sign * target_loss + stability_coef * nll
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if guidance_layer == "last":
            p_out = self.posterior_sample(
                model_output["x_prev"],
                t,
                attn_mask,
                infill_mask,
                corrupt_mask,
                gt_vals,
                sequence_output=(h + delta.data),
                bad_word_ids=bad_word_ids,
            )
            return {"x": p_out["x"], "probs": p_out["probs"]}

        x = x + delta.data
        probs = self.pred_xstart(x, t, attn_mask=attn_mask, bad_word_ids=bad_word_ids)["probs"]
        return {"x": x, "probs": probs}

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

        gt_vals = self.get_embeds(infill_seed)
        infill_m = infill_mask.bool().unsqueeze(-1)
        corrupt_m = corrupt_mask.bool().unsqueeze(-1)

        indices = list(range(self.schedule.timesteps))[::-1]
        t0 = torch.full((b,), indices[0], device=device, dtype=torch.long)
        noisy_gt = self.schedule.q_sample(gt_vals, t0)
        noisy_gt = torch.where(corrupt_m, noisy_gt, gt_vals)

        x = self.schedule.noise_scale * torch.randn(
            (b, gt_vals.shape[1], gt_vals.shape[-1]), device=device
        )
        x = torch.where(infill_m, x, noisy_gt)
        attn_mask = infill_seed != self.pad_id

        gkw = dict(guidance_kwargs) if guidance_kwargs is not None else None
        return_best = bool(gkw.pop("return_best", False)) if gkw is not None else False
        guidance_sign = float(gkw.pop("guidance_sign", 1.0)) if gkw is not None else 1.0

        traj_ids = []
        traj_scores = []
        for i in indices:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            with torch.no_grad():
                f_out = self.forward(x, t, attn_mask=attn_mask)
                p_out = self.posterior_sample(
                    x,
                    t,
                    attn_mask,
                    infill_m,
                    corrupt_m,
                    gt_vals,
                    sequence_output=f_out["sequence_output"],
                    bad_word_ids=bad_word_ids,
                )
                p_out["x_prev"] = x
                out = {**f_out, **p_out}
                x = out["x"]
                probs = out["probs"]

            if gkw is not None:
                g_out = self.guidance_steps(
                    out,
                    t,
                    attn_mask,
                    infill_m,
                    bad_word_ids,
                    corrupt_mask=corrupt_m,
                    gt_vals=gt_vals,
                    guidance_sign=guidance_sign,
                    **gkw,
                )
                x = g_out["x"]
                probs = g_out["probs"]

            with torch.no_grad():
                pred_ids = probs.argmax(-1)
                pred_ids = torch.where(infill_mask.bool(), pred_ids, infill_seed)
                traj_ids.append(pred_ids)
                if gkw is not None:
                    scores = self.guidance_score(self.get_embeds(pred_ids), t, attn_mask)
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
        stability_coef: float = 1.0,
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
            infill_seed=sequences,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=1,
            guidance_kwargs=guidance_kwargs,
            bad_word_ids=self.bad_word_ids,
        )


# --------------------------------------------------------------------------- helpers (shared with NOS_D)
def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _random_contig_infill_mask(pad_mask: torch.Tensor, edit_frac: float = 0.1) -> torch.Tensor:
    """RNA analog of CDR masking: random contiguous span over non-pad sites."""
    b, _l = pad_mask.shape
    out = torch.zeros_like(pad_mask)
    for i in range(b):
        idx = pad_mask[i].nonzero(as_tuple=False).view(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        span = max(1, int(round(edit_frac * n)))
        span = min(span, n)
        start = int(torch.randint(0, n - span + 1, (1,)).item())
        out[i, idx[start : start + span]] = True
    return out


def _get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class _GaussianDiffusionSchedule:
    """Official NOS continuous (Gaussian) corruption schedule."""

    def __init__(self, timesteps: int = 1000, noise_schedule: str = "cosine", noise_scale: float = 10.0):
        betas = np.array(_get_named_beta_schedule(noise_schedule, timesteps), dtype=np.float64)
        self.betas = betas
        self.timesteps = int(betas.shape[0])
        self.noise_scale = float(noise_scale)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = self.noise_scale * torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        posterior_variance = (self.noise_scale**2) * posterior_variance
        posterior_log_variance_clipped = 2 * np.log(self.noise_scale) + posterior_log_variance_clipped
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class _DiscreteCorruptionSchedule:
    """Official NOS discrete MASK corruption schedule (cosine / linear)."""

    def __init__(self, mask_id: int, timesteps: int = 64, noise_schedule: str = "cosine"):
        self.timesteps = int(timesteps)
        self.mask_id = int(mask_id)
        if noise_schedule == "linear":
            self.mask_rates = np.linspace(0, 1, self.timesteps, dtype=np.float64)
        elif noise_schedule == "cosine":
            self.mask_rates = 1.0 - (np.cos(np.pi * np.linspace(0, 1, self.timesteps)) + 1.0) / 2.0
        else:
            raise NotImplementedError(noise_schedule)

    def sample_prior(self, shape, device):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=device)

    def corrupt(self, input_ids: torch.Tensor, timesteps: torch.Tensor, corrupt_mask=None):
        mask_nums = torch.rand_like(input_ids, dtype=torch.float32)
        mask = torch.zeros_like(mask_nums, dtype=torch.bool)
        for i, t in enumerate(timesteps):
            mask[i] = mask_nums[i] < self.mask_rates[int(t.item())]
        if corrupt_mask is not None:
            mask = (mask * corrupt_mask).bool()
        new_ids = copy.deepcopy(input_ids)
        new_ids = torch.where(mask, torch.full_like(new_ids, self.mask_id), new_ids)
        return new_ids, mask


if __name__ == "__main__":
    torch.manual_seed(0)
    model = NOS_C(seq_len=32, max_timesteps=8, device="cpu", noise_scale=1.0)
    x = torch.randint(1, 5, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    loss, *_ = model.compute_loss(x, targets=torch.zeros(2, 1), mask=mask)
    assert torch.isfinite(loss)
    out = model.optimize(x, target_direction="increase", n_langevin=2, return_best=True)
    assert out.shape == x.shape
    out_neg = model.optimize(x, target_direction="decrease", n_langevin=2, return_best=True)
    assert out_neg.shape == x.shape
    print("NOS_C unit tests passed.")
