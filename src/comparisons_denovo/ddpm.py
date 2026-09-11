"""De novo DDPM baseline (adapted from mRNA-translation ``baselines/rna_ddpm.py``)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from comparisons_denovo._backbone import (
    ConvBlock1D,
    ProgressiveDecoder1D,
    mlp_regressor,
    tokens_to_one_hot,
)


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class _Denoiser1DConv(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, seq_len: int):
        super().__init__()
        enc_c, enc_l = 32, 4
        t_embed_dim = enc_c * enc_l
        self.t_emb = nn.Sequential(
            _SinusoidalPosEmb(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )
        self.encoder = nn.Sequential(
            ConvBlock1D(vocab_size, 16),
            ConvBlock1D(16, enc_c),
            nn.AdaptiveAvgPool1d(enc_l),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_c * enc_l, latent_dim),
            ProgressiveDecoder1D(
                latent_dim=latent_dim,
                target_length=seq_len,
                output_channels=vocab_size,
            ),
        )

    def forward(self, x_bcl: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.t_emb(t)
        z = self.encoder(x_bcl)
        z = z + temb.view_as(z)
        return self.decoder(z)

    def encode(self, x_bcl: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_bcl)


class DDPM(nn.Module):
    """DDPM over continuous one-hot RNA sequences (unconditional de novo).

    Paper: Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)
    Adapted from: mRNA-translation ``RNA_DDPM``
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        latent_dim: int = 32,
        num_timesteps: int = 100,
        pad_token_id: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.T = int(num_timesteps)
        self.pad_token_id = int(pad_token_id)

        betas = torch.linspace(1e-4, 2e-2, self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoiser = _Denoiser1DConv(self.latent_dim, self.vocab_size, self.seq_len)
        with torch.no_grad():
            dummy = torch.zeros(1, self.vocab_size, self.seq_len)
            flat = self.denoiser.encode(dummy)
            self.flatten_dim = int(flat.shape[1] * flat.shape[2])
        self.regression_head = mlp_regressor(self.flatten_dim)
        self.to(self.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_acp * x_start + sqrt_om * noise

    def p_losses(self, x_blv: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_bcl = x_blv.permute(0, 2, 1)
        noise = torch.randn_like(x_bcl)
        x_noisy = self.q_sample(x_bcl, t, noise=noise)
        return F.mse_loss(self.denoiser(x_noisy, t), noise)

    def encode(self, x_blv: torch.Tensor) -> torch.Tensor:
        return self.denoiser.encode(x_blv.permute(0, 2, 1))

    def regress(self, z_bcl: torch.Tensor) -> torch.Tensor:
        return self.regression_head(z_bcl.flatten(1)).squeeze(-1)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del mask
        tokens = x_0.long()
        oh = tokens_to_one_hot(tokens, self.vocab_size)
        t = torch.randint(0, self.T, (tokens.shape[0],), device=tokens.device, dtype=torch.long)
        gen = self.p_losses(oh, t)
        y_hat = self.regress(self.encode(oh))
        if targets is None:
            prop = torch.zeros((), device=tokens.device)
        else:
            prop = F.mse_loss(y_hat, targets.float().view(-1))
        total = float(recon_weight) * gen + prop
        return total, gen, prop, y_hat

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[t][:, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_recip = 1.0 / torch.sqrt(self.alphas[t][:, None, None])
        noise_pred = self.denoiser(x_t, t)
        x_prev = sqrt_recip * (x_t - beta_t / sqrt_om * noise_pred)
        if int(t[0].item()) > 0:
            x_prev = x_prev + torch.sqrt(self.betas[t][:, None, None]) * torch.randn_like(x_t)
        return x_prev

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        x = torch.randn(batch_size, self.vocab_size, self.seq_len, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.permute(0, 2, 1).argmax(dim=-1)

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        pad_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        del target_direction, pad_mask, _kwargs
        return self.sample(int(sequences.shape[0]))
