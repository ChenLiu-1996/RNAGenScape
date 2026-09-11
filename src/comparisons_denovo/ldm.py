"""De novo LDM baseline (adapted from mRNA-translation ``baselines/rna_ldm.py``)."""

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


class _DenoiserLinear(nn.Module):
    def __init__(self, t_embed_dim: int):
        super().__init__()
        self.t_emb = nn.Sequential(
            _SinusoidalPosEmb(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )
        self.encoder = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim // 2),
            nn.Linear(t_embed_dim // 2, t_embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim // 2),
            nn.Linear(t_embed_dim // 2, t_embed_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.t_emb(t)
        h = self.encoder(z) + temb
        return self.decoder(h)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)


class LDM(nn.Module):
    """Latent diffusion model over RNA one-hots (unconditional de novo).

    Paper: High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., CVPR 2022)
    Adapted from: mRNA-translation ``RNA_LDM``
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

        self.latent_encoder = nn.Sequential(
            ConvBlock1D(self.vocab_size, 16),
            ConvBlock1D(16, 64),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, self.latent_dim),
        )
        self.latent_decoder = ProgressiveDecoder1D(
            latent_dim=self.latent_dim,
            target_length=self.seq_len,
            output_channels=self.vocab_size,
        )

        betas = torch.linspace(1e-4, 2e-2, self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoiser = _DenoiserLinear(t_embed_dim=self.latent_dim)
        self.regression_head = mlp_regressor(self.latent_dim)
        self.to(self.device)

    def q_sample(self, z_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_acp * z_start + sqrt_om * noise

    def encode_latent(self, x_blv: torch.Tensor) -> torch.Tensor:
        return self.latent_encoder(x_blv.permute(0, 2, 1))

    def p_losses(self, x_blv: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_start = self.encode_latent(x_blv)
        noise = torch.randn_like(z_start)
        z_noisy = self.q_sample(z_start, t, noise=noise)
        return F.mse_loss(self.denoiser(z_noisy, t), noise)

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return self.regression_head(z).squeeze(-1)

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
        z = self.encode_latent(oh)
        y_hat = self.regress(z)
        if targets is None:
            prop = torch.zeros((), device=tokens.device)
        else:
            prop = F.mse_loss(y_hat, targets.float().view(-1))
        total = float(recon_weight) * gen + prop
        return total, gen, prop, y_hat

    @torch.no_grad()
    def p_sample(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip = 1.0 / torch.sqrt(self.alphas[t][:, None])
        noise_pred = self.denoiser(z_t, t)
        z_prev = sqrt_recip * (z_t - beta_t / sqrt_om * noise_pred)
        if int(t[0].item()) > 0:
            z_prev = z_prev + torch.sqrt(self.betas[t][:, None]) * torch.randn_like(z_t)
        return z_prev

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        # Match old RNA_LDM: encode random one-hot noise, then reverse in latent space.
        x = torch.randn(batch_size, self.vocab_size, self.seq_len, device=self.device)
        z = self.latent_encoder(x)
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            z = self.p_sample(z, t)
        logits = self.latent_decoder(z).permute(0, 2, 1)
        return logits.argmax(dim=-1)

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
