"""De novo VAE baseline (adapted from mRNA-translation ``baselines/rna_vae.py``)."""

from __future__ import annotations

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


class VAE(nn.Module):
    """Variational autoencoder over one-hot RNA sequences (unconditional de novo).

    Paper: Auto-Encoding Variational Bayes (Kingma & Welling, ICLR 2014)
    Adapted from: mRNA-translation ``RNA_VAE``
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        latent_dim: int = 32,
        kld_weight: float = 1e-5,
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
        self.kld_weight = float(kld_weight)
        self.pad_token_id = int(pad_token_id)

        self.encoder = nn.Sequential(
            ConvBlock1D(self.vocab_size, 16),
            ConvBlock1D(16, 64),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, self.latent_dim),
        )
        self.decoder = ProgressiveDecoder1D(
            latent_dim=self.latent_dim,
            target_length=self.seq_len,
            output_channels=self.vocab_size,
        )
        self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.var_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.regression_head = mlp_regressor(self.latent_dim)
        self.to(self.device)

    def encode_distribution(self, x_blv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_blv.permute(0, 2, 1))
        return self.mu_layer(z), self.var_layer(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def encode(self, x_blv: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode_distribution(x_blv)
        return self.reparameterize(mu, log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).permute(0, 2, 1)

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return self.regression_head(z).squeeze(-1)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE ELBO (CE + weighted KL) + property MSE. ``x_0`` is token ids [B, L]."""
        tokens = x_0.long()
        oh = tokens_to_one_hot(tokens, self.vocab_size)
        mu, log_var = self.encode_distribution(oh)
        z = self.reparameterize(mu, log_var)
        logits = self.decode(z)
        recon = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / max(tokens.shape[0], 1)
        gen = recon + self.kld_weight * kld

        y_hat = self.regress(z)
        if targets is None:
            prop = torch.zeros((), device=tokens.device)
        else:
            targets = targets.float().view(-1)
            prop = F.mse_loss(y_hat, targets)
        total = float(recon_weight) * gen + prop
        return total, gen, prop, y_hat

    @torch.no_grad()
    def sample(self, batch_size: int, *, return_logits: bool = False):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        logits = self.decode(z)
        if return_logits:
            return logits
        return logits.argmax(dim=-1)

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        pad_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """De novo: ignore seeds / direction; return unconditional samples sized to the batch."""
        del target_direction, pad_mask, _kwargs
        return self.sample(int(sequences.shape[0]))
