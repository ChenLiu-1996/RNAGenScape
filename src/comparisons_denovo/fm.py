"""De novo flow-matching baseline (adapted from mRNA-translation ``baselines/rna_flowmatching.py``)."""

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


class FM(nn.Module):
    """Conditional flow matching over continuous one-hot RNA (unconditional de novo).

    Paper: Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
    Adapted from: mRNA-translation ``RNA_FlowMatching``
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        latent_dim: int = 32,
        ode_steps: int = 50,
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
        self.ode_steps = int(ode_steps)
        self.pad_token_id = int(pad_token_id)

        self.time_embedding = nn.Linear(1, self.vocab_size)
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
        self.regression_head = mlp_regressor(self.latent_dim)
        self.to(self.device)

    def encode(self, x_blv: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_blv.permute(0, 2, 1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).permute(0, 2, 1)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict vector field v_theta(x_t, t); ``x_t`` is [B, L, V], ``t`` is [B]."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_exp = t.expand(-1, x_t.shape[1]).unsqueeze(-1).reshape(-1, 1)
        temb = self.time_embedding(t_exp).reshape(x_t.shape[0], x_t.shape[1], -1)
        z = self.encode(x_t + temb)
        return self.decode(z)

    def flow_matching_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0], 1), device=x_1.device) + 1e-8
        t_exp = t.expand(-1, x_1.shape[1]).unsqueeze(-1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        v_gt = (x_1 - x_0) / t_exp
        v_gt = v_gt / (torch.norm(v_gt, dim=-1, keepdim=True) + 1e-8)
        v_pred = self.velocity(x_t, t.squeeze(-1))
        return F.mse_loss(v_pred, v_gt)

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
        gen = self.flow_matching_loss(oh)
        y_hat = self.regress(self.encode(oh))
        if targets is None:
            prop = torch.zeros((), device=tokens.device)
        else:
            prop = F.mse_loss(y_hat, targets.float().view(-1))
        total = float(recon_weight) * gen + prop
        return total, gen, prop, y_hat

    @torch.no_grad()
    def sample(self, batch_size: int, *, ode_steps: Optional[int] = None) -> torch.Tensor:
        """Euler integration of the learned vector field from noise to data."""
        steps = int(self.ode_steps if ode_steps is None else ode_steps)
        x = torch.randn(batch_size, self.seq_len, self.vocab_size, device=self.device)
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            x = x + self.velocity(x, t) * dt
        return x.argmax(dim=-1)

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
