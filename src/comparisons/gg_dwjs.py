"""RNA-adapted gradient-guided discrete Walk-Jump Sampling (gg-dWJS).

Official gg-dWJS ([zarifikram/gg-dWJS](https://github.com/zarifikram/gg-dWJS))
trains a Gaussian denoiser on continuous one-hots and a separate property
discriminator on the smoothed manifold, then runs underdamped Langevin walk
steps with score + property-gradient guidance, followed by a Bayes jump back
to discrete sequences (TMLR 2024).

This RNA baseline keeps that recipe with a small Transformer backbone over
nucleotide one-hots (PAD/A/G/C/T/U/N), joint training of denoiser + property
head (matching our other comparison trainers), and no Lightning/Hydra/ByteNet.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class gg_dWJS(nn.Module):
    """Gradient-guided discrete Walk-Jump Sampling for RNA sequences."""

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        sigma: float = 1.0,
        num_properties: int = 1,
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
        self.hidden_dim = int(hidden_dim)
        self.sigma = float(sigma)
        self.num_properties = int(num_properties)

        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)
        self.pos_encoding = _SinusoidalPositionalEncoding(hidden_dim, seq_len + 100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.denoise_head = nn.Linear(hidden_dim, self.vocab_size)
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_properties),
        )
        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(
        self,
        continuous_x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode continuous one-hots ``[B, L, V]`` -> hidden ``[B, L, H]``."""
        h = self.input_proj(continuous_x)
        h = self.pos_encoding(h)
        # Transformer key_padding_mask: True means ignore.
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = ~pad_mask.bool()
        h = self.backbone(h, src_key_padding_mask=key_padding_mask)
        return self.output_norm(h)

    def denoise(
        self,
        y: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict clean continuous one-hot ``xhat`` from noisy ``y``."""
        return self.denoise_head(self._encode(y, pad_mask=pad_mask))

    def score(
        self,
        y: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score ``s(y) = (D(y) - y) / sigma^2``."""
        return (self.denoise(y, pad_mask=pad_mask) - y) / (self.sigma ** 2)

    def predict_property(
        self,
        y: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property prediction from continuous (possibly noisy) one-hots."""
        h = self._encode(y, pad_mask=pad_mask)
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            weights = pad_mask.float().unsqueeze(-1)
            pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.property_head(pooled)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
        property_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint denoiser + property loss on smoothed one-hots.

        Args:
            x_0: token ids ``[B, L]``
            targets: property labels ``[B]`` or ``[B, P]``
            mask: bool pad mask True=valid ``[B, L]``
            recon_weight / property_weight: loss weights

        Returns:
            total_loss, recon_loss, property_loss, property_pred ``[B, P]``
        """
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")
        x = F.one_hot(x_0.long(), num_classes=self.vocab_size).float()
        y = x + self.sigma * torch.randn_like(x)

        xhat = self.denoise(y, pad_mask=mask)
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            recon_loss = ((xhat - x) ** 2 * m).sum() / (m.sum() * self.vocab_size).clamp(min=1.0)
        else:
            recon_loss = F.mse_loss(xhat, x)

        prop_pred = self.predict_property(y, pad_mask=mask)
        if targets is None:
            prop_loss = torch.zeros((), device=x.device)
        else:
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * recon_loss + property_weight * prop_loss
        return total, recon_loss, prop_loss, prop_pred

    def _property_grad(
        self,
        y: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor],
        direction: float,
        guide_scale: float,
    ) -> torch.Tensor:
        """``direction * guide_scale * ∇_y property(y) / sigma^2``."""
        y_req = y.detach().requires_grad_(True)
        pred = self.predict_property(y_req, pad_mask=pad_mask)
        # Ascend property when direction > 0; descend when direction < 0.
        objective = (direction * pred).sum()
        grad = torch.autograd.grad(objective, y_req, create_graph=False)[0]
        return guide_scale * grad / (self.sigma ** 2)

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        delta: float = 0.5,
        friction: float = 1.0,
        lipschitz: float = 1.0,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Walk-jump sample, optionally property-guided.

        If ``seed_tokens`` is None, starts from random discrete tokens.
        ``delta`` is the external step size; effective walk step is ``delta * sigma``
        (official convention).
        """
        if seed_tokens is None:
            seed_tokens = torch.randint(
                0, self.vocab_size, (batch_size, self.seq_len), device=self.device
            )
        else:
            seed_tokens = seed_tokens.to(self.device)
            batch_size = seed_tokens.shape[0]

        x0 = F.one_hot(seed_tokens.long(), num_classes=self.vocab_size).float()
        if pad_mask is None:
            pad_mask = seed_tokens != 0

        y = x0 + self.sigma * torch.randn_like(x0)
        v = torch.zeros_like(y)
        dt = float(delta) * self.sigma
        u = 1.0 / float(lipschitz)
        zeta1 = math.exp(-float(friction))
        zeta2 = math.exp(-2.0 * float(friction))
        traj = []

        for _ in range(int(num_steps)):
            y = y + 0.5 * dt * v
            with torch.no_grad():
                psi = self.score(y, pad_mask=pad_mask)
            if guidance and self.num_properties > 0:
                psi = psi + self._property_grad(
                    y,
                    pad_mask=pad_mask,
                    direction=float(direction),
                    guide_scale=float(guide_scale),
                )
            noise = torch.randn_like(y)
            v = v + 0.5 * u * dt * psi
            v = zeta1 * v + 0.5 * u * dt * psi + math.sqrt(u * (1.0 - zeta2)) * noise
            y = y + 0.5 * dt * v
            if return_traj:
                traj.append(self.denoise(y, pad_mask=pad_mask).argmax(dim=-1).detach())

        with torch.no_grad():
            xhat = self.denoise(y, pad_mask=pad_mask)
        if guidance and self.num_properties > 0:
            xhat = xhat + self._property_grad(
                y,
                pad_mask=pad_mask,
                direction=float(direction),
                guide_scale=float(guide_scale),
            )

        tokens = xhat.argmax(dim=-1)
        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        num_steps: int = 50,
        delta: float = 0.5,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided walk-jump starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            num_steps=num_steps,
            delta=delta,
            guidance=True,
            direction=direction,
            guide_scale=guide_scale,
            pad_mask=pad_mask,
            return_traj=False,
        )


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 32, 7
    model = gg_dWJS(vocab_size=vocab_size, seq_len=seq_len, device="cpu")
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, recon, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert recon.ndim == 0 and prop.ndim == 0

    generated = model.sample(batch_size=batch_size, num_steps=5, guidance=False)
    assert generated.shape == (batch_size, seq_len)

    optimized = model.optimize(sequences, target_direction="increase", num_steps=5)
    assert optimized.shape == sequences.shape
    print("gg_dWJS unit tests passed.")
