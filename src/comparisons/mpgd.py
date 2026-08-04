"""RNA-adapted Manifold Preserving Guided Diffusion (MPGD).

Official MPGD ([KellyYutongHe/mpgd_pytorch](https://github.com/KellyYutongHe/mpgd_pytorch/),
ICLR 2024, arXiv:2311.16424) is a *training-free* guided sampler for pretrained
pixel / latent diffusion models. The shortcut updates the DDIM clean-data
estimate ``x0|t`` with a guidance gradient, optionally projecting that update
onto the data manifold via an autoencoder (MPGD-AE / MPGD-Z), then forms
``x_{t-1}``.

This RNA baseline keeps that sampling recipe on continuous nucleotide one-hots
(PAD/A/G/C/T/U/N). Because we do not ship a pretrained image DM + VQGAN, we
jointly train a small Transformer denoiser, a bottleneck sequence AE (manifold
projector), and a property head used as the guidance loss ``L`` at sample time.
Intentional simplifications vs the official code: no DPS/FreeDoM/LDM stack,
VQGAN, or DDIM schedule engineering; AE is a compact seq AE rather than VQGAN.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPGD(nn.Module):
    """Manifold-preserving guided diffusion for RNA sequences."""

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        latent_dim: int = 32,
        ae_weight: float = 0.1,
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
        self.num_timesteps = int(num_timesteps)
        self.latent_dim = int(latent_dim)
        self.ae_weight = float(ae_weight)
        self.num_properties = int(num_properties)

        # Denoiser backbone over continuous one-hots.
        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)
        self.pos_encoding = _SinusoidalPositionalEncoding(hidden_dim, seq_len + 100)
        self.time_embedding = _TimeEmbedding(hidden_dim)
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
        self.noise_head = nn.Linear(hidden_dim, self.vocab_size)

        # Bottleneck AE for MPGD-AE manifold projection (clean continuous one-hots).
        self.ae_enc = nn.Sequential(
            nn.Linear(self.vocab_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.ae_dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.vocab_size),
        )

        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_properties),
        )

        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

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

    @staticmethod
    def _tokens_to_continuous(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        return F.one_hot(token_ids.long(), num_classes=vocab_size).float() * 2.0 - 1.0

    def _encode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoding(h) + self.time_embedding(t).unsqueeze(1)
        key_padding_mask = None if pad_mask is None else ~pad_mask.bool()
        h = self.backbone(h, src_key_padding_mask=key_padding_mask)
        return self.output_norm(h)

    def _pool(self, h: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is None:
            return h.mean(dim=1)
        w = pad_mask.float().unsqueeze(-1)
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.noise_head(self._encode(x_t, t, pad_mask=pad_mask))

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM / VP clean estimate ``x0|t`` from noisy ``x_t``."""
        if eps is None:
            eps = self.predict_noise(x_t, t, pad_mask=pad_mask)
        a_bar = self.alpha_bars[t].view(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar.clamp(min=1e-8))

    def ae_project(self, x: torch.Tensor) -> torch.Tensor:
        """MPGD-AE style reconstruction ``D(E(x))`` (position-wise bottleneck)."""
        z = self.ae_enc(x)
        return self.ae_dec(z)

    def predict_property(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property from continuous ``x`` (uses ``t=0`` features by default)."""
        if t is None:
            t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = self._encode(x, t, pad_mask=pad_mask)
        return self.property_head(self._pool(h, pad_mask=pad_mask))

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return x_t, noise

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diffusion noise MSE (+ AE recon) and property MSE.

        Returns:
            total_loss, gen_loss (noise[+AE]), property_loss, property_pred ``[B, P]``
        """
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        x0 = self._tokens_to_continuous(x_0, self.vocab_size)
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        x_t, noise = self.add_noise(x0, t)

        eps_pred = self.predict_noise(x_t, t, pad_mask=mask)
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            noise_mse = ((eps_pred - noise) ** 2 * m).sum() / (
                m.sum() * self.vocab_size
            ).clamp(min=1.0)
        else:
            noise_mse = F.mse_loss(eps_pred, noise)

        x0_hat_ae = self.ae_project(x0)
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            ae_recon = ((x0_hat_ae - x0) ** 2 * m).sum() / (
                m.sum() * self.vocab_size
            ).clamp(min=1.0)
        else:
            ae_recon = F.mse_loss(x0_hat_ae, x0)

        gen_loss = noise_mse + self.ae_weight * ae_recon

        prop_pred = self.predict_property(
            x0, pad_mask=mask, t=torch.zeros(b, dtype=torch.long, device=x0.device)
        )
        if targets is None:
            prop_loss = torch.zeros((), device=x0.device)
        else:
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    def _guidance_step(
        self,
        x0_hat: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor],
        direction: float,
        guide_scale: float,
        use_ae: bool,
        a_bar_t: torch.Tensor,
    ) -> torch.Tensor:
        """MPGD update of ``x0|t``: ``x0 <- x0 - c_t ∇ L`` (optionally MPGD-AE)."""
        x0 = x0_hat.detach().requires_grad_(True)
        if use_ae:
            # Grad through D(E(x0)) so update lies closer to the AE manifold.
            x_for_loss = self.ae_project(x0)
        else:
            x_for_loss = x0
        pred = self.predict_property(
            x_for_loss,
            pad_mask=pad_mask,
            t=torch.zeros(x0.shape[0], dtype=torch.long, device=x0.device),
        )
        # Maximize property when direction > 0 => minimize L = -direction * pred.
        loss = (-float(direction) * pred).sum()
        grad = torch.autograd.grad(loss, x0, create_graph=False)[0]
        # Scale similar to official ``scale / sqrt(a_bar)``.
        step = float(guide_scale) / torch.sqrt(a_bar_t.view(-1, *([1] * (x0.ndim - 1))).clamp(min=1e-8))
        return (x0.detach() - step * grad).clamp(-1.0, 1.0)

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        use_ae_proj: bool = True,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """MPGD shortcut sampling (Algorithm 1 + optional MPGD-AE)."""
        if seed_tokens is None:
            x = torch.randn(batch_size, self.seq_len, self.vocab_size, device=self.device)
            if pad_mask is None:
                pad_mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        else:
            seed_tokens = seed_tokens.to(self.device)
            batch_size = seed_tokens.shape[0]
            x0 = self._tokens_to_continuous(seed_tokens, self.vocab_size)
            # Start from moderately noised seed (local refinement / optimization).
            t0 = torch.full(
                (batch_size,),
                max(self.num_timesteps // 2, 1),
                device=self.device,
                dtype=torch.long,
            )
            x, _ = self.add_noise(x0, t0)
            if pad_mask is None:
                pad_mask = seed_tokens != 0

        steps = max(int(num_steps), 1)
        # Uniform DDIM-like timestep grid.
        times = torch.linspace(
            self.num_timesteps - 1, 0, steps, device=self.device
        ).long()
        traj = []

        for i, t_val in enumerate(times):
            t = torch.full((batch_size,), int(t_val.item()), device=self.device, dtype=torch.long)
            a_bar_t = self.alpha_bars[t]
            if i + 1 < len(times):
                t_prev = int(times[i + 1].item())
                a_bar_prev = self.alpha_bars[t_prev]
            else:
                a_bar_prev = torch.ones((), device=self.device)

            with torch.enable_grad():
                eps = self.predict_noise(x, t, pad_mask=pad_mask)
                x0_hat = self.predict_x0(x, t, pad_mask=pad_mask, eps=eps)
                if guidance and self.num_properties > 0 and guide_scale != 0.0:
                    x0_hat = self._guidance_step(
                        x0_hat,
                        pad_mask=pad_mask,
                        direction=float(direction),
                        guide_scale=float(guide_scale),
                        use_ae=bool(use_ae_proj),
                        a_bar_t=a_bar_t,
                    )
                    # Optional hard AE re-projection after the guided step (MPGD-AE).
                    if use_ae_proj:
                        x0_hat = self.ae_project(x0_hat).clamp(-1.0, 1.0)

            # DDIM-style transition using the *original* noise prediction.
            a_bar_prev_b = a_bar_prev if a_bar_prev.ndim > 0 else a_bar_prev.expand_as(a_bar_t)
            a_bar_prev_b = a_bar_prev_b.view(-1, *([1] * (x.ndim - 1)))
            x = (
                torch.sqrt(a_bar_prev_b) * x0_hat
                + torch.sqrt((1.0 - a_bar_prev_b).clamp(min=0.0)) * eps.detach()
            ).clamp(-1.0, 1.0)
            if return_traj:
                traj.append(x.argmax(dim=-1).detach())

        tokens = x.argmax(dim=-1)
        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        num_steps: int = 50,
        guide_scale: float = 1.0,
        use_ae_proj: bool = True,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided MPGD starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            num_steps=num_steps,
            guidance=True,
            direction=direction,
            guide_scale=guide_scale,
            use_ae_proj=use_ae_proj,
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


class _TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 32, 7
    model = MPGD(vocab_size=vocab_size, seq_len=seq_len, num_timesteps=50, device="cpu")
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, gen, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert gen.ndim == 0 and prop.ndim == 0
    assert torch.isfinite(total)

    generated = model.sample(batch_size=batch_size, num_steps=5, guidance=False)
    assert generated.shape == (batch_size, seq_len)

    optimized = model.optimize(sequences, target_direction="increase", num_steps=5)
    assert optimized.shape == sequences.shape
    print("MPGD unit tests passed.")
