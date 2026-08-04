"""RNA-adapted Energy Matching (EM) baseline.

Official Energy Matching ([m1balcerak/EnergyMatching](https://github.com/m1balcerak/EnergyMatching),
NeurIPS 2025, arXiv:2504.10612) learns a *time-independent scalar potential*
``V_theta(x)`` whose negative gradient ``-∇V`` transports noise to data
(OT / flow matching), while contrastive divergence near the data shapes
``V`` into an unnormalized log-likelihood (EBM). Protein inverse design in
the official repo runs EM in a pretrained VAE latent with a separate
fitness CNN for guided sampling.

This RNA baseline keeps the EM recipe on continuous nucleotide one-hots
(PAD/A/G/C/T/U/N), with a small Transformer potential + joint property head
(matching our other comparison trainers). Intentional simplifications vs the
official code: no VAE / UNet1D / torchcfm / multi-GPU CD warm-up schedule;
minibatch OT uses ``scipy.optimize.linear_sum_assignment``; CD is optional
(``lambda_cd``, default 0 = flow-only warm-up).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@contextmanager
def _math_sdp() -> Iterator[None]:
    """Force math SDPA (needed for create_graph through Transformer attention)."""
    cm = None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        cm = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        cm = None
    if cm is not None:
        with cm:
            yield
        return
    # Fallback for older PyTorch: flip CUDA SDP backend flags.
    flash = torch.backends.cuda.flash_sdp_enabled()
    mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_on = torch.backends.cuda.math_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_mem_efficient_sdp(mem)
        torch.backends.cuda.enable_math_sdp(math_on)


class EM(nn.Module):
    """Energy Matching for RNA sequences (scalar potential + property head)."""

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        output_scale: float = 1.0,
        time_cutoff: float = 0.9,
        epsilon_max: float = 0.01,
        lambda_cd: float = 0.0,
        n_gibbs: int = 0,
        dt_gibbs: float = 0.01,
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
        self.output_scale = float(output_scale)
        self.time_cutoff = float(time_cutoff)
        self.epsilon_max = float(epsilon_max)
        self.lambda_cd = float(lambda_cd)
        self.n_gibbs = int(n_gibbs)
        self.dt_gibbs = float(dt_gibbs)
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
        # Scalar potential head V(x) -> (B,).
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
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
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = ~pad_mask.bool()
        h = self.backbone(h, src_key_padding_mask=key_padding_mask)
        return self.output_norm(h)

    def _pool(
        self,
        h: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pad_mask is None:
            return h.mean(dim=1)
        weights = pad_mask.float().unsqueeze(-1)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def potential(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Time-independent scalar potential ``V(x)`` with shape ``[B]``.

        ``t`` is unused (static field); kept so callers can pass a time tensor.
        """
        del t  # time-independent field
        h = self._encode(x, pad_mask=pad_mask)
        v = self.energy_head(self._pool(h, pad_mask=pad_mask)).squeeze(-1)
        return v * self.output_scale

    def velocity(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``-∇_x V(x)`` with the same shape as ``x``."""
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        # Fused SDPA backward has no higher-order grads; math SDPA does.
        with _math_sdp():
            v = self.potential(x_req, t=t, pad_mask=pad_mask)
            grad = torch.autograd.grad(
                outputs=v.sum(),
                inputs=x_req,
                create_graph=True,
                retain_graph=True,
            )[0]
        return -grad

    def predict_property(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property prediction from continuous (possibly noisy) one-hots."""
        h = self._encode(x, pad_mask=pad_mask)
        return self.property_head(self._pool(h, pad_mask=pad_mask))

    @staticmethod
    def _tokens_to_continuous(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Map discrete tokens to continuous targets in ``[-1, 1]``."""
        one_hot = F.one_hot(token_ids.long(), num_classes=vocab_size).float()
        return one_hot * 2.0 - 1.0

    @staticmethod
    def _ot_couple(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Permute ``x0`` via minibatch squared-Euclidean OT to match ``x1``."""
        b = x0.shape[0]
        if b == 1:
            return x0
        flat0 = x0.detach().reshape(b, -1)
        flat1 = x1.detach().reshape(b, -1)
        # Cost[i, j] = ||x0_i - x1_j||^2
        cost = (
            (flat0 ** 2).sum(dim=1, keepdim=True)
            + (flat1 ** 2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat0 @ flat1.T
        )
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        # Map each data index j to noise index i with OT pair (i -> j).
        # We want x0' such that path goes x0'[j] -> x1[j].
        inv = np.empty_like(col_ind)
        inv[col_ind] = row_ind
        return x0[torch.as_tensor(inv, device=x0.device)]

    @staticmethod
    def _flow_weight(t: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Gate flow loss: 1 for t < cutoff, linear decay to 0 at t=1."""
        w = torch.ones_like(t)
        decay = (t >= cutoff) & (t < 1.0)
        w[decay] = 1.0 - (t[decay] - cutoff) / max(1.0 - cutoff, 1e-8)
        w[t >= 1.0] = 0.0
        return w

    def _epsilon(self, t_val: float) -> float:
        """Piecewise temperature schedule used in EM sampling / CD."""
        cutoff = self.time_cutoff
        eps_max = self.epsilon_max
        if t_val < cutoff:
            return 0.0
        if t_val < 1.0:
            return ((t_val - cutoff) / max(1.0 - cutoff, 1e-8)) * eps_max
        return eps_max

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
        property_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy-matching flow loss (+ optional CD) and property MSE.

        Args:
            x_0: token ids ``[B, L]``
            targets: property labels ``[B]`` or ``[B, P]``
            mask: bool pad mask True=valid ``[B, L]``
            recon_weight: weight on generative (flow + CD) loss
            property_weight: weight on property MSE

        Returns:
            total_loss, gen_loss (flow[+CD]), property_loss, property_pred ``[B, P]``
        """
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        x1 = self._tokens_to_continuous(x_0, self.vocab_size)
        x0 = torch.randn_like(x1)
        x0 = self._ot_couple(x0, x1)

        t = torch.rand(x1.shape[0], device=x1.device)
        # Broadcast t for interpolation: x_t = (1-t) x0 + t x1
        t_ = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_) * x0 + t_ * x1
        ut = x1 - x0

        vt = self.velocity(xt, t=t, pad_mask=mask)
        flow_mse = (vt - ut).square()
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            per = (flow_mse * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)).clamp(min=1.0) * self.vocab_size)
        else:
            per = flow_mse.reshape(flow_mse.shape[0], -1).mean(dim=1)
        w = self._flow_weight(t, cutoff=self.time_cutoff)
        flow_loss = (w * per).mean()

        gen_loss = flow_loss
        if self.lambda_cd > 0.0 and self.n_gibbs > 0:
            pos_energy = self.potential(x1, t=torch.ones_like(t), pad_mask=mask)
            n = x1.shape[0]
            half = n // 2
            at_data = torch.zeros(n, dtype=torch.bool, device=x1.device)
            at_data[:half] = True
            at_data = at_data[torch.randperm(n, device=x1.device)]
            x_neg = self._gibbs_time_sweep(x1.detach(), at_data_mask=at_data, pad_mask=mask)
            neg_energy = self.potential(x_neg, t=torch.ones_like(t), pad_mask=mask)
            cd_loss = self.lambda_cd * (pos_energy.mean() - neg_energy.mean())
            gen_loss = flow_loss + cd_loss

        prop_pred = self.predict_property(x1, pad_mask=mask)
        if targets is None:
            prop_loss = torch.zeros((), device=x1.device)
        else:
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * gen_loss + property_weight * prop_loss
        return total, gen_loss, prop_loss, prop_pred

    def _gibbs_time_sweep(
        self,
        x_init: torch.Tensor,
        *,
        at_data_mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MALA negatives with piecewise epsilon(t) (official protein CD)."""
        samples = x_init.clone().detach()
        n_steps = max(int(self.n_gibbs), 1)
        dt = float(self.dt_gibbs)
        for i in range(n_steps):
            t_val = i * dt
            e = torch.zeros(samples.shape[0], device=samples.device, dtype=samples.dtype)
            # Not-at-data: schedule; at-data: epsilon_max.
            scheduled = self._epsilon(t_val)
            e[~at_data_mask] = scheduled
            e[at_data_mask] = self.epsilon_max
            noise_std = torch.sqrt(torch.clamp(2.0 * dt * e, min=0.0))

            samples = samples.detach().requires_grad_(True)
            v = self.potential(samples, pad_mask=pad_mask)
            grad_v = torch.autograd.grad(v.sum(), samples, create_graph=False)[0]
            with torch.no_grad():
                noise = torch.randn_like(samples) * noise_std.view(-1, *([1] * (samples.ndim - 1)))
                samples = (samples - dt * grad_v + noise).clamp(-1.0, 1.0)
        return samples.detach()

    def _property_grad(
        self,
        x: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor],
        direction: float,
        guide_scale: float,
    ) -> torch.Tensor:
        """``direction * guide_scale * ∇_x property(x)``."""
        x_req = x.detach().requires_grad_(True)
        pred = self.predict_property(x_req, pad_mask=pad_mask)
        objective = (direction * pred).sum()
        return guide_scale * torch.autograd.grad(objective, x_req, create_graph=False)[0]

    @torch.no_grad()
    def _decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous ``[-1,1]`` one-hots back to token ids."""
        return x.argmax(dim=-1)

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        t_end: float = 1.0,
        dt: float = 0.01,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Euler–Maruyama sampling of ``dx = -∇V dt + √(2 ε(t) dt) dW``.

        If ``seed_tokens`` is given, start from a noisy continuous embedding of
        those sequences (optimization / local refinement); otherwise from
        Gaussian noise (de novo).
        """
        if seed_tokens is None:
            x = torch.randn(batch_size, self.seq_len, self.vocab_size, device=self.device)
            if pad_mask is None:
                pad_mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        else:
            seed_tokens = seed_tokens.to(self.device)
            batch_size = seed_tokens.shape[0]
            x = self._tokens_to_continuous(seed_tokens, self.vocab_size)
            x = x + 0.1 * torch.randn_like(x)
            if pad_mask is None:
                pad_mask = seed_tokens != 0

        times = torch.arange(0.0, float(t_end) + 1e-8, float(dt), device=self.device)
        traj = []
        for t_val in times:
            e = self._epsilon(float(t_val.item()))
            # Velocity / property grads need autograd even under @torch.no_grad sample.
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                v = self.potential(x_req, pad_mask=pad_mask)
                drift = -torch.autograd.grad(v.sum(), x_req, create_graph=False)[0]
                if guidance and self.num_properties > 0 and guide_scale != 0.0:
                    # Ascend property when direction > 0.
                    drift = drift + self._property_grad(
                        x,
                        pad_mask=pad_mask,
                        direction=float(direction),
                        guide_scale=float(guide_scale),
                    )
            noise = torch.randn_like(x)
            sigma = math.sqrt(max(2.0 * e * float(dt), 0.0))
            x = (x + float(dt) * drift.detach() + sigma * noise).clamp(-1.0, 1.0)
            if return_traj:
                traj.append(self._decode_tokens(x).detach())

        tokens = self._decode_tokens(x)
        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        t_end: float = 1.0,
        dt: float = 0.01,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided EM sampling starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            t_end=t_end,
            dt=dt,
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
    model = EM(vocab_size=vocab_size, seq_len=seq_len, device="cpu", lambda_cd=0.0)
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, gen, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert gen.ndim == 0 and prop.ndim == 0
    assert torch.isfinite(total)

    generated = model.sample(batch_size=batch_size, t_end=0.2, dt=0.05, guidance=False)
    assert generated.shape == (batch_size, seq_len)

    optimized = model.optimize(sequences, target_direction="increase", t_end=0.2, dt=0.05)
    assert optimized.shape == sequences.shape
    print("EM unit tests passed.")
