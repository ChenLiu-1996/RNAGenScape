"""RNA-adapted Metric Flow Matching (MFM) baseline.

Official MFM ([kkapusniak/metric-flow-matching](https://github.com/kkapusniak/metric-flow-matching),
NeurIPS 2024, arXiv:2405.14780) learns smooth interpolations between
*populations* on a data manifold: a geopath network bends CondOT paths
with gamma(t)*g_theta(x0,x1,t), an optional data-dependent (LAND/RBF) metric
regularizes path velocity, and a flow network matches the resulting
conditional velocity.

Baseline adaptation (RNA property optimization):
* Partition each training batch into low / high property subpopulations by
  the median ground-truth label (MFM needs two populations).
* Train flows low -> high (maximization) and high -> low (minimization) with
  minibatch OT coupling, MetricFlowMatcher paths (geopath + LAND velocity
  regularizer), and a joint property head (same trainer protocol as EM/MPGD).
* At optimize / sample time, integrate the direction-conditioned velocity
  field from seed (test) sequences.

Intentional simplifications vs the official code: no pytorch-lightning /
torchcfm / torchdyn / RBF metric pretraining / image VAE; CondOT uses
``scipy.optimize.linear_sum_assignment``; LAND metric is applied on flattened
continuous one-hots; geopath + flow train jointly in ``compute_loss``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar/batch times ``t`` to ``x``'s trailing dims."""
    if t.ndim == 0:
        t = t.view(1).expand(x.shape[0])
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


def _land_metric_diag(
    x: torch.Tensor,
    samples: torch.Tensor,
    gamma: float,
    rho: float,
) -> torch.Tensor:
    """Diagonal LAND inverse-metric at ``x`` using reference ``samples``.

    Args:
        x: ``[B, D]``
        samples: ``[N, D]``
        gamma, rho: LAND bandwidth / ridge (official geo_metrics/land.py).

    Returns:
        ``M_inv`` of shape ``[B, D]`` (diagonal inverse metric).
    """
    # weights[b,n] = exp(-||x_b - s_n||^2 / (2 gamma^2))
    pairwise_sq = ((x[:, None, :] - samples[None, :, :]) ** 2).sum(dim=-1)
    weights = torch.exp(-pairwise_sq / (2.0 * gamma * gamma + 1e-12))
    differences = samples[None, :, :] - x[:, None, :]
    squared = differences**2
    m_diag = torch.einsum("bn,bnd->bd", weights, squared) + float(rho)
    return 1.0 / m_diag.clamp(min=1e-8)


class MFM(nn.Module):
    """Metric Flow Matching for RNA sequences (population low <-> high)."""

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 1.0,
        land_gamma: float = 0.5,
        land_rho: float = 1.0,
        geopath_weight: float = 0.1,
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
        self.alpha = float(alpha)
        self.land_gamma = float(land_gamma)
        self.land_rho = float(land_rho)
        self.geopath_weight = float(geopath_weight)
        self.num_properties = int(num_properties)

        # ---- Flow network: v_theta(x_t, t, direction) ----
        self.flow_input = nn.Linear(self.vocab_size, hidden_dim)
        self.flow_pos = _SinusoidalPositionalEncoding(hidden_dim, seq_len + 100)
        self.flow_time = _TimeEmbedding(hidden_dim)
        self.flow_dir = nn.Linear(1, hidden_dim)
        flow_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.flow_backbone = nn.TransformerEncoder(flow_layer, num_layers=num_layers)
        self.flow_norm = nn.LayerNorm(hidden_dim)
        self.flow_head = nn.Linear(hidden_dim, self.vocab_size)

        # ---- Geopath network: g_theta(x0, x1, t) ----
        self.geo_input = nn.Linear(2 * self.vocab_size, hidden_dim)
        self.geo_pos = _SinusoidalPositionalEncoding(hidden_dim, seq_len + 100)
        self.geo_time = _TimeEmbedding(hidden_dim)
        geo_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.geo_backbone = nn.TransformerEncoder(geo_layer, num_layers=max(1, num_layers // 2))
        self.geo_norm = nn.LayerNorm(hidden_dim)
        self.geo_head = nn.Linear(hidden_dim, self.vocab_size)

        # ---- Property head (joint, same as other baselines) ----
        self.prop_pool_proj = nn.Linear(self.vocab_size, hidden_dim)
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

    @staticmethod
    def _tokens_to_continuous(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        return F.one_hot(token_ids.long(), num_classes=vocab_size).float() * 2.0 - 1.0

    @staticmethod
    def _gamma(t: torch.Tensor) -> torch.Tensor:
        """Official gamma(t) on [0,1]: 1 - t^2 - (1-t)^2."""
        return 1.0 - t.square() - (1.0 - t).square()

    @staticmethod
    def _d_gamma(t: torch.Tensor) -> torch.Tensor:
        """d/dt gamma(t) = 2*(-2t + 1) on [0,1]."""
        return 2.0 * (-2.0 * t + 1.0)

    @staticmethod
    def _ot_couple(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Permute ``x0`` to min-cost matching against ``x1`` (flat L2)."""
        with torch.no_grad():
            b = x0.shape[0]
            a = x0.reshape(b, -1)
            c = x1.reshape(b, -1)
            cost = torch.cdist(a, c, p=2).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            inv = np.empty_like(col_ind)
            inv[col_ind] = row_ind
            return x0[torch.as_tensor(inv, device=x0.device)]

    def geopath(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Geopath correction ``g_theta(x0,x1,t)`` with shape ``[B, L, V]``."""
        t = t.reshape(-1).to(dtype=x0.dtype, device=x0.device)
        h = self.geo_input(torch.cat([x0, x1], dim=-1))
        h = self.geo_pos(h) + self.geo_time(t).unsqueeze(1)
        key_padding_mask = None if pad_mask is None else ~pad_mask.bool()
        h = self.geo_backbone(h, src_key_padding_mask=key_padding_mask)
        return self.geo_head(self.geo_norm(h))

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        direction: float,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Direction-conditioned velocity ``v_theta(x_t, t, d)``."""
        t = t.reshape(-1).to(dtype=x_t.dtype, device=x_t.device)
        d = torch.full((x_t.shape[0], 1), float(direction), device=x_t.device, dtype=x_t.dtype)
        h = self.flow_input(x_t)
        h = self.flow_pos(h) + self.flow_time(t).unsqueeze(1) + self.flow_dir(d).unsqueeze(1)
        key_padding_mask = None if pad_mask is None else ~pad_mask.bool()
        h = self.flow_backbone(h, src_key_padding_mask=key_padding_mask)
        return self.flow_head(self.flow_norm(h))

    def predict_property(
        self,
        continuous_x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.prop_pool_proj(continuous_x)
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            w = pad_mask.float().unsqueeze(-1)
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.property_head(pooled)

    def _sample_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (t, x_t, u_t, g) under MetricFlowMatcher (t_min=0, t_max=1)."""
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        t_ = _pad_t_like_x(t, x0)
        g = self.geopath(x0, x1, t, pad_mask=pad_mask)
        gamma = _pad_t_like_x(self._gamma(t), x0)
        d_gamma = _pad_t_like_x(self._d_gamma(t), x0)
        mu = (1.0 - t_) * x0 + t_ * x1 + self.alpha * gamma * g
        # Probabilistic path: optional small isotropic noise (sigma=0 CondOT).
        xt = mu
        ut = (x1 - x0) + self.alpha * d_gamma * g
        return t, xt, ut, g

    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        err = (pred - target).square()
        if pad_mask is None:
            return err.mean()
        m = pad_mask.float().unsqueeze(-1)
        return (err * m).sum() / (m.sum() * pred.shape[-1]).clamp(min=1.0)

    def _land_velocity_loss(
        self,
        xt: torch.Tensor,
        ut: torch.Tensor,
        ref_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared LAND speed of ``ut`` at ``xt`` (geopath regularizer)."""
        xt_f = xt.reshape(xt.shape[0], -1)
        ut_f = ut.reshape(ut.shape[0], -1)
        ref_f = ref_samples.reshape(ref_samples.shape[0], -1)
        # Cap reference size for stability / cost.
        n_ref = min(ref_f.shape[0], 64)
        if ref_f.shape[0] > n_ref:
            idx = torch.randperm(ref_f.shape[0], device=ref_f.device)[:n_ref]
            ref_f = ref_f[idx]
        m_inv = _land_metric_diag(xt_f, ref_f.detach(), self.land_gamma, self.land_rho)
        # Mean over dims (not sum): keeps the regularizer scale-stable for large L*V.
        speed2 = ((ut_f**2) * m_inv).mean(dim=-1)
        return speed2.mean()

    def _partition_populations(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split continuous sequences into low / high property groups."""
        yf = y.float().view(-1)
        med = yf.median()
        low_idx = torch.nonzero(yf <= med, as_tuple=False).view(-1)
        high_idx = torch.nonzero(yf >= med, as_tuple=False).view(-1)
        # Drop median ties from one side if both sides grabbed them.
        if low_idx.numel() + high_idx.numel() > yf.numel():
            low_idx = torch.nonzero(yf < med, as_tuple=False).view(-1)
            high_idx = torch.nonzero(yf > med, as_tuple=False).view(-1)
            eq = torch.nonzero(yf == med, as_tuple=False).view(-1)
            # Assign ties round-robin to balance.
            for i, j in enumerate(eq):
                if i % 2 == 0:
                    low_idx = torch.cat([low_idx, j.view(1)])
                else:
                    high_idx = torch.cat([high_idx, j.view(1)])

        if low_idx.numel() < 1 or high_idx.numel() < 1:
            # Degenerate labels: random half split.
            perm = torch.randperm(x.shape[0], device=x.device)
            mid = max(x.shape[0] // 2, 1)
            low_idx, high_idx = perm[:mid], perm[mid:]
            if high_idx.numel() < 1:
                high_idx = low_idx.clone()

        # Equalize counts for OT (pair min size).
        n = int(min(low_idx.numel(), high_idx.numel()))
        low_idx = low_idx[:n]
        high_idx = high_idx[:n]
        x_low, x_high = x[low_idx], x[high_idx]
        m_low = pad_mask[low_idx] if pad_mask is not None else None
        m_high = pad_mask[high_idx] if pad_mask is not None else None
        return x_low, x_high, m_low, m_high

    def _directional_flow_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        direction: float,
        pad_mask: Optional[torch.Tensor],
        ref_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OT-coupled Metric CFM loss (+ LAND geopath regularizer) for one direction."""
        x0 = self._ot_couple(x0, x1)
        t, xt, ut, _g = self._sample_path(x0, x1, pad_mask)
        vt = self.velocity(xt, t, direction=direction, pad_mask=pad_mask)
        flow_loss = self._masked_mse(vt, ut.detach() if self.geopath_weight > 0 else ut, pad_mask)
        # Geopath should reduce LAND tangential velocity; keep flow target stable.
        if self.alpha != 0.0 and self.geopath_weight > 0.0:
            geo_loss = self._land_velocity_loss(xt, ut, ref_samples)
        else:
            geo_loss = torch.zeros((), device=x0.device)
        return flow_loss, geo_loss

    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
        property_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Population Metric-CFM (both directions) + property MSE.

        Args:
            x_0: token ids ``[B, L]``
            targets: property labels ``[B]`` or ``[B, P]`` (required for split)
            mask: bool pad mask True=valid ``[B, L]``
            recon_weight: weight on generative (flow + geopath) loss
            property_weight: weight on property MSE

        Returns:
            total_loss, gen_loss, property_loss, property_pred ``[B, P]``
        """
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")
        if targets is None:
            raise ValueError("MFM requires property targets to partition populations.")

        x = self._tokens_to_continuous(x_0, self.vocab_size)
        targets = targets.float()
        y_vec = targets.view(targets.shape[0], -1)[:, 0]

        x_low, x_high, m_low, m_high = self._partition_populations(x, y_vec, mask)
        # Shared pad mask for paired minibatches (intersection of valid positions).
        if m_low is not None and m_high is not None:
            pair_mask = m_low & m_high
        else:
            pair_mask = m_low if m_low is not None else m_high

        ref = torch.cat([x_low, x_high], dim=0)
        flow_fwd, geo_fwd = self._directional_flow_loss(
            x_low, x_high, direction=1.0, pad_mask=pair_mask, ref_samples=ref
        )
        flow_bwd, geo_bwd = self._directional_flow_loss(
            x_high, x_low, direction=-1.0, pad_mask=pair_mask, ref_samples=ref
        )
        flow_loss = 0.5 * (flow_fwd + flow_bwd)
        geo_loss = 0.5 * (geo_fwd + geo_bwd)
        gen_loss = flow_loss + self.geopath_weight * geo_loss

        prop_pred = self.predict_property(x, pad_mask=mask)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * gen_loss + property_weight * prop_loss
        return total, gen_loss, prop_loss, prop_pred

    @torch.no_grad()
    def _decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1)

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        direction: float = 1.0,
        num_steps: int = 50,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Euler integrate ``dx/dt = v_theta(x,t,direction)`` on ``t in [0,1]``.

        ``direction=+1`` maximizes property (low->high flow); ``-1`` minimizes.
        """
        if seed_tokens is None:
            x = torch.randn(batch_size, self.seq_len, self.vocab_size, device=self.device)
            if pad_mask is None:
                pad_mask = torch.ones(
                    batch_size, self.seq_len, dtype=torch.bool, device=self.device
                )
        else:
            seed_tokens = seed_tokens.to(self.device)
            batch_size = seed_tokens.shape[0]
            x = self._tokens_to_continuous(seed_tokens, self.vocab_size)
            if pad_mask is None:
                pad_mask = seed_tokens != 0

        dt = 1.0 / max(int(num_steps), 1)
        traj = []
        for i in range(max(int(num_steps), 1)):
            t_val = float(i) * dt
            t = torch.full((batch_size,), t_val, device=self.device, dtype=x.dtype)
            v = self.velocity(x, t, direction=float(direction), pad_mask=pad_mask)
            x = (x + dt * v).clamp(-1.0, 1.0)
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
        num_steps: int = 50,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the population flow starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            direction=direction,
            num_steps=num_steps,
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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding; ``t`` shape ``[B]`` in [0,1]."""
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
        return self.mlp(emb.to(dtype=next(self.parameters()).dtype))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 8, 32, 7
    model = MFM(vocab_size=vocab_size, seq_len=seq_len, device="cpu")
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, gen, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert torch.isfinite(total), total

    generated = model.sample(batch_size=4, num_steps=5, direction=1.0)
    assert generated.shape == (4, seq_len)

    optimized = model.optimize(sequences[:4], target_direction="increase", num_steps=5)
    assert optimized.shape == (4, seq_len)
    print("MFM unit tests passed.")
