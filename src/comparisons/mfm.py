from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MFM(nn.Module):
    """MFM transports sequences along property tertiles via latent metric flow matching.

    Paper: Metric Flow Matching for Smooth Interpolations on the Data Manifold (NeurIPS 2024)
    Github: https://github.com/kkapusniak/metric-flow-matching
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 1.0,
        land_gamma: float = -1.0,
        land_rho: float = 1e-2,
        geopath_weight: float = 1.0,
        n_metric_samples: int = 4096,
        q_lo: float = 1.0 / 3.0,
        q_hi: float = 2.0 / 3.0,
        num_properties: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        del num_layers, num_heads, dropout, num_properties  # API compat with other baselines
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.land_gamma = float(land_gamma)  # <0 => auto from train latents
        self.land_rho = float(land_rho)
        self.geopath_weight = float(geopath_weight)
        self.n_metric_samples = int(n_metric_samples)
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self.use_geopath = self.alpha != 0.0

        self.ae = _ConvAE(vocab_size=self.vocab_size, length=self.seq_len, latent_dim=self.latent_dim)
        # Separate flow + geopath per direction.
        self.flow_pos = _VelocityNet(self.latent_dim, hidden_dims=(256, 256, 256))
        self.flow_neg = _VelocityNet(self.latent_dim, hidden_dims=(256, 256, 256))
        if self.use_geopath:
            self.geopath_pos = _GeoPathMLP(self.latent_dim, hidden_dims=(256, 256))
            self.geopath_neg = _GeoPathMLP(self.latent_dim, hidden_dims=(256, 256))
        else:
            self.geopath_pos = None
            self.geopath_neg = None

        # Latent standardization from full train encodings.
        self.register_buffer("z_mu", torch.zeros(self.latent_dim))
        self.register_buffer("z_sd", torch.ones(self.latent_dim))
        self.register_buffer("_stats_count", torch.zeros((), dtype=torch.long))

        # Fixed train latent pool / tertile frames (set by build_train_latent_pool).
        self._pool_z: Optional[torch.Tensor] = None
        self._frame_low: Optional[torch.Tensor] = None
        self._frame_mid: Optional[torch.Tensor] = None
        self._frame_high: Optional[torch.Tensor] = None
        self._land_gamma_resolved: Optional[float] = None

        self.to(self.device)

    # ------------------------------------------------------------------ AE / stats / pool
    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mu) / self.z_sd.clamp_min(1e-6)

    def _unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.z_sd.clamp_min(1e-6) + self.z_mu

    def _auto_land_gamma(self, latents: torch.Tensor, n_probe: int = 1024) -> float:
        n = latents.shape[0]
        idx = torch.randperm(n, device=latents.device)[: min(n_probe, n)]
        d = torch.cdist(latents[idx], latents[idx])
        iu = torch.triu_indices(d.size(0), d.size(0), offset=1, device=latents.device)
        return float(d[iu[0], iu[1]].median().item()) * 0.5

    def _partition_frames(
        self, latents: torch.Tensor, labels: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Tertile split of latents by property labels."""
        if latents.shape[0] < 6:
            return None
        y = labels.detach().float().cpu().numpy()
        lo, hi = float(np.quantile(y, self.q_lo)), float(np.quantile(y, self.q_hi))
        mask_low = y <= lo
        mask_mid = (y > lo) & (y <= hi)
        mask_high = y > hi
        if mask_low.sum() < 1 or mask_mid.sum() < 1 or mask_high.sum() < 1:
            return None
        return latents[mask_low], latents[mask_mid], latents[mask_high]

    @torch.no_grad()
    def build_train_latent_pool(self, loader, *, to_tokens) -> Dict[str, float]:
        """Encode the full training loader once; set latent stats and tertile frames.

        ``to_tokens`` maps a batch ``x`` tensor to token ids ``[B, L]``.
        """
        self.ae.eval()
        zs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for x, y in loader:
            tokens = to_tokens(x).to(self.device)
            z = self.ae.encode(tokens)
            zs.append(z.detach().cpu())
            ys.append(y.detach().float().view(-1).cpu())
        latents = torch.cat(zs, dim=0)
        labels = torch.cat(ys, dim=0)
        z_mu = latents.mean(dim=0)
        z_sd = latents.std(dim=0).clamp_min(1e-6)
        self.z_mu.copy_(z_mu.to(self.device))
        self.z_sd.copy_(z_sd.to(self.device))
        self._stats_count.fill_(latents.shape[0])

        z_norm = ((latents - z_mu) / z_sd).to(self.device)
        labels_d = labels.to(self.device)
        frames = self._partition_frames(z_norm, labels_d)
        if frames is None:
            raise RuntimeError("MFM tertile partition failed on the full train latent pool.")
        self._frame_low, self._frame_mid, self._frame_high = frames
        self._pool_z = z_norm
        if self.land_gamma > 0:
            self._land_gamma_resolved = float(self.land_gamma)
        else:
            self._land_gamma_resolved = self._auto_land_gamma(z_norm)
        return {
            "n_train": float(latents.shape[0]),
            "n_low": float(self._frame_low.shape[0]),
            "n_mid": float(self._frame_mid.shape[0]),
            "n_high": float(self._frame_high.shape[0]),
            "land_gamma": float(self._land_gamma_resolved),
        }

    def has_latent_pool(self) -> bool:
        return (
            self._pool_z is not None
            and self._frame_low is not None
            and self._frame_mid is not None
            and self._frame_high is not None
        )

    @staticmethod
    def _sample_frame(frame: torch.Tensor, batch_size: int) -> torch.Tensor:
        n = frame.shape[0]
        if n <= 0:
            raise ValueError("empty frame")
        if n <= batch_size:
            idx = torch.randint(0, n, (batch_size,), device=frame.device)
        else:
            idx = torch.randperm(n, device=frame.device)[:batch_size]
        return frame[idx]

    @staticmethod
    def _ot_couple(x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact OT minibatch plan."""
        with torch.no_grad():
            cost = torch.cdist(x0, x1, p=2).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            # torchcfm sample_plan returns reordered (x0[i], x1[pi(i)]); keep both aligned.
            return x0[torch.as_tensor(row_ind, device=x0.device)], x1[
                torch.as_tensor(col_ind, device=x1.device)
            ]

    # ------------------------------------------------------------------ Metric path
    @staticmethod
    def _gamma(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
        span = max(t_max - t_min, 1e-8)
        return 1.0 - ((t - t_min) / span) ** 2 - ((t_max - t) / span) ** 2

    @staticmethod
    def _d_gamma(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
        span = max(t_max - t_min, 1e-8)
        return 2.0 * (-2.0 * t + t_max + t_min) / (span**2)

    def _sample_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        geopath: Optional[nn.Module],
        t_min: float,
        t_max: float,
        *,
        training_geopath: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(t, x_t, u_t)`` on ``[t_min, t_max]`` (sigma=0 CondOT)."""
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, dtype=x0.dtype) * (t_max - t_min) + t_min
        t_ = _pad_t_like_x(t, x0)
        span = max(t_max - t_min, 1e-8)
        mu = ((t_max - t_) / span) * x0 + ((t_ - t_min) / span) * x1
        ut = (x1 - x0) / span
        if self.use_geopath and geopath is not None:
            g = geopath(x0, x1, t)
            if training_geopath:
                # LAND objective needs grads through g -> ut.
                pass
            gamma = _pad_t_like_x(self._gamma(t, t_min, t_max), x0)
            d_gamma = _pad_t_like_x(self._d_gamma(t, t_min, t_max), x0)
            mu = mu + gamma * g
            ut = ut + d_gamma * g
        return t, mu, ut

    def _land_loss(
        self,
        xt: torch.Tensor,
        ut: torch.Tensor,
        ref_samples: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Mean LAND tangential energy ``sum_d u_d^2 / M_dd``."""
        n_ref = min(ref_samples.shape[0], self.n_metric_samples)
        ref = ref_samples
        if ref.shape[0] > n_ref:
            idx = torch.randperm(ref.shape[0], device=ref.device)[:n_ref]
            ref = ref[idx]
        m_inv = _land_metric_diag(xt, ref.detach(), gamma, self.land_rho)
        return ((ut**2) * m_inv).sum(dim=-1).mean()

    def _pair_losses(
        self,
        frames_in_order: Sequence[torch.Tensor],
        *,
        flow_net: nn.Module,
        geopath: Optional[nn.Module],
        ref_samples: torch.Tensor,
        batch_size: int,
        land_gamma: float,
        train_geopath: bool = True,
        train_flow: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjacent-frame OT MFM losses for one direction."""
        timesteps = torch.linspace(0.0, 1.0, len(frames_in_order)).tolist()
        flow_losses: List[torch.Tensor] = []
        geo_losses: List[torch.Tensor] = []
        for i in range(len(frames_in_order) - 1):
            x0 = self._sample_frame(frames_in_order[i], batch_size)
            x1 = self._sample_frame(frames_in_order[i + 1], batch_size)
            x0, x1 = self._ot_couple(x0, x1)
            t_min, t_max = float(timesteps[i]), float(timesteps[i + 1])

            if train_geopath and self.use_geopath and geopath is not None and self.geopath_weight > 0.0:
                _t, xt_g, ut_g = self._sample_path(
                    x0, x1, geopath, t_min, t_max, training_geopath=True
                )
                geo_losses.append(self._land_loss(xt_g, ut_g, ref_samples, land_gamma))

            if train_flow:
                t, xt, ut = self._sample_path(
                    x0, x1, geopath, t_min, t_max, training_geopath=False
                )
                if self.use_geopath and geopath is not None:
                    ut = ut.detach()
                    xt = xt.detach()
                vt = flow_net(t, xt)
                flow_losses.append(F.mse_loss(vt, ut))

        if flow_losses:
            flow = sum(flow_losses) / len(flow_losses)
        else:
            flow = torch.zeros((), device=ref_samples.device)
        if geo_losses:
            geo = sum(geo_losses) / len(geo_losses)
        else:
            geo = torch.zeros((), device=ref_samples.device)
        return flow, geo

    def ae_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _z = self.ae(tokens)
        return F.cross_entropy(logits, tokens)

    def d2d_loss(
        self,
        *,
        batch_size: int,
        train_geopath: bool,
        train_flow: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flow/geopath losses on the fixed full-train tertile pool. Returns flow, geo, total."""
        if not self.has_latent_pool():
            raise RuntimeError("MFM d2d_loss requires build_train_latent_pool() first.")
        assert self._pool_z is not None and self._land_gamma_resolved is not None
        assert self._frame_low is not None and self._frame_mid is not None and self._frame_high is not None
        land_gamma = float(self._land_gamma_resolved)
        pair_bs = min(int(batch_size), 128)
        flow_pos, geo_pos = self._pair_losses(
            [self._frame_low, self._frame_mid, self._frame_high],
            flow_net=self.flow_pos,
            geopath=self.geopath_pos,
            ref_samples=self._pool_z,
            batch_size=pair_bs,
            land_gamma=land_gamma,
            train_geopath=train_geopath,
            train_flow=train_flow,
        )
        flow_neg, geo_neg = self._pair_losses(
            [self._frame_high, self._frame_mid, self._frame_low],
            flow_net=self.flow_neg,
            geopath=self.geopath_neg,
            ref_samples=self._pool_z,
            batch_size=pair_bs,
            land_gamma=land_gamma,
            train_geopath=train_geopath,
            train_flow=train_flow,
        )
        flow_loss = 0.5 * (flow_pos + flow_neg)
        geo_loss = 0.5 * (geo_pos + geo_neg)
        total = flow_loss + self.geopath_weight * geo_loss
        return flow_loss, geo_loss, total

    # ------------------------------------------------------------------ public API
    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eval helper: AE CE (+ d2d loss if the train latent pool is built). Property loss is 0."""
        del mask, targets
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")
        b = x_0.shape[0]
        ae_loss = self.ae_loss(x_0)
        flow_loss = torch.zeros((), device=x_0.device)
        geo_loss = torch.zeros((), device=x_0.device)
        if self.has_latent_pool():
            flow_loss, geo_loss, _ = self.d2d_loss(
                batch_size=b, train_geopath=True, train_flow=True
            )
        gen_loss = ae_loss + flow_loss + self.geopath_weight * geo_loss
        prop_loss = torch.zeros((), device=x_0.device)
        prop_pred = torch.zeros(b, 1, device=x_0.device)
        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        direction: float = 1.0,
        num_steps: int = 100,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Encode -> standardize -> Euler 0->1 with direction flow -> decode."""
        del pad_mask
        if seed_tokens is None:
            raise ValueError("MFM d2d sampling requires seed_tokens (data-to-data).")
        seed_tokens = seed_tokens.to(self.device)
        batch_size = seed_tokens.shape[0]
        flow_net = self.flow_pos if float(direction) >= 0.0 else self.flow_neg
        flow_net.eval()
        self.ae.eval()

        z = self._standardize(self.ae.encode(seed_tokens))
        dt = 1.0 / max(int(num_steps), 1)
        traj = []
        for i in range(max(int(num_steps), 1)):
            t = torch.full((batch_size,), float(i) * dt, device=self.device, dtype=z.dtype)
            v = flow_net(t, z)
            z = z + dt * v
            if return_traj:
                ids = self.ae.decode(self._unstandardize(z)).argmax(dim=1)
                traj.append(ids.detach())

        tokens = self.ae.decode(self._unstandardize(z)).argmax(dim=1)
        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        num_steps: int = 100,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the direction-specific latent flow from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            direction=direction,
            num_steps=num_steps,
            pad_mask=pad_mask,
            return_traj=False,
        )


# --------------------------------------------------------------------------- helpers / submodules
class _ConvAE(nn.Module):
    """1D-conv autoencoder over token sequences."""

    def __init__(self, vocab_size: int = 7, length: int = 150, latent_dim: int = 64):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.length = int(length)
        self.latent_dim = int(latent_dim)
        self.enc_conv = nn.Sequential(
            nn.Conv1d(vocab_size, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        enc_len = (length + 1) // 2
        enc_len = (enc_len + 1) // 2
        self.enc_len = enc_len
        self.enc_proj = nn.Linear(64 * enc_len, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, 64 * enc_len)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, vocab_size, 3, stride=2, padding=1, output_padding=0),
        )

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(ids.long(), num_classes=self.vocab_size).float().transpose(1, 2)
        return self.enc_proj(self.enc_conv(onehot).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(z).view(z.size(0), 64, self.enc_len)
        logits = self.dec_conv(h)
        if logits.size(-1) < self.length:
            logits = F.pad(logits, (0, self.length - logits.size(-1)))
        elif logits.size(-1) > self.length:
            logits = logits[..., : self.length]
        return logits

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(ids)
        return self.decode(z), z


class _SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "silu",
    ):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.GELU
        dims = [input_size, *list(hidden_dims), target_size]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _VelocityNet(nn.Module):
    """Zip ``VelocityNet``: MLP on ``concat(t, z)``."""

    def __init__(self, dim: int, hidden_dims: Sequence[int] = (256, 256, 256)):
        super().__init__()
        self.net = _SimpleDenseNet(dim + 1, dim, hidden_dims=hidden_dims)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        return self.net(torch.cat([t, x], dim=-1))


class _GeoPathMLP(nn.Module):
    """Zip ``GeoPathMLP`` with ``time_geopath=False``."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _SimpleDenseNet(2 * input_dim, input_dim, hidden_dims=hidden_dims)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(torch.cat([x0, x1], dim=-1))


def _pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
    """Diagonal LAND inverse-metric."""
    pairwise_sq = ((x[:, None, :] - samples[None, :, :]) ** 2).sum(dim=-1)
    weights = torch.exp(-pairwise_sq / (2.0 * gamma * gamma + 1e-12))
    differences = samples[None, :, :] - x[:, None, :]
    m_diag = torch.einsum("bn,bnd->bd", weights, differences**2) + float(rho)
    return 1.0 / m_diag.clamp(min=1e-8)


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, seq_len, b = 7, 48, 32
    model = MFM(vocab_size=vocab_size, seq_len=seq_len, device="cpu")
    tokens = torch.randint(1, vocab_size, (b, seq_len))
    y = torch.linspace(-1, 1, b)
    # Fake full-train pool: encode once then set frames via build_train_latent_pool API.
    class _Loader:
        def __iter__(self):
            yield tokens, y

    info = model.build_train_latent_pool(_Loader(), to_tokens=lambda x: x)
    assert info["n_train"] == float(b)
    assert model.has_latent_pool()
    ae = model.ae_loss(tokens)
    flow, geo, total = model.d2d_loss(batch_size=16, train_geopath=True, train_flow=True)
    assert ae.ndim == 0 and total.ndim == 0
    out = model.optimize(tokens[:4], target_direction="increase", num_steps=5)
    assert out.shape == (4, seq_len)
    out_neg = model.optimize(tokens[:4], target_direction="decrease", num_steps=5)
    assert out_neg.shape == (4, seq_len)
    print("MFM unit tests passed.")
