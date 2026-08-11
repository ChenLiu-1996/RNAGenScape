from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class EM(nn.Module):
    """Energy Matching unifies flow matching and energy-based modeling for generation and inverse design.

    Paper: Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling
    (NeurIPS 2025)
    Github: https://github.com/m1balcerak/EnergyMatching
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
        pot_hidden: int = 512,
        pot_layers: int = 4,
        output_scale: float = 1.0,
        time_cutoff: float = 0.9,
        epsilon_max: float = 0.1,
        lambda_cd: float = 1e-4,
        n_gibbs: int = 200,
        dt_gibbs: float = 0.01,
        cd_clamp: float = 1.0,
        phase1_steps: int = 10000,
        phase2_steps: int = 1000,
        ema_decay: float = 0.999,
        ema_decay_cd: float = 0.99,
        tau_s: float = 1.7,
        zeta: float = 0.03,
        num_properties: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        del hidden_dim, num_layers, num_heads, dropout, num_properties  # API compat
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.output_scale = float(output_scale)
        self.time_cutoff = float(time_cutoff)
        self.epsilon_max = float(epsilon_max)
        self.lambda_cd = float(lambda_cd)
        self.n_gibbs = int(n_gibbs)
        self.dt_gibbs = float(dt_gibbs)
        self.cd_clamp = float(cd_clamp)
        self.phase1_steps = int(phase1_steps)
        self.phase2_steps = int(phase2_steps)
        self.ema_decay = float(ema_decay)
        self.ema_decay_cd = float(ema_decay_cd)
        self.tau_s = float(tau_s)
        self.zeta = float(zeta)

        self.ae = _ConvAE(vocab_size=self.vocab_size, length=self.seq_len, latent_dim=self.latent_dim)
        self.predictor = _BaseCNN(n_tokens=self.vocab_size)
        self.potential = _Potential(
            dim=self.latent_dim,
            hidden=pot_hidden,
            n_layers=pot_layers,
            output_scale=self.output_scale,
        )
        # EMA copy used at sampling time.
        self.ema_potential = copy.deepcopy(self.potential)
        for p in self.ema_potential.parameters():
            p.requires_grad_(False)

        self.register_buffer("z_mu", torch.zeros(self.latent_dim))
        self.register_buffer("z_sd", torch.ones(self.latent_dim))
        self.register_buffer("_stats_count", torch.zeros((), dtype=torch.long))
        # Predictor label range in the same space as ``targets`` (usually z-scored).
        self.register_buffer("y_lo", torch.zeros(()))
        self.register_buffer("y_hi", torch.ones(()))
        self.register_buffer("_y_count", torch.zeros((), dtype=torch.long))

        # Fixed train latent pool (set by build_train_latent_pool).
        self._pool_z: Optional[torch.Tensor] = None

        self.to(self.device)

    # ------------------------------------------------------------------ stats / pool / EMA
    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mu) / self.z_sd.clamp_min(1e-6)

    def _unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.z_sd.clamp_min(1e-6) + self.z_mu

    @torch.no_grad()
    def build_train_latent_pool(self, loader, *, to_tokens) -> Dict[str, float]:
        """Encode the full training loader once; set latent stats and label range.

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
        self._pool_z = z_norm

        y_lo = float(labels.min().item())
        y_hi = float(labels.max().item())
        self.y_lo.fill_(y_lo)
        self.y_hi.fill_(y_hi)
        self._y_count.fill_(labels.numel())
        return {
            "n_train": float(latents.shape[0]),
            "y_lo": y_lo,
            "y_hi": y_hi,
        }

    def has_latent_pool(self) -> bool:
        return self._pool_z is not None and self._pool_z.shape[0] >= 2

    def _sample_pool(self, batch_size: int) -> torch.Tensor:
        if not self.has_latent_pool():
            raise RuntimeError("EM requires build_train_latent_pool() first.")
        assert self._pool_z is not None
        n = self._pool_z.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self._pool_z.device)
        return self._pool_z[idx]

    def _ema_update(self, *, phase2: bool) -> None:
        decay = self.ema_decay_cd if phase2 else self.ema_decay
        with torch.no_grad():
            for ps, pt in zip(self.potential.parameters(), self.ema_potential.parameters()):
                pt.copy_(pt * decay + ps * (1.0 - decay))

    # ------------------------------------------------------------------ OT / schedules
    @staticmethod
    def _ot_couple(x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact minibatch OT coupling."""
        with torch.no_grad():
            if x0.shape[0] == 1:
                return x0, x1
            cost = torch.cdist(x0, x1, p=2).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            return (
                x0[torch.as_tensor(row_ind, device=x0.device)],
                x1[torch.as_tensor(col_ind, device=x1.device)],
            )

    @staticmethod
    def _flow_weight(t: torch.Tensor, cutoff: float) -> torch.Tensor:
        w = torch.ones_like(t)
        decay = (t >= cutoff) & (t < 1.0)
        w[decay] = 1.0 - (t[decay] - cutoff) / (1.0 - cutoff)
        w[t >= 1.0] = 0.0
        return w

    def _epsilon(self, t_val: float) -> float:
        cutoff = self.time_cutoff
        eps_max = self.epsilon_max
        if t_val < cutoff:
            return 0.0
        if t_val < 1.0:
            return ((t_val - cutoff) / (1.0 - cutoff)) * eps_max
        return eps_max

    def _langevin_negatives(
        self,
        x_init: torch.Tensor,
        at_data_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alg. 2 negatives: half data (eps=eps_max), half noise (eps(t)). No latent clamp."""
        x = x_init.clone().detach()
        n_steps = max(int(self.n_gibbs), 1)
        dt = float(self.dt_gibbs)
        at_data_mask = at_data_mask.to(device=x.device, dtype=torch.bool)
        for i in range(n_steps):
            t_val = i * dt
            e_noise = self._epsilon(t_val)
            e = torch.where(
                at_data_mask,
                torch.full((), self.epsilon_max, device=x.device, dtype=x.dtype),
                torch.full((), e_noise, device=x.device, dtype=x.dtype),
            )
            x = x.detach().requires_grad_(True)
            v = self.potential.potential(x)
            grad_v = torch.autograd.grad(v.sum(), x, create_graph=False)[0]
            with torch.no_grad():
                noise = torch.randn_like(x) * torch.sqrt(2.0 * dt * e).view(-1, 1)
                x = x - dt * grad_v + noise
        return x.detach()

    # ------------------------------------------------------------------ stage losses
    def ae_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _z = self.ae(tokens)
        return F.cross_entropy(logits, tokens)

    def predictor_loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MSE on labels; returns (loss, pred [B, 1])."""
        y_vec = targets.float().view(tokens.shape[0], -1)[:, 0]
        pred = self.predictor(tokens).view(tokens.shape[0], 1)
        return F.mse_loss(pred.view(-1), y_vec), pred

    def potential_loss(
        self,
        *,
        batch_size: int,
        train_cd: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """OT (+ optional CD) on the fixed train latent pool. Returns ot, cd, total."""
        if not self.has_latent_pool():
            raise RuntimeError("EM potential_loss requires build_train_latent_pool() first.")
        x1 = self._sample_pool(batch_size)
        x0 = torch.randn_like(x1)
        x0, x1 = self._ot_couple(x0, x1)

        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t_ = t.view(-1, 1)
        xt = (1.0 - t_) * x0 + t_ * x1
        ut = x1 - x0
        vt = self.potential.velocity(xt)
        per = (vt - ut).square().mean(dim=-1)
        ot_loss = (self._flow_weight(t, self.time_cutoff) * per).mean()

        cd_loss = torch.zeros((), device=x1.device)
        if train_cd and self.lambda_cd > 0.0 and self.n_gibbs > 0:
            half = x1.shape[0] // 2
            at_data = torch.zeros(x1.shape[0], dtype=torch.bool, device=x1.device)
            at_data[:half] = True
            x_init = x1.clone()
            if half < x1.shape[0]:
                x_init[half:] = torch.randn_like(x1[half:])
            x_neg = self._langevin_negatives(x_init, at_data)
            cd = self.potential.potential(x1).mean() - self.potential.potential(x_neg).mean()
            cd_loss = torch.clamp(cd, min=-self.cd_clamp)

        total = ot_loss + self.lambda_cd * cd_loss
        return ot_loss, cd_loss, total

    # ------------------------------------------------------------------ public API
    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eval helper: AE CE + predictor MSE (+ OT on pool if built)."""
        del mask
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        b = x_0.shape[0]
        ae_loss = self.ae_loss(x_0)
        if targets is None:
            prop_pred = torch.zeros(b, 1, device=x_0.device)
            prop_loss = torch.zeros((), device=x_0.device)
        else:
            prop_loss, prop_pred = self.predictor_loss(x_0, targets)

        flow_loss = torch.zeros((), device=x_0.device)
        if self.has_latent_pool():
            # OT only; CD Langevin requires grads.
            flow_loss, _cd, _ = self.potential_loss(batch_size=b, train_cd=False)

        gen_loss = ae_loss + flow_loss
        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        t_end: Optional[float] = None,
        dt: float = 0.01,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        zeta: Optional[float] = None,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Alg. 3: data-seeded Langevin on U = V + (eps/zeta^2)*0.5*(ghat-y*)^2 with eps=eps_max."""
        del pad_mask
        if seed_tokens is None:
            raise ValueError("EM latent sampling requires seed_tokens (data-initialised Alg. 3).")
        seed_tokens = seed_tokens.to(self.device)
        batch_size = seed_tokens.shape[0]
        tau_s = float(self.tau_s if t_end is None else t_end)
        dt = float(dt)
        n_steps = max(int(round(tau_s / dt)), 1)
        zeta = float(self.zeta if zeta is None else zeta)
        y_target = 1.0 if float(direction) >= 0.0 else 0.0
        y_lo = float(self.y_lo.item())
        y_hi = float(self.y_hi.item())
        if abs(y_hi - y_lo) < 1e-8:
            y_lo, y_hi = 0.0, 1.0

        self.ae.eval()
        self.predictor.eval()
        pot = self.ema_potential
        pot.eval()

        z = self._standardize(self.ae.encode(seed_tokens))
        e = float(self.epsilon_max)
        noise_scale = math.sqrt(max(2.0 * e * dt, 0.0))
        traj = []
        for _ in range(n_steps):
            if return_traj:
                ids = self.ae.decode(self._unstandardize(z)).argmax(dim=1)
                traj.append(ids.detach())
            z = z.detach().requires_grad_(True)
            v = pot.potential(z).sum()
            logits = self.ae.decode(self._unstandardize(z)).transpose(1, 2)  # [B, L, V]
            y_hat = self.predictor.forward_soft(logits)
            y01 = (y_hat - y_lo) / (y_hi - y_lo)
            lik = 0.5 * ((y01 - y_target) ** 2).sum()
            if guidance and guide_scale != 0.0 and zeta > 0.0:
                u = v + float(guide_scale) * (e / (zeta ** 2)) * lik
            else:
                u = v
            g = torch.autograd.grad(u, z, create_graph=False)[0]
            with torch.no_grad():
                z = z - dt * g + noise_scale * torch.randn_like(z)

        tokens = self.ae.decode(self._unstandardize(z.detach())).argmax(dim=1)
        if return_traj:
            traj.append(tokens.detach())
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        t_end: Optional[float] = None,
        dt: float = 0.01,
        guide_scale: float = 1.0,
        zeta: Optional[float] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided Alg. 3 from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            t_end=t_end,
            dt=dt,
            guidance=True,
            direction=direction,
            guide_scale=guide_scale,
            zeta=zeta,
            pad_mask=pad_mask,
            return_traj=False,
        )


# --------------------------------------------------------------------------- submodules
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


class _Potential(nn.Module):
    """Time-independent SiLU MLP potential over latents."""

    def __init__(self, dim: int, hidden: int = 512, n_layers: int = 4, output_scale: float = 1.0):
        super().__init__()
        layers = []
        d = dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.output_scale = float(output_scale)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).sum(-1) * self.output_scale

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            v = self.potential(x)
            g = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        return -g


class _LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(F.relu(self.layer(x)), dim=1)[0]


class _BaseCNN(nn.Module):
    """Conv1d property predictor (Kirjner et al. BaseCNN)."""

    def __init__(self, n_tokens: int = 7, kernel_size: int = 5, input_size: int = 256):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = _LengthMaxPool1D(in_dim=input_size, out_dim=input_size * 2)
        self.decoder = nn.Linear(input_size * 2, 1)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, L]
        x = self.encoder(x).permute(0, 2, 1)
        return self.decoder(self.embedding(x)).squeeze(-1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(ids.long(), self.n_tokens).float().permute(0, 2, 1)
        return self._head(oh)

    def forward_soft(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, L, V]
        return self._head(torch.softmax(logits, dim=-1).permute(0, 2, 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 8, 32, 7
    model = EM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        device="cpu",
        n_gibbs=2,
        phase1_steps=2,
        phase2_steps=2,
        tau_s=0.05,
    )
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)

    class _Loader:
        def __iter__(self):
            yield sequences, targets.view(-1)

    info = model.build_train_latent_pool(_Loader(), to_tokens=lambda x: x)
    assert info["n_train"] == float(batch_size)
    assert model.has_latent_pool()
    ae = model.ae_loss(sequences)
    prop, pred = model.predictor_loss(sequences, targets)
    ot, cd, total = model.potential_loss(batch_size=batch_size, train_cd=True)
    assert ae.ndim == 0 and prop.ndim == 0 and total.ndim == 0 and pred.shape == (batch_size, 1)
    model._ema_update(phase2=True)

    model.eval()
    out = model.optimize(sequences[:4], target_direction="increase", t_end=0.05, dt=0.01)
    assert out.shape == (4, seq_len)
    out_neg = model.optimize(sequences[:4], target_direction="decrease", t_end=0.05, dt=0.01)
    assert out_neg.shape == (4, seq_len)
    print("EM unit tests passed.")
